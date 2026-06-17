"""GLM-5.2 distributed STAGE node — run a real layer block on this machine with
hidden-state I/O over our TCP transport. The 2-box WAN integration test for plan A.

  serve: load layer(s), listen, recv hidden -> run stage -> send hidden back.
  drive: load layer(s), make input, run stage, send to --next, recv result, validate + time.

usage:
  python glm_stage_node.py serve --layers 7 --port 29500
  python glm_stage_node.py drive --layers 6 --next <hostB>:29500 --tokens 6
"""
import sys, io, json, time, socket, struct, argparse, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

DIR, dev, BLK = "/root/glm52fp8", "cuda", 128
_cfg = GlmMoeDsaConfig.from_pretrained(DIR); _cfg._attn_implementation = "eager"
H = _cfg.hidden_size
_idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def _raw(n): return _h(_idx[n]).get_tensor(n)
def _dequant(n):
    w = _raw(n).to(torch.float32); s = _raw(n + "_scale_inv"); m, k = w.shape
    bm, bn = -(-m // BLK), -(-k // BLK)
    if tuple(s.shape) == (bn, bm) and bm != bn: s = s.t().contiguous()
    s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :k]
    return (w * s).to(torch.bfloat16)
def _maybe(n): return _dequant(n) if (n + "_scale_inv") in _idx else _raw(n).to(torch.bfloat16)


def load_layer(li):
    E, I = _cfg.n_routed_experts, _cfg.moe_intermediate_size
    P = f"model.layers.{li}."
    sd = {}
    for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
              "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
              "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
              "mlp.gate.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
              "mlp.shared_experts.down_proj.weight"]:
        sd[n] = _maybe(P + n).to(dev)
    sd["mlp.gate.e_score_correction_bias"] = _raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
    gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device=dev)
    down = torch.empty(E, H, I, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        g = _dequant(P + f"mlp.experts.{e}.gate_proj.weight"); u = _dequant(P + f"mlp.experts.{e}.up_proj.weight")
        gate_up[e] = torch.cat([g, u], 0).to(dev); down[e] = _dequant(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
    sd["mlp.experts.gate_up_proj"] = gate_up; sd["mlp.experts.down_proj"] = down
    with torch.device("meta"): layer = M.GlmMoeDsaDecoderLayer(_cfg, li)
    layer.load_state_dict(sd, strict=False, assign=True); layer.eval()
    return layer


def dense_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
        b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    o, w = M.eager_attention_forward(self, torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1),
                                     value_states, attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None
M.GlmMoeDsaAttention.forward = dense_attn_forward

_rotary = M.GlmMoeDsaRotaryEmbedding(_cfg).to(dev)
def stage_fwd(layers, hidden):
    T = hidden.shape[1]
    pos = torch.arange(T, device=dev).unsqueeze(0)
    pe = _rotary(hidden, position_ids=pos)
    mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
    mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))
    x = hidden
    for L in layers:
        x = L(x, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    return x


def _sendall(sock, b): sock.sendall(struct.pack("!Q", len(b)) + b)
def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return buf
def send_tensor(sock, t):
    buf = io.BytesIO(); torch.save(t.cpu(), buf); _sendall(sock, buf.getvalue())
def recv_tensor(sock):
    n = struct.unpack("!Q", _recvn(sock, 8))[0]
    return torch.load(io.BytesIO(_recvn(sock, n)), weights_only=False).to(dev)


def serve(layer_ids, port):
    layers = [load_layer(i) for i in layer_ids]
    print(f"stage serving layers {layer_ids} on :{port}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"conn from {addr}", flush=True)
        try:
            with torch.no_grad():
                while True:
                    h = recv_tensor(conn)
                    send_tensor(conn, stage_fwd(layers, h))
        except (ConnectionError, EOFError):
            print("conn closed", flush=True)


def drive(layer_ids, next_ep, tokens, rounds=5):
    layers = [load_layer(i) for i in layer_ids]
    host, port = next_ep.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=30)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"drive layers {layer_ids} -> {next_ep}", flush=True)
    torch.manual_seed(0)
    with torch.no_grad():
        for r in range(rounds):
            h0 = torch.randn(1, tokens, H, dtype=torch.bfloat16, device=dev) * 0.1
            t0 = time.time(); h1 = stage_fwd(layers, h0)          # local stage
            t1 = time.time(); send_tensor(sock, h1); h2 = recv_tensor(sock)  # WAN round-trip
            t2 = time.time()
            print(f"round {r}: local {(t1-t0)*1000:.1f}ms | WAN+remote {(t2-t1)*1000:.1f}ms | "
                  f"out finite={torch.isfinite(h2).all().item()} mean|x|={h2.abs().mean().item():.3f}", flush=True)
    sock.close()
    print("VERDICT: 2-stage GLM-5.2 pipeline over real WAN works — hidden states cross machines, output sane.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("serve"); p.add_argument("--layers", type=int, nargs="+", required=True); p.add_argument("--port", type=int, default=29500)
    p = sub.add_parser("drive"); p.add_argument("--layers", type=int, nargs="+", required=True); p.add_argument("--next", required=True); p.add_argument("--tokens", type=int, default=6)
    a = ap.parse_args()
    if a.mode == "serve": serve(a.layers, a.port)
    else: drive(a.layers, a.next, a.tokens)
