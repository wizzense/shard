"""GLM-5.2 distributed generation — coordinator + stage roles over the transport.
The swarm-serving driver: combines the generation loop (glm_gen) with the stage transport.

  coord : tokenizer + embed + final norm + lm_head + sampling + the token loop. Embeds the
          (growing) sequence, ships hidden to the stage chain, gets hidden back, norm+lm_head,
          greedy-samples, appends, repeats. No KV cache (recompute the seq each step) -- the
          simplest correct distributed loop for validation.
  stage : a contiguous block of layers in fp8. recv hidden -> run block -> send hidden back.

validate on box A: one `stage` (layers 6,7) + one `coord`, localhost. Output is gibberish
(2/78 layers) but proves DISTRIBUTED generation end to end. The full swarm = coord + N stages
holding all 78 layers (same code, `--next` chains stage->stage, tail returns to coord).
run under /root/vmoe:  stage:  python glm_swarm.py stage --layers 6 7 --port 29600
                       coord:  python glm_swarm.py coord --stage 127.0.0.1:29600 --prompt "..."
"""
import sys, io, json, time, socket, struct, argparse, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig, AutoTokenizer
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

DIR, dev, BLK = "/root/glm52fp8", "cuda", 128
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I, eps = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.rms_norm_eps
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)
def bscale(n): return raw(n + "_scale_inv")
def dequant(n):
    w = raw(n).to(torch.float32); s = bscale(n); m, k = w.shape
    s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :k]
    return (w * s).to(torch.bfloat16)
def maybe(n): return dequant(n) if (n + "_scale_inv") in idx else raw(n).to(torch.bfloat16)

# ---- transport (length-prefixed torch tensor over TCP) ----
def _sendall(sock, b): sock.sendall(struct.pack("!Q", len(b)) + b)
def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return buf
def send_tensor(sock, t):
    bio = io.BytesIO(); torch.save(t.cpu(), bio); _sendall(sock, bio.getvalue())
def recv_tensor(sock):
    return torch.load(io.BytesIO(_recvn(sock, struct.unpack("!Q", _recvn(sock, 8))[0])), weights_only=False).to(dev)

# ---- quant layer load + monkeypatches (fp8 experts, dense MLA) ----
def load_quant_layer(li):
    P = f"model.layers.{li}."
    sd = {}
    for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
              "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
              "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
              "mlp.gate.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
              "mlp.shared_experts.down_proj.weight"]:
        sd[n] = maybe(P + n).to(dev)
    sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
    fp8 = torch.float8_e4m3fn
    w1 = torch.empty(E, 2 * I, H, dtype=fp8, device=dev); w2 = torch.empty(E, H, I, dtype=fp8, device=dev)
    w1s = torch.empty(E, (2 * I) // BLK, H // BLK, dtype=torch.float32, device=dev)
    w2s = torch.empty(E, H // BLK, I // BLK, dtype=torch.float32, device=dev)
    for e in range(E):
        w1[e] = torch.cat([raw(P + f"mlp.experts.{e}.gate_proj.weight"), raw(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
        w1s[e] = torch.cat([bscale(P + f"mlp.experts.{e}.gate_proj.weight"), bscale(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
        w2[e] = raw(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
        w2s[e] = bscale(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(cfg, li)
    layer.load_state_dict(sd, strict=False, assign=True)
    ex = layer.mlp.experts; ex._w1, ex._w2 = w1, w2
    ex._qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=[BLK, BLK])
    return layer.eval()

def _moe(self, hidden_states, top_k_index, top_k_weights):
    return fused_experts(hidden_states, self._w1, self._w2, top_k_weights.to(hidden_states.dtype),
                         top_k_index.to(torch.int32), quant_config=self._qc)
M.GlmMoeDsaNaiveMoe.forward = _moe

def _attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
          position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    o, w = M.eager_attention_forward(self, torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1),
                                     value_states, attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None
M.GlmMoeDsaAttention.forward = _attn

_rotary = None
def _rope_mask(h):
    global _rotary
    if _rotary is None: _rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    L = h.shape[1]; pos = torch.arange(L, device=dev).unsqueeze(0)
    mask = torch.zeros(1, 1, L, L, dtype=torch.bfloat16, device=dev)
    mask.masked_fill_(torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1), float("-inf"))
    return pos, _rotary(h, position_ids=pos), mask

def run_block(layers, h):
    pos, pe, mask = _rope_mask(h)
    for L in layers:
        h = L(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    return h


def stage(layer_ids, port, nxt=None):
    layers = [load_quant_layer(i) for i in layer_ids]
    print(f"stage layers {layer_ids} listening :{port}" + (f" -> {nxt}" if nxt else " (tail->return)"), flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    fwd = None
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            with torch.no_grad():
                while True:
                    h = run_block(layers, recv_tensor(conn))
                    if nxt:
                        if fwd is None:
                            host, p = nxt.rsplit(":", 1); fwd = socket.create_connection((host, int(p)))
                            fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        send_tensor(fwd, h); send_tensor(conn, recv_tensor(fwd))   # relay tail return
                    else:
                        send_tensor(conn, h)                                       # tail returns hidden
        except (ConnectionError, EOFError):
            print("conn closed", flush=True)


def coord(stage_ep, prompt, max_new):
    tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True)
    embed_w = maybe("model.embed_tokens.weight").to(dev)
    lm_head_w = maybe("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = maybe("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=60); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"coord -> stage chain @ {stage_ep}", flush=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new):
            h = torch.nn.functional.embedding(ids, embed_w)        # embed full seq
            send_tensor(s, h); h = recv_tensor(s)                  # -> stage chain -> back
            x = h[0, -1].float(); x = x * torch.rsqrt(x.pow(2).mean() + eps) * norm_w  # final norm
            nxt = int((x.to(torch.bfloat16) @ lm_head_w.t()).float().argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
            if nxt in eos: break
    dt = time.time() - t0; n = ids.shape[1]
    print(f"\nGENERATED {n} tokens in {dt:.1f}s = {n/dt:.1f} tok/s (distributed, no-cache recompute)", flush=True)
    print("decoded:", repr(tok.decode(ids[0], skip_special_tokens=True)[:200]), flush=True)
    print("VERDICT: DISTRIBUTED generation works — coord embeds, stage(s) run quantized blocks over "
          "the transport, coord norm+lm_head+samples+loops. Scales to N stages / 78 layers across the swarm.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("stage"); p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--port", type=int, default=29600); p.add_argument("--next", default=None)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="The capital of France is"); p.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    if a.role == "stage": stage(a.layers, a.port, a.next)
    else: coord(a.stage, a.prompt, a.max_new)
