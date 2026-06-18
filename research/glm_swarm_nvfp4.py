"""GLM-5.2 NVFP4 distributed generation — coordinator + stage roles over the transport.
The NVFP4 swarm-serving driver: same architecture as glm_swarm.py (fp8) but the layer
execution runs the proven NVFP4 FusedMoE kernel (~5GB/layer -> full GLM-5.2 on ~16 5090s).

  coord : tokenizer + embed + final norm + lm_head + sampling + token loop. Ships hidden to
          the stage chain, gets hidden back, norm+lm_head, greedy-samples, appends, repeats.
  stage : a contiguous block of layers in NVFP4. recv hidden -> run block -> send hidden back.
          dense layers (0..first_k_dense_replace-1) = 1-expert FusedMoE; MoE layers = 256-expert
          routed FusedMoE + 1-expert shared FusedMoE. MLA/norms/gate stay bf16.

run under /root/vmoe:
  stage:  python glm_swarm_nvfp4.py stage --layers 6 7 8 --port 29600 [--next host:port]
  coord:  python glm_swarm_nvfp4.py coord --stage host:29600 --prompt "..." --max-new 32
"""
import os, io, json, time, socket, struct, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig, AutoTokenizer
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

# NVFP4 MoE kernel: we force the PRECOMPILED VLLM_CUTLASS backend (see _vllm_ctx, sets
# kernel_config.moe_backend="cutlass"), so flashinfer's JIT (which OOM-kills/deadlocks on fresh
# nodes) is never invoked. The no-op below is a belt-and-suspenders fallback, off by default.
if os.environ.get("GLM_SKIP_FLASHINFER_BUILD", "0") == "1":
    try:
        import flashinfer.jit.core as _fijc
        _fijc.JitSpec.build = lambda self, *a, **k: None
    except Exception as _e:
        print("warn: flashinfer build no-op patch failed:", _e, flush=True)

DIR, dev = os.environ.get("GLM_DIR", "/root/glm52nvfp4"), "cuda"
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I, Idense, K, eps = (cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size,
                           cfg.intermediate_size, cfg.num_experts_per_tok, cfg.rms_norm_eps)
NDENSE = cfg.first_k_dense_replace
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)

# ---- transport (length-prefixed torch tensor over TCP) ----
def _sendall(sock, b): sock.sendall(struct.pack("!Q", len(b)) + b)
def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return bytes(buf)
def send_tensor(sock, t):
    bio = io.BytesIO(); torch.save(t.cpu(), bio); _sendall(sock, bio.getvalue())
def recv_tensor(sock):
    return torch.load(io.BytesIO(_recvn(sock, struct.unpack("!Q", _recvn(sock, 8))[0])), weights_only=False).to(dev)

# ====================== NVFP4 execution (stage role) ======================
_VC = None; _CTXMGR = None
def _vllm_ctx():
    """Set up the vLLM runtime context cascade once (FusedMoE needs it). Keep a GLOBAL ref to
    the config context manager — if it's GC'd the config un-sets and FusedMoE init fails."""
    global _VC, _CTXMGR
    if _VC is not None: return _VC
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    # Force the precompiled VLLM_CUTLASS nvfp4 MoE kernel (in the wheel) instead of the default
    # FLASHINFER_CUTLASS, which JIT-compiles fused_moe_120 and gets OOM-killed on fresh nodes.
    try:
        vcfg.kernel_config.moe_backend = "cutlass"
    except Exception as e:
        print("warn: could not set moe_backend=cutlass:", e, flush=True)
    _CTXMGR = set_current_vllm_config(vcfg); _CTXMGR.__enter__()
    from vllm.config import get_current_vllm_config
    print(f"[cfg] moe_backend = {get_current_vllm_config().kernel_config.moe_backend}", flush=True)
    initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
    _VC = vcfg; return vcfg

def shared_routing(*a, **kw):
    hs = kw["hidden_states"]; T = hs.shape[0]
    return (torch.ones(T, 1, dtype=torch.bfloat16, device=hs.device),
            torch.zeros(T, 1, dtype=torch.int32, device=hs.device))

def _build_moe(base, n_exp, inter, routed):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
    if routed:
        eb = raw(base.replace("mlp.experts.", "mlp.gate.") + "e_score_correction_bias").float().to(dev)
        m = FusedMoE(num_experts=n_exp, top_k=K, hidden_size=H, intermediate_size=inter, params_dtype=torch.bfloat16,
                     renormalize=cfg.norm_topk_prob, use_grouped_topk=True, num_expert_group=cfg.n_group,
                     topk_group=cfg.topk_group, scoring_func="sigmoid", routed_scaling_factor=cfg.routed_scaling_factor,
                     e_score_correction_bias=eb, quant_config=qnv, prefix=base).to(dev)
    else:
        m = FusedMoE(num_experts=1, top_k=1, hidden_size=H, intermediate_size=inter, params_dtype=torch.bfloat16,
                     renormalize=False, custom_routing_function=shared_routing, quant_config=qnv, prefix=base).to(dev)
    pp = dict(m.named_parameters())
    for e in (range(n_exp) if routed else [None]):
        for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            nbase = f"{base}{e}.{proj}." if routed else f"{base}{proj}."
            for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
                n = nbase + suf
                if n in idx: m.weight_loader(pp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, e if routed else 0)
    m.quant_method.process_weights_after_loading(m)
    return m

def load_layer(li):
    P = f"model.layers.{li}."
    is_dense = li < NDENSE
    names = ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
             "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
             "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    sd = {n: raw(P + n).to(torch.bfloat16).to(dev) for n in names}
    if not is_dense:
        sd["mlp.gate.weight"] = raw(P + "mlp.gate.weight").to(torch.bfloat16).to(dev)
        sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(cfg, li)
    layer.load_state_dict(sd, strict=False, assign=True); layer.eval()
    if is_dense:
        dmoe = _build_moe(P + "mlp.", 1, Idense, routed=False)
        def fwd(self, x, _d=dmoe):
            shp = x.shape; h = x.view(-1, H)
            ones = torch.ones(h.shape[0], 1, dtype=torch.bfloat16, device=h.device)
            return _d(h, ones).view(shp)
        layer.mlp.forward = fwd.__get__(layer.mlp)
    else:
        rmoe = _build_moe(P + "mlp.experts.", E, I, routed=True)
        smoe = _build_moe(P + "mlp.shared_experts.", 1, I, routed=False)
        def fwd(self, x, _r=rmoe, _s=smoe):
            shp = x.shape; h = x.view(-1, H)
            rl = torch.nn.functional.linear(h, self.gate.weight)
            ones = torch.ones(h.shape[0], 1, dtype=torch.bfloat16, device=h.device)
            return (_r(h, rl) + _s(h, ones)).view(shp)
        layer.mlp.forward = fwd.__get__(layer.mlp)
    return layer

def dense_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
               position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
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
M.GlmMoeDsaAttention.forward = dense_attn

_rotary = None
def _rope_mask(h):
    global _rotary
    if _rotary is None: _rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    L = h.shape[1]; pos = torch.arange(L, device=dev).unsqueeze(0)
    mask = torch.zeros(1, 1, L, L, dtype=torch.bfloat16, device=dev)
    mask.masked_fill_(torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1), float("-inf"))
    return pos, _rotary(h, position_ids=pos), mask

def run_block(layers, h, vcfg):
    from vllm.forward_context import set_forward_context
    pos, pe, mask = _rope_mask(h)
    with set_forward_context(None, vcfg):
        for L in layers:
            h = L(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    return h

def stage(layer_ids, port, nxt=None):
    vcfg = _vllm_ctx()
    layers = [load_layer(i) for i in layer_ids]
    mem = torch.cuda.memory_allocated() / 1e9
    # WARMUP: trigger flashinfer/cutlass JIT now (first forward compiles on CPU, GPU idle ~1-2min)
    # so "listening" means truly ready and the first real token isn't cold.
    print(f"stage layers {layer_ids} loaded ({mem:.1f} GB) — warming up (JIT)...", flush=True)
    with torch.no_grad():
        _ = run_block(layers, torch.randn(1, 8, H, dtype=torch.bfloat16, device=dev) * 0.1, vcfg)
    torch.cuda.synchronize()
    print(f"stage layers {layer_ids} WARM, listening :{port}" +
          (f" -> {nxt}" if nxt else " (tail->return)"), flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    fwd = None
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            with torch.no_grad():
                while True:
                    h = run_block(layers, recv_tensor(conn), vcfg)
                    if nxt:
                        if fwd is None:
                            host, p = nxt.rsplit(":", 1); fwd = socket.create_connection((host, int(p)))
                            fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        send_tensor(fwd, h); send_tensor(conn, recv_tensor(fwd))
                    else:
                        send_tensor(conn, h)
        except (ConnectionError, EOFError):
            print("conn closed", flush=True); fwd = None

# ====================== coordinator role ======================
def coord(stage_ep, prompt, max_new):
    tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True)
    embed_w = raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = raw("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=300); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(300)
    print(f"coord -> stage chain @ {stage_ep}", flush=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    n_prompt = ids.shape[1]; t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new):
            h = torch.nn.functional.embedding(ids, embed_w)
            send_tensor(s, h); h = recv_tensor(s)
            x = h[0, -1].float(); x = x * torch.rsqrt(x.pow(2).mean() + eps) * norm_w
            nxt = int((x.to(torch.bfloat16) @ lm_head_w.t()).float().argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
            if nxt in eos: break
    dt = time.time() - t0; gen = ids.shape[1] - n_prompt
    print(f"\nGENERATED {gen} tokens in {dt:.1f}s = {gen/dt:.2f} tok/s (NVFP4 distributed, no-cache recompute)", flush=True)
    print("decoded:", repr(tok.decode(ids[0], skip_special_tokens=True)[:400]), flush=True)
    return gen / dt, tok.decode(ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("stage"); p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--port", type=int, default=29600); p.add_argument("--next", default=None)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="The capital of France is"); p.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    if a.role == "stage": stage(a.layers, a.port, a.next)
    else: coord(a.stage, a.prompt, a.max_new)
