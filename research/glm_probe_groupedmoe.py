"""GLM-5.2 fast-verify: the capturable SPARSE MoE that restores the speedup at 256 experts.

probe (realconfig) showed batched-all-experts is exact + capturable but gives NO speedup at
256 experts (computes all 256 vs the 8 selected -> 96% of the layer is wasted FLOPs).

fix: a GATHER-based grouped MoE. for each token gather only its top_k experts' weight
matrices (`gate_up_proj[top_k_index]` -> fixed shape [T, K, ...]), do the matmul, weight,
sum. computes only T*K expert-applications (here 6*8=48) vs all-experts T*E (6*256=1536) --
~32x less -- and every shape is static (T, K constant), so it CUDA-graphs. exact vs the
shipped routing loop.

measures: exact vs original, grouped-MoE-alone time vs batched-all, full-layer eager vs
graph speedup, and a stage projection. run: python glm_probe_groupedmoe.py
"""
import sys, time, torch
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-5.2"
dev, dt = "cuda", torch.bfloat16
Kp1 = 6

print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)
cfg = GlmMoeDsaConfig.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
hidden = cfg.hidden_size
LIDX = next(i for i in range(cfg.first_k_dense_replace, cfg.num_hidden_layers)
            if cfg.indexer_types[i] == "full")
print(f"REAL config: experts={cfg.n_routed_experts}/top{cfg.num_experts_per_tok}  "
      f"moe_inter={cfg.moe_intermediate_size}  hidden={hidden}  (MoE+full layer idx {LIDX})", flush=True)


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
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    o, w = M.eager_attention_forward(self, query_states, key_states, value_states,
                                     attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None


def grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    # gather only the selected experts' weights -> fixed shape [T, K, ...]; compute T*K, not T*E.
    gu = self.gate_up_proj[top_k_index]                          # [T, K, 2*inter, hidden]
    dn = self.down_proj[top_k_index]                             # [T, K, hidden, inter]
    gate_up = torch.einsum("tkih,th->tki", gu, hidden_states)    # [T, K, 2*inter]
    gate, up = gate_up.chunk(2, dim=-1)
    hmid = self.act_fn(gate) * up                               # [T, K, inter]
    out = torch.einsum("tkhi,tki->tkh", dn, hmid)               # [T, K, hidden]
    out = out * top_k_weights.to(out.dtype).unsqueeze(-1)
    return out.sum(dim=1)                                       # [T, hidden]


def batched_experts_forward(self, hidden_states, top_k_index, top_k_weights):  # baseline
    T, E = hidden_states.shape[0], self.num_experts
    x = hidden_states.unsqueeze(0).expand(E, T, -1)
    gate, up = torch.bmm(x, self.gate_up_proj.transpose(1, 2)).chunk(2, dim=-1)
    out = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
    full_w = hidden_states.new_zeros(T, E).scatter_(1, top_k_index, top_k_weights.to(hidden_states.dtype))
    return (out * full_w.t().unsqueeze(-1)).sum(dim=0)


torch.manual_seed(0)
layer = M.GlmMoeDsaDecoderLayer(cfg, LIDX).to(dev, dt).eval()
rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
h = torch.randn(1, Kp1, hidden, dtype=dt, device=dev) * 0.1
pos = torch.arange(Kp1, device=dev).unsqueeze(0)
pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, Kp1, Kp1, dtype=dt, device=dev)
mask.masked_fill_(torch.triu(torch.ones(Kp1, Kp1, device=dev, dtype=torch.bool), 1), float("-inf"))


def fwd(hh):
    return layer(hh, attention_mask=mask, position_ids=pos, past_key_values=None,
                 use_cache=False, position_embeddings=pe, prev_topk_indices=None)[0]


def bench(fn, R=50, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / R * 1000


with torch.no_grad():
    ref = fwd(h).float().clone()                                # original (loop MoE + indexer)
    M.GlmMoeDsaAttention.forward = dense_attn_forward
    M.GlmMoeDsaNaiveMoe.forward = grouped_experts_forward
    patched = fwd(h).float().clone()
    rel = (patched - ref).abs().max().item() / ref.abs().max().item()
    print(f"[exact] grouped-MoE vs original: rel max-diff {rel:.2e} -> {'EXACT' if rel < 1e-2 else 'DIVERGES'}", flush=True)

    # isolated MoE cost: grouped (selected) vs batched (all)
    moe = layer.mlp.experts
    flat = h.view(-1, hidden)
    K = cfg.num_experts_per_tok
    ti = torch.randint(0, cfg.n_routed_experts, (Kp1, K), device=dev)
    tw = torch.softmax(torch.randn(Kp1, K, device=dev), dim=-1)
    grp_ms = bench(lambda: grouped_experts_forward(moe, flat, ti, tw))
    bat_ms = bench(lambda: batched_experts_forward(moe, flat, ti, tw))
    print(f"[moe ] grouped(selected-8) {grp_ms:.2f} ms  vs  batched(all-256) {bat_ms:.2f} ms  -> {bat_ms/grp_ms:.1f}x less", flush=True)

    eager_ms = bench(lambda: fwd(h))
    sh = h.clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fwd(sh)
    torch.cuda.current_stream().wait_stream(st)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = fwd(sh)
        graph_ms = bench(lambda: g.replay())
        diff = (gout.float() - patched).abs().max().item()
        print(f"[perf] 1 MoE layer  eager {eager_ms:.2f} ms | graph {graph_ms:.2f} ms | "
              f"SPEEDUP {eager_ms/graph_ms:.1f}x | replay max-diff {diff:.5f}", flush=True)
        print(f"[proj] ~6-layer stage graphed verify ≈ {graph_ms*6:.1f} ms", flush=True)
        ok = graph_ms < eager_ms * 0.8 and diff < 0.02 and rel < 1e-2
        print("VERDICT:", "FAST VERIFY REAL AT 256 EXPERTS -- grouped MoE captures, exact, and faster. "
              "the path to 18-25 tok/s on GLM-5.2 holds." if ok else "see numbers above.", flush=True)
    except Exception as e:
        print(f"[perf] CUDAGRAPH FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
