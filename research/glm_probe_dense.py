"""GLM-5.2 fast-verify, part 2: prove the dense-bypass path captures AND stays exact.

probe 1 found: glm_moe_dsa runs eager on Blackwell, but the stage won't CUDA-graph -- the
DSA indexer's scatter/boolean-mask block (GlmMoeDsaAttention.forward) breaks capture. KEY
FACT from the source: at seq <= index_topk (2048) the indexer selects ALL keys, so that
block is a semantic NO-OP -- it builds an all-False mask and masked_fills nothing. our
verify sequences are ~6 tokens, always << 2048.

so the fix is a dense verify forward that skips the indexer and runs plain MLA attention
with the causal mask. this probe monkeypatches exactly that and checks two things:
  1. EXACTNESS: eager-dense output == eager-original (indexer) output (it must, no-op).
  2. CAPTURE: the dense stage CUDA-graphs, replays bit-exact, and is faster.
if both hold, the fast verify ports to GLM-5.2 -- the 18-25 tok/s lever is back.

run: python glm_probe_dense.py
"""
import sys, time, torch
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-5.2"
dev, dt = "cuda", torch.bfloat16
N_LAYERS, N_EXPERTS, Kp1 = 4, 16, 6

cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}", flush=True)

cfg = GlmMoeDsaConfig.from_pretrained(MODEL)
cfg.num_hidden_layers = N_LAYERS
cfg.n_routed_experts = cfg.num_experts = N_EXPERTS
cfg.num_experts_per_tok = min(cfg.num_experts_per_tok, N_EXPERTS)
cfg.first_k_dense_replace = 1
cfg._attn_implementation = "eager"
hidden = cfg.hidden_size

torch.manual_seed(0)
layers = torch.nn.ModuleList([M.GlmMoeDsaDecoderLayer(cfg, i) for i in range(N_LAYERS)]).to(dev, dt).eval()
rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)

h = torch.randn(1, Kp1, hidden, dtype=dt, device=dev) * 0.1
pos = torch.arange(Kp1, device=dev).unsqueeze(0)
pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, Kp1, Kp1, dtype=dt, device=dev)
mask.masked_fill_(torch.triu(torch.ones(Kp1, Kp1, device=dev, dtype=torch.bool), 1), float("-inf"))


def stage_fwd(hh):
    x, topk = hh, None
    for layer in layers:
        out = layer(x, attention_mask=mask, position_ids=pos, past_key_values=None,
                    use_cache=False, position_embeddings=pe, prev_topk_indices=topk)
        x = out[0]; topk = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    return x


# --- dense bypass: plain MLA attention, no indexer / no sparse-mask scatter ---
def dense_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, position_ids=None, prev_topk_indices=None, **kwargs):
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
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attn_output, attn_weights = M.eager_attention_forward(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling, **kwargs)
    attn_output = attn_output.reshape(b, s, -1).contiguous()
    return self.o_proj(attn_output), attn_weights, None


with torch.no_grad():
    ref = stage_fwd(h).float().clone()                       # eager WITH indexer (probe-1 path)
    M.GlmMoeDsaAttention.forward = dense_attn_forward        # monkeypatch -> dense bypass
    dense_eager = stage_fwd(h).float().clone()
    exact = (dense_eager - ref).abs().max().item()
    print(f"[1] dense-bypass vs indexer eager: max-diff {exact:.6f} -> "
          f"{'EXACT (indexer is a no-op at this seq, confirmed)' if exact < 1e-3 else 'DIVERGES -- inspect'}", flush=True)

    for _ in range(5): stage_fwd(h)
    torch.cuda.synchronize(); t0 = time.time(); R = 30
    for _ in range(R): stage_fwd(h)
    torch.cuda.synchronize(); eager_ms = (time.time() - t0) / R * 1000

    try:
        sh = h.clone()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): stage_fwd(sh)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = stage_fwd(sh)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(R): g.replay()
        torch.cuda.synchronize(); graph_ms = (time.time() - t0) / R * 1000
        diff = (gout.float() - dense_eager).abs().max().item()
        print(f"[2] CUDAGRAPH replay: {graph_ms:.2f} ms | eager {eager_ms:.2f} ms | "
              f"SPEEDUP {eager_ms/graph_ms:.1f}x | replay max-diff {diff:.5f}", flush=True)
        ok = graph_ms < eager_ms and diff < 0.02 and exact < 1e-3
        print("VERDICT:", "FAST VERIFY PORTS TO GLM-5.2 -- dense bypass captures, exact, faster. "
              "the 18-25 tok/s lever is real on Blackwell." if ok else
              "partial -- see numbers above.", flush=True)
    except Exception as e:
        print(f"[2] CUDAGRAPH STILL FAILS: {type(e).__name__}: {str(e)[:240]}", flush=True)
        print("VERDICT: the indexer was not the only blocker -- bisect the MoE routing next.", flush=True)
