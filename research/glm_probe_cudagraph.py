"""GLM-5.2 (glm_moe_dsa) fast-verify feasibility probe -- the make-or-break before any
fleet spend. mirrors phase0/probe_cudagraph.py (gpt-oss), adapted to the new arch.

the whole 18-25 tok/s result rests on CUDA-graphing the stage forward (the fast verify).
gpt-oss graphed because its MoE routing captures (batched all-experts, no dynamic gather).
glm_moe_dsa adds two unknowns this probe answers, on real Blackwell silicon:

  1. DSA (DeepSeek Sparse Attention) indexer: source shows `index_scores.topk(...).indices`
     (capturable) BUT also scatter + boolean indexing in sparse-mask construction (NOT
     capturable -> breaks the graph). KEY BET: at verify seq lengths (K+1 ~= 6 tokens)
     << index_topk (2048), the indexer selects ALL tokens, so the sparse/dynamic path
     should be dormant and the layer collapses to dense MLA. this probe runs the
     verify-shaped forward to confirm the dynamic path doesn't fire at our lengths.
  2. MLA latent cache: attention calls `past_key_values.update(k, v, layer_idx)` with
     full k/v shaped [batch, seq, num_heads, head_dim] (NOTE: seq axis = 1, unlike
     gpt-oss's [batch, heads, seq, dim]). the static-cache port must index that axis.

design choices so this runs on ONE 5090 with NO 376GB download:
  - real config (GlmMoeDsaConfig.from_pretrained) so MLA/DSA/rope dims are exact and
    internally consistent, but num_hidden_layers + n_routed_experts shrunk: graph-ability
    is a property of control flow, not weight values or expert/layer COUNT. random init.
  - tests the no-cache stage forward first (isolates "does it run + capture on Blackwell"),
    exactly like the gpt-oss probe's first pass. the static-MLA-cache graph (the
    fastverify_graph.py analog) is the follow-on once this confirms capture + we read the
    real cache axis order off the first run.

HONEST CAVEAT: written against the v5.12 source signatures, not yet executed (boxes
paused). the mask format / prev_topk threading / head-dim interplay may need a first-run
shakeout -- the probe is built to SURFACE those (loud diagnostics, try/except) and turn
them into measured facts on the first cheap GPU-hour. run: python glm_probe_cudagraph.py
"""
import sys, time, torch
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-5.2"
dev = "cuda"; dt = torch.bfloat16
N_LAYERS = 4          # enough to include dense (first_k_dense_replace) + MoE layers
N_EXPERTS = 16        # shrunk from 256; routing mechanism (gate/top-k/scatter) unchanged
Kp1 = 6               # verify-shaped: cur + K draft tokens (the path that must capture)

cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  torch {torch.__version__} cuda {torch.version.cuda}", flush=True)

cfg = GlmMoeDsaConfig.from_pretrained(MODEL)
cfg.num_hidden_layers = N_LAYERS
cfg.n_routed_experts = N_EXPERTS
cfg.num_experts = N_EXPERTS
cfg.num_experts_per_tok = min(cfg.num_experts_per_tok, N_EXPERTS)
cfg.first_k_dense_replace = 1                     # layer 0 dense, 1..3 MoE
hidden = cfg.hidden_size
print(f"config: L={cfg.num_hidden_layers} H={hidden} experts={cfg.n_routed_experts}/top{cfg.num_experts_per_tok} "
      f"MLA(kv_lora={cfg.kv_lora_rank} q_lora={cfg.q_lora_rank} v_dim={cfg.v_head_dim}) "
      f"DSA(index_topk={getattr(cfg,'index_topk','?')} index_heads={getattr(cfg,'index_n_heads','?')})", flush=True)

torch.manual_seed(0)
layers = torch.nn.ModuleList([GlmMoeDsaDecoderLayer(cfg, i) for i in range(N_LAYERS)]).to(dev, dt).eval()
rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev)

h = (torch.randn(1, Kp1, hidden, dtype=dt, device=dev) * 0.1)
pos = torch.arange(Kp1, device=dev).unsqueeze(0)
pe = rotary(h, position_ids=pos)                  # (cos, sin)
# additive causal mask [B, 1, q, kv]; -inf above the diagonal
mask = torch.full((1, 1, Kp1, Kp1), 0.0, dtype=dt, device=dev)
mask.masked_fill_(torch.triu(torch.ones(Kp1, Kp1, device=dev, dtype=torch.bool), 1), float("-inf"))


def stage_fwd(hh):                                # no kv-cache; verify-shaped seq
    x, topk = hh, None
    for layer in layers:
        out = layer(x, attention_mask=mask, position_ids=pos, past_key_values=None,
                    use_cache=False, position_embeddings=pe, prev_topk_indices=topk)
        x = out[0] if isinstance(out, (tuple, list)) else out
        topk = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    return x


with torch.no_grad():
    # ---- test A: does the glm_moe_dsa stage even run on this GPU? ----
    try:
        for _ in range(5): stage_fwd(h)
        torch.cuda.synchronize(); t0 = time.time(); R = 30
        for _ in range(R): stage_fwd(h)
        torch.cuda.synchronize(); eager_ms = (time.time() - t0) / R * 1000
        eager_out = stage_fwd(h).clone()
        print(f"EAGER {N_LAYERS}-layer stage forward (seq={Kp1}): {eager_ms:.2f} ms "
              f"({eager_ms/N_LAYERS:.2f} ms/layer) -- RUNS on this GPU", flush=True)
    except Exception as e:
        print(f"EAGER FORWARD FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        print("VERDICT: glm_moe_dsa stage does not run as-built here -- shakeout needed "
              "(mask format / head_dim / DSA path). fix before the graph test.", flush=True)
        sys.exit(1)

    # ---- test B: capture the verify-shaped forward as a CUDA graph ----
    try:
        static_h = h.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): stage_fwd(static_h)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = stage_fwd(static_h)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(R): g.replay()
        torch.cuda.synchronize(); graph_ms = (time.time() - t0) / R * 1000
        diff = (static_out.float() - eager_out.float()).abs().max().item()
        print(f"CUDAGRAPH replay: {graph_ms:.2f} ms | SPEEDUP {eager_ms/graph_ms:.1f}x | max-diff {diff:.4f}", flush=True)
        ok = graph_ms < eager_ms and diff < 0.05
        print("VERDICT:", "FEASIBLE -- DSA dormant at verify seq, MLA+MoE graph + faster + correct. "
              "fast verify ports to GLM-5.2." if ok else
              "graphs but no win / output drift -- inspect.", flush=True)
    except Exception as e:
        print(f"CUDAGRAPH FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        print("VERDICT: NOT capturable as-built -- almost certainly the DSA indexer's "
              "scatter/boolean-index path firing. NEXT: a dense verify forward that skips "
              "the indexer when seq <= index_topk (our case). that is the real eng lift, "
              "and it is bounded -- own the small-seq attention path like gpt-oss.", flush=True)
