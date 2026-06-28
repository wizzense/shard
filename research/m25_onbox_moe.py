"""On-box: is M2.5's NVFP4 cutlass FusedMoE TOKEN-COUNT INVARIANT? (Phase 5 concurrency crux.)

gpt-oss's mxfp4 MoE was NOT invariant — token 0's MLP output changed depending on how many tokens shared
the batch (kernel grouping/padding) — which blocked LOSSLESS batched concurrency (forced per-stream MoE).
This measures whether M2.5's NVFP4 MoE has the same property. Verdict decides Phase 5's design:
  invariant  -> full batched MoE is lossless (~2x aggregate)
  NON-invar  -> per-stream MoE (attention still batches) or keep RING_LOCK + a queue.

  python m25_onbox_moe.py [layer]
"""
import os, sys, torch
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S
from vllm.forward_context import set_forward_context

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 28
S.vllm_ctx()
L = S.Layer(LAYER)
dev = "cuda"
torch.manual_seed(0)

print(f"[moe] layer {LAYER}: token-0 MLP output, batched(B) vs alone(1) — same token, varying batch size", flush=True)
worst = 0.0
with torch.no_grad(), set_forward_context(None, S._CTX[1]):   # FusedMoE requires vLLM's forward context
    base = torch.randn(1, 16, S.H, dtype=torch.bfloat16, device=dev) * 0.5
    o1 = L.mlp(base[:, :1, :])                       # token 0 computed ALONE
    for B in (2, 4, 8, 16):
        oB = L.mlp(base[:, :B, :])                   # token 0 computed in a B-token batch
        d = (oB[:, :1, :].float() - o1.float()).abs().max().item()
        rel = d / (o1.float().abs().max().item() + 1e-9)
        worst = max(worst, d)
        print(f"    B={B:2}: max|diff(token0)| = {d:.3e}   rel = {rel:.2e}", flush=True)
verdict = "INVARIANT (batched MoE is lossless — Phase 5 can use full batched verify)" if worst < 1e-3 \
    else "NON-INVARIANT (need per-stream MoE or a request queue — same class as the mxfp4 issue)"
print(f"[moe] worst max|diff| = {worst:.3e}  ->  {verdict}", flush=True)
