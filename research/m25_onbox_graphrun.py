"""On-box validation of the PRODUCTION GraphRunner (varying start_pos): bit-equivalence vs eager
run_block at every position (the spec-decode VERIFY-path gate) + speedup, across bucket boundaries.
  M25_CUDA_GRAPH=1 python m25_onbox_graphrun.py
"""
import os, torch, time
os.environ["M25_CUDA_GRAPH"] = "1"                  # forces M25_STATIC_KV
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S

S.vllm_ctx()
dev = "cuda"
LAYERS = list(range(28, 32))
layers = [S.Layer(i) for i in LAYERS]
S.get_pe()
Kp1, vcfg = 9, S._CTX[1]

# eager-prefill committed context [0, P) into the static KV
P = 2030
with torch.no_grad():
    gp = torch.Generator().manual_seed(0)
    S.run_block(layers, 0, (torch.randn(1, P, S.H, generator=gp, dtype=torch.bfloat16) * 0.3).to(dev), vcfg)

gr = S.GraphRunner(layers, vcfg, Kp1)
worst = 0.0
with torch.no_grad():
    # contiguous verify blocks that CROSS the 2048 bucket boundary (2030->2048->... = bucket 2048 then 4096)
    starts = list(range(P, P + 6 * Kp1, Kp1))
    for start in starts:
        gx = torch.Generator().manual_seed(start)
        x = (torch.randn(1, Kp1, S.H, generator=gx, dtype=torch.bfloat16) * 0.3).to(dev)
        o_eager = S.run_block(layers, start, x, vcfg).clone()    # writes [start,start+Kp1), reads :start+Kp1
        o_graph = gr.run(start, x).clone()                       # same KV write, bucketed read + additive mask
        d = (o_eager.float() - o_graph.float()).abs().max().item()
        worst = max(worst, d)
        print(f"  start={start:5} bucket={gr._bucket(start+Kp1):5}  graph-vs-eager max|diff|={d:.3e}  "
              f"{'OK' if d == 0 else 'DIFF'}", flush=True)
print(f"[graphrun] {'BIT-EQUIV across positions + buckets' if worst == 0 else f'DIVERGE worst={worst:.2e}'}", flush=True)

def bench(fn, n=50):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
with torch.no_grad():
    xe = (torch.randn(1, Kp1, S.H, dtype=torch.bfloat16) * 0.3).to(dev)
    e = bench(lambda: S.run_block(layers, starts[-1], xe, vcfg))
g = bench(lambda: gr.run(starts[-1], xe))
print(f"[graphrun] per-block: eager {e:.2f}ms  graph {g:.2f}ms  -> {e/g:.2f}x ({len(LAYERS)} layers; stage ~13)", flush=True)
