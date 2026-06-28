"""On-box 3b feasibility probe: can run_block (static KV + NVFP4 cutlass MoE + SDPA) be CUDA-graph
captured and replayed BIT-IDENTICALLY, and is it faster? The runbook flags the NVFP4 FusedMoE
graph-safety as the UNVERIFIED gating risk for the whole CUDA-graph throughput lever.

Captures at a FIXED (start_pos, K+1) shape — the varying-position machinery (tensor cp + bucketed
additive mask) is the full implementation; this probe answers the gating questions first:
  (a) does capture even succeed (MoE graph-safe)?   (b) is replay bit-equivalent to eager?   (c) speedup?

  python m25_onbox_graph.py
"""
import os, torch, time, traceback
os.environ.setdefault("M25_DIR", "/root/m25")
os.environ["M25_STATIC_KV"] = "1"          # fixed KV addresses are required for capture
import m25_stage as S

S.vllm_ctx()
dev = "cuda"
LAYERS = list(range(28, 32))
S.get_pe()
layers = [S.Layer(i) for i in LAYERS]
Kp1, START = 9, 1024
vcfg = S._CTX[1]


def block(x):                               # the verify block: run_block over our layers at START
    return S.run_block(layers, START, x, vcfg)


with torch.no_grad():
    # prefill [0,START) so the static KV has real committed context for the block to attend over
    gp = torch.Generator().manual_seed(0)
    S.run_block(layers, 0, (torch.randn(1, START, S.H, generator=gp, dtype=torch.bfloat16) * 0.3).to(dev), vcfg)
    gx = torch.Generator().manual_seed(7)
    xblk = (torch.randn(1, Kp1, S.H, generator=gx, dtype=torch.bfloat16) * 0.3).to(dev)
    o_eager = block(xblk).clone()           # eager reference (block re-writes [START,START+Kp1) each call)

    # --- capture ---
    static_x = torch.zeros(1, Kp1, S.H, dtype=torch.bfloat16, device=dev)
    sidestream = torch.cuda.Stream(); sidestream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sidestream):
        for _ in range(3):
            static_x.copy_(xblk); block(static_x)
    torch.cuda.current_stream().wait_stream(sidestream); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            static_out = block(static_x)
        captured = True
    except Exception as e:
        captured = False
        print("[graph] CAPTURE FAILED — NVFP4 MoE or SDPA is not graph-safe at this shape:", flush=True)
        print("        " + "".join(traceback.format_exception_only(type(e), e)).strip(), flush=True)

if captured:
    static_x.copy_(xblk); graph.replay(); torch.cuda.synchronize()
    d = (static_out.float() - o_eager.float()).abs().max().item()
    gate = "BIT-EQUIV" if d == 0 else ("WITHIN-ULP" if d < 1e-2 else "DIVERGE (graph corrupts verify!)")
    # timing: eager vs replay
    def bench(fn, n=30):
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1000
    with torch.no_grad():
        eager_ms = bench(lambda: block(xblk))
    replay_ms = bench(lambda: graph.replay())
    print(f"[graph] CAPTURE OK | graph-vs-eager max|diff|={d:.3e} -> {gate}", flush=True)
    print(f"[graph] per-block: eager {eager_ms:.2f}ms  graph {replay_ms:.2f}ms  -> {eager_ms/replay_ms:.2f}x "
          f"({len(LAYERS)} layers; a full stage is ~13)", flush=True)
