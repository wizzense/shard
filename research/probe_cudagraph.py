"""fast-verify feasibility probe: can we CUDA-graph a gpt-oss 120B layer block?
the eager-MXFP4 verify is ~34 ms/token, mostly per-layer Python + kernel-launch
overhead (the draft's 62->5 ms under vLLM graphs was almost all overhead). if a
9-layer stage forward captures into a CUDA graph AND replays faster AND matches
eager output, the fast verify -> tree -> ~10-15 tok/s path is real. the open risk
is gpt-oss's data-dependent MoE routing, which may not capture."""
import sys, time, torch
sys.path.insert(0, "/root")
from pipeline import load_stage, _causal_mask

parts = load_stage("/root/models/gpt-oss-120b", 1, 4, device="cuda")
dev = "cuda"
hidden = parts["_model"].config.hidden_size
T = 8                                                      # tokens through the block
h = (torch.randn(1, T, hidden, dtype=torch.bfloat16, device=dev) * 0.1)
pos = torch.arange(T, device=dev).unsqueeze(0)
pe = parts["rotary"](h, pos)
sliding, win_sz = parts.get("sliding"), parts.get("window", 0)
full = _causal_mask(T, T, 0, 0, h.dtype, dev)
win = _causal_mask(T, T, 0, win_sz, h.dtype, dev) if (sliding and win_sz) else full

def stage_fwd(hh):                                         # 9 layers, no kv-cache
    x = hh
    for i, layer in enumerate(parts["layers"]):
        mask = win if (sliding and sliding[i]) else full
        out = layer(x, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    return x

with torch.no_grad():
    for _ in range(5): stage_fwd(h)
    torch.cuda.synchronize(); t0 = time.time(); N = 30
    for _ in range(N): stage_fwd(h)
    torch.cuda.synchronize(); eager_ms = (time.time() - t0) / N * 1000
    eager_out = stage_fwd(h).clone()
    print(f"EAGER 9-layer forward (T={T}): {eager_ms:.1f} ms ({eager_ms/9:.2f} ms/layer)", flush=True)
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
        for _ in range(N): g.replay()
        torch.cuda.synchronize(); graph_ms = (time.time() - t0) / N * 1000
        diff = (static_out - eager_out).abs().max().item()
        print(f"CUDAGRAPH replay: {graph_ms:.1f} ms | SPEEDUP {eager_ms/graph_ms:.1f}x | output max-diff {diff:.4f}", flush=True)
        print("VERDICT:", "FEASIBLE (graphs work + faster + correct)" if (graph_ms < eager_ms and diff < 0.05)
              else "graphs work but no win / output drift", flush=True)
    except Exception as e:
        print(f"CUDAGRAPH FAILED: {type(e).__name__}: {str(e)[:240]}", flush=True)
        print("VERDICT: NOT graph-able (likely MoE dynamic routing) -> fast verify needs custom kernels", flush=True)
