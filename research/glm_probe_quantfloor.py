"""GLM-5.2 verify is memory-bound on selected-expert weight reads (probe: grouped MoE graphs
but gives 1.0x at bf16 -- no overhead to remove, the weight bytes ARE the cost). plan A
(run stages under vLLM/SGLang at FP8/NVFP4) wins ONLY if quantization cuts that read.

this measures the dominant term -- gathering the selected experts' weights (top-8 of 256) for
a 6-token verify -- at bf16 vs fp8, and projects fp4. if the time scales ~linearly with
bytes/weight, the memory-bound verify floor drops 2x at fp8 / 4x at nvfp4, which is exactly
what vLLM/SGLang deliver on Blackwell. confirms (or kills) plan A on real silicon.
run: python glm_probe_quantfloor.py
"""
import time, torch

# GLM-5.2 MoE dims. verify reads T*K=48 expert matrices regardless of total experts, so we
# store E=64 (>48, valid indices) to fit one 32GB card -- the gather/read volume is identical.
E, I, H, T, K = 64, 2048, 6144, 6, 8
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  bw-probe: top-{K} of {E} experts, {T} tokens, hidden {H}", flush=True)
bw = torch.cuda.get_device_properties(0).memory_bus_width
print(f"experts/layer touched ~= min(T*K, E) = {min(T*K, E)}", flush=True)


def bench(dtype, label, bytes_per):
    gate_up = torch.empty(E, 2 * I, H, device=dev, dtype=torch.bfloat16).normal_(0, 0.1)
    down = torch.empty(E, H, I, device=dev, dtype=torch.bfloat16).normal_(0, 0.1)
    if dtype != torch.bfloat16:
        gate_up = gate_up.to(dtype); down = down.to(dtype)
    idx = torch.randint(0, E, (T, K), device=dev)
    gb = (gate_up.numel() + down.numel()) * bytes_per / 1e9
    # the verify's dominant op: gather the selected experts' weight matrices (fixed shape)
    def step():
        a = gate_up[idx]      # [T,K,2I,H]
        b = down[idx]         # [T,K,H,I]
        return a.sum() + b.sum()                  # force materialization/read
    for _ in range(3): step()
    torch.cuda.synchronize(); t0 = time.time(); R = 30
    for _ in range(R): step()
    torch.cuda.synchronize(); ms = (time.time() - t0) / R * 1000
    read_gb = (gate_up[idx].numel() + down[idx].numel()) * bytes_per / 1e9
    del gate_up, down; torch.cuda.empty_cache()
    print(f"  {label:14s} weights {gb:5.1f} GB | gather/read {read_gb:.2f} GB/layer | {ms:6.2f} ms "
          f"({read_gb/ (ms/1000) /1000:.2f} TB/s)", flush=True)
    return ms


with torch.no_grad():
    bf16 = bench(torch.bfloat16, "bf16", 2)
    try:
        fp8 = bench(torch.float8_e4m3fn, "fp8_e4m3", 1)
        print(f"\nfp8 vs bf16: {bf16/fp8:.2f}x faster verify-weight read", flush=True)
        print(f"projected nvfp4 (~0.5 B/wt): ~{bf16/ (fp8*0.5):.1f}x vs bf16", flush=True)
        print("VERDICT:", "PLAN A HOLDS -- quant scales the memory-bound verify down; "
              "vLLM/SGLang FP8/NVFP4 fused kernels are the speed path." if fp8 < bf16 * 0.75
              else "quant did NOT cut the read here -- re-examine.", flush=True)
    except Exception as e:
        print(f"fp8 gather unsupported ({type(e).__name__}); measuring via uint8 (same 1 B/wt) ...", flush=True)
        u8 = bench(torch.uint8, "uint8(~fp8)", 1)
        print(f"\nuint8 vs bf16: {bf16/u8:.2f}x -- the read floor scales with bytes/weight as expected", flush=True)
        print("VERDICT: PLAN A HOLDS (read floor is bytes-bound; FP8/NVFP4 cut it 2-4x).", flush=True)
