"""plan-A validation: real vLLM fused MoE at GLM-5.2 dims for a 6-token verify.
bf16 fused_experts vs our hand-rolled grouped einsum (6.85ms) / batched (11.62ms), and the
fp8 path if the quant-config API cooperates. run under /root/vmoe.
"""
import time, torch
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk

E, I, H, T, K = 256, 2048, 6144, 6, 8
dev = "cuda"
print(f"GPU {torch.cuda.get_device_name(0)} | fused_experts @ GLM dims: {E}/top{K}, {T} tokens, H{H}", flush=True)


def build(dtype):
    w1 = torch.empty(E, 2 * I, H, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)
    w2 = torch.empty(E, H, I, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)
    if dtype != torch.bfloat16:
        w1f = w1.to(dtype); del w1; torch.cuda.empty_cache()
        w2f = w2.to(dtype); del w2; torch.cuda.empty_cache()
        return w1f, w2f
    return w1, w2


def bench(fn, R=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / R * 1000


x = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1
gate = torch.randn(T, E, device=dev, dtype=torch.bfloat16)
tw, tid, _ = fused_topk(x, gate, K, renormalize=True)

w1, w2 = build(torch.bfloat16)
bf16 = bench(lambda: fused_experts(x, w1, w2, tw, tid))
print(f"fused_experts bf16: {bf16:.3f} ms  (our einsum grouped 6.85 / batched 11.62 ms)", flush=True)
del w1, w2; torch.cuda.empty_cache()

# discover the fp8 quant-config helper
import vllm.model_executor.layers.fused_moe.config as fmc
helpers = [n for n in dir(fmc) if "quant_config" in n.lower() or "fp8" in n.lower()]
print("fp8 quant-config helpers:", helpers, flush=True)

try:
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    w1f, w2f = build(torch.float8_e4m3fn)
    s1 = torch.ones(E, 1, 1, device=dev, dtype=torch.float32)
    s2 = torch.ones(E, 1, 1, device=dev, dtype=torch.float32)
    qc = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, per_act_token_quant=False)
    fp8 = bench(lambda: fused_experts(x, w1f, w2f, tw, tid, quant_config=qc))
    print(f"fused_experts fp8 : {fp8:.3f} ms  | {bf16/fp8:.2f}x vs bf16", flush=True)
    print(f"[proj] ~6-layer stage MoE verify: bf16 {bf16*6:.1f} -> fp8 {fp8*6:.1f} ms "
          f"(nvfp4 ~{fp8*6/2:.1f} ms)", flush=True)
except Exception as e:
    print(f"fp8 attempt: {type(e).__name__}: {str(e)[:240]}", flush=True)
    print("(bf16 fused number already shows fused-kernel vs hand-rolled; fp8 needs the right qc args.)", flush=True)
