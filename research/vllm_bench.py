"""P1 de-risk: can vLLM give a cheap gpt-oss-20b draft on Ada (4090)?
our transformers draft is 62 ms/tok (MXFP4-MoE overhead the stack won't remove).
vLLM has CUDA-graphed, Ada-compatible gpt-oss kernels. if this lands ~12-20 ms/tok
single-stream, P1 (optimized-kernel draft, no training) is the path to a cheap draft."""
import time
from vllm import LLM, SamplingParams

llm = LLM(model="/root/models/gpt-oss-20b", max_model_len=1024,
          gpu_memory_utilization=0.85, enforce_eager=False)   # CUDA graphs on
sp = SamplingParams(temperature=0, max_tokens=128, min_tokens=128, ignore_eos=True)
llm.generate(["warmup the graphs"], sp)                        # warm
t0 = time.time()
out = llm.generate(["The quick brown fox jumps over the lazy dog and then runs"], sp)
dt = time.time() - t0
n = len(out[0].outputs[0].token_ids)
print(f"VLLM_20B_DECODE: {n/dt:.1f} tok/s, {dt/n*1000:.1f} ms/tok (single-stream, CUDA graphs)", flush=True)
print(f"  vs transformers eager: 16.2 tok/s, 61.7 ms/tok", flush=True)
