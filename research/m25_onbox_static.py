"""On-box: M25_STATIC_KV bit-identical to cat on real hardware + memory. Single-mode (vLLM FusedMoE
registers layer names globally, so a layer can only be built once per process). Run TWICE:
  M25_STATIC_KV=0 python m25_onbox_static.py   # cat
  M25_STATIC_KV=1 python m25_onbox_static.py   # static
and compare the printed hash (identical => bit-identical end-to-end through SDPA-flash).
"""
import os, torch, hashlib
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S

S.vllm_ctx()
dev = "cuda"
LAYERS = list(range(28, 32))
S.get_pe()
SEQ = [(0, 512), (512, 9), (521, 9), (523, 9), (532, 9)]   # prefill, verify, verify, ROLLBACK(523<530), resume
layers = [S.Layer(i) for i in LAYERS]
outs = []
with torch.no_grad():
    for start, s in SEQ:
        g = torch.Generator().manual_seed(start * 1000 + s)   # identical inputs across both modes
        x = (torch.randn(1, s, S.H, generator=g, dtype=torch.bfloat16) * 0.3).to(dev)
        outs.append(S.run_block(layers, start, x, S._CTX[1]).float().cpu().reshape(-1))
flat = torch.cat(outs)
mode = "static" if S.M25_STATIC_KV else "cat"
ml = S.M25_KV_MAXLEN if S.M25_STATIC_KV else "-"
print(f"[static] mode={mode:6} hash {hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]} "
      f"peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB MAXLEN={ml} ({len(LAYERS)} layers)", flush=True)
