"""Decisive: is the StaticCache+mask-patch draft identical to clean DynamicCache on a PURE FORWARD
(no rollback)? If yes -> the g=1.5 is purely a rollback artifact (target the rollback). If no -> the
patch/StaticCache changes the draft fundamentally (pivot to a hand-rolled draft).  python glm_cg_fwdcmp.py"""
import torch
import transformers.models.glm4.modeling_glm4 as G
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, StaticCache
DRAFT = "/root/glm4_9b_draft"; dev = "cuda"; ML = 128
tok = AutoTokenizer.from_pretrained(DRAFT, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
ids = tok("def quicksort(arr):", return_tensors="pt").input_ids.to(dev); L = ids.shape[1]; N = 24

# A: clean DynamicCache, UNPATCHED (the g=1.94 baseline forward)
dc = DynamicCache(); A = []
with torch.no_grad():
    t = int(m(input_ids=ids, past_key_values=dc, use_cache=True).logits[0, -1].argmax()); A.append(t)
    for i in range(N - 1):
        t = int(m(input_ids=torch.tensor([[t]], device=dev), past_key_values=dc, use_cache=True).logits[0, -1].argmax()); A.append(t)

# B: StaticCache + position_ids-based mask patch, monotonic forward
def patch(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
    dt = inputs_embeds.dtype; d = inputs_embeds.device
    qp = position_ids.reshape(-1); kp = torch.arange(ML, device=d)
    allow = kp.unsqueeze(0) <= qp.unsqueeze(1)
    return torch.where(allow, torch.zeros((), dtype=dt, device=d), torch.full((), torch.finfo(dt).min, dtype=dt, device=d))[None, None]
G.create_causal_mask = patch
sc = StaticCache(config=m.config, max_cache_len=ML, device=dev, dtype=torch.bfloat16); B = []
with torch.no_grad():
    t = int(m(input_ids=ids, past_key_values=sc, cache_position=torch.arange(L, device=dev),
              position_ids=torch.arange(L, device=dev)[None], use_cache=True).logits[0, -1].argmax()); B.append(t)
    for i in range(N - 1):
        p = L + i
        t = int(m(input_ids=torch.tensor([[t]], device=dev), past_key_values=sc, cache_position=torch.tensor([p], device=dev),
                  position_ids=torch.tensor([[p]], device=dev), use_cache=True).logits[0, -1].argmax()); B.append(t)

match = sum(1 for a, b in zip(A, B) if a == b)
print("A (dynamic, clean):  ", A[:12], flush=True)
print("B (static + patch):  ", B[:12], flush=True)
print(f"PURE-FORWARD match: {match}/{N} ->", "PATCH IS CLEAN (g-loss is rollback)" if match == N else "PATCH CHANGES THE FORWARD (fundamental)", flush=True)
