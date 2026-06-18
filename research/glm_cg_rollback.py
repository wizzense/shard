"""Decisive isolated test for plan B: patch GLM-4's causal mask to be built from position_ids
(key_pos <= query position) instead of HF's max-written length. position_ids is derived from
cache_position, so a rewound cache_position (speculative rollback) -> the right mask AND RoPE, and
it's CUDA-graph-safe (only the position varies; arange(MAXLEN) is const). Test: does a re-draft after
rewind bit-match a fresh draft, eager AND torch.compiled? + ms/tok.   python glm_cg_rollback.py"""
import time, torch
import transformers.models.glm4.modeling_glm4 as G
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
DRAFT = "/root/glm4_9b_draft"; dev = "cuda"; MAXLEN = 256

def cg_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
    dtype = inputs_embeds.dtype; d = inputs_embeds.device
    qpos = position_ids.reshape(-1)                                  # [q_len]; query at qpos[i]
    kp = torch.arange(MAXLEN, device=d)
    allow = kp.unsqueeze(0) <= qpos.unsqueeze(1)                     # [q_len, MAXLEN]: key j attended iff j <= qpos
    neg = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=d), torch.full((), neg, dtype=dtype, device=d))[None, None]
G.create_causal_mask = cg_causal_mask

tok = AutoTokenizer.from_pretrained(DRAFT, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager").to(dev).eval()
ids = tok("def quicksort(arr):", return_tensors="pt").input_ids.to(dev); L = ids.shape[1]; N = 16

def new_cache_cur():
    c = StaticCache(config=m.config, max_cache_len=MAXLEN, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        cur = int(m(input_ids=ids, past_key_values=c, cache_position=torch.arange(L, device=dev), use_cache=True).logits[0, -1].argmax())
    return c, cur

def run(stepfn, warm=False):
    if warm:
        c, cur = new_cache_cur()
        for w in range(8): stepfn(c, cur, L + w)                      # warm the cudagraph
    c, cur = new_cache_cur(); fresh = []; t = cur                     # fresh monotonic draft
    for i in range(N): t = stepfn(c, t, L + i); fresh.append(t)
    c, cur = new_cache_cur(); t = cur                                 # rollback run: draft 8 along the same path
    for i in range(8): stepfn(c, t, L + i); t = fresh[i]
    t = fresh[3]; rb = []                                             # rewind to L+3, re-draft
    for p in range(L + 3, L + N - 1): t = stepfn(c, t, p); rb.append(t)
    match = sum(1 for a, b in zip(rb, fresh[4:]) if a == b)
    return fresh, rb, match

def estep(c, t, p):
    return int(m(input_ids=torch.tensor([[t]], device=dev), past_key_values=c,
                 cache_position=torch.tensor([p], device=dev), use_cache=True).logits[0, -1].argmax())
fresh, rb, match = run(estep)
print(f"EAGER  patch: fresh[4:10]={fresh[4:10]} rb[:6]={rb[:6]} match={match}/{N-4}", flush=True)

cm = torch.compile(m, mode="reduce-overhead", fullgraph=False)
_inp = torch.zeros((1, 1), dtype=torch.long, device=dev); _cp = torch.zeros((1,), dtype=torch.long, device=dev)
def cstep(c, t, p):
    _inp[0, 0] = t; _cp[0] = p
    return int(cm(input_ids=_inp, past_key_values=c, cache_position=_cp, use_cache=True).logits[0, -1].argmax())
fresh, rb, match = run(cstep, warm=True)
print(f"COMPILED patch: fresh[4:10]={fresh[4:10]} rb[:6]={rb[:6]} match={match}/{N-4}", flush=True)
c, cur = new_cache_cur()
for w in range(8): cstep(c, cur, L + w)
c, cur = new_cache_cur(); torch.cuda.synchronize(); t0 = time.time(); t = cur
for i in range(128): t = cstep(c, t, L + i)
torch.cuda.synchronize()
print(f"COMPILED speed: {(time.time()-t0)/128*1000:.1f} ms/tok", flush=True)
