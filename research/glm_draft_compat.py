"""Does a small GLM model share GLM-5.2's exact token IDs? The draft's lm_head output index i must
equal GLM-5.2's token i, else the draft's proposals are garbage to the target (g->0). Compare the
tokenizers' id<->token mapping directly (encode several strings, no special tokens).
  python glm_draft_compat.py [hf_id]"""
import sys
from transformers import AutoTokenizer
import glm_swarm_nvfp4_kv as KV

CAND = sys.argv[1] if len(sys.argv) > 1 else "THUDM/glm-edge-1.5b-chat"
t52 = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
tc = AutoTokenizer.from_pretrained(CAND, trust_remote_code=True)
print(f"vocab  GLM-5.2={t52.vocab_size}  {CAND}={tc.vocab_size}", flush=True)
tests = ["def quicksort(arr):", "        return quicksort(left) + [pivot] + quicksort(right)",
         "import numpy as np", "The quick brown fox jumps over the lazy dog.",
         "    middle = [x for x in arr if x == pivot]", "if __name__ == '__main__':"]
allok = True
for s in tests:
    a = t52(s, add_special_tokens=False).input_ids
    b = tc(s, add_special_tokens=False).input_ids
    ok = (a == b); allok &= ok
    print(f"  {'OK ' if ok else 'DIFF'}  5.2={a[:10]}  cand={b[:10]}", flush=True)
# also spot-check the inverse: do specific ids decode to the same piece?
ids = [262, 421, 2422, 10934, 510, 286, 220, 16]
print("id decode 5.2:", [t52.decode([i]) for i in ids], flush=True)
print("id decode cand:", [tc.decode([i]) for i in ids], flush=True)
print("VERDICT:", "COMPATIBLE — usable as a drop-in draft" if allok else "INCOMPATIBLE — different token IDs, cannot use", flush=True)
