"""Per-node selective fetch for the GLM-5.2-NVFP4 swarm: download ONLY the safetensors files
that hold this node's assigned layers (~4 layers ~20GB), not the full 410GB. The coordinator
additionally grabs embed/norm/lm_head + tokenizer.

  stage node:  python node_fetch.py --layers 6 7 8 9
  coord node:  python node_fetch.py --coord [--layers 0 1 2 3 4 5]
"""
import os, sys, json, time, argparse
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

def fetch(repo, f, tries=8):
    """Download with resume + backoff — no HF token means throttling/429 under fleet load."""
    for i in range(tries):
        try:
            return hf_hub_download(repo, f, local_dir=D)
        except Exception as e:
            wait = min(60, 5 * (i + 1))
            print(f"  retry {f} ({i+1}/{tries}) after {wait}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {f} after {tries} tries")

REPO = "Mapika/GLM-5.2-NVFP4"; TOK_REPO = "zai-org/GLM-5.2"; D = "/root/glm52nvfp4"

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, nargs="*", default=[])
ap.add_argument("--coord", action="store_true")
a = ap.parse_args()

os.makedirs(D, exist_ok=True)
# index + config first
for f in ["model.safetensors.index.json", "config.json"]:
    fetch(REPO, f)
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]

want = set()
for L in a.layers:
    for w, f in idx.items():
        if w.startswith(f"model.layers.{L}."):
            want.add(f)
if a.coord:
    for w in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        want.add(idx[w])

print(f"node fetch: layers {a.layers} coord={a.coord} -> {len(want)} files", flush=True)
for f in sorted(want):
    print("  ", f, flush=True)
    fetch(REPO, f)

# tokenizer (from the base GLM repo; nvfp4 repo ships none)
if a.coord:
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        try:
            fetch(TOK_REPO, f, tries=3)
        except Exception as e:
            print("tokenizer fetch warn:", e, flush=True)
print("NODE_FETCH_DONE", flush=True)
