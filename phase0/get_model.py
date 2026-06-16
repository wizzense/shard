"""robust HF model fetch for the big gpt-oss checkpoints.

the HF python client gets rate-throttled to a crawl on the 57GB 120B shards;
wget on the resolve URL does not (~200MB/s). snapshot_download also orphans
half-finished + duplicate blobs and fills the disk. so: list the repo, keep only
what transformers loads (root-level safetensors + config + tokenizer, NOT the
`original/` bf16 duplicates), and wget each shard -c (resumable) into a flat dir.
load with from_pretrained("/root/models/<name>").

  python get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b
"""
import sys, os, subprocess
from huggingface_hub import list_repo_files

repo, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
token = os.environ.get("HF_TOKEN", "")                 # authenticated => HF doesn't throttle
hdr = ["--header", f"Authorization: Bearer {token}"] if token else []
SKIP_DIR = ("original/", "metal/", "onnx/", "gguf/")
KEEP_EXT = (".safetensors", ".json", ".jinja", ".txt", ".model")
files = [f for f in list_repo_files(repo, token=token or None)
         if not any(f.startswith(d) for d in SKIP_DIR) and f.endswith(KEEP_EXT)]
print(f"{len(files)} files to fetch from {repo} ({'auth' if token else 'anon'})", flush=True)
for f in files:
    dst = os.path.join(outdir, f)
    os.makedirs(os.path.dirname(dst) or outdir, exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("have", f, flush=True); continue
    url = f"https://huggingface.co/{repo}/resolve/main/{f}"
    print("get", f, flush=True)
    r = subprocess.run(["wget", "-c", "--tries=50", "--read-timeout=25",
                        "--progress=dot:giga"] + hdr + ["-O", dst, url])
    if r.returncode != 0:
        print("FAIL", f, flush=True); sys.exit(1)
print("DONE", outdir, flush=True)
