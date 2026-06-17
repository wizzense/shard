#!/bin/bash
# fresh RTX 5090 (Blackwell sm_120, CUDA 13) box -> env for the GLM-5.2 probe.
# the cudagraph probe needs NO model weights (random-init from the real config.json,
# ~5KB), so this env + network to HF is all it needs. the real-weights / MTP probe
# (later) additionally pulls only the checkpoint shards for the layers it tests.
#
# usage: bash glm_bootstrap.sh && /root/glmenv/bin/python glm_probe_cudagraph.py
set -e

python3 -m venv /root/glmenv
. /root/glmenv/bin/activate
pip install -U pip

# Blackwell (sm_120) needs a cu13 torch build; stable by mid-2026.
pip install --index-url https://download.pytorch.org/whl/cu130 torch
# transformers >=5.12 is the first release with the glm_moe_dsa architecture.
pip install "transformers>=5.12" accelerate safetensors huggingface_hub

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| dev", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer  # import-check
print("glm_moe_dsa import OK -- ready for glm_probe_cudagraph.py")
PY
