#!/bin/bash
# fresh vast box (vastai/base-image:cuda-13.2.1-auto) -> shard runtime for gpt-oss.
# kernels==0.14.1 is the load-bearing pin: 0.15.x breaks mxfp4 ("Either a revision
# or a version must be specified"). triton + kernels are what make MXFP4 load on Ada.
set -e
pip install --break-system-packages -q --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
pip install --break-system-packages -q transformers==5.6.0 huggingface_hub safetensors accelerate kernels==0.14.1 triton
python3 - <<'PY'
import torch, transformers, kernels
print("torch", torch.__version__, "| tf", transformers.__version__,
      "| kernels", kernels.__version__, "| gpus", torch.cuda.device_count())
PY
