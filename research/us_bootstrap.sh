#!/bin/bash
# in-house draft+coordinator box: vLLM (its own venv) + the 20B draft. the
# coordinator runs in the vLLM venv too (it has torch+transformers for the imports
# and the tokenizer; it never loads the 120B).
python3 -m venv /root/vllmenv
/root/vllmenv/bin/pip install -q --upgrade pip >/dev/null 2>&1
/root/vllmenv/bin/pip install -q vllm > /root/vllm_install.log 2>&1
echo "vllm $(/root/vllmenv/bin/python -c 'import vllm;print(vllm.__version__)' 2>&1)" >> /root/vllm_install.log
/root/vllmenv/bin/python /root/get_model.py openai/gpt-oss-20b /root/models/gpt-oss-20b > /root/dl.log 2>&1
echo "20b rc=$? shards=$(ls /root/models/gpt-oss-20b/*.safetensors 2>/dev/null | wc -l)" > /root/ready.txt
/root/vllmenv/bin/python -c "import vllm" 2>&1 && echo "vllm import OK" >> /root/ready.txt
