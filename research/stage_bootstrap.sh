#!/bin/bash
# swarm stage box: transformers stack + the 120B (authenticated download, no throttle).
bash /root/setup_box.sh > /root/setup.log 2>&1
echo "DEPS rc=$?" >> /root/setup.log
python3 /root/get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b > /root/dl.log 2>&1
echo "120b rc=$? shards=$(ls /root/models/gpt-oss-120b/*.safetensors 2>/dev/null | wc -l)/15" > /root/ready.txt
du -sh /root/models/gpt-oss-120b >> /root/ready.txt 2>&1
