#!/bin/bash
# launch this box's single 120B stage (detached). usage:
#   launch_node.sh <stage> <nstages> <next_host:port|-> <head|->
# kill targets the python invocation (not this wrapper) to avoid the self-match trap.
STAGE=$1; NS=$2; NEXT=$3; HEAD=$4
M=/root/models/gpt-oss-120b
pkill -9 -f "python3 -u /root/specpipe"; sleep 2
ARGS="--stage $STAGE --nstages $NS --model $M --listen-port 29501 --timeout 600"
[ "$NEXT" != "-" ] && ARGS="$ARGS --next $NEXT"
[ "$HEAD" = "head" ] && ARGS="$ARGS --served-head"
setsid python3 -u /root/specpipe.py $ARGS </dev/null >/root/stage.log 2>&1 &
echo "launched stage $STAGE pid $!"
