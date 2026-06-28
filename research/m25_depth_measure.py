"""Measure the depth-pipelining fix for batched serving over the warm ring. Compares:
  - single-stream pipelined (depth=4)            -> the existing ~20 tok/s baseline
  - batched B=2,4 x depth=1 (synchronous, OLD)   -> the regression (~B/L)
  - batched B=2,4 x depth=4 (the fix)            -> WAN hidden -> aggregate ~ B x single-stream
One persistent connection (reset/reset_batch per job clears the stages). Copy task = draftable -> meaningful tok/s.
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_depth_measure.py
"""
import socket, os
import m25_stage as S
import m25_pipe as P
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)

# M2.5's chat template forces "<think>\n" so generation starts inside a reasoning block -> novel text ->
# n-gram accept=0 (depth-pipelining has nothing to speculate). For a DRAFTABLE throughput measurement, close
# the think block immediately so the model emits content (the repetition) from token 1. Patch render_ids
# (m25_pipe imported it by name) to append "</think>\n\n".
_THINK_SKIP = tok("</think>\n\n", add_special_tokens=False)["input_ids"]
_orig_render = P.render_ids
def _render_noskip(t, messages, tools=None, add_generation_prompt=True):
    return _orig_render(t, messages, tools=tools, add_generation_prompt=add_generation_prompt) + _THINK_SKIP
P.render_ids = _render_noskip

HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=600); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=600); ret.setsockopt(*P.NODELAY); ret.settimeout(600)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

# draftable steady-state task (the engine's target workload): a repetition the n-gram drafter nails after a
# few tokens, so accept is high once past M2.5's short forced-<think> preamble. max_new large => steady state dominates.
PROMPT = [{"role": "user", "content": "Write the word 'swarm' exactly 400 times, separated by single spaces. Output only the words, nothing else."}]
K = 8; MAXNEW = 256

def single():
    dr = NgramDrafter(ng=3)
    return P.coordinate_pipe(pipe, tok, PROMPT, K, MAXNEW, 600, 4, ret_sock=ret, local_draft=dr, prefill_chunk=2048, max_ctx=131072)

def batched(B, depth):
    drs = [NgramDrafter(ng=3) for _ in range(B)]
    return P.coordinate_pipe_batch(pipe, tok, [PROMPT] * B, K, MAXNEW, 600, ret, drs, depth=depth, prefill_chunk=2048, max_ctx=131072)

print("=== single-stream pipelined (depth=4) baseline ===", flush=True)
r = single()
base = r["tok_s"]
print(f"  tok/s={r['tok_s']:.2f}  g={r['toks_per_traversal']:.1f}  accept={r['mean_accept']/K*100:.0f}%", flush=True)

print("\n=== batched: depth=1 (synchronous, OLD) vs depth=4 (fix) ===", flush=True)
print(f"{'B':>2} {'depth':>5} {'agg tok/s':>10} {'per-stream':>10} {'vs single':>9} {'rounds':>7} {'wasted':>7}", flush=True)
for B in (2, 4):
    for depth in (1, 4):
        r = batched(B, depth)
        agg = r["agg_tok_s"]
        print(f"{B:>2} {depth:>5} {agg:>10.2f} {agg/B:>10.2f} {agg/base:>8.2f}x {r['rounds']:>7} {r['wasted']:>7}", flush=True)
print("\n[depth-measure] done", flush=True)
