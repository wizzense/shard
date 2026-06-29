"""Realistic batched-serving check: 4 DIFFERENT real user prompts (trick-reasoning, code, TOOL-CALL,
explainer), NO think-skip (the model reasons normally -> lower n-gram accept than the copy task = the
honest real-traffic number). Compares batched B=4 (concurrent) vs single-stream served sequentially, and
verifies coherence + that the tool-call stream still emits a structured call under batching.
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_realistic_batch.py
"""
import socket, os, time
import m25_stage as S
import m25_pipe as P
from m25_tools import parse_completion
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=1800); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=1800); ret.setsockopt(*P.NODELAY); ret.settimeout(1800)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

WEATHER = [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}},
                           "required": ["city"]}}}]
PROMPTS = [
    [{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Think it through, then give the number."}],
    [{"role": "user", "content": "Write a Python function is_palindrome(s) that returns True if s is a palindrome ignoring punctuation, spaces and case. Include a short docstring."}],
    [{"role": "user", "content": "What's the weather in Tokyo right now? Use the get_weather tool."}],
    [{"role": "user", "content": "Explain the difference between TCP and UDP in about four sentences."}],
]
K = 8; MAXNEW = 256
labels = ["reasoning", "code", "tool-call", "explain"]

print("=== single-stream (each served on its own, sequential) ===", flush=True)
tot_tok = 0; tot_t = 0.0
for m, lab in zip(PROMPTS, labels):
    dr = NgramDrafter(ng=3)
    t0 = time.time()
    r = P.coordinate_pipe(pipe, tok, m, K, MAXNEW, 1800, 4, ret_sock=ret, local_draft=dr, tools=WEATHER,
                          prefill_chunk=2048, max_ctx=131072)
    dt = time.time() - t0; tot_tok += r["n_tokens"]; tot_t += dt
    p = parse_completion(r["text"])
    print(f"  [{lab:>9}] {r['n_tokens']:>3}tok {r['tok_s']:>6.1f} tok/s  accept={r['mean_accept']/K*100:>3.0f}%  "
          f"tool={bool(p['tool_calls'])}  :: {((p['content'] or r['text']) or '').strip()[:64]!r}", flush=True)
print(f"  --> single-stream aggregate over 4 sequential requests: {tot_tok/max(tot_t,1e-9):.1f} tok/s", flush=True)

print("\n=== batched B=4 (all 4 concurrent, real mixed prompts, depth=4) ===", flush=True)
drs = [NgramDrafter(ng=3) for _ in range(4)]
rb = P.coordinate_pipe_batch(pipe, tok, PROMPTS, K, MAXNEW, 1800, ret, drs, depth=4, tools=WEATHER,
                             prefill_chunk=2048, max_ctx=131072)
for i, (s, lab) in enumerate(zip(rb["streams"], labels)):
    p = parse_completion(s["text"])
    print(f"  [{lab:>9}] {s['n_tokens']:>3}tok  tool={bool(p['tool_calls'])}  "
          f":: {((p['content'] or s['text']) or '').strip()[:64]!r}", flush=True)
print(f"  --> batched aggregate: {rb['agg_tok_s']:.1f} tok/s  (prefill {rb.get('prefill_s',0):.1f}s excluded)", flush=True)
print(f"\n[realistic] batched/single ratio: {rb['agg_tok_s']/max(tot_tok/max(tot_t,1e-9),1e-9):.2f}x", flush=True)
print("[realistic] done", flush=True)
