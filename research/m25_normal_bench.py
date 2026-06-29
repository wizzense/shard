"""NORMAL-USAGE benchmark (the honest one): real tasks over a real, varied long-context document
(the engine's own source code), NOT repeated "swarm". Measures single-stream tok/s across context
with reasoning OFF (the fast path) — the regime the engine is actually intended for (code/RAG/agentic
long-context, where the answer reuses the context so the prompt-lookup drafter accepts it). Also shows
the reasoning-ON tax and one genuinely-novel task (low overlap = the hard case) for contrast.

  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_normal_bench.py
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

# real, varied long document = the engine's own source (genuine code, no artificial repetition)
DOC = ""
for f in ["/root/m25_stage.py", "/root/m25_pipe.py", "/root/m25_gateway.py"]:
    try:
        DOC += open(f).read() + "\n\n"
    except OSError:
        pass
DOC_IDS = tok(DOC, add_special_tokens=False)["input_ids"]
print(f"[normal] document = {len(DOC_IDS)} tokens of real engine source", flush=True)

K = 8; MAXNEW = 160


def doc_prompt(ctx, instruction):
    src = tok.decode(DOC_IDS[:ctx])
    return [{"role": "user", "content": f"Here is some source code:\n\n{src}\n\n{instruction}"}]


def run(messages, reasoning):
    dr = NgramDrafter(ng=3)
    return P.coordinate_pipe(pipe, tok, messages, K, MAXNEW, 1800, 4, ret_sock=ret, local_draft=dr,
                             prefill_chunk=2048, max_ctx=131072, reasoning=reasoning)


# 1) CONTEXT SWEEP — realistic code-summarization/extraction over real long context, reasoning OFF
INSTR = "List the main functions defined in this code and give a one-sentence description of each. Quote each function's def line."
print(f"\n=== single-stream, NORMAL task (extract+quote functions), reasoning OFF, across context ===", flush=True)
print(f"{'ctx_tok':>8} {'tok/s':>7} {'accept':>7} {'g':>5} {'>=20?':>6}", flush=True)
for ctx in [1000, 4000, 8000, 16000]:
    if ctx > len(DOC_IDS):
        break
    r = run(doc_prompt(ctx, INSTR), reasoning=False)
    g = r["toks_per_traversal"]
    print(f"{r['prompt_tokens']:>8} {r['tok_s']:>7.1f} {r['mean_accept']/K*100:>6.0f}% {g:>5.1f} "
          f"{'YES' if r['tok_s']>=20 else 'no':>6}", flush=True)

# 2) REASONING ON vs OFF tax at a fixed large context (same task)
print(f"\n=== reasoning ON vs OFF tax (extract task @ 8000 ctx) ===", flush=True)
for rsn in (True, False):
    r = run(doc_prompt(8000, INSTR), reasoning=rsn)
    print(f"  reasoning={str(rsn):>5}: {r['tok_s']:>6.1f} tok/s  accept={r['mean_accept']/K*100:>3.0f}%  "
          f"g={r['toks_per_traversal']:.1f}  out={ (parse_completion(r['text'])['content'] or r['text']).strip()[:60]!r}", flush=True)

# 3) TASK-TYPE spread at 8000 ctx, reasoning OFF (overlap varies by task)
print(f"\n=== task-type spread @ 8000 ctx, reasoning OFF ===", flush=True)
TASKS = [
    ("summarize", "Summarize what this code does in 4 sentences."),
    ("qa-quote", "What does the coordinate_pipe function do? Quote the key lines from it."),
    ("continue", "Write a new Python helper function, in the same style, that counts how many layers each stage holds."),
    ("novel(no-ctx)", "__NOVEL__"),
]
for name, instr in TASKS:
    if instr == "__NOVEL__":
        m = [{"role": "user", "content": "Explain how TCP congestion control works (slow start, congestion avoidance, fast recovery) in about 6 sentences."}]
    else:
        m = doc_prompt(8000, instr)
    r = run(m, reasoning=False)
    print(f"  {name:>14}: {r['tok_s']:>6.1f} tok/s  accept={r['mean_accept']/K*100:>3.0f}%  g={r['toks_per_traversal']:.1f}", flush=True)

print("\n[normal] done", flush=True)
