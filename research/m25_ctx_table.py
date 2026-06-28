"""Decode tok/s as a function of CONTEXT, for pipeline (single-stream depth=4) vs batched (B=4 depth=4).
Draftable continuation task + think-skip so accept is high (meaningful tok/s, not the g=1 WAN floor).
One persistent connection. tok_s is DECODE rate (excludes prefill).
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_ctx_table.py
"""
import socket, os
import m25_stage as S
import m25_pipe as P
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
# think-skip: close the forced <think> so the model emits draftable content from token 1
_TS = tok("</think>\n\n", add_special_tokens=False)["input_ids"]
_orig = P.render_ids
P.render_ids = lambda t, m, tools=None, add_generation_prompt=True: _orig(t, m, tools=tools, add_generation_prompt=add_generation_prompt) + _TS

HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=1200); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=1200); ret.setsockopt(*P.NODELAY); ret.settimeout(1200)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

# ~N tokens of a draftable repeated phrase -> the model continues it -> high n-gram accept at any context length
WORD = "swarm "
def prompt_ctx(n):
    ids = tok(WORD * (n + 8), add_special_tokens=False)["input_ids"][:n]
    return [{"role": "user", "content": "Continue this sequence exactly, same word repeated:\n" + tok.decode(ids)}]

CTXS = [512, 2048, 8192, 16384, 26000]
K = 8; MAXNEW = 64
print(f"{'ctx_tok':>8} {'pipeline tok/s':>14} {'batched B=4 agg':>16} {'per-stream':>11} {'batch/pipe':>11} {'accept':>7}", flush=True)
for n in CTXS:
    m = prompt_ctx(n)
    dr = NgramDrafter(ng=3)
    rs = P.coordinate_pipe(pipe, tok, m, K, MAXNEW, 1200, 4, ret_sock=ret, local_draft=dr, prefill_chunk=2048, max_ctx=131072)
    drs = [NgramDrafter(ng=3) for _ in range(4)]
    rb = P.coordinate_pipe_batch(pipe, tok, [m] * 4, K, MAXNEW, 1200, ret, drs, depth=4, prefill_chunk=2048, max_ctx=131072)
    sp = rs["tok_s"]; ag = rb["agg_tok_s"]
    print(f"{rs['prompt_tokens']:>8} {sp:>14.2f} {ag:>16.2f} {ag/4:>11.2f} {ag/max(sp,1e-9):>10.2f}x {rs['mean_accept']/K*100:>6.0f}%", flush=True)
print("[ctx-table] done", flush=True)
