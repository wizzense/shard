"""Offline proof for the fault-tolerance RESUME primitive ported into m25_pipe.coordinate_pipe.

Over a deterministic in-memory ring (same harness as m25_confsched_test), NO GPU:
 1. CONTINUITY — running with resume_ids=<committed tokens> re-prefills prompt+committed and continues;
    output_ids starts byte-identical with the committed tokens and equals the same greedy stream a
    from-scratch run produces. So a mid-request node death costs ONE re-prefill, not a restart, and the
    user's committed output is preserved (continuation, not regeneration).
 2. RESUMABLE DUMP — on a mid-generation ring failure, resumable=True returns {ok:False, output_ids:
    <committed prefix>} (so the control plane can heal+resume) instead of raising; resumable=False raises.

  python research/m25_resume_test.py
"""
import sys, os, types, threading, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))

PROMPT_IDS = list(range(2, 22))     # 20-token prompt
EOS = 999999


def truth(p):
    return (p * 2654435761) % 5000


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


class _Chan:
    def __init__(self): self.q = queue.Queue()
    def settimeout(self, t): pass


def _recv(sock):
    v = sock.q.get(timeout=15)
    if v == "__DIE__":
        raise ConnectionError("simulated mid-generation node death")
    return v


_stub("m25_stage", H=3072, DIR="/tmp/none", EPS=1e-6, raw=lambda *a, **k: None,
      vllm_ctx=lambda *a, **k: None, Layer=object, run_block=lambda *a, **k: None, _CTX=(None, None))
_stub("m25_tools", render_ids=lambda tok, messages, tools=None: list(PROMPT_IDS),
      parse_completion=lambda t: {"content": t, "reasoning_content": "", "tool_calls": []})
_stub("node_kv", send_msg=lambda sock, obj: sock.q.put(obj), recv_msg=_recv,
      EDGE_ERRORS=(ConnectionError, queue.Empty), TransportError=RuntimeError)
_stub("receipt", ReceiptSigner=None, load_or_make_node_key=lambda *a, **k: None,
      verify_receipt=lambda *a, **k: None, verify_coverage=lambda *a, **k: None)

import m25_pipe   # noqa: E402


class _FakeTok:
    eos_token_id = EOS
    def decode(self, ids, skip_special_tokens=True): return ",".join(map(str, ids))


class _TruthDrafter:
    def request(self, ids, k): self._ids = list(ids); self._k = k
    def fetch(self): b = len(self._ids); return [truth(b + i) for i in range(self._k)]


def _ring(pipe_in, ret_out, stop, die_after=None):
    """Greedy tail oracle. If die_after is set, sends __DIE__ instead of the response on the Nth verify."""
    nver = 0
    while not stop.is_set():
        try:
            msg = pipe_in.q.get(timeout=0.25)
        except queue.Empty:
            continue
        op = msg.get("op")
        if op == "reset":
            ret_out.q.put("ok")
        elif op == "receipt":
            ret_out.q.put([])
        else:
            nver += 1
            if die_after is not None and nver > die_after:
                ret_out.q.put("__DIE__"); return
            start = msg["start"]; n = len(msg["token_ids"])
            ret_out.q.put([truth(start + j + 1) for j in range(n)])


def _run(resume_ids=None, resumable=False, die_after=None, max_new=40, K=8, depth=4):
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_ring, args=(pipe, ret, stop, die_after), daemon=True); t.start()
    try:
        return m25_pipe.coordinate_pipe(
            pipe_sock=pipe, tok=_FakeTok(), messages=[{"role": "user", "content": "x"}],
            K=K, max_new=max_new, timeout=15, depth=depth, ret_sock=ret, local_draft=_TruthDrafter(),
            tools=None, prefill_chunk=0, max_ctx=0, resume_ids=resume_ids, resumable=resumable)
    finally:
        stop.set(); t.join(timeout=2)


def test_resume_continuity():
    full = _run()["output_ids"]                            # from-scratch greedy stream
    R = 12                                                 # pretend 12 tokens were committed before a drop
    committed = full[:R]
    res = _run(resume_ids=committed, max_new=40)
    out = res["output_ids"]
    assert out[:R] == committed, f"continuation NOT byte-preserved: {out[:R]} vs {committed}"
    # length-independent: the resumed output IS the same greedy stream (it overshoots max_new by up to a
    # K-block, so don't compare lengths — compare content against the position-keyed truth stream).
    assert out == [truth(len(PROMPT_IDS) + i) for i in range(len(out))], "resumed stream != greedy truth stream"
    assert res["resume_tokens"] == R, f"resume_tokens={res['resume_tokens']} != {R}"
    assert len(out) >= 40, f"resumed run should reach max_new, got {len(out)}"
    print(f"[resume] PASS 1 — resume re-prefills {R} committed tokens + continues byte-preserved "
          f"(out[:{R}]==committed, same greedy stream, {len(out)} total)")


def test_resumable_dump_on_failure():
    res = _run(resumable=True, die_after=3, max_new=200)   # ring dies mid-gen after 3 verifies
    assert res["ok"] is False and res.get("resumable") is True, f"expected resumable dump, got {res.get('ok')}"
    committed = res["output_ids"]
    assert len(committed) >= 1, "no committed tokens handed back"
    full = _run()["output_ids"]
    assert committed == full[:len(committed)], "committed dump is not a prefix of the greedy stream"
    print(f"[resume] PASS 2 — mid-gen death w/ resumable=True returns {len(committed)} committed tokens "
          f"(prefix of truth), no raise")

    # resumable=False must RAISE (TransportError, stubbed as RuntimeError) on the same failure
    raised = False
    try:
        _run(resumable=False, die_after=3, max_new=200)
    except Exception:
        raised = True
    assert raised, "resumable=False must raise on a dead edge, not return"
    print("[resume] PASS 3 — resumable=False raises on a dead edge (no silent partial)")


if __name__ == "__main__":
    test_resume_continuity()
    test_resumable_dump_on_failure()
    print("\n[resume] ALL PASS — m25_pipe FT resume primitive: byte-preserved continuation + resumable dump")
