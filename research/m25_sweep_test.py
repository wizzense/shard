"""Standalone unit test for m25_pipe._sweep_summary — the K/depth sweep table + winner selection.

m25_pipe imports m25_stage / m25_tools / node_kv, which load the model dir (and a GPU ctx) at import
time, so it can't be imported off-box. We inject lightweight stub modules into sys.modules first, then
import m25_pipe and test the PURE sweep-summary logic against the REAL function (not a copy). No GPU,
no model, no network.

  python research/m25_sweep_test.py
"""
import sys, os, types


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# on-box-only deps -> stubs so `import m25_pipe` succeeds anywhere
_stub("m25_stage", H=3072, DIR="/tmp/none", EPS=1e-6, raw=lambda *a, **k: None,
      vllm_ctx=lambda *a, **k: None, Layer=object, run_block=lambda *a, **k: None, _CTX=(None, None))
_stub("m25_tools", render_ids=lambda *a, **k: [],
      parse_completion=lambda t: {"content": t, "reasoning_content": "", "tool_calls": []})
_stub("node_kv", send_msg=lambda *a, **k: None, recv_msg=lambda *a, **k: None,
      EDGE_ERRORS=(Exception,), TransportError=Exception)
_stub("receipt", ReceiptSigner=None, load_or_make_node_key=lambda *a, **k: None,
      verify_receipt=lambda *a, **k: None, verify_coverage=lambda *a, **k: None)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
import m25_pipe


def row(K, depth, tok_s, ok=True, g=3.0, accept=0.6, ntok=200):
    return {"K": K, "depth": depth, "tok_s": tok_s, "g": g, "accept": accept,
            "prefill_s": 1.0, "ntok": ntok, "h_kb": (K + 1) * 3072 * 2 / 1024, "ok": ok, "text": "hi"}


_n = 0
def check(cond, msg):
    global _n
    _n += 1
    assert cond, f"FAIL {_n}: {msg}"
    print(f"  ok {_n}: {msg}")


# 1. winner = highest decode tok/s among ok configs
_, best = m25_pipe._sweep_summary([row(6, 4, 15.0), row(12, 4, 18.5), row(16, 4, 17.0)])
check(best["K"] == 12 and best["depth"] == 4, "winner = highest tok/s (K=12)")

# 2. a FAILED config never wins, even with a higher (bogus) tok/s, and is flagged in the table
table, best = m25_pipe._sweep_summary([row(6, 4, 15.0), row(16, 4, 99.0, ok=False)])
check(best["K"] == 6, "failed config excluded from winner despite higher tok_s")
check("<-- FAIL" in table, "failed config flagged in table")

# 3. all-failed sweep -> best is None, no crash (empty-safe)
_, best = m25_pipe._sweep_summary([row(6, 4, 0.0, ok=False)])
check(best is None, "all-failed sweep returns best=None")

# 4. an ok-but-zero-tok/s config is excluded by the tok_s>0 guard
_, best = m25_pipe._sweep_summary([row(6, 4, 0.0, ok=True), row(8, 4, 5.0, ok=True)])
check(best["K"] == 8, "ok-but-zero-tok/s config excluded (tok_s>0 guard)")

# 5. table renders a header + one line per config + BEST footer
table, _ = m25_pipe._sweep_summary([row(4, 2, 10.0), row(8, 2, 12.0)])
check("tok/s" in table and table.count("\n") >= 6, "table has header + rows")
check("BEST: K=8" in table, "BEST footer names the winner")

# 6. 2-D sweep (K x depth): winner is the global max, not per-row
_, best = m25_pipe._sweep_summary([row(8, 2, 14.0), row(8, 4, 16.0), row(12, 2, 15.0), row(12, 4, 13.0)])
check(best["K"] == 8 and best["depth"] == 4, "2-D winner = global max over K x depth")

print(f"\nVERDICT: _sweep_summary {_n}/{_n} — winner selection, fail-exclusion, zero-guard, empty-safe, 2-D.")
