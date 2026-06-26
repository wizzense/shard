"""Local (no-GPU) proof of the trustless-verification path: real ed25519 signing + the coordinator's
exact verification (verify_receipt + verify_coverage), exercising the receipt accumulation the
m25_pipe direct-return pipeline produces, plus the fail-closed negatives.

Run: venv/bin/python research/m25_receipt_test.py   (needs `cryptography`)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shard"))
from receipt import ReceiptSigner, verify_receipt, verify_coverage, ReceiptError
from manifest import gen_key

P = F = 0
def ok(c, name):
    global P, F
    if c: P += 1; print(f"  PASS {name}")
    else: F += 1; print(f"  FAIL {name}")

LAYERS = 62
BLOCKS = [(0, 21), (21, 42), (42, 62)]          # the 3-stage M2.5 tiling
keys = [gen_key() for _ in BLOCKS]

# --- simulate one job exactly as the pipeline does: signer per stage on reset, observe per chunk ---
signers = [ReceiptSigner(keys[i], "swarm-x", "job-7", lo, hi) for i, (lo, hi) in enumerate(BLOCKS)]
for step in range(6):                            # 6 verify chunks through the ring
    for s in signers:
        s.observe(f"in-{step}".encode(), f"out-{step}".encode())

# accumulate forward (head -> middle -> tail), tail labelled "tail" — the wire order the coord receives
receipts = []
for i, s in enumerate(signers):
    receipts.append({"stage": ("tail" if i == len(signers) - 1 else i), **s.finalize()})
bodies = [{k: v for k, v in r.items() if k != "stage"} for r in receipts]

print("== happy path ==")
ok([r["stage"] for r in receipts] == [0, 1, "tail"], "accumulation order head->middle->tail")
ok(all(b["n_chunks"] == 6 for b in bodies), "each stage chained all 6 chunks")
all_valid = True
for b in bodies:
    try: verify_receipt(b)
    except ReceiptError: all_valid = False
ok(all_valid, "every per-stage signature VALID")
ok(len({b["pubkey"] for b in bodies}) == 3, "three distinct signer pubkeys (no coordinator forgery)")
cov_ok = True
try: verify_coverage(bodies, LAYERS)
except ReceiptError: cov_ok = False
ok(cov_ok, "coverage tiles [0:62] no gap/overlap")

print("== fail closed: tampered signature ==")
bad = dict(bodies[1]); bad["out_root"] = "0" * 64       # alter attested output, keep old sig
raised = False
try: verify_receipt(bad)
except ReceiptError: raised = True
ok(raised, "tampered out_root -> ReceiptError")

print("== fail closed: coordinator forgery (swap pubkey) ==")
forge = dict(bodies[0]); forge["pubkey"] = bodies[2]["pubkey"]   # claim another node signed it
raised = False
try: verify_receipt(forge)
except ReceiptError: raised = True
ok(raised, "pubkey swap breaks signature -> ReceiptError")

print("== fail closed: coverage gap (missing middle block) ==")
raised = False
try: verify_coverage([bodies[0], bodies[2]], LAYERS)
except ReceiptError: raised = True
ok(raised, "missing [21:42] -> coverage ReceiptError")

print("== fail closed: coverage overlap ==")
ov = ReceiptSigner(keys[0], "swarm-x", "job-7", 10, 42); ov.observe(b"i", b"o")
raised = False
try: verify_coverage([bodies[0], {k: v for k, v in ov.finalize().items()}, bodies[2]], LAYERS)
except ReceiptError: raised = True
ok(raised, "overlapping blocks -> coverage ReceiptError")

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
