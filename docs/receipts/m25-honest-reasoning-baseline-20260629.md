# Honest reasoning-ON baseline (n-gram only) — 2026-06-29

Scattered 6-box EU ring (FR-SE2-GB-CZ-SE-NO, libp2p), single-stream, REASONING ON, K=8 depth=4,
n-gram drafter only. research/m25_honest_bench.py. NO copy-repetition, NO think-skip.

   category    p_tok  tok/s  accept  g    prefill  TTFT   visible   think  answer
 reason-math      67    1.8     0%  1.0    1.9s    2.0s   68.6s     162     41
 reason-logic     88    2.1     2%  1.1    4.4s    4.6s    >max       0    259
 open-chat        58    2.6     0%  1.0    2.7s    2.9s    >max       0    256
 code-edit      3563    3.3     0%  1.0    9.1s    9.3s    >max       0    256
 rag-quote      3562    8.7    19%  2.5    5.0s    5.1s    >max       0    257
 agentic-tool    202    3.3     0%  1.0    1.3s    1.5s   10.3s      29     26
 decode-weighted mean = 2.8 tok/s

Takeaway: reasoning-ON normal usage ~2-3 tok/s on a same-continent scattered ring; n-gram only helps
VERBATIM-reuse (rag-quote). The <think> block dominates latency (reason-math: 68s to first visible answer).
This is the "before" for the EAGLE hybrid-drafter work.
