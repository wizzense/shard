# Deploy-readiness — what's left to make the engine usable & deployable

Status after 2026-06-23: the concrete bar is met (28.2 tok/s decode at >100k on a copy/retrieval
task, 3-node WA/WA/TX WAN swarm, greedy-exact, signed per-stage receipts live — see STATE.md). This
is the honest gap list between "works in a niche" and "deployable general engine."

Tags: 🔴 blocking for any honest deploy · 🟡 needed to escape the niche · 🟢 scale-hardening.
**E**ngineering (shard repo) · **R**esearch · **I**ntegration (c0mpute repo / not this engine).

## 1. Latency (what users feel)
- 🔴 **E — Time-to-first-token at long ctx.** 100k prefill is ~10 min (~180 tok/s, sequential through
  the ring). *Done when:* TTFT(100k) < ~60s. Levers: overlap/parallelize prefill across stages, fp8/int8
  KV (bandwidth), prefill-specialized path. Until then it's batch-only, never interactive.
- 🟡 **E — Decode on NOVEL generation.** 28 tok/s is on copy/retrieval; novel prose at 100k is ~2–3
  tok/s (n-gram has nothing to copy). *Done when:* ≥20 tok/s at 100k on open-ended generation. Needs a
  **real long-context draft** — small (1–3B) draft with fp8 windowed KV that doesn't OOM, or an
  EAGLE-style trained head matched to gpt-oss. **Biggest single "make it general" unlock.**
- 🟢 **E — fp8/int8 KV + weights** to cut the per-stage 100k-attention bottleneck (the fat node is the floor).

## 2. Generality (does what people pay for)
- 🔴 **E — Sampling, not just greedy.** Need lossless speculative *sampling* (temperature/top-p) at parity.
- 🔴 **E — Concurrent request batching.** One stream at a time today; throughput economics need continuous batching.
- 🟡 **I — >1 model in the catalog** (incl. an uncensored one — the actual differentiator). Manifest/fetch supports it.

## 3. Reliability (survives a real consumer-GPU swarm)
- 🔴 **R — Mid-request fault tolerance.** A node dropping mid-gen fails the request today. *Done when:* kill a
  node mid-stream and the request still completes (KV migration or fast re-prefill of just the dropped block).
  The hardest non-crypto item; churn is constant on volunteer GPUs.
- 🔴 **E/I — Live heal + hot spares** (pre-warmed) so a drop is a <few-sec blip, not a cold relaunch.
- 🟡 **E — SLA behavior.** Graceful degradation + health so the orchestrator routes around flaky nodes.

## 4. It's a live permissionless network, not a hand-deployed engine
- 🔴 **I — One-command join** (deps + driver check + pull only the assigned block + register + serve), home-NAT.
- 🔴 **I — Live scheduler/control-plane** — wire `shard/scheduler.py` into the c0mpute orchestrator; swarms form
  from the live pool automatically.
- 🔴 **I — PAY** — per-node USDC on `worker_earnings`, keyed on verified receipts. The line between engine and network.

## 5. Trust enforcement (the moat bites)
- 🔴 **I — Enforce the layer-block challenge live** (random redundant recompute on a trusted node → strike).
  Primitive built (`shard/challenge.py`); policy loop is c0mpute-side.
- 🔴 **I — Stake + slash + graded reputation** the scheduler consumes (c0mpute rep is binary today).
- 🟢 **R — Crypto proof-of-compute** (ZK/commitments) to replace recompute-and-compare. Long horizon; the
  receipt's in/out-root slot is the drop-in point. Economic enforcement covers launch.

## 6. Privacy (currently an unaddressed leak)
- 🔴 **E/I — Boundary-layer pinning** — keep leaky embed/final blocks on staked/trusted nodes; untrusted
  volunteers hold only deep middle blocks.
- 🟡 **I — Per-request trusted-only routing** for sensitive jobs. Don't sell for sensitive use until done.

## 7. Economics & ops
- 🔴 **I — Metered pricing lane** (flat per-tier mis-prices a slow frontier swarm).
- 🟡 **E — Supply** (enough idle/volunteer GPUs that a swarm forms without renting).
- 🟢 **E — Production ops:** harden the supervisor, monitoring, security pass on transport + rendezvous.

---

## Minimum for a first real (niche) deploy
§3 fault tolerance · §4 join + scheduler + PAY · §5 challenge enforcement + slashing · §2 sampling · §7 pricing
→ uncensored/private long-context retrieval on volunteer GPUs, provably honest.

## To be a general alternative to centralized AI
Add §1 (TTFT + real draft for novel gen) · §2 batching · §6 privacy. Even then: competes on **access,
idle-compute cost, and trustless verification — never raw speed** (WAN latency floor). Never message "faster
than OpenAI"; the truthful pitch is "frontier models on terms they won't give you, provably honest."

## Highest-leverage ENGINE-side (shard repo) next steps
1. Real long-context draft model → novel-gen ≥20 tok/s (turns a niche tool into a general one).
2. Mid-request fault tolerance → survives a real swarm.
3. Faster prefill / TTFT → interactive, not batch-only.
4. Lossless speculative sampling → real workloads, not just greedy.
(PAY / live scheduler / challenge-enforcement / pricing are c0mpute-repo integration, a separate effort.)
