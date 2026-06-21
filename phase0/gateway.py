"""shard gateway — a shared LIVE TERMINAL into the gpt-oss-120B swarm.

Everyone who opens shard.c0mpute.ai watches the SAME stream over one SSE channel; prompts go
through a public FIFO queue and run one at a time on the single-stream swarm. Tokens arrive in
BATCHES (pipelined spec-decode commits K accepted tokens per ring traversal) — the UI emits the
DELTA + batch size per commit and animates each batch landing, so the distributed-batch nature is
visible instead of faked as smooth char streaming.

Aesthetic matches c0mpute: black, white/grey, argent-pixel-cf (Typekit kwe2dpm) + Courier, thin
white borders, subtle radius, no gradients.

  SHARD_PSK=... python3 gateway.py --head IP:PORT --tail IP:PORT --port 29600 --nodes-file nodes.json
  GATEWAY_MOCK=1 python3 gateway.py --head x --tail x   # local UI/stream/queue test, no swarm
"""
import argparse, collections, itertools, json, os, queue, random, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--head", required=True)
ap.add_argument("--tail", required=True)
ap.add_argument("--draft", default="127.0.0.1:8200")
ap.add_argument("--model", default="/root/models/gpt-oss-20b")
ap.add_argument("--port", type=int, default=29600)
ap.add_argument("--nodes", default="[]")
ap.add_argument("--nodes-file", default="")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--max-ctx", type=int, default=8192)
A = ap.parse_args()
MOCK = bool(os.environ.get("GATEWAY_MOCK"))
if A.nodes_file:
    A.nodes = open(A.nodes_file).read()
NODES = json.loads(A.nodes)
MAXQ, MAXPROMPT = 12, 400000   # ~95k tokens of input (paste a whole document); long prefill is slow but supported

if not MOCK:
    sys.path.insert(0, "/root")
    import wire; wire.key_from_env()
    import specpipe as sp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    HEAD = (A.head.split(":")[0], int(A.head.split(":")[1]))
    TAIL = (A.tail.split(":")[0], int(A.tail.split(":")[1]))
    DRAFT = (A.draft.split(":")[0], int(A.draft.split(":")[1]))
    SOCKS = {}
    def connect():
        for s in SOCKS.values():
            try: s.close()
            except Exception: pass
        SOCKS.clear()
        d = socket.socket(); d.connect(DRAFT)
        p = socket.socket(); p.settimeout(1800); p.connect(HEAD)
        r = socket.socket(); r.settimeout(1800); r.connect(TAIL)
        sp.send_msg(r, {"op": "hello_return"})
        SOCKS.update(draft=d, pipe=p, ret=r)

# ---- shared broadcast + queue state ----
CLIENTS, CLIENTS_LOCK, CID = {}, threading.Lock(), itertools.count(1)
JOBS, JOBS_LOCK, WAKE, JID = collections.deque(), threading.Lock(), threading.Event(), itertools.count(1)
STATE = {"phase": "idle", "jid": None, "prompt": None, "name": None, "text": "",
         "ptoks": 0, "stats": {}, "batches": 0, "ip": None}
IP_LAST = {}                       # client ip -> last submit ts (rate limit + memory-bounded)
RATE_SECS = 10                     # min seconds between an IP's submits

def broadcast(ev):
    with CLIENTS_LOCK:
        for q in list(CLIENTS.values()):
            try: q.put_nowait(ev)
            except queue.Full: pass

def waiting_list():
    with JOBS_LOCK:
        return [{"jid": j["jid"], "name": j["name"]} for j in JOBS]

def running_ref():
    return {"jid": STATE["jid"], "name": STATE["name"]} if STATE["phase"] == "running" else None

def queue_ev():
    return {"t": "queue", "waiting": waiting_list(), "running": running_ref()}

def snapshot_ev():
    return {"t": "snapshot", "phase": STATE["phase"], "jid": STATE["jid"], "prompt": STATE["prompt"],
            "name": STATE["name"], "text": STATE["text"], "stats": STATE["stats"],
            "batches": STATE["batches"], "waiting": waiting_list(), "running": running_ref()}

def emit_batch(full_text, ptoks, n, dt):
    """one spec-decode commit: broadcast just the NEW suffix + how many tokens landed this batch."""
    delta = full_text[len(STATE["text"]):]
    k = n - STATE["stats"].get("n", 0)
    STATE["text"] = full_text
    STATE["batches"] += 1
    STATE["stats"] = {"n": n, "tok_s": round(n / max(dt, 1e-9), 2), "ctx": ptoks + n}
    broadcast({"t": "batch", "delta": delta, "k": k, "n": n,
               "tok_s": STATE["stats"]["tok_s"], "ctx": ptoks + n, "batches": STATE["batches"]})

def run_real(job):
    cb = {"ptoks": 0}
    def on_commit(ev):
        if ev["phase"] == "prefilled":
            cb["ptoks"] = ev["prompt_tokens"]; STATE["ptoks"] = ev["prompt_tokens"]
            broadcast({"t": "prefilled", "prompt_tokens": ev["prompt_tokens"]})
        else:
            full = tok.decode(ev["out"], skip_special_tokens=True)
            emit_batch(full, cb["ptoks"], len(ev["out"]), ev["dt"])
    for attempt in (1, 2):
        try:
            if "pipe" not in SOCKS or attempt == 2:
                print(f"[gw] connect (attempt {attempt})", flush=True); connect()
            broadcast({"t": "prefilling"})
            r = sp.coordinate_pipe(SOCKS["draft"], SOCKS["pipe"], tok, job["prompt"], 4, A.max_new,
                                   1800, 4, ret_sock=SOCKS["ret"], prefill_chunk=4096,
                                   draft_ctx=8000, on_commit=on_commit, reasoning="low", max_ctx=A.max_ctx)
            broadcast({"t": "done", "n": r["n_tokens"], "tok_s": round(r["tok_s"], 2),
                       "accept": round(r["mean_accept"], 2), "tpt": round(r["toks_per_traversal"], 2)})
            return
        except Exception as e:
            import traceback; traceback.print_exc()
            SOCKS.clear()
            if attempt == 2:
                broadcast({"t": "error", "error": f"{type(e).__name__}: {str(e)[:140]}"})

def run_mock(job):
    # marker glued like the real decode ("...analysis<reasoning>assistantfinal<answer>"), streamed in
    # small char-slices so MARK straddles batch boundaries -> exercises the buffered split.
    text = ("analysisWe briefly consider: " + job["prompt"][:60] +
            ".assistantfinal## Sharded inference\n\n**Sharded inference** splits a model too big for one GPU "
            "across several scattered devices; each holds a slice of the weights and passes activations to "
            "the next, so the model runs as if it were whole.\n\nKey ideas:\n\n- **Memory** — no single "
            "device holds the whole model\n- **Speculative decoding** — each ring traversal commits a *batch* "
            "of tokens at once\n- **Dynamic context** — every decode step stays cheap as the answer grows\n\n"
            "| Split type | What it shards |\n|---|---|\n| Tensor | weight matrices |\n| Pipeline | layers |\n\n"
            "That's why text lands in bursts and streams at a steady rate to the end.")
    broadcast({"t": "prefilling"}); time.sleep(0.4)
    ptoks = len(job["prompt"].split()) + 8; STATE["ptoks"] = ptoks
    broadcast({"t": "prefilled", "prompt_tokens": ptoks}); time.sleep(0.2)
    i, n, t0, full = 0, 0, time.time(), ""
    while i < len(text):
        step = random.randint(8, 18); chunk = text[i:i + step]; i += step
        full += chunk; n += max(1, len(chunk) // 4)
        time.sleep(0.12)
        emit_batch(full, ptoks, n, time.time() - t0)
    broadcast({"t": "done", "n": n, "tok_s": round(n / max(time.time() - t0, 1e-9), 2),
               "accept": 3.04, "tpt": 4.04})

def warmup():
    """ONE-TIME at boot (not per-prompt): compile the draft's Triton attention/slot kernels across
    the seqlen range it will hit, plus the swarm's flex prefill + decode graph. Without this the
    draft JIT-compiles mid-generation the first time it meets a new shape -> the latency spikes you
    see as 'hiccups at the coord'. Runs once before serving; after it, every prompt is warm."""
    STATE["phase"] = "warming"; broadcast({"t": "warming"})
    try:
        connect()
        print("[gw] warmup: draft Triton kernels (seqlen sweep)…", flush=True)
        for L in (128, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8000):   # cover the draft's decode
            sp.send_msg(SOCKS["draft"], {"ids": list(range(100, 100 + L)), "k": 8}); sp.recv_msg(SOCKS["draft"])  # kernels up to draft_ctx
        print("[gw] warmup: swarm flex prefill + decode graph…", flush=True)
        sp.coordinate_pipe(SOCKS["draft"], SOCKS["pipe"], tok, "Briefly describe distributed computing.",
                           4, 32, 1800, 4, ret_sock=SOCKS["ret"], prefill_chunk=4096, draft_ctx=8000,
                           reasoning="low", max_ctx=A.max_ctx, ignore_eos=True)
        print("[gw] warmup done — coord + swarm warm", flush=True)
    except Exception as e:
        print(f"[gw] warmup failed (continuing): {type(e).__name__}: {e}", flush=True); SOCKS.clear()
    finally:
        STATE["phase"] = "idle"; broadcast({"t": "idle"})

def worker():
    if not MOCK:
        warmup()
    runner = run_mock if MOCK else run_real
    while True:
        WAKE.wait()
        while True:
            with JOBS_LOCK:
                job = JOBS.popleft() if JOBS else None
                if not job: WAKE.clear()
            if not job: break
            STATE.update(phase="running", jid=job["jid"], prompt=job["prompt"], name=job["name"],
                         text="", ptoks=0, stats={}, batches=0, ip=job.get("ip"))
            broadcast({"t": "start", "jid": job["jid"], "prompt": job["prompt"], "name": job["name"]})
            broadcast(queue_ev())
            try:
                runner(job)
            except Exception as e:
                broadcast({"t": "error", "error": f"{type(e).__name__}: {str(e)[:140]}"})
            STATE["phase"] = "idle"; STATE["jid"] = None
            broadcast({"t": "idle"}); broadcast(queue_ev())

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self): self.send_header("Access-Control-Allow-Origin", "*")
    def _bytes(self, b, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self._cors()
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/live": return self.sse()
        if self.path == "/nodes": return self._bytes(json.dumps(NODES).encode(), "application/json")
        self._bytes(PAGE.encode(), "text/html; charset=utf-8")
    def do_POST(self):
        if self.path != "/submit":
            return self._bytes(b'{"error":"not found"}', "application/json", 404)
        n = int(self.headers.get("Content-Length", 0))
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: body = {}
        prompt = (body.get("prompt") or "").strip()[:MAXPROMPT]
        name = (str(body.get("name") or "anon"))[:16]
        if not prompt:
            return self._bytes(b'{"error":"empty prompt"}', "application/json", 400)
        ip = (self.headers.get("X-Forwarded-For") or self.headers.get("X-Real-IP")
              or self.client_address[0] or "?").split(",")[0].strip()      # real client (nginx forwards it)
        with JOBS_LOCK:
            running_ip = STATE["ip"] if STATE["phase"] == "running" else None
            if running_ip == ip or any(j.get("ip") == ip for j in JOBS):   # ONE prompt per ip at a time (anti-monopolize)
                return self._bytes(b'{"error":"you already have a prompt running or queued -- one at a time"}', "application/json", 429)
            now = time.time()
            if now - IP_LAST.get(ip, 0) < RATE_SECS:                       # cooldown between an ip\'s submits
                return self._bytes(b'{"error":"easy -- wait a few seconds between prompts"}', "application/json", 429)
            if len(JOBS) >= MAXQ:
                return self._bytes(b'{"error":"queue full, try again shortly"}', "application/json", 429)
            IP_LAST[ip] = now
            if len(IP_LAST) > 4000:                                        # bound memory: drop stale entries in place
                for k in [k for k, v in IP_LAST.items() if now - v > 3600]: IP_LAST.pop(k, None)
            jid = next(JID); JOBS.append({"jid": jid, "prompt": prompt, "name": name, "ip": ip}); pos = len(JOBS)
        WAKE.set(); broadcast(queue_ev())
        self._bytes(json.dumps({"jid": jid, "position": pos}).encode(), "application/json")
    def sse(self):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive")
        self._cors(); self.end_headers()
        cid = next(CID); q = queue.Queue(maxsize=256)
        with CLIENTS_LOCK: CLIENTS[cid] = q
        try:
            self._ev(snapshot_ev())
            while True:
                try: ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                self._ev(ev)
        except Exception:
            pass
        finally:
            with CLIENTS_LOCK: CLIENTS.pop(cid, None)
    def _ev(self, ev):
        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>shard · c0mpute</title>
<link rel="stylesheet" href="https://use.typekit.net/kwe2dpm.css">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<style>
:root{--mono:'Courier New',Courier,monospace;--pix:"argent-pixel-cf",'Courier New',monospace;
--fg:#d9d9d9;--white:#fff;--mut:rgba(255,255,255,.5);--mut2:rgba(255,255,255,.32);
--ln:rgba(255,255,255,.12);--ln2:rgba(255,255,255,.2);--card:#0a0a0a;--live:#22c55e;--err:#ef4444;--r:10px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#000;color:var(--fg);font-family:var(--mono);height:100vh;overflow:hidden;
-webkit-font-smoothing:antialiased}
.pix{font-family:var(--pix);font-smooth:never;-webkit-font-smoothing:none;text-rendering:optimizeSpeed}
header{display:flex;align-items:center;gap:14px;padding:15px 24px;border-bottom:1px solid var(--ln)}
.wm{font-family:var(--pix);color:#fff;font-size:21px;letter-spacing:.01em;display:flex;align-items:baseline;text-decoration:none}
header a{text-decoration:none}
.wm .z{font-size:1.8em;line-height:1;margin-top:-.3em}
.div{width:1px;height:18px;background:var(--ln2);margin:0 4px}
.tag{color:var(--mut);font-size:13px;letter-spacing:.04em}
.right{margin-left:auto;display:flex;align-items:center;gap:16px}
.live{display:flex;align-items:center;gap:7px;color:var(--mut);font-size:12px;letter-spacing:.05em}
.dot{width:7px;height:7px;border-radius:50%;background:var(--mut2)}
.dot.on{background:var(--live);box-shadow:0 0 9px var(--live);animation:bl 1.2s infinite}
@keyframes bl{50%{opacity:.35}}
a.gh{color:var(--mut);transition:.15s}a.gh:hover{color:#fff}
main{display:flex;height:calc(100vh - 57px)}
#mapwrap{flex:1.45;position:relative;min-width:0;border-right:1px solid var(--ln)}
svg{width:100%;height:100%}
.state{fill:#070707;stroke:rgba(255,255,255,.07);stroke-width:.6}
.edge{stroke:rgba(255,255,255,.14);stroke-width:1.3;fill:none}
.edge.on{stroke:#fff;stroke-width:2;filter:drop-shadow(0 0 5px rgba(255,255,255,.7))}
.halo{fill:none;stroke:rgba(255,255,255,.10);stroke-width:.7}
.nd{fill:#fff;filter:drop-shadow(0 0 3px rgba(255,255,255,.5))}.nd.coord{fill:#000;stroke:#fff;stroke-width:1.3}
.lbl{fill:rgba(255,255,255,.6);font:11px var(--mono)}.lbl .r{fill:var(--mut2);font-size:10px}
.hop{fill:rgba(255,255,255,.42);font:9.5px var(--mono)}
.maptag{position:absolute;left:18px;bottom:16px;color:var(--mut2);font-size:11px;line-height:1.6;background:rgba(0,0,0,.55);border:1px solid var(--ln);border-radius:8px;padding:9px 12px;-webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px);max-width:340px}
.maptag a{color:rgba(138,180,255,.88);text-decoration:none}.maptag a:hover{text-decoration:underline}
.maptag .nip{color:rgba(255,255,255,.72)}.maptag .nl{color:var(--mut2)}.maptag b{color:#fff}
aside{flex:1;max-width:540px;min-width:400px;display:flex;flex-direction:column;padding:18px;gap:13px;background:#000}
.card{border:1px solid var(--ln);border-radius:var(--r);background:var(--card)}
.serving{padding:12px 14px;display:flex;flex-direction:column;gap:5px}
.serving .lab{color:var(--mut2);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase}
.serving .pr{color:#fff;font-size:13.5px;line-height:1.45;max-height:42px;overflow:hidden}
.serving .who{color:var(--mut);font-size:11px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.stat{border:1px solid var(--ln);border-radius:var(--r);background:var(--card);padding:9px 11px}
.stat .k{color:var(--mut2);font-size:10px;letter-spacing:.08em;text-transform:uppercase}
.stat .v{font-size:19px;margin-top:3px;color:#fff}.stat .v.acc{color:var(--live)}
.outwrap{flex:1;position:relative;display:flex;min-height:0}
#out{flex:1;overflow:auto;border:1px solid var(--ln);border-radius:var(--r);background:var(--card);
padding:15px 16px;font-size:13.5px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.think{color:var(--mut);font-size:12.5px}
.ans{color:#e3e3e3}
.ans>*:first-child{margin-top:0}
.ans h1,.ans h2,.ans h3,.ans h4{color:#fff;font-weight:700;line-height:1.25;margin:.85em 0 .4em}
.ans h1{font-size:1.28em}.ans h2{font-size:1.14em}.ans h3{font-size:1.04em}
.ans p{margin:.5em 0}
.ans strong,.ans b{color:#fff;font-weight:700}
.ans ul,.ans ol{margin:.45em 0;padding-left:1.4em}.ans li{margin:.18em 0}
.ans code{font-family:var(--mono);background:rgba(255,255,255,.09);padding:1px 5px;border-radius:4px;font-size:.92em}
.ans pre{background:#050505;border:1px solid var(--ln);border-radius:8px;padding:11px 13px;overflow-x:auto;margin:.6em 0}
.ans pre code{background:none;padding:0}
.ans table{border-collapse:collapse;margin:.7em 0;display:block;overflow-x:auto;font-size:.94em}
.ans th,.ans td{border:1px solid var(--ln2);padding:5px 11px;text-align:left;white-space:nowrap}
.ans th{background:rgba(255,255,255,.06);color:#fff;font-weight:700}
.ans a{color:#8ab4ff;text-decoration:underline}
.ans blockquote{border-left:2px solid var(--ln2);padding-left:11px;margin:.5em 0;color:var(--mut)}
.divider{display:block;height:1px;background:var(--ln);margin:11px 0}
.batch{border-radius:3px;animation:land 1s ease-out}
@keyframes land{0%{background:rgba(34,197,94,.30);color:#fff}55%{background:rgba(34,197,94,.10)}100%{background:transparent}}
.cur{display:inline-block;width:7px;height:14px;background:var(--live);vertical-align:-2px;margin-left:1px;animation:bl .9s infinite}
#badge{position:absolute;top:10px;right:12px;font-size:11px;color:var(--live);border:1px solid rgba(34,197,94,.4);
border-radius:6px;padding:2px 8px;opacity:0;pointer-events:none;background:rgba(34,197,94,.08)}
#badge.pop{animation:pop 1s ease-out}
@keyframes pop{0%{opacity:0;transform:translateY(5px)}18%{opacity:1;transform:none}80%{opacity:1}100%{opacity:0}}
.qrow{display:flex;align-items:center;gap:10px;color:var(--mut);font-size:11.5px;letter-spacing:.03em;min-height:16px}
.qrow .you{color:var(--live)}
.inrow{display:flex;gap:9px;align-items:stretch}
textarea{flex:1;height:62px;resize:none;background:#050505;color:var(--fg);border:1px solid var(--ln2);
border-radius:var(--r);padding:11px 12px;font:13px/1.5 var(--mono);outline:none;transition:.15s}
textarea:focus{border-color:rgba(255,255,255,.4)}
button{background:#fff;color:#000;border:0;border-radius:var(--r);padding:0 18px;font:700 13px var(--mono);
letter-spacing:.02em;cursor:pointer;transition:.12s;white-space:nowrap}
button:hover{background:rgba(255,255,255,.88)}button:disabled{opacity:.4;cursor:not-allowed}
.foot{color:var(--mut2);font-size:10.5px;text-align:center;letter-spacing:.04em}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.15);border-radius:3px}
@media(max-width:880px){#mapwrap{display:none}aside{max-width:none}}
</style></head>
<body>
<header>
  <a href="https://c0mpute.ai" class="wm pix">C<span class=z>0</span>MPUTE</a>
  <div class=div></div><div class="tag pix">shard · live swarm</div>
  <div class=right>
    <div class=live><span class=dot id=dot></span><span id=livet>connecting</span></div>
    <a class=gh href="https://github.com/leyten/c0mpute" target=_blank aria-label=github>
      <svg width=16 height=16 viewBox="0 0 24 24" fill=currentColor><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
    </a>
  </div>
</header>
<main>
  <div id=mapwrap><svg id=svg viewBox="0 0 960 600" preserveAspectRatio="xMidYMid meet"></svg>
    <div class=maptag id=maptag>gpt-oss-120B · 36 layers split across scattered RTX 4090s</div>
  </div>
  <aside>
    <div class="card serving">
      <div class=lab>now serving · live to everyone</div>
      <div class=pr id=serv_pr>swarm idle — submit a prompt below</div>
      <div class=who id=serv_who></div>
    </div>
    <div class=stats>
      <div class=stat><div class=k>tok/s</div><div class="v acc" id=s_tps>—</div></div>
      <div class=stat><div class=k>tokens</div><div class=v id=s_n>—</div></div>
      <div class=stat><div class=k>context</div><div class=v id=s_ctx>—</div></div>
      <div class=stat><div class=k>batches</div><div class=v id=s_b>—</div></div>
    </div>
    <div class=outwrap>
      <div id=out><span style="color:var(--mut2)">the swarm's output streams here — in batches, the way pipelined spec-decode actually commits them.</span></div>
      <div id=badge></div>
    </div>
    <div class=qrow id=qrow></div>
    <div class=inrow>
      <textarea id=prompt placeholder="ask the swarm anything — or paste a long document (up to ~95k tokens). runs in the public queue."></textarea>
      <button id=go>run →</button>
    </div>
    <div class=foot id=foot>shared terminal · one stream at a time · be kind, it's real hardware</div>
  </aside>
</main>
<script>
const $=id=>document.getElementById(id);
const ME=(localStorage.shard_h||(localStorage.shard_h="anon-"+Math.random().toString(36).slice(2,5)));
let RING=[],HOPS=[],LOOP=0,myJid=null,answering=false,pending="",leadStripped=false,thinkStr="",ansStr="";
// ---- map ----
const svg=d3.select("#svg"),proj=d3.geoAlbersUsa().scale(1160).translate([480,300]);
fetch("/nodes").then(r=>r.json()).then(data=>{
 const ns=data.nodes||data; HOPS=data.hops||[]; LOOP=data.loop_ms||0;
 d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json").then(us=>{
  svg.append("g").selectAll("path").data(topojson.feature(us,us.objects.states).features)
     .join("path").attr("class","state").attr("d",d3.geoPath(proj));
  const pts=ns.map(n=>({...n,xy:proj([n.lon,n.lat])})).filter(n=>n.xy);
  const coord=pts.find(p=>p.role==="coord"),stages=pts.filter(p=>p.role!=="coord").sort((a,b)=>a.idx-b.idx);
  RING=[coord,...stages,coord].filter(Boolean);
  const eg=svg.append("g");
  for(let i=0;i<RING.length-1;i++)eg.append("path").attr("class","edge").attr("id","e"+i)
    .attr("d",`M${RING[i].xy[0]},${RING[i].xy[1]}L${RING[i+1].xy[0]},${RING[i+1].xy[1]}`);
  if(HOPS.length===RING.length-1){ const el=svg.append("g");   // MEASURED per-hop RTT on each edge (real latency)
   for(let i=0;i<RING.length-1;i++){ const a=RING[i].xy,b=RING[i+1].xy;
    el.append("text").attr("class","hop").attr("x",(a[0]+b[0])/2).attr("y",(a[1]+b[1])/2-3)
      .attr("text-anchor","middle").text(HOPS[i]+"ms"); } }
  const g=svg.append("g");
  g.selectAll(".halo").data(pts).join("circle").attr("class","halo").attr("cx",d=>d.xy[0]).attr("cy",d=>d.xy[1]).attr("r",6.5);
  g.selectAll(".nd").data(pts).join("circle").attr("class",d=>"nd"+(d.role==="coord"?" coord":"")).attr("cx",d=>d.xy[0]).attr("cy",d=>d.xy[1]).attr("r",3.5);
  g.selectAll(".lbl").data(pts).join("text").attr("class","lbl").attr("x",d=>d.xy[0]+11).attr("y",d=>d.xy[1]+4)
    .html(d=>`<tspan>${d.city}</tspan><tspan class=r dx=6>${d.role}${d.layers&&d.layers!=='draft'?' L'+d.layers:''}</tspan>`);
  // verifiable node panel: real public IPs anyone can whois/geolocate — different ISPs, different states
  let h=(LOOP?`<b>ring loop ${LOOP}ms</b> · gpt-oss-120B · ${stages.length} scattered 4090s<br>`:"");
  h+=RING.slice(0,-1).map(n=>`<span class=nip>${n.city}</span> ${n.ip?`<a href="https://ipinfo.io/${n.ip}" target=_blank rel=noopener>${n.ip} ↗</a>`:""} <span class=nl>${n.layers&&n.layers!=='draft'?'L'+n.layers:n.role}</span>`).join("<br>");
  h+=`<br><span class=nl>↗ check the IPs yourself — real consumer ISPs, not one datacenter</span>`;
  $("maptag").innerHTML=h;
 });
});
function pulseRing(){ if(RING.length<2)return; let k=0;
 (function hop(){ if(k>=RING.length-1)return; svg.select("#e"+k).classed("on",true);
   const a=RING[k].xy,b=RING[k+1].xy;
   const dur=Math.max(45,Math.min(420,(HOPS[k]||25)*5));      // spark travels proportional to the MEASURED hop RTT
   svg.append("circle").attr("r",3).attr("fill","#fff").attr("cx",a[0]).attr("cy",a[1])
     .transition().duration(dur).ease(d3.easeLinear).attr("cx",b[0]).attr("cy",b[1]).remove();
   const kk=k; setTimeout(()=>svg.select("#e"+kk).classed("on",false),dur+40); k++; setTimeout(hop,dur+4); })();
}
// ---- output rendering: append each BATCH as an animated span ----
const out=$("out");
function esc(s){return(s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const MARK="assistantfinal";   // gpt-oss channel boundary (decoded, specials stripped)
// answer is markdown -> render + SANITIZE (shared public terminal: never inject another viewer's DOM)
const md=t=>{ try{ return DOMPurify.sanitize(marked.parse(t||"",{breaks:true,gfm:true})); }catch(e){ return esc(t); } };
function resetRender(){ thinkStr=""; ansStr=""; answering=false; pending=""; leadStripped=false; renderOut(); }
function renderOut(cursor){
 let h="";
 if(thinkStr) h+=`<div class=think>${esc(thinkStr)}</div>`;
 if(answering){ if(thinkStr) h+=`<div class=divider></div>`; h+=`<div class=ans>${md(ansStr)}</div>`; }
 if(cursor!==false) h+=`<span class=cur></span>`;
 out.innerHTML=h; out.scrollTop=out.scrollHeight;
}
// buffer text so the MARK is detected even when it straddles two batches; hold back a possible
// partial marker while still in the thinking channel, then split think vs answer.
function land(delta,k){
 pending+=delta;
 if(!leadStripped){ pending=pending.replace(/^analysis/,""); leadStripped=true; }
 if(!answering){
   const i=pending.indexOf(MARK);
   if(i>=0){ thinkStr+=pending.slice(0,i); answering=true; ansStr+=pending.slice(i+MARK.length); pending=""; }
   else { const safe=Math.max(0,pending.length-(MARK.length-1)); thinkStr+=pending.slice(0,safe); pending=pending.slice(safe); }
 } else { ansStr+=pending; pending=""; }
 renderOut();
 if(k){ const bd=$("badge"); bd.textContent="batch +"+k; bd.classList.remove("pop"); void bd.offsetWidth; bd.classList.add("pop"); }
 pulseRing();
}
function finishRender(){ if(pending){ if(answering)ansStr+=pending; else thinkStr+=pending; pending=""; } renderOut(false); }
// ---- live SSE ----
function setLive(on,txt){ $("dot").className="dot"+(on?" on":""); $("livet").textContent=txt; }
function setServing(prompt,name){ $("serv_pr").textContent=prompt||"swarm idle — submit a prompt below";
 $("serv_who").textContent=name?("submitted by "+name+(name===ME?" (you)":"")):""; }
function setStats(s){ if(!s)return; if(s.tok_s!=null)$("s_tps").textContent=s.tok_s;
 if(s.n!=null)$("s_n").textContent=s.n; if(s.ctx!=null)$("s_ctx").textContent=s.ctx; }
function setQueue(waiting,running){
 const n=waiting.length; let s=running?("serving "+(running.name===ME?"your prompt":running.name)):"swarm idle";
 if(n)s+=" · "+n+" in queue";
 const mine=waiting.findIndex(w=>w.jid===myJid);
 const q=$("qrow"); q.innerHTML="QUEUE · "+s+(mine>=0?` · <span class=you>you're #${mine+1}, ~${Math.max(1,(mine+1)*12)}s</span>`:"");
 if(running&&running.jid===myJid&&myJid!=null)myJid=null;
}
function connect(){
 const es=new EventSource("/live");
 es.onopen=()=>setLive(true,"live");
 es.onerror=()=>{setLive(false,"reconnecting");};
 es.onmessage=e=>{ const ev=JSON.parse(e.data); handle(ev); };
}
function handle(ev){
 switch(ev.t){
  case "snapshot":
   if(ev.phase==="warming"){ setServing("swarm warming up — first prompts ready in a moment…",null); setLive(true,"warming"); }
   else setServing(ev.phase==="running"?ev.prompt:null, ev.phase==="running"?ev.name:null);
   setStats(ev.stats); $("s_b").textContent=ev.batches||"—";
   if(ev.text){ resetRender(); land(ev.text,0); }
   setQueue(ev.waiting||[],ev.running); setLive(true,"live"); break;
  case "start":
   resetRender();
   ["s_tps","s_n","s_ctx","s_b"].forEach(k=>$(k).textContent="—");
   $("foot").style.color=""; setServing(ev.prompt,ev.name); break;
  case "warming": setServing("swarm warming up — first prompts ready in a moment…",null); setLive(true,"warming"); break;
  case "prefilling": $("foot").textContent="prefilling context across the ring…"; $("foot").style.color=""; break;
  case "prefilled": $("s_ctx").textContent=ev.prompt_tokens; break;
  case "batch": land(ev.delta,ev.k); setStats({tok_s:ev.tok_s,n:ev.n,ctx:ev.ctx}); $("s_b").textContent=ev.batches; break;
  case "done": { finishRender();
    $("s_tps").textContent=ev.tok_s; $("s_n").textContent=ev.n;
    $("foot").textContent=`done · ${ev.n} tok @ ${ev.tok_s} tok/s · accept ${ev.accept} · ${ev.tpt} tok/traversal`; break; }
  case "idle": { finishRender(); break; }
  case "queue": setQueue(ev.waiting||[],ev.running); break;
  case "error": $("foot").textContent="error: "+ev.error; $("foot").style.color="var(--err)"; break;
 }
}
// ---- submit ----
$("go").onclick=async()=>{
 const prompt=$("prompt").value.trim(); if(!prompt)return;
 $("go").disabled=true;
 try{
  const r=await fetch("/submit",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt,name:ME})});
  const j=await r.json();
  if(j.error){ $("qrow").innerHTML="QUEUE · <span style='color:var(--err)'>"+j.error+"</span>"; }
  else{ myJid=j.jid; $("prompt").value=""; $("foot").style.color="";
    $("qrow").innerHTML=`QUEUE · <span class=you>submitted — you're #${j.position}</span>`; }
 }catch(e){ $("qrow").textContent="submit failed: "+e; }
 finally{ $("go").disabled=false; }
};
$("prompt").addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); $("go").click(); } });
connect();
</script></body></html>"""

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    mode = "MOCK" if MOCK else f"head={A.head} tail={A.tail} draft={A.draft}"
    _nn = len(NODES["nodes"]) if isinstance(NODES, dict) else len(NODES)
    print(f"[gateway] :{A.port}  {mode}  nodes={_nn}  (shared live terminal)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", A.port), H).serve_forever()
