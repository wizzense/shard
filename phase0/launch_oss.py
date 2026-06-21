"""Launcher for the gpt-oss-120B scattered-4090 swarm (specpipe fast-verify).

Brings up the swarm and runs the SYNC baseline then the PIPELINED coordinator on the
SAME warm stages, so the two tok/s numbers are a clean A/B on one topology.

  python launch_oss.py --stages id0,id1,id2,id3 --coord COORDID \
      --prompt "Explain decentralized computing in two sentences." \
      --max-new 160 --sweep 3,4 --depth 4

Flow: eps -> launch tail-first (stage3=tail, stage0=served-head; all --fast --direct-return)
      -> draft_server (vLLM gpt-oss-20b) on coord -> warm barrier -> coord SYNC sweep, PIPE sweep.
Teardown is manual (vastai stop/destroy). Stage order is taken as given (fixed across both runs).
"""
import os, sys, json, time, subprocess, argparse, concurrent.futures as cf

KEY = "/root/.ssh/vast_c0mpute"
# transport secret — NEVER hardcode (this file is in a public repo). Read from env or ~/.shard_psk (gitignored).
PSK = (os.environ.get("SHARD_PSK") or open(os.path.expanduser("~/.shard_psk")).read().strip())
PORT = 29600
DRAFT_PORT = 8200
M120 = "/root/models/gpt-oss-120b"
M20 = "/root/models/gpt-oss-20b"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=20"]


def instances():
    r = subprocess.run(["vastai", "show", "instances-v1", "--raw"], capture_output=True, text=True)
    o = r.stdout
    try:
        d = json.loads(o)                       # v1 raw is a JSON object/array
    except Exception:
        k = o.find("[")                         # tolerate a leading non-JSON line
        d = json.loads(o[k:]) if k >= 0 else []
    lst = d if isinstance(d, list) else d.get("instances", [])
    return {i["id"]: i for i in lst}


def ep(inst):
    """public ip + mapped host port for container :29600 (cross-box reachable)."""
    p = (inst.get("ports") or {}).get(f"{PORT}/tcp")
    return inst["public_ipaddr"], (p[0]["HostPort"] if p else None)


def ssh22(inst):
    p22 = (inst.get("ports") or {}).get("22/tcp")
    host = inst["public_ipaddr"] if p22 else inst["ssh_host"]
    port = p22[0]["HostPort"] if p22 else inst["ssh_port"]
    return host, port


def rssh(inst, cmd, timeout=120):
    h, p = ssh22(inst)
    return subprocess.run(["ssh", "-i", KEY, "-p", str(p)] + SSHO + [f"root@{h}", cmd],
                          capture_output=True, text=True, timeout=timeout)


def fire(inst, cmd, timeout=25):
    """launch a detached (setsid) process; tolerate the ssh not returning -- vLLM/torch
    can hold the channel open via child fds, but the setsid survives and readiness is
    polled separately. so a TimeoutExpired here is not a failure."""
    try:
        rssh(inst, cmd, timeout)
    except subprocess.TimeoutExpired:
        pass


def launch_stage(inst, stage, nstages, nxt_ep, served_head, max_ctx=2048):
    is_tail = stage == nstages - 1
    nextarg = f" --next {nxt_ep}" if nxt_ep else ""
    head = " --served-head" if served_head else ""
    # kill prior GPU procs (NEVER pkill -f specpipe -> would kill this ssh's match); free the port
    cmd = (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
           f"fuser -k {PORT}/tcp 2>/dev/null; sleep 2; rm -f /root/stage.log; cd /root && "
           f"SHARD_PSK={PSK} setsid bash -c 'python3 specpipe.py --stage {stage} --nstages {nstages} "
           f"--model {M120} --listen-port {PORT}{nextarg}{head} --fast --direct-return "
           f"--max-ctx {max_ctx} --timeout 600 > /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")
    fire(inst, cmd)


def warm_stage(inst, label):
    for _ in range(80):
        r = rssh(inst, "grep -ciE 'listening' /root/stage.log 2>/dev/null; "
                       "grep -ciE 'Traceback|Error|exit status' /root/stage.log 2>/dev/null", 30)
        nums = [x for x in r.stdout.split() if x.isdigit()]
        if nums and nums[0] != "0":
            return label, True
        if len(nums) > 1 and nums[1] != "0":
            tail = rssh(inst, "tail -5 /root/stage.log", 20).stdout
            return f"{label} ERR: {tail[-400:]}", False
        time.sleep(10)
    return f"{label} TIMEOUT", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma ids in chain order (head..tail)")
    ap.add_argument("--coord", type=int, required=True)
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--ks", default="4")
    ap.add_argument("--depths", default="2,4,8")
    ap.add_argument("--tree", default="", help="run FAST graphed tree spec 'w,d' instead of the linear compare")
    ap.add_argument("--dump", action="store_true", help="write the warm run (ids+hash+sync-match) to /root/run.json for the receipt")
    ap.add_argument("--skip-draft", action="store_true", help="reuse a draft already running on coord")
    ap.add_argument("--gateway", action="store_true", help="after warm, launch the streaming gateway on the coord "
                    "as the sole coordinator (persisted via the same setsid/fire path as the draft) and write "
                    "nodes.json from the live topology; implies no self-test run.")
    ap.add_argument("--no-run", action="store_true", help="launch+warm stages and draft, then STOP (no self-test "
                    "coordinator). leaves the ring fresh so an external coordinator (e.g. the gateway) is the first "
                    "session -- the forward links are set once at launch and don't reconnect, so a self-test would "
                    "poison the ring for the next coordinator.")
    ap.add_argument("--max-ctx", type=int, default=2048, help="fast-verify static cache size (prompt+gen ceiling)")
    a = ap.parse_args()

    sids = [int(x) for x in a.stages.split(",")]
    nstages = len(sids)
    insts = instances()
    coord = insts[a.coord]
    stages = [insts[i] for i in sids]
    hops_ms, loop_ms = [], 0                         # measured per-hop RTT + loop, surfaced on the demo map
    try:                                            # RTT-optimal stage order: minimize coord->s0->..->sN->coord
        import itertools, launch_swarm
        alln = [coord] + stages
        print("[mesh] measuring all-pairs RTT to order the ring...", flush=True)
        M = launch_swarm.mesh_rtt(alln)
        best = border = None
        for perm in itertools.permutations(range(1, len(alln))):
            c = M[0][perm[0]] + sum(M[perm[i]][perm[i + 1]] for i in range(len(perm) - 1)) + M[perm[-1]][0]
            if best is None or c < best:
                best, border = c, perm
        if best is not None and best < 9000:        # all hops probed (9999 = a failed probe)
            stages = [alln[i] for i in border]
            loop_ms = int(best)
            o = list(border)                        # ring edges coord->s0->..->sN->coord, real measured RTT each
            hops_ms = [int(M[0][o[0]])] + [int(M[o[i]][o[i + 1]]) for i in range(len(o) - 1)] + [int(M[o[-1]][0])]
            print(f"[mesh] optimal loop {best:.0f}ms; order {[s.get('geolocation') for s in stages]}; hops {hops_ms}", flush=True)
        else:
            print("[mesh] probe inconclusive; keeping given order", flush=True)
    except Exception as e:
        print(f"[mesh] skipped ({type(e).__name__}: {e}); keeping given order", flush=True)
    eps = [ep(s) for s in stages]
    print(f"[topology] coord={a.coord} ({coord.get('geolocation')})", flush=True)
    for k, s in enumerate(stages):
        ip, hp = eps[k]
        print(f"  stage{k} {s['id']} ({s.get('geolocation')}) {ip}:{hp}"
              f"{' [served-head]' if k == 0 else ''}{' [tail]' if k == nstages-1 else ''}", flush=True)

    if a.skip_draft:
        print("[draft] --skip-draft: reusing the draft already running on coord", flush=True)
    else:
        print("[draft] starting vLLM gpt-oss-20b on coord (loads in parallel with stages)...", flush=True)
        fire(coord, f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
                    f"fuser -k {DRAFT_PORT}/tcp 2>/dev/null; sleep 2; rm -f /root/draft.log; cd /root && "
                    f"SHARD_PSK={PSK} CUDA_VISIBLE_DEVICES=0 setsid bash -c "
                    f"'/root/vllmenv/bin/python draft_server.py --model {M20} --port {DRAFT_PORT} "
                    f"--max-len {min(a.max_ctx, 8192)} > /root/draft.log 2>&1' </dev/null >/dev/null 2>&1 &")

    print("[launch] stages tail-first; wait each to listen before launching its predecessor "
          "(120B partial load ~1-2min/node, so a predecessor never connects to a dead successor)...", flush=True)
    for k in range(nstages - 1, -1, -1):
        nxt = f"{eps[k+1][0]}:{eps[k+1][1]}" if k < nstages - 1 else None
        launch_stage(stages[k], k, nstages, nxt, served_head=(k == 0), max_ctx=a.max_ctx)
        label, ok = warm_stage(stages[k], f"stage{k} {stages[k]['id']}")
        print(f"  {'OK ' if ok else 'FAIL '}{label}", flush=True)
        if not ok:
            print("[abort] stage failed to warm", flush=True); return

    # wait for the draft server
    for _ in range(60):
        r = rssh(coord, "grep -ciE 'ready, listening' /root/draft.log 2>/dev/null", 20)
        if r.stdout.strip().split() and r.stdout.strip().split()[0] != "0":
            print("[draft] ready", flush=True); break
        time.sleep(10)
    else:
        print("[draft] not ready:", rssh(coord, "tail -6 /root/draft.log", 20).stdout[-500:], flush=True); return

    head_ep = f"{eps[0][0]}:{eps[0][1]}"
    tail_ep = f"{eps[nstages-1][0]}:{eps[nstages-1][1]}"

    if a.gateway:
        # launch the streaming gateway on the coord, as the FIRST (sole) coordinator on the
        # fresh ring, via the SAME fire()/setsid path that persists the draft+stages (a plain
        # ssh-backgrounded process gets reaped when the ssh session closes on these boxes).
        # write nodes.json from the live topology so the map shows the real ring.
        LATLON = {"California": (37.4, -122.0), "North Carolina": (35.5, -79.0),
                  "Washington": (47.4, -120.5), "Utah": (39.3, -111.7), "Oregon": (44.0, -120.5),
                  "Texas": (31.0, -99.0), "New York": (43.0, -75.0), "Virginia": (37.5, -78.5)}
        seen = {}
        def latlon(geo):
            st = (geo or "").split(",")[0].strip()
            base = LATLON.get(st, (39.0, -98.0))
            k = seen.get(st, 0); seen[st] = k + 1            # nudge duplicates so they don't overlap on the map
            return (base[0] + 0.0, base[1] + 3.0 * k)
        nodes = []
        span = 36 // nstages
        for k, s in enumerate(stages):
            la, lo = latlon(s.get("geolocation"))
            role = "head" if k == 0 else ("tail" if k == nstages - 1 else "mid")
            nodes.append({"idx": k, "city": (s.get("geolocation") or "US").split(",")[0],
                          "lat": la, "lon": lo, "layers": f"{k*span}-{(k+1)*span}", "role": role,
                          "ip": ep(s)[0]})                    # real public IP — verifiable on the map
        cla, clo = latlon(coord.get("geolocation"))
        nodes.append({"idx": 99, "city": (coord.get("geolocation") or "Utah").split(",")[0],
                      "lat": cla, "lon": clo, "layers": "draft", "role": "coord", "ip": ep(coord)[0]})
        import json as _json
        nodes_json = _json.dumps({"nodes": nodes, "hops": hops_ms, "loop_ms": loop_ms}).replace("'", "")
        rssh(coord, f"cat > /root/nodes.json <<'NODESEOF'\n{nodes_json}\nNODESEOF", 20)
        print(f"\n[gateway] launching on coord:29600 (sole coordinator) head={head_ep} tail={tail_ep}", flush=True)
        fire(coord, f"fuser -k 29600/tcp 2>/dev/null; sleep 2; rm -f /root/gateway.log; "
                    f"cd /root && SHARD_PSK={PSK} setsid bash -c 'exec python3 -u gateway.py --head {head_ep} "
                    f"--tail {tail_ep} --draft 127.0.0.1:{DRAFT_PORT} --port 29600 --max-new 4096 "
                    f"--max-ctx {a.max_ctx} "
                    f"--nodes-file /root/nodes.json > /root/gateway.log 2>&1' </dev/null >/dev/null 2>&1 &")
        for _ in range(30):                                  # wait for the gateway to bind (torch import ~15-20s)
            r = rssh(coord, "ss -ltnp 2>/dev/null | grep -c :29600", 20)
            if r.stdout.strip().startswith("1"):
                print("[gateway] UP on :29600 (proxied to shard.c0mpute.ai); first request warms the cold flex compile", flush=True)
                break
            time.sleep(3)
        else:
            print("[gateway] did NOT bind:", rssh(coord, "tail -5 /root/gateway.log", 20).stdout[-400:], flush=True)
        print("[done] stages + gateway warm; teardown: vastai destroy instance <id>", flush=True)
        return

    if a.no_run:
        print(f"\n[ready] ring warm, no self-test run.  HEAD={head_ep}  TAIL={tail_ep}", flush=True)
        print(f"[ready] point the gateway here:  --head {head_ep} --tail {tail_ep} --draft 127.0.0.1:{DRAFT_PORT}", flush=True)
        print("[done] stages still warm; teardown: vastai destroy instance <id>", flush=True)
        return

    base = (f"cd /root && SHARD_PSK={PSK} python3 specpipe.py --coordinator --nstages {nstages} "
            f"--model {M20} --draft-server 127.0.0.1:{DRAFT_PORT} --next {head_ep} "
            f"--direct-return --tail {tail_ep} --prompt \"{a.prompt}\" --max-new {a.max_new} --timeout 600")

    if a.tree:
        print(f"\n[run] === FAST TREE spec (w,d={a.tree}), cold+warm ===", flush=True)
        r = rssh(coord, base + f" --tree-fast {a.tree} 2>&1 | grep -viE 'INFO|WARNING|warn'", 1800)
    else:
        dump = " --dump /root/run.json" if a.dump else ""
        print(f"\n[run] === COMPARE: SYNC vs PIPE (ks={a.ks}, depths={a.depths}), cold+warm, ONE process ===", flush=True)
        r = rssh(coord, base + f" --compare --ks {a.ks} --depths {a.depths}{dump} 2>&1 | grep -viE 'INFO|WARNING|warn'", 1800)
    print(r.stdout[-2500:], flush=True)
    print("\n[done] stages still warm; teardown: vastai destroy instance <id>", flush=True)


if __name__ == "__main__":
    main()
