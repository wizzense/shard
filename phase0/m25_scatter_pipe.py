"""Launch the M2.5 PIPELINED direct-return ring across scattered vast boxes over libp2p, and run
the proven coordinate_pipe coordinator (depth chunks in flight = the GLM throughput lever).

Direct-return topology (per launch_libp2p): head sidecar forwards BOTH the ring (->s1) and the
coordinator return-channel (->tail); middle sidecars inbound+forward; tail inbound only. Stages run
m25_pipe (fire-forward); the coordinator (on the head box) dials the head engine locally and the tail
via the 29612 return tunnel.

  python m25_scatter_pipe.py --order CA:42545183:0:10 WA:..:10:23 MN:..:23:36 NJ:..:36:49 NC:..:49:62 \
      --K 6 --depth 4 --max-new 256 --prompt-file /root/copy_prompt.txt
"""
import sys, json, time, subprocess, argparse

KEY = "/root/.ssh/vast_c0mpute"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25", "-i", KEY]
REPO = "/root/.openclaw/workspace/shard"
LIBP2P, ENG_IN, FWD_RING, FWD_RET = 29600, 29610, 29611, 29612


def vinst(iid):
    return json.loads(subprocess.check_output(["vastai", "show", "instance", str(iid), "--raw"], text=True))


def sh(host, port, cmd, timeout=120):
    return subprocess.run(["ssh", *SSHO, "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)


def push_code(host, port):
    for f in ["phase0/m25_pipe.py", "phase0/m25_stage.py", "phase0/m25_tools.py", "phase0/ngram_draft.py",
              "phase0/node_kv.py", "shard/transport.py", "shard/receipt.py", "shard/manifest.py"]:
        dst = "/root/" + f.split("/")[-1]
        subprocess.run(["scp", *SSHO, "-P", str(port), f"{REPO}/{f}", f"root@{host}:{dst}"], capture_output=True, text=True)


def peerid(host, port):
    r = sh(host, port, "/tmp/sidecar -key /root/node.key -prove ping 2>/dev/null | grep PEERID")
    for ln in r.stdout.splitlines():
        if ln.startswith("PEERID "):
            return ln.split()[1]
    raise RuntimeError(f"no PeerId {host}:{port}: {r.stdout[-200:]}{r.stderr[-200:]}")


def launch_sidecar(host, port, announce, inbound, forwards):
    fw = " ".join(f"-forward {f}" for f in forwards)
    inb = f"-inbound {inbound}" if inbound else ""
    cmd = (f"pkill -9 -x sidecar 2>/dev/null; fuser -k {LIBP2P}/tcp {FWD_RING}/tcp {FWD_RET}/tcp 2>/dev/null; sleep 2; rm -f /root/sidecar.log; "
           f"setsid bash -c '/tmp/sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/{LIBP2P} "
           f"-announce {announce} {inb} {fw} > /root/sidecar.log 2>&1' </dev/null >/dev/null 2>&1 &")
    for attempt in range(5):
        sh(host, port, cmd, 30)
        for _ in range(4):
            time.sleep(3)
            up = sh(host, port, "grep -cE 'tunnel up|listening' /root/sidecar.log 2>/dev/null || echo 0", 20)
            bad = sh(host, port, "grep -c 'address already in use' /root/sidecar.log 2>/dev/null || echo 0", 20)
            if (up.stdout.strip().splitlines() or ["0"])[-1].strip() not in ("", "0") and \
               (bad.stdout.strip().splitlines() or ["0"])[-1].strip() in ("", "0"):
                return True
        print(f"  sidecar {host} retry {attempt+1}", flush=True)
    return False


def launch_stage(host, port, stage, nstages, lo, hi, is_tail, receipts=False):
    nxt = "" if is_tail else f"--next 127.0.0.1:{FWD_RING}"
    rc = "SHARD_RECEIPTS=1 " if receipts else ""
    cmd = (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
           f"fuser -k {ENG_IN}/tcp 2>/dev/null; sleep 4; rm -f /root/stage.log; cd /root && "
           f"{rc}SHARD_TRANSPORT=libp2p CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid bash -c "
           f"'/root/venv/bin/python /root/m25_pipe.py stage --stage {stage} --nstages {nstages} --lo {lo} --hi {hi} "
           f"--port {ENG_IN} {nxt} > /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")
    try:
        sh(host, port, cmd, 25)
    except subprocess.TimeoutExpired:
        pass


def warm(host, port, label, tries=80):
    for _ in range(tries):
        time.sleep(8)
        r = sh(host, port, "grep -c WARM /root/stage.log 2>/dev/null || echo 0", 20)
        if (r.stdout.strip().splitlines() or ["0"])[-1].strip() not in ("", "0"):
            return True
        e = sh(host, port, "grep -cE 'Traceback|Error|CUDA out' /root/stage.log 2>/dev/null || echo 0", 20)
        if (e.stdout.strip().splitlines() or ["0"])[-1].strip() not in ("", "0"):
            print(f"  {label} ERROR:\n" + sh(host, port, "tail -12 /root/stage.log", 20).stdout, flush=True)
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--prompt", default="Explain a decentralized inference swarm in 3 sentences.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--sweep", default=None); ap.add_argument("--sweep-depth", default=None)  # pass through to coord
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--receipts", action="store_true")
    a = ap.parse_args()
    nodes = []
    for spec in a.order:
        region, iid, lo, hi = spec.split(":")
        j = vinst(iid)
        ports = j.get("ports") or {}
        m = ports.get(f"{LIBP2P}/tcp")
        nodes.append(dict(region=region, iid=iid, host=j["ssh_host"], port=int(j["ssh_port"]),
                          pip=(j.get("public_ipaddr") or "").strip(), pport=m[0]["HostPort"] if m else None,
                          lo=int(lo), hi=int(hi)))
    n = len(nodes)
    print("[pipe] push code + PeerIds ...", flush=True)
    for nd in nodes:
        push_code(nd["host"], nd["port"])
        nd["pid"] = peerid(nd["host"], nd["port"])
        nd["maddr"] = f"/ip4/{nd['pip']}/tcp/{nd['pport']}/p2p/{nd['pid']}"
        print(f"  {nd['region']} {nd['pip']}:{nd['pport']} [{nd['lo']},{nd['hi']}) {nd['pid'][:14]}..", flush=True)

    print("[pipe] sidecars (direct-return: head forwards ring+ret) ...", flush=True)
    for k, nd in enumerate(nodes):
        announce = f"/ip4/{nd['pip']}/tcp/{nd['pport']}"
        inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""           # head's predecessor is the local coord
        forwards = []
        if k < n - 1:
            forwards.append(f"127.0.0.1:{FWD_RING}={nodes[k+1]['maddr']}")
        if k == 0:
            forwards.append(f"127.0.0.1:{FWD_RET}={nodes[-1]['maddr']}")   # head also tunnels coord-return -> tail
        ok = launch_sidecar(nd["host"], nd["port"], announce, inbound, forwards)
        print(f"  {'OK' if ok else 'FAIL'} {nd['region']}", flush=True)
        if not ok:
            print(sh(nd["host"], nd["port"], "tail -4 /root/sidecar.log", 20).stdout); return

    print("[pipe] stages tail-first ...", flush=True)
    for k in range(n - 1, -1, -1):
        launch_stage(nodes[k]["host"], nodes[k]["port"], k, n, nodes[k]["lo"], nodes[k]["hi"], k == n - 1, a.receipts)
    for k in range(n - 1, -1, -1):
        ok = warm(nodes[k]["host"], nodes[k]["port"], f"s{k} {nodes[k]['region']}")
        print(f"  {'WARM' if ok else 'FAIL'} s{k} {nodes[k]['region']}", flush=True)
        if not ok:
            return

    head = nodes[0]
    pf = f"--prompt-file {a.prompt_file}" if a.prompt_file else f'--prompt "{a.prompt}"'
    sw = (f"--sweep {a.sweep} " if a.sweep else "") + (f"--sweep-depth {a.sweep_depth} " if a.sweep_depth else "") + ("--validate " if a.validate else "")
    rc = "SHARD_RECEIPTS=1 " if a.receipts else ""
    print("[pipe] coordinator (pipelined) on head ...", flush=True)
    cmd = (f"cd /root && {rc}SHARD_TRANSPORT=libp2p CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 /root/venv/bin/python /root/m25_pipe.py coord "
           f"--head 127.0.0.1:{ENG_IN} --tail 127.0.0.1:{FWD_RET} --K {a.K} --depth {a.depth} --ngram-n {a.ngram_n} "
           f"--max-new {a.max_new} --prefill-chunk {a.prefill_chunk} {sw}{pf} 2>&1 | tee /root/coord.log | grep -vE 'INFO|WARNING|warn|instantiate'")
    r = sh(head["host"], head["port"], cmd, timeout=1800 if (a.sweep or a.sweep_depth or a.validate) else 1200)
    print(r.stdout, flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-700:], flush=True)


if __name__ == "__main__":
    main()
