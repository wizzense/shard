"""mesh latency harness — measure the real pairwise RTT among scattered swarm nodes,
then solve the minimum-latency pipeline loop (shard/topology.py).

c0mpute nodes sit on home links across the internet; RTT is asymmetric and does NOT track
geography (peering, routing). this measures the actual app-level round-trip over a
persistent TCP connection -- the same kind the pipeline holds open during decode, so the
number reflects a real per-hop activation cost, not an ICMP ping. each node measures its
own row (RTT from itself to every peer, incl. the coordinator), the rows assemble into the
asymmetric L matrix, and the solver orders the pipeline + picks the best-k-of-pool.

modes:
  serve    : run an echo endpoint so peers can probe this node.
  measure  : time median RTT from here to each peer; print this node's row as JSON.
  solve    : given all nodes' rows + the coordinator id, print the optimal pipeline order.
  selftest : serve + measure localhost, end to end (no network needed).

usage:
  python mesh.py serve   --port 7000
  python mesh.py measure --peers head=1.2.3.4:7000 mid=5.6.7.8:7000 coord=9.9.9.9:7000 > row.json
  python mesh.py solve   --rows rows.json --coordinator coord
"""
import argparse, json, os, socket, statistics, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import shard
PAYLOAD = b"x" * 64                              # tiny: we measure latency, not bandwidth


def serve(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(64)
    print(f"mesh echo listening on :{port}", flush=True)

    def handle(c):
        with c:
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while (d := c.recv(4096)):
                c.sendall(d)

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def rtt_to(host, port, samples=20, warmup=3, timeout=5.0):
    """median app-level round-trip (ms) over one persistent connection."""
    s = socket.create_connection((host, port), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    times = []
    try:
        for i in range(samples + warmup):
            t0 = time.perf_counter()
            s.sendall(PAYLOAD)
            got = 0
            while got < len(PAYLOAD):
                chunk = s.recv(len(PAYLOAD) - got)
                if not chunk:
                    raise ConnectionError("peer closed mid-probe")
                got += len(chunk)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    finally:
        s.close()
    return statistics.median(times)


def measure(peers, samples=20):
    row = {}
    for name, ep in peers.items():
        host, port = ep.rsplit(":", 1)
        try:
            row[name] = round(rtt_to(host, int(port), samples), 3)
        except Exception as e:
            row[name] = None
            print(f"  {name} {ep}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return row


def solve(rows, coordinator, k=None):
    """rows: {node_id: {peer_id: rtt_ms}}, including the coordinator's row."""
    from shard.topology import optimal_loop, select_and_order, loop_cost
    ids = [n for n in rows if n != coordinator]
    L = [[0.0 if a == b else rows[a][b] for b in ids] for a in ids]
    c_out = [rows[coordinator][a] for a in ids]     # coordinator -> node (entry hop)
    c_in = [rows[a][coordinator] for a in ids]      # node -> coordinator (direct return)
    idx = list(range(len(ids)))
    order, cost = (select_and_order(idx, L, c_out, c_in, k) if k else
                   optimal_loop(idx, L, c_out, c_in))
    naive = loop_cost(idx, L, c_out, c_in)
    return [ids[i] for i in order], cost, naive


def _parse_peers(items):
    return dict(it.split("=", 1) for it in items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=7000)
    p = sub.add_parser("measure")
    p.add_argument("--peers", nargs="+", required=True, help="name=host:port ...")
    p.add_argument("--samples", type=int, default=20)
    p = sub.add_parser("solve")
    p.add_argument("--rows", required=True); p.add_argument("--coordinator", required=True)
    p.add_argument("--k", type=int, default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()

    if a.mode == "serve":
        serve(a.port)
    elif a.mode == "measure":
        print(json.dumps(measure(_parse_peers(a.peers), a.samples)))
    elif a.mode == "solve":
        rows = json.load(open(a.rows))
        order, cost, naive = solve(rows, a.coordinator, a.k)
        print(f"optimal pipeline: {' -> '.join(order)}")
        print(f"loop latency: {cost:.1f} ms  (vs {naive:.1f} ms naive, {naive/cost:.2f}x)")
    elif a.mode == "selftest":
        threading.Thread(target=serve, args=(7099,), daemon=True).start()
        time.sleep(0.3)
        row = measure({"self": "127.0.0.1:7099"}, samples=50)
        print("selftest localhost RTT:", row, "ms -> mesh harness OK")
