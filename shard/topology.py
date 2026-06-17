"""latency-optimal pipeline ordering — the heart of serving a *scattered* swarm.

c0mpute nodes are random consumer GPUs on home links, never co-located. every token
traverses coordinator -> head -> ... -> tail -> (direct return) -> coordinator, so the
per-token WAN cost is:

    entry hop  +  sum of forward hops  +  return hop
    c_out[h]   +  sum L[node_i, node_i+1]  +  c_in[t]

the visit order is free (any node can hold any contiguous block), so the cheapest
pipeline is the minimum-latency Hamiltonian loop with the coordinator as the depot.
internet RTT is asymmetric and doesn't track geography (peering), so we optimize on the
*measured* mesh, not distance.

  - optimal_loop : exact (Held-Karp) for n<=16, nearest-neighbor + 2-opt above.
  - select_and_order : pick the best k of n online nodes AND order them (exact for n<=16).

both take L (L[i][j] = ms from node i to node j, asymmetric ok), c_out (coordinator->i),
c_in (i->coordinator). pure python, no deps; run `python -m shard.topology` for a demo.
"""
from itertools import combinations

INF = float("inf")


def loop_cost(order, L, c_out, c_in):
    """total per-traversal latency for a given node ordering (entry + hops + return)."""
    if not order:
        return 0.0
    cost = c_out[order[0]] + c_in[order[-1]]
    for a, b in zip(order, order[1:]):
        cost += L[a][b]
    return cost


def _held_karp(nodes, L, c_out, c_in):
    """exact min-latency loop over exactly `nodes` (indices). O(k^2 2^k), k<=~16."""
    idx = list(nodes)
    k = len(idx)
    if k == 1:
        return idx, c_out[idx[0]] + c_in[idx[0]]
    pos = {n: i for i, n in enumerate(idx)}                 # node -> bit position
    dp = [[INF] * k for _ in range(1 << k)]                 # dp[mask][j] = min cost ... ending at j
    par = [[-1] * k for _ in range(1 << k)]
    for j in range(k):
        dp[1 << j][j] = c_out[idx[j]]
    for mask in range(1 << k):
        for j in range(k):
            if dp[mask][j] == INF or not (mask >> j) & 1:
                continue
            base = dp[mask][j]
            for m in range(k):
                if (mask >> m) & 1:
                    continue
                nmask = mask | (1 << m)
                cand = base + L[idx[j]][idx[m]]
                if cand < dp[nmask][m]:
                    dp[nmask][m] = cand
                    par[nmask][m] = j
    full = (1 << k) - 1
    best, bj = INF, -1
    for j in range(k):
        c = dp[full][j] + c_in[idx[j]]
        if c < best:
            best, bj = c, j
    order, mask, j = [], full, bj                           # reconstruct
    while j != -1:
        order.append(idx[j])
        pj = par[mask][j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order, best


def _nn_2opt(nodes, L, c_out, c_in, rounds=4):
    """heuristic for large k: nearest-neighbor seed, then 2-opt segment reversals."""
    idx = list(nodes)
    # nearest-neighbor from the cheapest entry hop
    start = min(idx, key=lambda n: c_out[n])
    tour, rest = [start], set(idx) - {start}
    while rest:
        last = tour[-1]
        nxt = min(rest, key=lambda n: L[last][n])
        tour.append(nxt); rest.discard(nxt)
    best = loop_cost(tour, L, c_out, c_in)
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                c = loop_cost(cand, L, c_out, c_in)
                if c + 1e-9 < best:
                    tour, best, improved = cand, c, True
    return tour, best


def optimal_loop(nodes, L, c_out, c_in):
    """min-latency pipeline order over all `nodes`. exact <=16, heuristic above."""
    nodes = list(nodes)
    if len(nodes) <= 16:
        return _held_karp(nodes, L, c_out, c_in)
    return _nn_2opt(nodes, L, c_out, c_in)


def select_and_order(nodes, L, c_out, c_in, k):
    """pick the best k of n online nodes AND order them into the cheapest loop.

    exact for n<=16 (Held-Karp answer ranges over size-k subsets); above that, solve the
    full order then greedily drop the node whose removal helps most until k remain.
    """
    nodes = list(nodes)
    if k >= len(nodes):
        return optimal_loop(nodes, L, c_out, c_in)
    if len(nodes) <= 16:
        best_order, best_cost = None, INF
        for subset in combinations(nodes, k):
            order, cost = _held_karp(subset, L, c_out, c_in)
            if cost < best_cost:
                best_order, best_cost = order, cost
        return best_order, best_cost
    order, _ = _nn_2opt(nodes, L, c_out, c_in)              # greedy-drop from a good full tour
    while len(order) > k:
        drop = min(range(len(order)),
                   key=lambda i: loop_cost(order[:i] + order[i + 1:], L, c_out, c_in))
        order = order[:drop] + order[drop + 1:]
    return _nn_2opt(order, L, c_out, c_in)


# ---- demo: a scattered-US mesh, optimal loop vs naive ordering ----
if __name__ == "__main__":
    # ~ms one-way, lat/long-ish placement; internet RTT, not distance: a couple of
    # asymmetric peering quirks baked in so geography is NOT the answer.
    cities = ["WA", "OR", "CA", "TX", "KS", "IL", "GA", "NC", "VA", "NY"]
    xy = {"WA": (0, 9), "OR": (0, 7), "CA": (1, 3), "TX": (5, 1), "KS": (6, 5),
          "IL": (8, 6), "GA": (9, 2), "NC": (11, 3), "VA": (11, 5), "NY": (12, 8)}
    def base(a, b):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        return 4.0 + 2.3 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5      # ms
    n = len(cities)
    L = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            if i != j:
                L[i][j] = base(a, b) + (3.0 if (i + 2 * j) % 5 == 0 else 0.0)  # asym noise
    L[2][8] = L[8][2] = 9.0          # CA<->VA: a fat peering pipe, far but fast
    c_out = [base("WA", c) for c in cities]   # coordinator near WA (entry hop)
    c_in = [base("WA", c) * 0.9 for c in cities]  # direct return, slightly cheaper path

    nodes = list(range(n))
    geo = loop_cost(nodes, L, c_out, c_in)            # input order = a clean geographic guess
    join = [4, 9, 2, 7, 0, 5, 8, 3, 6, 1]            # arbitrary join order (the real case)
    join_cost = loop_cost(join, L, c_out, c_in)
    order, cost = optimal_loop(nodes, L, c_out, c_in)
    name = lambda o: " -> ".join(cities[i] for i in o)
    print(f"arbitrary join order {join_cost:6.1f} ms   {name(join)}")
    print(f"geographic guess     {geo:6.1f} ms   {name(nodes)}")
    print(f"OPTIMAL loop         {cost:6.1f} ms   {name(order)}")
    print(f"  -> {join_cost / cost:.2f}x vs how nodes actually join, {geo / cost:.2f}x vs a hand geo-guess")
    sub_order, sub_cost = select_and_order(nodes, L, c_out, c_in, k=6)
    print(f"best 6 of {n}          {sub_cost:6.1f} ms   {name(sub_order)}")
