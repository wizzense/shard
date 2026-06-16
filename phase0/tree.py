"""tree speculative decoding: the model-independent core.

a draft proposes a *tree* of candidate continuations rooted at the current token.
the target verifies the whole tree in ONE pipeline traversal (one WAN round-trip)
using a tree attention mask -- each node attends only to its ancestors -- and the
coordinator accepts the longest root-to-leaf path whose tokens match the target's
own greedy argmaxes. greedy acceptance => output identical to plain decode.

a tree is flattened to M nodes in any order with node 0 = root. each node carries:
  tok[i]    token id
  par[i]    parent index (par[0] = -1)
  dep[i]    depth (root = 0)
the verify feeds the M nodes on top of a cache holding the committed prefix
(prefix_len positions); node i sits at absolute position prefix_len + dep[i].
"""
import torch


def ancestors(par, i):
    a = []
    while i != -1:
        a.append(i); i = par[i]
    return a                                   # [i, parent(i), ..., root]


def tree_mask(par, dep, prefix_len, window, dtype, device):
    """additive mask, shape [1,1,M,prefix_len+M]. query node i (abs pos
    prefix_len+dep[i]) attends key j iff: j is in the committed prefix, OR j is a
    tree node that is an ancestor of i. window>0 also requires (pos_i - pos_j) <
    window (sliding layers)."""
    M = len(par)
    kv = prefix_len + M
    minv = torch.finfo(dtype).min
    m = torch.full((M, kv), minv, dtype=dtype, device=device)
    pos_i = [prefix_len + dep[i] for i in range(M)]
    for i in range(M):
        anc = set(ancestors(par, i))
        for j in range(prefix_len):            # prefix keys at position j
            if (not window) or (pos_i[i] - j) < window:
                m[i, j] = 0
        for tj in range(M):                    # tree keys
            if tj in anc:
                pj = prefix_len + dep[tj]
                if (not window) or (pos_i[i] - pj) < window:
                    m[i, prefix_len + tj] = 0
    return m[None, None]


def accept_tree(tok, par, children, targ):
    """walk the tree following the target's greedy argmaxes. targ[i] is the
    target's predicted next token at node i. start at root: the accepted next
    token is targ[root]; if a child of root carries that token, accept it and
    recurse, else stop with targ[root] as the correction. returns
    (committed_tokens, kept_node_indices) -- kept = root + accepted path nodes,
    in path order, for the cache gather."""
    committed, kept = [], [0]
    node = 0
    while True:
        want = targ[node]                      # target's greedy next token at `node`
        nxt = None
        for c in children[node]:
            if tok[c] == want:
                nxt = c; break
        committed.append(want)                 # always commit the target's token (exact)
        if nxt is None:
            break                              # draft had no matching child -> correction, stop
        kept.append(nxt); node = nxt           # accepted: descend
    return committed, kept


def gather_cache(cache, keep_abs, device):
    """keep only the given absolute cache positions (committed prefix + last
    round's accepted path), reindexed to 0..len-1, on the seq dim. avoids a second
    traversal to re-materialize the accepted KV. mirrors DynamicCache.crop, which
    truncates each layer's .keys/.values (transformers 5.x: cache.layers[i])."""
    idx = torch.tensor(keep_abs, device=device)
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()


if __name__ == "__main__":                     # self-test of the structure logic
    # tree:  root(0)=cur
    #        ├─ 1 ─ 2 ─ 3        branch A (linear)
    #        └─ 4 ─ 5            branch B (linear)
    tok = [100, 11, 12, 13, 21, 22]
    par = [-1, 0, 1, 2, 0, 4]
    dep = [0, 1, 2, 3, 1, 2]
    children = [[] for _ in tok]
    for i, p in enumerate(par):
        if p != -1: children[p].append(i)
    # target argmaxes: at root wants 11 (->node1), at 1 wants 12 (->2), at 2 wants 99 (no child) stop
    targ = {0: 11, 1: 12, 2: 99, 3: 0, 4: 0, 5: 0}
    committed, kept = accept_tree(tok, par, children, targ)
    assert committed == [11, 12, 99], committed          # accepted 11,12 + correction 99
    assert kept == [0, 1, 2], kept                       # root + the two accepted nodes
    # target that prefers branch B: root wants 21 (->node4), at 4 wants 22 (->5), at 5 wants 7 stop
    targ2 = {0: 21, 4: 22, 5: 7, 1: 0, 2: 0, 3: 0}
    committed2, kept2 = accept_tree(tok, par, children, targ2)
    assert committed2 == [21, 22, 7], committed2
    assert kept2 == [0, 4, 5], kept2                     # accepted path is NOT a flat prefix -> needs gather
    # mask: node 3 (pos prefix+3) attends prefix + {0,1,2,3}, not branch B {4,5}
    m = tree_mask(par, dep, prefix_len=2, window=0, dtype=torch.float32, device="cpu")[0, 0]
    row3 = (m[3] == 0).int().tolist()
    assert row3 == [1, 1, 1, 1, 1, 1, 0, 0], row3        # prefix(2) + tree nodes 0,1,2,3 yes; branch-B 4,5 no
    print("tree.py self-test PASSED: accept-path + non-prefix gather + ancestor mask all correct")
