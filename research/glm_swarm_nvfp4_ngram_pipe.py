"""GLM-5.2 NVFP4 swarm — N-GRAM (retrieval) draft over the PIPELINED ring. The draft is a free
suffix-match lookup (zero model forward), so the 9B-draft bottleneck disappears entirely and the loop
becomes ring-throughput-bound. Same overlap-by-1 pipeline as glm_swarm_nvfp4_pipe.py, but draft_k is
replaced by an n-gram lookup on the speculative stream: while the suffix keeps matching, we pipeline
`depth` chunks ahead; when no match is found we stall to a single sync traversal (take the target's
next token, which extends the stream and usually re-enables matching). Greedy accept => byte-identical
output. Strong on repetitive/structured text (code); weak on novel prose (low match rate).

  coord: python glm_swarm_nvfp4_ngram_pipe.py coord --stage head:port --ret-port 29600 \
         --depth 6 --K 4 --n 3 --prompt "def quicksort(arr):" --max-new 96
"""
import socket, time, argparse, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer

def ngram_draft(seq, K, n):
    """propose up to K tokens continuing seq by matching its length-n suffix to the most recent earlier occurrence."""
    if len(seq) < n + 1: return []
    suf = seq[-n:]
    for i in range(len(seq) - n - 1, n - 2, -1):
        if seq[i:i + n] == suf:
            return seq[i + n:i + n + K]
    return []

def coord(stage_ep, prompt, max_new, K, n, ret_port, depth):
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1); ret_conn = [None]
    print(f"coord(NGRAM-PIPE depth={depth} K={K} n={n}) -> head {stage_ep}; tail returns on :{ret_port}", flush=True)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]

    def send_chunk(start, toks):
        send_msg(fwd, start, torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w))
    def recv_logits():
        if ret_conn[0] is None:
            ret_conn[0], _ = ret_srv.accept(); ret_conn[0].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ret_conn[0].settimeout(300)
        _, hb = recv_msg(ret_conn[0])
        x = hb[0].float(); xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()

    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist(); L = len(ids)
    send_chunk(0, ids); r = recv_logits(); cur = r[-1]     # prefill verify -> first token
    stream = ids + [cur]                                    # stream[i] = token at position i; tail is always the last index
    pos = L; out = [cur]                                    # cur committed at position pos == len(stream)-1
    inflight = []; discard = 0
    valid = 0; accepted = 0; wasted = 0; stalls = 0; dt_recv = 0.0
    t0 = time.time(); done = False
    while not done:
        while len(inflight) < depth and not done:          # FILL via n-gram while the suffix keeps matching
            ds = ngram_draft(stream, K, n)
            send_chunk(len(stream) - 1, [stream[-1]] + ds)
            inflight.append((len(stream) - 1, ds))
            if ds: stream.extend(ds)
            else: stalls += 1; break                        # no match -> can't speculate; wait for the verify
        _tr = time.time(); r = recv_logits(); dt_recv += time.time() - _tr
        sp, ds = inflight.pop(0)
        if discard > 0:                                     # stale (post-divergence)
            discard -= 1; wasted += 1; continue
        n_acc = 0
        for j in range(len(ds)):
            if ds[j] == r[j]: n_acc += 1
            else: break
        valid += 1; accepted += n_acc
        if ds and n_acc == len(ds):                         # full accept -> commit drafts (overlap-by-1, no bonus)
            out.extend(ds); pos += len(ds); cur = ds[-1]
        else:                                               # divergence / empty chunk -> take target's correction
            committed = ds[:n_acc] + [r[n_acc]]
            out.extend(committed); cur = r[n_acc]; pos += n_acc + 1
            discard = len(inflight)                          # every chunk still in flight is stale
            del stream[pos:]; stream.append(cur)            # truncate stale speculation; stream[pos]=cur
        if len(out) >= max_new or cur in eos: done = True
    dt = time.time() - t0; ntok = len(out)
    if cur in eos and out and out[-1] in eos: out = out[:-1]
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | depth {depth} K {K} n {n} | "
          f"{valid} valid (+{wasted} stale, {stalls} stalls) | mean accept {accepted/max(valid,1):.2f} | "
          f"{(accepted+valid)/max(valid,1):.2f} tok/valid-traversal | recv-wait {dt_recv/dt:.0%}", flush=True)
    print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
    return ntok / dt

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--K", type=int, default=4); p.add_argument("--n", type=int, default=3)
    p.add_argument("--ret-port", type=int, default=29600); p.add_argument("--depth", type=int, default=6)
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.n, a.ret_port, a.depth)
