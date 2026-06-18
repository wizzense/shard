"""Differential debug for plan B's rewind bug. Forward is bit-clean (glm_cg_fwdcmp 24/24) but the
pipe's StaticCache rewind gives g=0.36 (vs DynamicCache 1.94). Here we run BOTH caches in lockstep:
StaticCache+patch (cudagraph path) and DynamicCache (known-good). We USE DynamicCache's draft so the
run stays correct (g high), and LOG the first positions where StaticCache disagrees -> that pinpoints
exactly where rewind corrupts the draft. Both caches see the same committed tokens (we feed dc's pick
to both), so a DIFF means StaticCache's attention/state is wrong after a rollback. Eager (no compile).
  coord: python glm_swarm_nvfp4_cg_diff.py coord --stage head:port --ret-port 29600 --depth 6 --K 2
"""
import socket, time, argparse, sys, torch
import transformers.models.glm4.modeling_glm4 as G
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache, DynamicCache

DRAFT = "/root/glm4_9b_draft"; _MAXLEN = 4096
_ORIG_MASK = G.create_causal_mask                                    # stock mask (used for the DynamicCache reference)

def cg_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
    dt = inputs_embeds.dtype; d = inputs_embeds.device               # StaticCache-only: full MAXLEN buffer
    qp = position_ids.reshape(-1); kp = torch.arange(_MAXLEN, device=d)
    allow = kp.unsqueeze(0) <= qp.unsqueeze(1)
    return torch.where(allow, torch.zeros((), dtype=dt, device=d), torch.full((), torch.finfo(dt).min, dtype=dt, device=d))[None, None]

def coord(stage_ep, prompt, max_new, K, ret_port, depth):
    global _MAXLEN
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    DVOCAB = draft.config.vocab_size
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1); ret_conn = [None]
    print(f"coord(CG-DIFF depth={depth} K={K}) -> head {stage_ep}", flush=True)
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
    _MAXLEN = max(2048, L + 4 * max_new + depth * K + 256)
    sc = StaticCache(config=draft.config, max_cache_len=_MAXLEN, device=dev, dtype=torch.bfloat16)
    dc = DynamicCache()
    _inp = torch.zeros((1, 1), dtype=torch.long, device=dev); _cp = torch.zeros((1,), dtype=torch.long, device=dev); _pid = torch.zeros((1, 1), dtype=torch.long, device=dev)
    ndiff = [0]; nstep = [0]
    def dstep_both(t, position):
        ti = t if t < DVOCAB else 0
        _inp[0, 0] = ti; _cp[0] = position; _pid[0, 0] = position
        G.create_causal_mask = cg_causal_mask
        ps = int(draft(input_ids=_inp, position_ids=_pid, cache_position=_cp, past_key_values=sc, use_cache=True).logits[0, -1].argmax())
        G.create_causal_mask = _ORIG_MASK
        pd = int(draft(input_ids=torch.tensor([[ti]], device=dev), past_key_values=dc, use_cache=True).logits[0, -1].argmax())
        nstep[0] += 1
        if ps != pd:
            ndiff[0] += 1
            if ndiff[0] <= 25: print(f"DIFF@pos={position} static={ps} dyn={pd} (dc_len={dc.get_seq_length()})", file=sys.stderr, flush=True)
        return pd
    with torch.no_grad():
        send_chunk(0, ids); r = recv_logits(); cur = r[-1]
        cids = torch.tensor([[min(t, DVOCAB - 1) for t in ids]], device=dev)
        G.create_causal_mask = cg_causal_mask
        draft(input_ids=cids, past_key_values=sc, cache_position=torch.arange(L, device=dev), position_ids=torch.arange(L, device=dev)[None], use_cache=True)
        G.create_causal_mask = _ORIG_MASK
        draft(input_ids=cids, past_key_values=dc, use_cache=True)
    out = [cur]; pos = L; inflight = []; discard = 0; send_pos = pos; tail_tok = cur
    valid = 0; accepted = 0; wasted = 0
    def draft_k():
        nonlocal tail_tok
        ds = []; t = tail_tok; p = send_pos
        for _ in range(K):
            t = dstep_both(t, p); ds.append(t); p += 1
        return ds
    t0 = time.time()
    with torch.no_grad():
        done = False
        while not done:
            while len(inflight) < depth and not done:
                ds = draft_k(); send_chunk(send_pos, [tail_tok] + ds)
                inflight.append((send_pos, ds)); tail_tok = ds[-1]; send_pos += K
            r = recv_logits(); sp, ds = inflight.pop(0)
            if send_pos > _MAXLEN - 2 * K - 2: print("SAFETY BREAK", file=sys.stderr, flush=True); break
            if discard > 0:
                discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            valid += 1; accepted += n
            if n == K:
                out.extend(ds); pos += K; cur = ds[-1]
            else:
                out.extend(ds[:n] + [r[n]]); cur = r[n]; pos += n + 1
                discard = len(inflight); tail_tok = cur; send_pos = pos
                dc.crop(pos)                                   # roll back BOTH: dc crops, sc resets its length tracker
                for lyr in sc.layers:                          # FIX: rewind cumulative_length to pos (else stays at max-written)
                    if torch.is_tensor(getattr(lyr, "cumulative_length", None)): lyr.cumulative_length.fill_(pos)
                print(f"ROLLBACK to pos={pos} (n={n}) sc_len={sc.get_seq_length()}", file=sys.stderr, flush=True)
            if len(out) >= max_new or cur in eos: done = True
    dt = time.time() - t0; ntok = len(out)
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | g(dyn)={accepted/max(valid,1):.2f} | "
          f"STATIC DISAGREEMENTS: {ndiff[0]}/{nstep[0]} draft steps", flush=True)
    print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:300]), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--K", type=int, default=2); p.add_argument("--ret-port", type=int, default=29600); p.add_argument("--depth", type=int, default=6)
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.ret_port, a.depth)
