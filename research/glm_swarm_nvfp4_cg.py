"""GLM-5.2 NVFP4 swarm — PIPELINED spec-decode with a CUDA-GRAPHED 9B draft (plan B).

Same overlap-by-1 pipeline as glm_swarm_nvfp4_pipe.py, but the draft uses StaticCache (cudagraph-able)
instead of DynamicCache. The blocker was rollback: StaticCache leaves rejected drafts in the buffer
and HF's causal mask keys off max-written length, so a re-draft attends the stale tail (g collapses).
FIX: monkeypatch GLM-4's create_causal_mask to build the mask from position_ids (key_pos <= query pos)
— position_ids is derived from cache_position, so rewinding cache_position on divergence gives both the
right mask AND the right RoPE, and it's cudagraph-safe (only the position varies; arange(MAXLEN) const).
No dcache.crop, no attention_mask kwarg. --compile wraps the draft in torch.compile(reduce-overhead).

  coord: python glm_swarm_nvfp4_cg.py coord --stage head:port --ret-port 29600 --depth 6 --K 2 [--compile]
"""
import socket, time, json, argparse, torch
import transformers.models.glm4.modeling_glm4 as G
import transformers.cache_utils as CU
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

DRAFT = "/root/glm4_9b_draft"
_MAXLEN = 4096
_WRITE_POS = None    # static-address tensor that controls the StaticCache write slot (set per dstep)

def _patched_static_update(self, key_states, value_states, *args, **kwargs):
    # stock StaticLayer.update writes at its own monotonic cumulative_length, IGNORING cache_position,
    # so rollback can't move the write. Write at _WRITE_POS instead -> rollback works AND it's cudagraph-safe
    # (the graph reads the fixed-address _WRITE_POS at replay). Our patched causal mask handles the stale tail.
    if not self.is_initialized: self.lazy_initialization(key_states, value_states)
    cp = torch.arange(key_states.shape[-2], device=self.device) + _WRITE_POS
    self.keys.index_copy_(2, cp, key_states); self.values.index_copy_(2, cp, value_states)
    return self.keys, self.values
CU.StaticLayer.update = _patched_static_update   # set per-run before the cache; arange(_MAXLEN) must be a compile-time constant

def cg_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
    dtype = inputs_embeds.dtype; d = inputs_embeds.device
    qpos = position_ids.reshape(-1)                                  # [q_len]; query at qpos[i]
    kp = torch.arange(_MAXLEN, device=d)
    allow = kp.unsqueeze(0) <= qpos.unsqueeze(1)                     # key j attended iff j <= query position
    neg = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=d), torch.full((), neg, dtype=dtype, device=d))[None, None]
G.create_causal_mask = cg_causal_mask

def coord(stage_ep, prompt, max_new, K, ret_port, depth, compile=False, dump=None, plain=False):
    global _MAXLEN, _WRITE_POS
    _WRITE_POS = torch.zeros((), dtype=torch.long, device=dev)
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1); ret_conn = [None]
    print(f"coord(CG depth={depth} K={K} compile={compile}) -> head {stage_ep}; tail returns on :{ret_port}", flush=True)
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
    if plain:                                     # REFERENCE: pure 1-token greedy over the ring — no draft, no spec, no cudagraph
        t0 = time.time(); send_chunk(0, ids); r = recv_logits(); cur = r[-1]; out = [cur]; pos = L
        with torch.no_grad():
            while len(out) < max_new and cur not in eos:
                send_chunk(pos, [cur]); r = recv_logits(); cur = r[0]; out.append(cur); pos += 1
        dt = time.time() - t0
        if cur in eos and out and out[-1] in eos: out = out[:-1]
        print(f"\nPLAIN GREEDY {len(out)} tokens in {dt:.1f}s = {len(out)/dt:.2f} tok/s (no draft/spec/cudagraph)", flush=True)
        print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
        if dump:
            json.dump({"prompt": prompt, "output_text": tok.decode(ids + out, skip_special_tokens=True),
                       "output_token_ids": ids + out, "tok_s_warm": round(len(out) / dt, 2)}, open(dump, "w"))
            print(f"dumped reference -> {dump}", flush=True)
        return len(out) / dt
    print("loading draft GLM-4-9B...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    print(f"draft loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)
    DVOCAB = draft.config.vocab_size
    _MAXLEN = max(2048, L + 4 * max_new + depth * K + 256)
    dcache = StaticCache(config=draft.config, max_cache_len=_MAXLEN, device=dev, dtype=torch.bfloat16)
    step = torch.compile(draft, mode="reduce-overhead", fullgraph=False) if compile else draft
    _inp = torch.zeros((1, 1), dtype=torch.long, device=dev); _cp = torch.zeros((1,), dtype=torch.long, device=dev)
    _pid = torch.zeros((1, 1), dtype=torch.long, device=dev)
    def dstep(t, position):                       # pass position_ids EXPLICITLY (= the rewound pos) so RoPE + the
        _inp[0, 0] = t if t < DVOCAB else 0       # patched mask both use it, not the cache's max-written seq length
        _cp[0] = position; _pid[0, 0] = position; _WRITE_POS.fill_(position)   # _WRITE_POS drives the cache write slot
        if compile: torch.compiler.cudagraph_mark_step_begin()   # else cudagraph-trees reuses buffers across calls -> corrupt drafts
        return int(step(input_ids=_inp, position_ids=_pid, past_key_values=dcache, cache_position=_cp, use_cache=True).logits[0, -1].argmax())
    with torch.no_grad():
        send_chunk(0, ids); r = recv_logits(); cur = r[-1]
        _WRITE_POS.fill_(0)                                       # prefill writes [0..L-1]
        draft(input_ids=torch.tensor([[min(t, DVOCAB - 1) for t in ids]], device=dev), past_key_values=dcache,
              cache_position=torch.arange(L, device=dev), position_ids=torch.arange(L, device=dev)[None], use_cache=True)
        if compile:
            for w in range(8): dstep(cur, L + w)
    out = [cur]; pos = L; inflight = []; discard = 0; send_pos = pos; tail_tok = cur
    valid = 0; accepted = 0; wasted = 0; dt_draft = 0.0; dt_recv = 0.0
    def draft_k():
        nonlocal tail_tok
        ds = []; t = tail_tok; p = send_pos
        for _ in range(K):
            t = dstep(t, p); ds.append(t); p += 1
        return ds
    t0 = time.time()
    with torch.no_grad():
        done = False
        while not done:
            while len(inflight) < depth and not done:
                _td = time.time(); ds = draft_k(); dt_draft += time.time() - _td
                send_chunk(send_pos, [tail_tok] + ds)
                inflight.append((send_pos, ds)); tail_tok = ds[-1]; send_pos += K
            _tr = time.time(); r = recv_logits(); dt_recv += time.time() - _tr
            sp, ds = inflight.pop(0)
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
                discard = len(inflight); tail_tok = cur; send_pos = pos   # rewind: _WRITE_POS (set per dstep) moves the write; patch masks the stale tail
            if len(out) >= max_new or cur in eos: done = True
    dt = time.time() - t0; ntok = len(out)
    if cur in eos and out and out[-1] in eos: out = out[:-1]
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | depth {depth} K {K} compile={compile} | "
          f"{valid} valid (+{wasted} stale) | mean accept {accepted/max(valid,1):.2f} | "
          f"{(accepted+valid)/max(valid,1):.2f} tok/valid-traversal", flush=True)
    print(f"  time split: draft {dt_draft:.1f}s ({dt_draft/dt:.0%}) | recv-wait {dt_recv:.1f}s ({dt_recv/dt:.0%})", flush=True)
    print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
    if dump:
        json.dump({"prompt": prompt, "output_text": tok.decode(ids + out, skip_special_tokens=True),
                   "output_token_ids": ids + out, "tok_s_warm": round(ntok / dt, 2),
                   "reference_source": "plain greedy KV decode (glm_swarm_nvfp4_kv.py)"}, open(dump, "w"))
        print(f"dumped run -> {dump}", flush=True)
    return ntok / dt

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--K", type=int, default=2); p.add_argument("--ret-port", type=int, default=29600)
    p.add_argument("--depth", type=int, default=6); p.add_argument("--compile", action="store_true")
    p.add_argument("--dump", default=None); p.add_argument("--plain", action="store_true")
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.ret_port, a.depth, a.compile, a.dump, a.plain)
