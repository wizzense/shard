"""GLM-5.2 NVFP4 swarm + MTP speculative decoding — the WAN-latency lever.

Each WAN round-trip through the ~N-stage chain costs the same whether it carries 1 token or K+1.
So the coordinator drafts K tokens locally with GLM-5.2's native MTP head (a full MLA+MoE layer,
fp8, EAGLE-style), ships [cur,d_1..d_K] through the chain in ONE traversal, and accepts the
longest prefix the target agrees with (greedy) -> output token-identical to plain decode, but
(accepted+1) tokens committed per round-trip. Correctness is independent of draft quality: a
rejected draft falls back to the target's own argmax. (acceptance g sets the speedup.)

Stages: run glm_swarm_nvfp4_kv.py stage (nvfp4 KV-cached, crops cache to start_pos -> rejected
drafts roll back for free). Coordinator: this file, holds embed/lm_head/norm + the MTP head.

  stage: python glm_swarm_nvfp4_kv.py stage --layers 6 7 8 9 --port 29600 [--next ...]
  coord: python glm_swarm_nvfp4_spec.py coord --stage host:port --prompt "..." --max-new 64 --K 4
"""
import os, json, time, socket, argparse, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import H, eps, dev, cfg, send_msg, recv_msg, _get_pe, run_block
from safetensors import safe_open
from transformers import AutoTokenizer

MTP_DIR = os.environ.get("MTP_DIR", "/root/glm52_mtp")
MTP_FILE = f"{MTP_DIR}/mtp_layer78.safetensors"
BLK = 128

# ---- MTP head (fp8): a full GLM-MoE-DSA decoder layer + EAGLE glue (enorm/hnorm/eh_proj/shared_head) ----
class MTP:
    def __init__(self, embed_w, lm_head_w):
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        self.fused_experts = fused_experts
        self.embed_w, self.lm_head_w = embed_w, lm_head_w
        sf = safe_open(MTP_FILE, "pt", device="cpu")
        self.sf = sf
        P = "model.layers.78."
        def g(n): return sf.get_tensor(P + n)
        def deq(n):                                   # block-fp8 -> bf16
            w = g(n).float(); s = g(n + "_scale_inv")
            m, k = w.shape
            s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :k]
            return (w * s).to(torch.bfloat16).to(dev)
        self.enorm = g("enorm.weight").to(torch.bfloat16).to(dev)
        self.hnorm = g("hnorm.weight").to(torch.bfloat16).to(dev)
        self.eh = g("eh_proj.weight").to(torch.bfloat16).to(dev)   # eh_proj is bf16 (not fp8)
        self.in_ln = g("input_layernorm.weight").to(torch.bfloat16).to(dev)
        self.post_ln = g("post_attention_layernorm.weight").to(torch.bfloat16).to(dev)
        self.sh_norm = g("shared_head.norm.weight").float().to(dev)
        # MLA (dequant to bf16)
        self.q_a = deq("self_attn.q_a_proj.weight"); self.q_a_ln = g("self_attn.q_a_layernorm.weight").to(torch.bfloat16).to(dev)
        self.q_b = deq("self_attn.q_b_proj.weight")
        self.kv_a = deq("self_attn.kv_a_proj_with_mqa.weight"); self.kv_a_ln = g("self_attn.kv_a_layernorm.weight").to(torch.bfloat16).to(dev)
        self.kv_b = deq("self_attn.kv_b_proj.weight"); self.o = deq("self_attn.o_proj.weight")
        self.gate = g("mlp.gate.weight").to(torch.bfloat16).to(dev)
        self.gate_bias = g("mlp.gate.e_score_correction_bias").float().to(dev)
        self.nheads = cfg.num_attention_heads; self.qk_nope = cfg.qk_nope_head_dim; self.qk_rope = cfg.qk_rope_head_dim
        self.qk_head = self.qk_nope + self.qk_rope; self.v_head = cfg.v_head_dim; self.kv_lora = cfg.kv_lora_rank
        self.scaling = self.qk_head ** -0.5; self.E = cfg.n_routed_experts; self.I = cfg.moe_intermediate_size
        # experts fp8 stacked
        fp8 = torch.float8_e4m3fn
        E, I = self.E, self.I
        self.w1 = torch.empty(E, 2 * I, H, dtype=fp8, device=dev); self.w2 = torch.empty(E, H, I, dtype=fp8, device=dev)
        w1s = torch.empty(E, (2 * I) // BLK, H // BLK, dtype=torch.float32, device=dev)
        w2s = torch.empty(E, H // BLK, I // BLK, dtype=torch.float32, device=dev)
        EP = "mlp.experts."   # g() already prepends P
        for e in range(E):
            self.w1[e] = torch.cat([g(f"{EP}{e}.gate_proj.weight"), g(f"{EP}{e}.up_proj.weight")], 0).to(dev)
            w1s[e] = torch.cat([g(f"{EP}{e}.gate_proj.weight_scale_inv"), g(f"{EP}{e}.up_proj.weight_scale_inv")], 0).to(dev)
            self.w2[e] = g(f"{EP}{e}.down_proj.weight").to(dev); w2s[e] = g(f"{EP}{e}.down_proj.weight_scale_inv").to(dev)
        self.qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=[BLK, BLK])
        # shared expert fp8 (1 expert)
        self.sw1 = torch.cat([deq("mlp.shared_experts.gate_proj.weight"), deq("mlp.shared_experts.up_proj.weight")], 0)
        self.sw2 = deq("mlp.shared_experts.down_proj.weight")
        self.kc = self.vc = None
        print(f"MTP head loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)

    def reset(self): self.kc = self.vc = None
    def crop(self, n):
        if self.kc is not None and self.kc.shape[2] > n:
            self.kc = self.kc[:, :, :n, :].contiguous(); self.vc = self.vc[:, :, :n, :].contiguous()
    def _rms(self, x, w):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

    def _attn(self, x, start_pos, pe):
        b, s = x.shape[:2]
        q = torch.nn.functional.linear(self._rms(torch.nn.functional.linear(x, self.q_a), self.q_a_ln), self.q_b)
        q = q.view(b, s, self.nheads, self.qk_head).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope, self.qk_rope], -1)
        ckv = torch.nn.functional.linear(x, self.kv_a)
        k_pass_c, k_rot = torch.split(ckv, [self.kv_lora, self.qk_rope], -1)
        k_pass = torch.nn.functional.linear(self._rms(k_pass_c, self.kv_a_ln), self.kv_b)
        k_pass = k_pass.view(b, s, self.nheads, self.qk_nope + self.v_head).transpose(1, 2)
        k_nope, value = torch.split(k_pass, [self.qk_nope, self.v_head], -1)
        k_rot = k_rot.view(b, 1, s, self.qk_rope)
        cos, sin = pe[0][start_pos:start_pos+s].unsqueeze(0), pe[1][start_pos:start_pos+s].unsqueeze(0)
        q_rot, k_rot = KV.M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(b, self.nheads, s, self.qk_rope)
        Q = torch.cat([q_pass, q_rot], -1); Knew = torch.cat([k_nope, k_rot], -1)
        if self.kc is not None and self.kc.shape[2] > start_pos:
            self.kc = self.kc[:, :, :start_pos, :].contiguous(); self.vc = self.vc[:, :, :start_pos, :].contiguous()
        if self.kc is None: self.kc, self.vc = Knew, value
        else: self.kc = torch.cat([self.kc, Knew], 2); self.vc = torch.cat([self.vc, value], 2)
        total = self.kc.shape[2]
        attn = torch.matmul(Q, self.kc.transpose(-1, -2)) * self.scaling
        qpos = torch.arange(s, device=dev).view(s, 1) + start_pos; kpos = torch.arange(total, device=dev).view(1, total)
        attn = attn + torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
        o = torch.matmul(torch.softmax(attn.float(), -1).to(value.dtype), self.vc).transpose(1, 2).reshape(b, s, -1)
        return torch.nn.functional.linear(o, self.o)

    def _moe(self, x):
        from vllm.model_executor.layers.fused_moe import fused_experts
        b, s = x.shape[:2]; h = x.view(-1, H)
        rl = torch.nn.functional.linear(h, self.gate).float()
        scores = rl.sigmoid() + self.gate_bias
        grp = scores.view(h.shape[0], cfg.n_group, -1)
        gt = grp.topk(2, -1).values.sum(-1)
        gmask = torch.zeros_like(gt).scatter_(1, gt.topk(cfg.topk_group, -1).indices, 1.0).bool()
        smask = gmask.unsqueeze(-1).expand(-1, -1, scores.shape[-1] // cfg.n_group).reshape(h.shape[0], -1)
        sel = scores.masked_fill(~smask, float("-inf"))
        tw, tid = sel.topk(cfg.num_experts_per_tok, -1)
        tw = (rl.sigmoid().gather(1, tid))
        if cfg.norm_topk_prob: tw = tw / tw.sum(-1, keepdim=True)
        tw = (tw * cfg.routed_scaling_factor).to(torch.bfloat16)
        routed = self.fused_experts(h, self.w1, self.w2, tw, tid.to(torch.int32), quant_config=self.qc)
        shared = torch.nn.functional.silu(torch.nn.functional.linear(h, self.sw1[:self.I]) ) * \
                 torch.nn.functional.linear(h, self.sw1[self.I:])
        shared = torch.nn.functional.linear(shared, self.sw2)
        return (routed + shared).view(b, s, H)

    def step(self, token, hprev, position, pe):
        """one EAGLE draft step: (token at `position`, prev feature) -> feature + next-token logit."""
        e = torch.nn.functional.embedding(torch.tensor([[token]], device=dev), self.embed_w)
        if position == 0: e = torch.zeros_like(e)
        e = self._rms(e, self.enorm); hn = self._rms(hprev.view(1, 1, H), self.hnorm)
        x = torch.nn.functional.linear(torch.cat([e, hn], -1), self.eh)
        x = x + self._attn(self._rms(x, self.in_ln), position, pe)
        x = x + self._moe(self._rms(x, self.post_ln))
        feat = x[0, -1]
        z = feat.float() * torch.rsqrt(feat.float().pow(2).mean() + eps) * self.sh_norm
        logit = (z.to(torch.bfloat16) @ self.lm_head_w.t()).float()
        return feat, int(logit.argmax())

# ---- coordinator: spec-decode loop ----
def coord_spec(stage_ep, prompt, max_new, Kd):
    KV._vllm_ctx()
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    mtp = MTP(embed_w, lm_head_w); pe = _get_pe()
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=300); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.settimeout(300)
    print(f"coord(SPEC K={Kd}) -> {stage_ep}", flush=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    def chain(start_pos, toks):                       # send tokens' embeds, return per-position target argmaxes + hiddens
        h = torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w)
        send_msg(s, start_pos, h); _, hb = recv_msg(s)
        x = hb[0].float(); xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        r = (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()
        return r, hb[0]                                # r[i] = target token after position start_pos+i ; hb[i] = hidden
    L = ids.shape[1]; t0 = time.time()
    r, hidden = chain(0, ids[0].tolist())             # prefill
    cur = r[-1]; h_seed = hidden[-1]; pos = L
    # sync MTP cache over the prompt: feed (token_p, h_{p-1}) so its attn has context
    mtp.reset()
    for pp in range(1, L):
        mtp.step(ids[0, pp].item(), hidden[pp - 1], pp - 1, pe)
    out = [cur]; rounds = 0; accepted = 0
    while len(out) < max_new and cur not in eos:
        # draft K: MTP runs one position behind the main model (predicts token_{i+2} from
        # (token_{i+1}, h_i)); first step uses the real seed hidden, then chains its own features.
        drafts = []; tk = cur; hp = h_seed
        for k in range(Kd):
            hp, nt = mtp.step(tk, hp, pos - 1 + k, pe)
            drafts.append(nt); tk = nt
        r, hidden = chain(pos, [cur] + drafts)        # verify [cur,d_1..dK] at positions pos..pos+K
        n = 0
        for j in range(Kd):
            if drafts[j] == r[j]: n += 1
            else: break
        committed = drafts[:n] + [r[n]]
        out.extend(committed)
        # re-sync the MTP cache for the accepted positions using the REAL main hiddens (not the
        # speculative self-features) so the next round's draft attends to accurate context.
        mtp.crop(pos)
        for j in range(n):
            mtp.step(committed[j], hidden[j], pos + j, pe)
        cur = r[n]; h_seed = hidden[n]; pos += n + 1
        rounds += 1; accepted += n
        if any(t in eos for t in committed): break
    dt = time.time() - t0; ntok = len(out)
    full = ids[0].tolist() + out
    if any(t in eos for t in out): full = full[:len(ids[0]) + out.index(next(t for t in out if t in eos))]
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | {rounds} traversals | "
          f"mean accept {accepted/max(rounds,1):.2f} | {(accepted+rounds)/max(rounds,1):.2f} tok/traversal (vs 1.0 plain)", flush=True)
    print("decoded:", repr(tok.decode(full, skip_special_tokens=True)[:500]), flush=True)
    return ntok / dt, (accepted + rounds) / max(rounds, 1), tok.decode(full, skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("stage"); p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--port", type=int, default=29600); p.add_argument("--next", default=None)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="The capital of France is"); p.add_argument("--max-new", type=int, default=32); p.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    if a.role == "stage": KV.stage(a.layers, a.port, a.next)
    else: coord_spec(a.stage, a.prompt, a.max_new, a.K)
