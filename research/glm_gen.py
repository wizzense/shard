"""GLM-5.2 generation driver — the missing end-to-end piece for swarm serving:
tokenize -> embed -> [quant layer stages] -> final norm -> lm_head -> greedy sample -> loop.

Validates the DRIVER MECHANISM on box A with the layers we have (6,7). Output text is
gibberish (only 2 of 78 layers), but proves the full generation pipeline runs: this is the
coordinator+tail logic the swarm deploy reuses (embed on stage 0, norm+lm_head on the tail).
No KV cache here (recompute the growing seq each step) -- simplest correct loop for validation.
run under /root/vmoe: python glm_gen.py
"""
import json, time, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig, AutoTokenizer
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

DIR, dev, BLK, LAYERS = "/root/glm52fp8", "cuda", 128, [6, 7]
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)
def bscale(n): return raw(n + "_scale_inv")
def dequant(n):
    w = raw(n).to(torch.float32); s = bscale(n); m, k = w.shape
    s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :k]
    return (w * s).to(torch.bfloat16)
def maybe(n): return dequant(n) if (n + "_scale_inv") in idx else raw(n).to(torch.bfloat16)


def load_quant_layer(li):
    P = f"model.layers.{li}."
    sd = {}
    for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
              "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
              "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
              "mlp.gate.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
              "mlp.shared_experts.down_proj.weight"]:
        sd[n] = maybe(P + n).to(dev)
    sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
    fp8 = torch.float8_e4m3fn
    w1 = torch.empty(E, 2 * I, H, dtype=fp8, device=dev); w2 = torch.empty(E, H, I, dtype=fp8, device=dev)
    w1s = torch.empty(E, (2 * I) // BLK, H // BLK, dtype=torch.float32, device=dev)
    w2s = torch.empty(E, H // BLK, I // BLK, dtype=torch.float32, device=dev)
    for e in range(E):
        w1[e] = torch.cat([raw(P + f"mlp.experts.{e}.gate_proj.weight"), raw(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
        w1s[e] = torch.cat([bscale(P + f"mlp.experts.{e}.gate_proj.weight"), bscale(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
        w2[e] = raw(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
        w2s[e] = bscale(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(cfg, li)
    layer.load_state_dict(sd, strict=False, assign=True)
    ex = layer.mlp.experts
    ex._w1, ex._w2 = w1, w2
    ex._qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=[BLK, BLK])
    return layer.eval()


def fused_moe_forward(self, hidden_states, top_k_index, top_k_weights):
    return fused_experts(hidden_states, self._w1, self._w2,
                         top_k_weights.to(hidden_states.dtype), top_k_index.to(torch.int32), quant_config=self._qc)
M.GlmMoeDsaNaiveMoe.forward = fused_moe_forward


def dense_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    o, w = M.eager_attention_forward(self, torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1),
                                     value_states, attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None
M.GlmMoeDsaAttention.forward = dense_attn_forward

print("loading driver: embed + layers + norm + lm_head + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True)
embed_w = maybe("model.embed_tokens.weight").to(dev)       # [vocab, H]
lm_head_w = maybe("lm_head.weight").to(dev)                 # [vocab, H]
norm_w = maybe("model.norm.weight").to(dev)                 # RMSNorm
layers = [load_quant_layer(li) for li in LAYERS]
rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
eps = cfg.rms_norm_eps


def rmsnorm(x, w):
    x = x.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(torch.bfloat16)


@torch.no_grad()
def generate(prompt, max_new=20):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(max_new):
        L = ids.shape[1]
        h = torch.nn.functional.embedding(ids, embed_w)                       # embed (stage 0)
        pos = torch.arange(L, device=dev).unsqueeze(0); pe = rotary(h, position_ids=pos)
        mask = torch.zeros(1, 1, L, L, dtype=torch.bfloat16, device=dev)
        mask.masked_fill_(torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1), float("-inf"))
        for layer in layers:                                                 # the swarm stages
            h = layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
        h = rmsnorm(h, norm_w)                                               # tail: final norm
        logits = h[0, -1] @ lm_head_w.t().to(torch.bfloat16)                 # tail: lm_head
        nxt = int(logits.float().argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
        if nxt in (cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]):
            break
    return ids


t0 = time.time()
out = generate("The capital of France is", max_new=20)
dt = time.time() - t0
n = out.shape[1]
print(f"\nGENERATED {n} tokens in {dt:.1f}s | pipeline ran end-to-end", flush=True)
print("decoded (gibberish expected — only 2/78 layers):", repr(tok.decode(out[0], skip_special_tokens=True)[:200]), flush=True)
print("VERDICT: generation driver works — embed->quant stages->norm->lm_head->sample->loop runs end to end. "
      "Ready to distribute across the swarm (stages on nodes, embed/lm_head on coordinator+tail).", flush=True)
