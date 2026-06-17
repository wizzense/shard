"""GLM-5.2 fp8 STAGE RUNNER — the plan-A crux on REAL weights.

loads one real GLM-5.2-FP8 layer (block-fp8, DeepSeek-V3 format: 128x128 weight_scale_inv),
block-dequantizes it, stacks the 256 separate experts into the module's fused param layout,
and runs the block forward with hidden-state in -> hidden-state out (what a swarm stage does).
validates the real weights load + the block produces sane finite output. speed at fp8 is
already shown separately (fused_experts 1.1ms); this is the correctness half.

run under /root/glmenv: python glm_stage_fp8.py
"""
import json, glob, time, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

DIR, dev, LAYER, BLK = "/root/glm52fp8", "cuda", 6, 128
cfg = GlmMoeDsaConfig.from_pretrained(DIR)
cfg._attn_implementation = "eager"
H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
print(f"layer {LAYER} | H={H} experts={E} moe_inter={I}", flush=True)

idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_handles = {}
def _h(shard):
    if shard not in _handles: _handles[shard] = safe_open(f"{DIR}/{shard}", "pt", device="cpu")
    return _handles[shard]
def raw(name): return _h(idx[name]).get_tensor(name)
def dequant(name):                                  # block-fp8 -> bf16
    w = raw(name).to(torch.float32)
    s = raw(name + "_scale_inv")
    m, n = w.shape
    bm, bn = -(-m // BLK), -(-n // BLK)
    if tuple(s.shape) == (bn, bm) and bm != bn:     # scale stored transposed
        s = s.t().contiguous()
    assert tuple(s.shape) == (bm, bn), f"{name}: w{(m,n)} s{tuple(s.shape)} exp{(bm,bn)}"
    s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :n]
    return (w * s).to(torch.bfloat16)
def maybe(name):                                    # dequant if fp8 else bf16 direct
    return dequant(name) if (name + "_scale_inv") in idx else raw(name).to(torch.bfloat16)

P = f"model.layers.{LAYER}."
sd = {}
for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
          "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
          "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
          "mlp.gate.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
          "mlp.shared_experts.down_proj.weight"]:
    sd[n] = maybe(P + n).to(dev)
sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").to(torch.float32).to(dev)

print("stacking 256 experts (block-dequant)...", flush=True)
gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device=dev)
down = torch.empty(E, H, I, dtype=torch.bfloat16, device=dev)
for e in range(E):
    g = dequant(P + f"mlp.experts.{e}.gate_proj.weight")
    u = dequant(P + f"mlp.experts.{e}.up_proj.weight")
    gate_up[e] = torch.cat([g, u], 0).to(dev)
    down[e] = dequant(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
sd["mlp.experts.gate_up_proj"] = gate_up
sd["mlp.experts.down_proj"] = down

with torch.device("meta"):                          # no alloc; assign real weights below
    layer = M.GlmMoeDsaDecoderLayer(cfg, LAYER)
missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
loaded = [k for k in sd]
print(f"loaded {len(loaded)} tensors | unexpected {len(unexpected)} | still-missing(non-indexer): "
      f"{[m for m in missing if 'indexer' not in m][:6]}", flush=True)
layer.eval()  # loaded params already on dev (assign=True); unused indexer stays meta (bypassed)


def dense_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
        b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    o, w = M.eager_attention_forward(self, query_states, key_states, value_states,
                                     attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None


M.GlmMoeDsaAttention.forward = dense_attn_forward
rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
T = 6
torch.manual_seed(0)
h = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev) * 0.1   # prev-stage hidden in
pos = torch.arange(T, device=dev).unsqueeze(0)
pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))

with torch.no_grad():
    out = layer(h, attention_mask=mask, position_ids=pos, past_key_values=None,
                use_cache=False, position_embeddings=pe, prev_topk_indices=None)[0]
    fin = torch.isfinite(out).all().item()
    print(f"\nSTAGE OUTPUT: shape {tuple(out.shape)} | finite={fin} | "
          f"mean|x|={out.abs().mean().item():.3f} max|x|={out.abs().max().item():.1f} "
          f"in|x|mean={h.abs().mean().item():.3f}", flush=True)
    for _ in range(3):
        layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 10 * 1000
    print(f"eager layer forward (NaiveMoe loop): {ms:.1f} ms (correctness run; fp8-fused MoE = 1.1ms)", flush=True)
    print("VERDICT:", "REAL GLM-5.2 fp8 WEIGHTS LOAD + STAGE FORWARD PRODUCES SANE OUTPUT — "
          "plan-A stage runner works on real weights." if fin and out.abs().max() < 1e4 else
          "output not sane — inspect.", flush=True)
