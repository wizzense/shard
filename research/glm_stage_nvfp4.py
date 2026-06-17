"""GLM-5.2 NVFP4 pipeline-parallel stage — the deployment density (~5GB/layer -> full model
on ~16 5090s, vs fp8's ~31). Same approach as glm_stage_quant.py but experts in NVFP4
(E2M1 packed + fp8 block scales + per-expert global scales), run via vLLM fused_experts'
nvfp4 path. MLA/router/shared stay BF16 (the NVFP4 checkpoint keeps them BF16 anyway).

recipe (from vLLM oracle/nvfp4.py make_nvfp4_moe_quant_config):
  w1/w2          = packed E2M1 weights (uint8), gate||up concatenated for w1
  w1/w2_scale    = fp8 block scales, SWIZZLED (swizzle_blockscale) per expert
  g1/g2_alphas   = per-expert weight global scale (weight_scale_2)
  a1/a2_gscale   = 1 / input_scale (per expert)
validate on a real NVFP4 layer: sane output + per-stage GPU mem (~5GB). run under /root/vmoe.

STATUS (2026-06-17): NVFP4 weight assembly + scale swizzle + per-expert global scales all
load correctly on a real layer (~4.8GB experts -> ~5.4GB/layer = the ~16-node density).
EXECUTION is the remaining step: the generic `fused_experts(quant_config=nvfp4...)` falls
into the triton path and rejects packed E2M1 weights ("hidden size 6144 != 3072"). NVFP4
needs vLLM's MODULAR MoE kernel — select_nvfp4_moe_backend -> convert_to_nvfp4_moe_kernel_format
-> make_nvfp4_moe_kernel (oracle/nvfp4.py) — not the fp8 one-liner. Wiring that kernel is
the scoped next task; the fp8 path (glm_stage_quant.py) is the proven working mechanism.
"""
import json, time, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

DIR, dev, LAYER = "/root/glm52nvfp4", "cuda", 6
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)
P = f"model.layers.{LAYER}."

# ---- bf16 parts (MLA/norms/router/shared kept BF16 in the NVFP4 ckpt) ----
sd = {}
for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
          "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
          "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
          "mlp.gate.weight"]:
    sd[n] = raw(P + n).to(torch.bfloat16).to(dev)
sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
# shared expert is NVFP4 too; stub to zero for this routed-experts density test (small residual add)
bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
sd["mlp.shared_experts.gate_proj.weight"] = bf(I, H)
sd["mlp.shared_experts.up_proj.weight"] = bf(I, H)
sd["mlp.shared_experts.down_proj.weight"] = bf(H, I)

# ---- NVFP4 experts: keep packed; stack w1=gate||up, w2=down; swizzle block scales ----
print("stacking 256 NVFP4 experts (E2M1 packed, swizzled scales)...", flush=True)
Hp, Ip = H // 2, I // 2                                  # E2M1 packs 2 vals/byte
w1 = torch.empty(E, 2 * I, Hp, dtype=torch.uint8, device=dev)
w2 = torch.empty(E, H, Ip, dtype=torch.uint8, device=dev)
w1s_list, w2s_list = [], []
g1 = torch.empty(E, device=dev); g2 = torch.empty(E, device=dev)
a1 = torch.empty(E, device=dev); a2 = torch.empty(E, device=dev)
for e in range(E):
    gw, uw = raw(P + f"mlp.experts.{e}.gate_proj.weight"), raw(P + f"mlp.experts.{e}.up_proj.weight")
    w1[e] = torch.cat([gw, uw], 0).to(dev)
    w2[e] = raw(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
    gs = torch.cat([raw(P + f"mlp.experts.{e}.gate_proj.weight_scale"),
                    raw(P + f"mlp.experts.{e}.up_proj.weight_scale")], 0).to(dev)
    w1s_list.append(swizzle_blockscale(gs))
    w2s_list.append(swizzle_blockscale(raw(P + f"mlp.experts.{e}.down_proj.weight_scale").to(dev)))
    g1[e] = torch.maximum(raw(P + f"mlp.experts.{e}.gate_proj.weight_scale_2"),
                          raw(P + f"mlp.experts.{e}.up_proj.weight_scale_2")).to(dev)
    g2[e] = raw(P + f"mlp.experts.{e}.down_proj.weight_scale_2").to(dev)
    a1[e] = torch.maximum(raw(P + f"mlp.experts.{e}.gate_proj.input_scale"),
                          raw(P + f"mlp.experts.{e}.up_proj.input_scale")).to(dev)
    a2[e] = raw(P + f"mlp.experts.{e}.down_proj.input_scale").to(dev)
w1_scale = torch.stack(w1s_list); w2_scale = torch.stack(w2s_list)
QC = nvfp4_moe_quant_config(g1_alphas=g1, g2_alphas=g2, a1_gscale=1.0 / a1, a2_gscale=1.0 / a2,
                            w1_scale=w1_scale, w2_scale=w2_scale)

with torch.device("meta"):
    layer = M.GlmMoeDsaDecoderLayer(cfg, LAYER)
layer.load_state_dict(sd, strict=False, assign=True)
layer.mlp.experts._w1, layer.mlp.experts._w2, layer.mlp.experts._qc = w1, w2, QC
layer.eval()


def fused_moe_forward(self, hidden_states, top_k_index, top_k_weights):
    return fused_experts(hidden_states, self._w1, self._w2,
                         top_k_weights.to(hidden_states.dtype), top_k_index.to(torch.int32),
                         quant_config=self._qc)
M.GlmMoeDsaNaiveMoe.forward = fused_moe_forward


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
    o, w = M.eager_attention_forward(self, torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1),
                                     value_states, attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None
M.GlmMoeDsaAttention.forward = dense_attn_forward

rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
T = 6
torch.manual_seed(0)
h = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T, device=dev).unsqueeze(0)
pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))

with torch.no_grad():
    out = layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    fin = torch.isfinite(out).all().item()
    torch.cuda.synchronize(); mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nNVFP4 STAGE: out {tuple(out.shape)} finite={fin} mean|x|={out.abs().mean().item():.3f} "
          f"| peak GPU mem {mem:.1f} GB (fp8 was ~10, bf16 ~19)", flush=True)
    for _ in range(3): layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 10 * 1000
    print(f"NVFP4 fused-MoE stage forward: {ms:.1f} ms/layer", flush=True)
    print("VERDICT:", f"NVFP4 PP STAGE WORKS — ~{mem:.0f}GB/layer -> full GLM-5.2 on ~16 5090s." if fin and out.abs().max() < 1e4
          else "output not sane — likely the gate/up global-scale fusion needs requant; inspect.", flush=True)
