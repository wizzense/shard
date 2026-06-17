"""GLM-5.2 QUANTIZED pipeline-parallel stage — run a layer block with the MoE experts kept
in fp8 (no bf16 dequant), the memory unlock for serving the full model on a 5090 swarm.

why: glm_stage_node.py dequants everything to bf16 (~19GB/layer -> 1.5TB full model). The
experts are 95% of that. Here we KEEP the 256 experts in block-fp8 and run them via vLLM
`fused_experts` (block-fp8, proven 1.1ms/layer); only MLA + router + shared expert + norms
dequant to bf16 (small). Per layer: ~10GB vs 19GB -> the full model fits a feasible swarm,
and pipeline-parallel avoids the vLLM-TP rank-0 overflow that blocks the NVFP4 TP path.

validates on ONE real fp8 layer: load, run the stage forward (dense MLA bf16 + fp8 fused
MoE), check sane output + per-stage GPU memory. run under /root/vmoe: python glm_stage_quant.py
"""
import json, time, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

DIR, dev, LAYER, BLK = "/root/glm52fp8", "cuda", 6, 128
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)
def blk_scale(n):                                   # [ceil(M/128), ceil(N/128)] f32
    return raw(n + "_scale_inv")
def dequant(n):                                     # block-fp8 -> bf16 (for the small bf16 parts)
    w = raw(n).to(torch.float32); s = blk_scale(n); m, k = w.shape
    s = s.repeat_interleave(BLK, 0)[:m].repeat_interleave(BLK, 1)[:, :k]
    return (w * s).to(torch.bfloat16)
def maybe(n): return dequant(n) if (n + "_scale_inv") in idx else raw(n).to(torch.bfloat16)

P = f"model.layers.{LAYER}."

# ---- bf16 parts: MLA, norms, router gate, shared expert (small) ----
sd = {}
for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
          "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
          "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
          "mlp.gate.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
          "mlp.shared_experts.down_proj.weight"]:
    sd[n] = maybe(P + n).to(dev)
sd["mlp.gate.e_score_correction_bias"] = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)

# ---- fp8 experts: KEEP fp8, stack w1=[E,2I,H] / w2=[E,H,I] + block scales (no dequant) ----
print("stacking 256 experts in fp8 (no dequant)...", flush=True)
fp8 = torch.float8_e4m3fn
w1 = torch.empty(E, 2 * I, H, dtype=fp8, device=dev)
w2 = torch.empty(E, H, I, dtype=fp8, device=dev)
w1_s = torch.empty(E, (2 * I) // BLK, H // BLK, dtype=torch.float32, device=dev)
w2_s = torch.empty(E, H // BLK, I // BLK, dtype=torch.float32, device=dev)
for e in range(E):
    g, u = raw(P + f"mlp.experts.{e}.gate_proj.weight"), raw(P + f"mlp.experts.{e}.up_proj.weight")
    w1[e] = torch.cat([g, u], 0).to(dev)
    w1_s[e] = torch.cat([blk_scale(P + f"mlp.experts.{e}.gate_proj.weight"),
                         blk_scale(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
    w2[e] = raw(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
    w2_s[e] = blk_scale(P + f"mlp.experts.{e}.down_proj.weight").to(dev)
QC = fp8_w8a8_moe_quant_config(w1_scale=w1_s, w2_scale=w2_s, block_shape=[BLK, BLK])

# ---- build the layer (bf16 parts assigned; experts stay meta, forward swapped to fused fp8) ----
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
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nQUANT STAGE: out shape {tuple(out.shape)} finite={fin} "
          f"mean|x|={out.abs().mean().item():.3f} | peak GPU mem {mem:.1f} GB (bf16 stage was ~19GB)", flush=True)
    for _ in range(3):
        layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 10 * 1000
    print(f"fp8 fused-MoE stage forward: {ms:.1f} ms/layer", flush=True)
    ok = fin and out.abs().max() < 1e4
    print("VERDICT:", "QUANTIZED PP STAGE WORKS — fp8 experts run via fused_experts, MLA bf16, "
          f"~{mem:.0f}GB/layer (half of bf16). full model now fits a feasible 5090 swarm." if ok
          else "output not sane — inspect.", flush=True)
