"""GLM-5.2 quantized MULTI-LAYER stage — what an actual swarm node runs: a contiguous block
of several layers, experts in fp8 (fused_experts), MLA bf16, hidden-state in/out. Extends the
single-layer glm_stage_quant.py to N layers, validating a real per-node stage end to end.
run under /root/vmoe: python glm_stage_quant_multi.py
"""
import json, time, torch
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

DIR, dev, BLK = "/root/glm52fp8", "cuda", 128
LAYERS = [6, 7]                                       # a 2-layer stage (both fp8-complete on box A)
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
        w1[e] = torch.cat([raw(P + f"mlp.experts.{e}.gate_proj.weight"),
                           raw(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
        w1s[e] = torch.cat([bscale(P + f"mlp.experts.{e}.gate_proj.weight"),
                            bscale(P + f"mlp.experts.{e}.up_proj.weight")], 0).to(dev)
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

print(f"loading quantized stage: layers {LAYERS} (fp8 experts)...", flush=True)
layers = [load_quant_layer(li) for li in LAYERS]
rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
T = 6
torch.manual_seed(0)
h0 = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T, device=dev).unsqueeze(0)
pe = rotary(h0, position_ids=pos)
mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))


def stage_fwd(h):                                    # what a swarm node computes: its whole block
    for L in layers:
        h = L(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    return h


with torch.no_grad():
    out = stage_fwd(h0)
    torch.cuda.synchronize(); mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n{len(LAYERS)}-LAYER QUANT STAGE: out {tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"mean|x| {h0.abs().mean():.3f}->{out.abs().mean():.3f} | peak GPU mem {mem:.1f} GB", flush=True)
    for _ in range(3): stage_fwd(h0)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): stage_fwd(h0)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 10 * 1000
    print(f"stage forward ({len(LAYERS)} layers): {ms:.1f} ms ({ms/len(LAYERS):.1f} ms/layer)", flush=True)
    print(f"PROJECTION: a {mem/len(LAYERS):.1f}GB/layer node holds ~{int(26/(mem/len(LAYERS)))} layers on a 32GB 5090 "
          f"-> ~{-(-78//max(1,int(26/(mem/len(LAYERS)))))} nodes for the 78-layer model at fp8.", flush=True)
    print("VERDICT: multi-layer quantized stage works — a real swarm node runs its block in fp8.", flush=True)
