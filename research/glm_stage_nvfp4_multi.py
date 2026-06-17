"""How many NVFP4 GLM-5.2 layers fit one 32GB 5090? This sets the swarm node count.
Load N contiguous MoE layers as full NVFP4 stages (routed MoE + shared + MLA), run a hidden
state through all of them in sequence, report resident weights + peak GPU. run under /root/vmoe:
  python glm_stage_nvfp4_multi.py 5
"""
import os, sys, json, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29581")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

DIR, dev = "/root/glm52nvfp4", "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# every NVFP4 layer is architecturally identical -> load layer 6 N times to measure the exact
# per-node footprint for N layers without downloading more shards.
LAYERS = [6] * N
torch.cuda.set_device(0)
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
vcfg = VllmConfig(); _ctx = set_current_vllm_config(vcfg); _ctx.__enter__()
initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))

cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I, K = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = idx[n]
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)

def shared_routing(*a, **kw):
    hs = kw["hidden_states"]; T = hs.shape[0]
    return (torch.ones(T, 1, dtype=torch.bfloat16, device=hs.device),
            torch.zeros(T, 1, dtype=torch.int32, device=hs.device))

_CTR = [0]
def build_moe(P, n_exp, inter, custom):
    _CTR[0] += 1; pfx = f"{P}#{_CTR[0]}"   # unique vLLM registration name per copy (weights still load from P)
    eb = raw(P.replace("mlp.experts.", "mlp.gate.") + "e_score_correction_bias").float().to(dev) if custom is None else None
    kw = dict(num_experts=n_exp, top_k=(K if custom is None else 1), hidden_size=H, intermediate_size=inter,
              params_dtype=torch.bfloat16, quant_config=qnv, prefix=pfx)
    if custom is None:
        kw.update(renormalize=cfg.norm_topk_prob, use_grouped_topk=True, num_expert_group=cfg.n_group,
                  topk_group=cfg.topk_group, scoring_func="sigmoid",
                  routed_scaling_factor=cfg.routed_scaling_factor, e_score_correction_bias=eb)
    else:
        kw.update(renormalize=False, custom_routing_function=custom)
    m = FusedMoE(**kw).to(dev); pp = dict(m.named_parameters())
    rng = range(n_exp) if custom is None else [None]
    for e in rng:
        for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            base = f"{P}{e}.{proj}." if custom is None else f"{P}{proj}."
            for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
                n = base + suf
                if n in idx: m.weight_loader(pp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, e if custom is None else 0)
    m.quant_method.process_weights_after_loading(m)
    return m, eb

def dense_attn(self, hidden_states, position_embeddings, attention_mask, **kw):
    b, s = hidden_states.shape[:-1]
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
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
M.GlmMoeDsaAttention.forward = dense_attn

stages = []
for L in LAYERS:
    P = f"model.layers.{L}."
    rmoe, eb = build_moe(P + "mlp.experts.", E, I, None)
    smoe, _ = build_moe(P + "mlp.shared_experts.", 1, I, shared_routing)
    sd = {n: raw(P + n).to(torch.bfloat16).to(dev) for n in
          ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
           "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
           "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
           "mlp.gate.weight"]}
    sd["mlp.gate.e_score_correction_bias"] = eb
    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(cfg, L)
    layer.load_state_dict(sd, strict=False, assign=True); layer.eval()
    def mk(rmoe, smoe):
        def fwd(self, hidden_states):
            shp = hidden_states.shape; h = hidden_states.view(-1, H)
            rl = torch.nn.functional.linear(h, self.gate.weight)
            ones = torch.ones(h.shape[0], 1, dtype=torch.bfloat16, device=h.device)
            return (rmoe(h, rl) + smoe(h, ones)).view(shp)
        return fwd
    layer.mlp.forward = mk(rmoe, smoe).__get__(layer.mlp)
    stages.append(layer)
    print(f"  loaded layer {L}  | resident {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
T = 8; torch.manual_seed(0)
h = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T, device=dev).unsqueeze(0); pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), set_forward_context(None, vcfg):
    for layer in stages:
        h = layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
resident = torch.cuda.memory_allocated() / 1e9; peak = torch.cuda.max_memory_allocated() / 1e9
fin = torch.isfinite(h).all().item()
print(f"\n{N} NVFP4 layers: finite={fin} | resident {resident:.1f} GB | peak {peak:.1f} GB "
      f"| {resident/N:.2f} GB/layer", flush=True)
fits = peak < 30
nodes = -(-78 // N)
print(f"VERDICT: {N} layers/node {'FITS' if fits else 'TIGHT/OOM'} a 32GB 5090 (peak {peak:.1f}GB). "
      f"=> ~78 layers / {N} = {nodes} nodes." if fin else "not finite — inspect.", flush=True)
