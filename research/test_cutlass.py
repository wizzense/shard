"""Definitive test: force the precompiled VLLM_CUTLASS nvfp4 MoE backend (no flashinfer JIT).
Confirms the config value sticks, then builds a real layer-6 FusedMoE + forwards. If it picks
VLLM_CUTLASS and runs without compiling -> that's the fleet fix. run foreground under /root/vmoe."""
import os, json, torch
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29571")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

DIR, dev, LAYER = "/root/glm52nvfp4", "cuda", 6
torch.cuda.set_device(0)
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
vcfg = VllmConfig()
vcfg.kernel_config.moe_backend = "cutlass"   # force precompiled VLLM_CUTLASS
_ctx = set_current_vllm_config(vcfg); _ctx.__enter__()
print("CONFIRM current moe_backend =", get_current_vllm_config().kernel_config.moe_backend, flush=True)
initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))

cfg = GlmMoeDsaConfig.from_pretrained(DIR)
H, E, I, K = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = idx[n]
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)
P = f"model.layers.{LAYER}.mlp.experts."
eb = raw(f"model.layers.{LAYER}.mlp.gate.e_score_correction_bias").float().to(dev)
moe = FusedMoE(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
               renormalize=cfg.norm_topk_prob, use_grouped_topk=True, num_expert_group=cfg.n_group,
               topk_group=cfg.topk_group, scoring_func="sigmoid", routed_scaling_factor=cfg.routed_scaling_factor,
               e_score_correction_bias=eb, quant_config=qnv, prefix=P).to(dev)
pp = dict(moe.named_parameters())
for e in range(E):
    for proj, shard in [("gate_proj","w1"),("up_proj","w3"),("down_proj","w2")]:
        grp = "w2" if shard=="w2" else "w13"
        for suf in ["weight","weight_scale","weight_scale_2","input_scale"]:
            n=f"{P}{e}.{proj}.{suf}"
            if n in idx: moe.weight_loader(pp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, e)
moe.quant_method.process_weights_after_loading(moe)
print("experts loaded; forwarding...", flush=True)
torch.manual_seed(0)
x = torch.randn(6, H, dtype=torch.bfloat16, device=dev)*0.1
rl = torch.randn(6, E, dtype=torch.bfloat16, device=dev)
with torch.no_grad(), set_forward_context(None, vcfg):
    out = moe(x, rl)
print(f"FORWARD OK finite={torch.isfinite(out).all().item()} mean|x|={out.abs().mean().item():.4f}", flush=True)
print("VERDICT: VLLM_CUTLASS nvfp4 MoE runs WITHOUT flashinfer JIT." if torch.isfinite(out).all() else "not finite", flush=True)
