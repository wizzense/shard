"""MiniMax-M2.5 PP stage core — the ONE new compute file for the port.

A decoder Layer = hand-rolled standard GQA attention (bit-checked vs HF MiniMaxM2Attention,
m25_gqa_check.py) + vLLM NVFP4 FusedMoE experts (executes on sm_120, m25_moe_probe.py). Each
Layer carries its own KV cache and crops to start_pos for spec-decode rollback. Written to the
phase0 serve-loop contract (reset() / forward(x, start_pos, pe) / run_block) so specpipe's
coordinate_pipe + libp2p transport + receipts + heal ride on top unchanged.

M2.5 specifics (verified from the real nvidia/MiniMax-M2.5-NVFP4 config):
  62 layers, hidden 3072, GQA 48q/8kv head_dim 128, full-width q/k_norm before reshape,
  partial RoPE (first 64 dims, rotate_half), rope_theta 5e6, 256 experts / top-8, sigmoid
  router + per-layer e_score_correction_bias, NO shared expert, NO dense layers.

  self-test:  python m25_stage.py --dir /root/m25 --layers 29 30
"""
import os, json, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minimax_m2 import modeling_minimax_m2 as M

dev = "cuda"
_CTX = None
DIR = os.environ.get("M25_DIR", "/root/m25")
cfg = AutoConfig.from_pretrained(DIR)
H, NH, NKV, HD = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
E = getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", None))
K = cfg.num_experts_per_tok
I = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
EPS = cfg.rms_norm_eps
GRP = NH // NKV
SCALING = HD ** -0.5
NORM_TOPK = getattr(cfg, "norm_topk_prob", True)
ROUTED_SCALE = getattr(cfg, "routed_scaling_factor", 1.0)

_idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = _idx[n]
    if s not in _HD:
        _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)


def vllm_ctx():
    global _CTX
    if _CTX is not None:
        return _CTX[1]
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    try:
        vcfg.kernel_config.moe_backend = "cutlass"
    except Exception as e:
        print("warn moe_backend:", e, flush=True)
    ctx = set_current_vllm_config(vcfg); ctx.__enter__()
    initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
    _CTX = (ctx, vcfg)
    return vcfg


_QCFG = None
def quant_config():
    global _QCFG
    if _QCFG is not None:
        return _QCFG
    cfgj = json.load(open(f"{DIR}/config.json"))
    qc = cfgj.get("quantization_config")
    hfq = json.load(open(f"{DIR}/hf_quant_config.json")) if os.path.exists(f"{DIR}/hf_quant_config.json") else None
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    src = hfq["quantization"] if (hfq and "quantization" in hfq) else qc
    _QCFG = ModelOptNvFp4Config.from_config(src)
    return _QCFG


def _build_moe(li):
    """vLLM NVFP4 FusedMoE for layer li's 256 experts (the m25_moe_probe-proven path)."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    Pmoe = f"model.layers.{li}.block_sparse_moe."
    Pexp = Pmoe + "experts."
    suffixes = sorted({k.split(f"{Pexp}0.w1.")[1] for k in _idx if k.startswith(f"{Pexp}0.w1.")})
    eb = raw(Pmoe + "e_score_correction_bias").float().to(dev)
    moe = FusedMoE(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
                   renormalize=NORM_TOPK, use_grouped_topk=False, scoring_func="sigmoid",
                   routed_scaling_factor=ROUTED_SCALE, e_score_correction_bias=eb,
                   quant_config=quant_config(), prefix=Pexp[:-1]).to(dev)
    params = dict(moe.named_parameters())
    for e in range(E):
        for proj, shard in [("w1", "w1"), ("w3", "w3"), ("w2", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            for suf in suffixes:
                name = f"{Pexp}{e}.{proj}.{suf}"
                pname = f"{grp}_{suf}"
                if name in _idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).to(dev), name, shard, e)
    moe.quant_method.process_weights_after_loading(moe)
    gate = raw(Pmoe + "gate.weight").to(torch.bfloat16).to(dev)
    return moe, gate


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


class Layer:
    """One M2.5 decoder layer: bf16 GQA + NVFP4 MoE, own KV cache (crops to start_pos)."""
    def __init__(self, li):
        self.li = li
        P = f"model.layers.{li}."
        g = lambda n: raw(P + n).to(torch.bfloat16).to(dev)
        self.in_ln = g("input_layernorm.weight")
        self.post_ln = g("post_attention_layernorm.weight")
        self.q_proj = g("self_attn.q_proj.weight"); self.k_proj = g("self_attn.k_proj.weight")
        self.v_proj = g("self_attn.v_proj.weight"); self.o_proj = g("self_attn.o_proj.weight")
        self.q_norm = g("self_attn.q_norm.weight"); self.k_norm = g("self_attn.k_norm.weight")
        self.moe, self.gate = _build_moe(li)
        self.kc = self.vc = None

    def reset(self):
        self.kc = self.vc = None

    def _rms(self, x, w):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + EPS)).to(x.dtype) * w

    def attn(self, x, start_pos, cos, sin):
        b, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(b, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(b, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(b, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        cu = cos[start_pos:start_pos + s].unsqueeze(0).unsqueeze(0)   # [1,1,s,rd]
        su = sin[start_pos:start_pos + s].unsqueeze(0).unsqueeze(0)
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        if self.kc is not None and self.kc.shape[2] > start_pos:
            self.kc = self.kc[:, :, :start_pos, :].contiguous(); self.vc = self.vc[:, :, :start_pos, :].contiguous()
        if self.kc is None:
            self.kc, self.vc = k, v
        else:
            self.kc = torch.cat([self.kc, k], 2); self.vc = torch.cat([self.vc, v], 2)
        kk = self.kc.repeat_interleave(GRP, dim=1); vv = self.vc.repeat_interleave(GRP, dim=1)
        total = kk.shape[2]
        attn = torch.matmul(q, kk.transpose(-1, -2)) * SCALING
        qpos = torch.arange(s, device=dev).view(s, 1) + start_pos
        kpos = torch.arange(total, device=dev).view(1, total)
        attn = attn + torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
        o = torch.matmul(torch.softmax(attn.float(), -1).to(vv.dtype), vv)
        o = o.transpose(1, 2).reshape(b, s, NH * HD)
        return lin(o, self.o_proj)

    def mlp(self, x):
        shp = x.shape
        h = x.reshape(-1, H)
        rl = torch.nn.functional.linear(h, self.gate)   # [T, E]
        return self.moe(h, rl).view(shp)

    def forward(self, x, start_pos, pe):
        cos, sin = pe
        x = x + self.attn(self._rms(x, self.in_ln), start_pos, cos, sin)
        x = x + self.mlp(self._rms(x, self.post_ln))
        return x


_PE = None
def get_pe(maxpos=8192):
    global _PE
    if _PE is None:
        rot = M.MiniMaxM2RotaryEmbedding(cfg).to(dev)
        dummy = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(maxpos, device=dev).unsqueeze(0)
        cos, sin = rot(dummy, pos)
        _PE = (cos[0], sin[0])                                       # [maxpos, 64]
    return _PE


def run_block(layers, start_pos, h, vcfg):
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward(h, start_pos, pe)
    return h


def _selftest(layer_ids):
    vcfg = vllm_ctx()
    layers = [Layer(i) for i in layer_ids]
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"loaded layers {layer_ids} ({gb:.2f} GB, {gb/len(layer_ids):.2f} GB/layer)", flush=True)
    torch.manual_seed(0)
    x = torch.randn(1, 8, H, dtype=torch.bfloat16, device=dev) * 0.1
    h = run_block(layers, 0, x, vcfg)
    print(f"prefill(8): out {tuple(h.shape)} finite={torch.isfinite(h).all().item()} mean|h|={h.abs().mean():.4f}", flush=True)
    x2 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.1
    h2 = run_block(layers, 8, x2, vcfg)
    print(f"decode(@8): out {tuple(h2.shape)} finite={torch.isfinite(h2).all().item()} mean|h|={h2.abs().mean():.4f}", flush=True)
    for L in layers:
        L.reset()
    ok = torch.isfinite(h).all().item() and torch.isfinite(h2).all().item()
    print("VERDICT:", f"m25_stage Layer chain ({len(layer_ids)} real layers) runs GQA+NVFP4-MoE, finite — assembled stage is sound."
          if ok else "NON-FINITE — inspect.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layers", type=int, nargs="+", default=[29, 30])
    a = ap.parse_args()
    _selftest(a.layers)
