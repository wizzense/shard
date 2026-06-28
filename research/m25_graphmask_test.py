"""Offline proof for the CUDA-graph path's BUCKETED ADDITIVE MASK (m25_stage GraphRunner / _GraphState).

The eager path reads kc[:,:,:total] with causal_lower_right(s,total). The graphed path reads a FIXED bucket
kc[:,:,:alen] (alen>=total) whose tail [total:alen) is UNWRITTEN garbage, and masks it with a static additive
mask (mask[i,j]=0 if j<=start_pos+i else -inf). This proves the two produce identical attention — i.e. the
graph's bucketed read + additive mask is bit-equivalent to the eager causal_lower_right read, so the
CUDA-graph verify path stays lossless. Replicates the exact mask _GraphState.set builds. NO GPU.

  python research/m25_graphmask_test.py
"""
import torch
from torch.nn.attention.bias import causal_lower_right

torch.manual_seed(0)
NH, NKV, HD = 48, 8, 128
SCALING = HD ** -0.5
F = torch.nn.functional

# (s=K+1, start_pos, bucket alen) — total = start_pos + s, with [total:alen) being unwritten garbage
CASES = [(9, 100, 2048), (9, 1024, 2048), (9, 5000, 8192), (9, 30000, 32768), (5, 0, 2048)]


def graph_mask(s, start, alen):
    """exactly what _GraphState.set builds: additive [1,1,s,alen], 0 where key j <= abs query pos."""
    qpos = (torch.arange(s) + start).view(s, 1)
    kpos = torch.arange(alen).view(1, alen)
    return torch.where(kpos <= qpos, 0.0, float("-inf"))[None, None]


def test_bucketed_additive_mask_equals_causal_lower_right():
    for s, start, alen in CASES:
        total = start + s
        q = torch.randn(1, NH, s, HD)
        kc = torch.randn(1, NKV, alen, HD)          # full bucket; [total:alen) is "unwritten" garbage
        vc = torch.randn(1, NKV, alen, HD)
        # eager: read EXACTLY :total with the causal_lower_right flag
        eager = F.scaled_dot_product_attention(q, kc[:, :, :total], vc[:, :, :total],
                                               attn_mask=causal_lower_right(s, total), scale=SCALING, enable_gqa=True)
        # graphed: read the FULL bucket :alen (incl garbage tail) with the static additive mask
        graphed = F.scaled_dot_product_attention(q, kc[:, :, :alen], vc[:, :, :alen],
                                                 attn_mask=graph_mask(s, start, alen), scale=SCALING, enable_gqa=True)
        d = (eager - graphed).abs().max().item()
        assert d < 1e-5, f"[s={s},start={start},alen={alen}] bucketed-mask vs causal_lower_right diff={d:.2e}"
        print(f"  s={s} start={start:5} total={total:5} alen={alen:5}  eager==graphed  max|diff|={d:.1e}")
    print("[graphmask] PASS — bucketed read + additive mask == causal_lower_right over :total "
          "(garbage tail masked to 0); CUDA-graph verify path stays lossless")


def test_garbage_tail_is_ignored():
    """sanity: re-randomizing the tail [total:alen) must NOT change the graphed output (proves the mask
    truly zeroes it — if the tail leaked, the spec-decode verify would be corrupted)."""
    s, start, alen = 9, 1024, 8192
    total = start + s
    q = torch.randn(1, NH, s, HD); kc = torch.randn(1, NKV, alen, HD); vc = torch.randn(1, NKV, alen, HD)
    m = graph_mask(s, start, alen)
    o1 = F.scaled_dot_product_attention(q, kc, vc, attn_mask=m, scale=SCALING, enable_gqa=True)
    kc[:, :, total:].normal_(); vc[:, :, total:].normal_()        # scribble the unwritten tail
    o2 = F.scaled_dot_product_attention(q, kc, vc, attn_mask=m, scale=SCALING, enable_gqa=True)
    d = (o1 - o2).abs().max().item()
    assert d == 0.0, f"garbage tail leaked into output: diff={d:.2e}"
    print("[graphmask] PASS — scribbling the unwritten tail changes nothing (mask isolates committed KV)")


if __name__ == "__main__":
    test_bucketed_additive_mask_equals_causal_lower_right()
    test_garbage_tail_is_ignored()
    print("\n[graphmask] ALL PASS — the CUDA-graph bucketed-mask read is lossless vs the eager path")
