"""Characterize: (A) graph vs eager-ADDITIVE (same op -> should be ~bit-identical = graph is faithful),
(B) graph replay time vs eager-causal_lower_right (flash) -> the real speedup question."""
import os, torch, time
os.environ["M25_CUDA_GRAPH"] = "1"
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S
S.vllm_ctx(); dev="cuda"
layers=[S.Layer(i) for i in range(28,32)]; cos,sin=S.get_pe(); vcfg=S._CTX[1]
Kp1, P, start = 9, 2030, 2030+18
with torch.no_grad():
    gp=torch.Generator().manual_seed(0); S.run_block(layers,0,(torch.randn(1,P,S.H,generator=gp,dtype=torch.bfloat16)*0.3).to(dev),vcfg)
    gr=S.GraphRunner(layers, vcfg, Kp1)
    gx=torch.Generator().manual_seed(start); x=(torch.randn(1,Kp1,S.H,generator=gx,dtype=torch.bfloat16)*0.3).to(dev)
    o_graph = gr.run(start, x).clone()
    # eager-ADDITIVE: set _GR to the same state, run_block WITHOUT graph
    alen=gr._bucket(start+Kp1); st=S._GraphState(Kp1,alen,cos.shape[-1],dev); st.set(start,cos,sin)
    S._GR=st; o_eadd = S.run_block(layers, start, x, vcfg).clone(); S._GR=None
    o_ecausal = S.run_block(layers, start, x, vcfg).clone()   # eager causal_lower_right (flash)
print(f"(A) graph vs eager-ADDITIVE   max|diff|={(o_graph.float()-o_eadd.float()).abs().max().item():.3e}  (faithful if ~0)")
print(f"    graph vs eager-CAUSAL     max|diff|={(o_graph.float()-o_ecausal.float()).abs().max().item():.3e}  (backend bf16 drift)")
def bench(fn,n=60):
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
with torch.no_grad():
    te=bench(lambda: S.run_block(layers,start,x,vcfg))          # eager causal (flash)
tg=bench(lambda: gr.run(start,x))                               # graph (additive)
print(f"(B) eager-causal(flash) {te:.2f}ms   graph(additive) {tg:.2f}ms   -> {te/tg:.2f}x")
