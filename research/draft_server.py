"""in-house draft service: vLLM-served gpt-oss-20b behind a tiny socket protocol.

the c0mpute entry node runs this; the spec-decode driver queries it for K greedy
draft tokens from the committed prefix. vLLM prefix-caching makes each round's
re-proposal cheap (the growing committed prefix stays cached) -- that's the
spec-decode rollback handled for free, no manual draft KV management. the draft
holds NO authority (the distributed target verifies every token), so centralizing
it as a managed service is safe and is the right call for the c0mpute network.

runs in the vLLM venv on the entry node's draft GPU:
  CUDA_VISIBLE_DEVICES=0 /root/vllmenv/bin/python draft_server.py --port 8200

wire protocol matches node_kv.send_msg/recv_msg (8-byte !Q length prefix + pickle):
  request  {"ids": [committed token ids...], "k": K}
  response [K greedy draft token ids]
"""
import argparse, socket, struct, pickle
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _recvall(c, n):
    b = b""
    while len(b) < n:
        x = c.recv(n - len(b))
        if not x:
            raise ConnectionError("peer closed")
        b += x
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gpt-oss-20b")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac, max_model_len=args.max_len,
              enforce_eager=False, enable_prefix_caching=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port)); srv.listen(4)
    print(f"[draft] vLLM gpt-oss-20b ready, listening on :{args.port}", flush=True)
    def build_tree(ids, width, depth):
        """tree rooted at cur (=ids[-1]): top-`width` next tokens, each greedily
        continued to `depth`. returns flattened {tok,par,dep} (node 0 = root)."""
        sp0 = SamplingParams(temperature=0, max_tokens=1, logprobs=width)
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp0, use_tqdm=False)
        lp = o[0].outputs[0].logprobs[0]                       # {tok_id: Logprob} at position 1
        kids = [t for t, _ in sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)[:width]]
        tok, par, dep = [ids[-1]], [-1], [0]                   # node 0 = root = cur
        for ct in kids:
            spc = SamplingParams(temperature=0, max_tokens=max(depth - 1, 0), ignore_eos=True)
            cont = llm.generate([TokensPrompt(prompt_token_ids=ids + [ct])], spc, use_tqdm=False)
            chain = [ct] + list(cont[0].outputs[0].token_ids)
            parent = 0
            for d, t in enumerate(chain):
                tok.append(t); par.append(parent); dep.append(d + 1); parent = len(tok) - 1
        return {"tok": tok, "par": par, "dep": dep}

    while True:
        conn, _ = srv.accept()
        try:
            while True:
                (n,) = struct.unpack("!Q", _recvall(conn, 8))
                req = pickle.loads(_recvall(conn, n))
                if "tree" in req:                             # tree request
                    resp = build_tree(req["ids"], req["tree"]["width"], req["tree"]["depth"])
                else:                                         # linear request
                    sp = SamplingParams(temperature=0, max_tokens=req["k"], min_tokens=req["k"], ignore_eos=True)
                    out = llm.generate([TokensPrompt(prompt_token_ids=req["ids"])], sp, use_tqdm=False)
                    resp = list(out[0].outputs[0].token_ids)
                data = pickle.dumps(resp)
                conn.sendall(struct.pack("!Q", len(data)) + data)
        except (ConnectionError, EOFError, OSError):
            try: conn.close()
            except OSError: pass


if __name__ == "__main__":
    main()
