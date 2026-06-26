"""Local (no-GPU) test of phase0/m25_gateway.py in MOCK mode — proves the OpenAI /v1 contract:
models list, non-stream chat, non-stream tool_calls, streaming content deltas, streaming tool_calls.

Run: python3 research/m25_gateway_test.py
"""
import json, os, socket, subprocess, sys, time, http.client

HERE = os.path.dirname(os.path.abspath(__file__))
GW = os.path.join(HERE, "..", "phase0", "m25_gateway.py")
PORT = 29677
P = F = 0
def ok(c, name):
    global P, F
    if c: P += 1; print(f"  PASS {name}")
    else: F += 1; print(f"  FAIL {name}")

def wait_port(port, t=10):
    end = time.time() + t
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.3): return True
        except OSError: time.sleep(0.1)
    return False

def req(method, path, body=None, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
    hdr = {"Content-Type": "application/json"}
    c.request(method, path, json.dumps(body) if body is not None else None, hdr)
    r = c.getresponse()
    if stream:
        data = r.read().decode()
        c.close(); return r.status, data
    data = r.read().decode()
    c.close()
    return r.status, json.loads(data) if data else None

TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "Search the web",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

env = dict(os.environ, M25_GATEWAY_MOCK="1")
proc = subprocess.Popen([sys.executable, GW, "--head", "x:1", "--tail", "x:1", "--port", str(PORT)],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try:
    if not wait_port(PORT):
        print("server did not start:\n", proc.stdout.read().decode()); sys.exit(1)

    print("== /v1/models ==")
    st, j = req("GET", "/v1/models")
    ok(st == 200 and j["data"][0]["id"] == "minimax-m2.5", "models list")

    print("== non-stream chat (no tools) ==")
    st, j = req("POST", "/v1/chat/completions",
                {"model": "minimax-m2.5", "messages": [{"role": "user", "content": "Hi there"}]})
    ch = j["choices"][0]
    ok(st == 200 and j["object"] == "chat.completion", "200 + object")
    ok(ch["message"]["role"] == "assistant" and ch["message"]["content"], "assistant content present")
    ok(ch["finish_reason"] == "stop" and "tool_calls" not in ch["message"], "finish stop, no tool_calls")
    ok(j["usage"]["total_tokens"] == j["usage"]["prompt_tokens"] + j["usage"]["completion_tokens"], "usage sums")
    ok("reasoning_content" in ch["message"], "reasoning_content surfaced")

    print("== non-stream chat WITH tools ==")
    st, j = req("POST", "/v1/chat/completions",
                {"model": "minimax-m2.5", "messages": [{"role": "user", "content": "search cats"}], "tools": TOOLS})
    ch = j["choices"][0]
    ok(ch["finish_reason"] == "tool_calls", "finish_reason tool_calls")
    tc = ch["message"]["tool_calls"][0]
    ok(tc["type"] == "function" and tc["function"]["name"] == "web_search", "tool_call function name")
    ok(isinstance(tc["function"]["arguments"], str) and "query" in json.loads(tc["function"]["arguments"]), "arguments is JSON string")

    print("== streaming (no tools) ==")
    st, data = req("POST", "/v1/chat/completions",
                   {"model": "minimax-m2.5", "messages": [{"role": "user", "content": "stream please"}], "stream": True}, stream=True)
    chunks = [json.loads(l[6:]) for l in data.splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    ok(data.rstrip().endswith("[DONE]"), "ends with [DONE]")
    ok(any(d.get("role") == "assistant" for d in deltas), "first role delta")
    ok(any("content" in d for d in deltas), "content deltas streamed")
    ok(any("reasoning_content" in d for d in deltas), "reasoning deltas streamed")
    ok(any(c["choices"][0].get("finish_reason") == "stop" for c in chunks if c.get("choices")), "finish stop chunk")
    ok(any(c.get("usage") for c in chunks), "usage chunk present")
    full = "".join(d.get("content", "") for d in deltas)
    ok("<minimax:tool_call>" not in full, "no tool-call XML leaked into content")

    print("== streaming WITH tools ==")
    st, data = req("POST", "/v1/chat/completions",
                   {"model": "minimax-m2.5", "messages": [{"role": "user", "content": "search dogs"}], "tools": TOOLS, "stream": True}, stream=True)
    chunks = [json.loads(l[6:]) for l in data.splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    tc_chunks = [c for c in chunks if c.get("choices") and "tool_calls" in c["choices"][0]["delta"]]
    ok(len(tc_chunks) == 1, "one tool_calls delta chunk")
    ok(tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "web_search", "streamed tool name")
    ok(any(c["choices"][0].get("finish_reason") == "tool_calls" for c in chunks if c.get("choices")), "finish tool_calls")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
