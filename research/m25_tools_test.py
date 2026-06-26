"""Local (no-GPU) tests for phase0/m25_tools.py against the REAL MiniMax-M2.5 tokenizer/template.

Run: M25_TOK_DIR=/path/to/tokenizer venv/bin/python research/m25_tools_test.py
(tokenizer dir needs tokenizer.json + tokenizer_config.json + chat_template.jinja + special_tokens_map.json)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from m25_tools import render_ids, parse_completion, to_openai_message, TOOLCALL_BEGIN, TOOLCALL_END

P = F = 0
def ok(cond, name):
    global P, F
    if cond: P += 1; print(f"  PASS {name}")
    else: F += 1; print(f"  FAIL {name}")

TOK_DIR = os.environ.get("M25_TOK_DIR")
tok = None
if TOK_DIR:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]

print("== parse_completion ==")

# 1. reasoning + single tool call, JSON-typed (days=3) and string (city) args
out1 = ('Let me look that up.\n</think>\n\nChecking the weather.'
        f'{TOOLCALL_BEGIN}\n<invoke name="get_weather">\n'
        '<parameter name="city">Paris</parameter>\n<parameter name="days">3</parameter>\n'
        f'</invoke>\n{TOOLCALL_END}')
r1 = parse_completion(out1)
ok(r1["reasoning_content"] == "Let me look that up.", "1.reasoning extracted")
ok(r1["content"] == "Checking the weather.", "1.content before call")
ok(len(r1["tool_calls"]) == 1 and r1["tool_calls"][0]["name"] == "get_weather", "1.one call, right name")
ok(r1["tool_calls"][0]["arguments"] == {"city": "Paris", "days": 3}, "1.args: string raw + int via json")

# 2. multiple invokes in one wrapper
out2 = (f'</think>\n{TOOLCALL_BEGIN}\n'
        '<invoke name="a">\n<parameter name="x">1</parameter>\n</invoke>\n'
        '<invoke name="b">\n<parameter name="y">hello</parameter>\n</invoke>\n'
        f'{TOOLCALL_END}')
r2 = parse_completion(out2)
ok([c["name"] for c in r2["tool_calls"]] == ["a", "b"], "2.two calls in order")
ok(r2["tool_calls"][0]["arguments"] == {"x": 1} and r2["tool_calls"][1]["arguments"] == {"y": "hello"}, "2.args coerced")

# 3. plain text answer, no tools, no think marker
r3 = parse_completion("The capital of France is Paris.")
ok(r3["tool_calls"] == [] and r3["content"] == "The capital of France is Paris.", "3.plain content")

# 4. reasoning then plain content, no tool call
r4 = parse_completion("thinking hard\n</think>\n\nFinal answer: 42.")
ok(r4["reasoning_content"] == "thinking hard" and r4["content"] == "Final answer: 42.", "4.reasoning+content no call")

# 5. truncated: begin token but NO end token (hit max_new) — invoke bounded by </invoke>
out5 = f'</think>\n{TOOLCALL_BEGIN}\n<invoke name="get_weather">\n<parameter name="city">Berlin</parameter>\n</invoke>'
r5 = parse_completion(out5)
ok(len(r5["tool_calls"]) == 1 and r5["tool_calls"][0]["arguments"] == {"city": "Berlin"}, "5.missing end token tolerated")

# 6. JSON object-valued argument
out6 = f'</think>\n{TOOLCALL_BEGIN}\n<invoke name="f">\n<parameter name="cfg">{{"a": 1, "b": [2, 3]}}</parameter>\n</invoke>\n{TOOLCALL_END}'
r6 = parse_completion(out6)
ok(r6["tool_calls"][0]["arguments"] == {"cfg": {"a": 1, "b": [2, 3]}}, "6.object arg via json")

# 7. leading <think> present (no forced-open assumption)
r7 = parse_completion("<think>\nweigh options\n</think>\n\ndone")
ok(r7["reasoning_content"] == "weigh options" and r7["content"] == "done", "7.explicit <think> stripped")

print("== to_openai_message ==")
msg, fin = to_openai_message(r1)
ok(fin == "tool_calls", "oai.finish_reason tool_calls")
ok(msg["tool_calls"][0]["function"]["name"] == "get_weather", "oai.function name")
ok(json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris", "days": 3}, "oai.arguments is json string")
msg3, fin3 = to_openai_message(r3)
ok(fin3 == "stop" and msg3["content"] == "The capital of France is Paris." and "tool_calls" not in msg3, "oai.plain -> stop")

if tok is not None:
    print("== render_ids (REAL tokenizer) ==")
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Weather in Paris?"}]
    s = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    ok("<tools>" in s and "get_weather" in s and TOOLCALL_BEGIN in s, "render.tools block injected")
    ids = render_ids(tok, msgs, tools=TOOLS)
    ok(isinstance(ids, list) and all(isinstance(i, int) for i in ids) and len(ids) > 10, "render.flat int ids")
    ok(render_ids(tok, msgs, tools=None) != ids, "render.no-tools differs from with-tools")
    # multi-turn re-render: assistant tool_call + tool result must template without error and chain
    convo = msgs + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "get_weather",
             "arguments": json.dumps({"city": "Paris", "days": 3})}}]},
        {"role": "tool", "content": "18C, clear"},
        {"role": "user", "content": "And tomorrow?"}]
    ids2 = render_ids(tok, convo, tools=TOOLS)
    ok(len(ids2) > len(ids), "render.multi-turn (asst tool_call + tool result) re-renders & grows")
else:
    print("(skipping real-tokenizer render tests; set M25_TOK_DIR to enable)")

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
