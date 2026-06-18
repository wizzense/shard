import transformers, os
f = os.path.join(os.path.dirname(transformers.__file__), "models/glm4/modeling_glm4.py")
lines = open(f).read().split("\n")
print("FILE:", f, "lines:", len(lines))
def show(a, b):
    for i in range(max(0, a), min(b, len(lines))):
        print(f"{i+1:4d}| {lines[i]}")
def find(sub):
    return [i for i, l in enumerate(lines) if sub in l]
ei = find("def eager_attention_forward")
print("=== eager_attention_forward ===")
if ei: show(ei[0], ei[0] + 24)
ai = find("class Glm4Attention")
print("=== Glm4Attention.forward ===")
if ai:
    fwd = [i for i in find("def forward") if i > ai[0]][0]
    show(fwd, fwd + 50)
mi = find("create_causal_mask") + find("_update_causal_mask")
print("=== mask creation sites ===")
for j in sorted(set(mi))[:6]: show(j - 1, j + 4)
