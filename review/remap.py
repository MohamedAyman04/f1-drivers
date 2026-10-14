"""Maps cell numbers of the notebook at commit 1925b7f (110 cells) to the fixed notebook, by aligning cell contents."""
import json, difflib, sys
OLD = sys.argv[1]; NEW = "../notebooks/f1_podium_prediction.ipynb"
old = json.load(open(OLD))["cells"]; new = json.load(open(NEW))["cells"]
src = lambda c: "".join(c["source"])
key = lambda c: (c["cell_type"], next((l.strip() for l in src(c).split("\n") if l.strip()), "")[:30])
sm = difflib.SequenceMatcher(None, [key(c) for c in old], [key(c) for c in new], autojunk=False)
m = {}
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        for k in range(i2 - i1): m[i1 + k] = j1 + k
    elif tag == "replace":                      # edited cells: pair by best content similarity inside the block
        free = list(range(j1, j2))
        for i in range(i1, i2):
            if not free: break
            best = max(free, key=lambda j: difflib.SequenceMatcher(None, src(old[i])[:400], src(new[j])[:400]).ratio())
            if difflib.SequenceMatcher(None, src(old[i])[:400], src(new[best])[:400]).ratio() > 0.5:
                m[i] = best; free.remove(best)
missing = [i for i in range(len(old)) if i not in m]
print("mapped", len(m), "of", len(old), "| unmapped old cells:", missing)
json.dump(m, open("cell_remap.json", "w"))
