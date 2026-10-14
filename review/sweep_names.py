"""Sweep step for OUT_OF_SCOPE.md: list every name the notebook code uses and whether the lab material
(lab PDFs + guide notebooks) contains it. Run: .venv/bin/python review/sweep_names.py"""
import ast, json, re, glob, collections, sys
SCR="review/lab_text/"   # plain text extracted from labs/lab_manual1-4.pdf
lab_text="\n".join(open(f).read() for f in sorted(glob.glob(SCR+"lab_manual*.clean")))
for f in sorted(glob.glob("guide_notebooks/*.ipynb")):
    nb=json.load(open(f))
    lab_text+="\n"+"\n".join("".join(c["source"]) for c in nb["cells"])
lab_words=set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*",lab_text))
nb=json.load(open("notebooks/f1_podium_prediction.ipynb"))
uses=collections.defaultdict(lambda: collections.defaultdict(set))   # kind -> name -> cells
def add(kind,name,i): uses[kind][name].add(i)
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    src="\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith(("%","!")))
    try: tree=ast.parse(src)
    except SyntaxError as e: print("parse fail",i,e); continue
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            for a in n.names: add("import",a.name,i)
        elif isinstance(n,ast.ImportFrom):
            for a in n.names: add("import",f"{n.module}.{a.name}",i)
        elif isinstance(n,ast.Attribute): add("attr",n.attr,i)
        elif isinstance(n,ast.keyword) and n.arg: add("kwarg",n.arg,i)
        elif isinstance(n,ast.Call) and isinstance(n.func,ast.Name): add("func",n.func.id,i)
out={}
for kind in uses:
    for name,cells in uses[kind].items():
        last=name.split(".")[-1]
        out[(kind,name)]=(last in lab_words, sorted(cells))
json.dump([{"kind":k,"name":n,"in_labs":v[0],"cells":v[1]} for (k,n),v in sorted(out.items())],open("review/sweep_names.json","w"),indent=1)
miss=[(k,n,v[1]) for (k,n),v in sorted(out.items()) if not v[0]]
print(len(out),"names;",len(miss),"not found in lab material")
for k,n,c in miss: print(f"{k:7} {n:42} {c[:8]}")
