import re, nbformat
src = open("review/verify_cells.py").read()
parts = re.split(r"^# %%(?: \[markdown\])?\n", src, flags=re.M)
heads = re.findall(r"^# %%( \[markdown\])?\n", src, flags=re.M)
nb = nbformat.v4.new_notebook()
for h, body in zip(heads, parts[1:]):
    if h: nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().split("\n"))))
    else: nb.cells.append(nbformat.v4.new_code_cell(body.strip()))
nb.metadata["kernelspec"] = {"name": "csen903", "display_name": "Python (csen903 f1)", "language": "python"}
nbformat.write(nb, "review/verify.ipynb"); print(len(nb.cells), "cells")
