"""Builds the board's HTML from the template, oos_items.json and findings_data.py. Usage: python build_board.py OUT.html"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from findings_data import FINDINGS
import cellmap
TEMPLATE = Path(sys.argv[2]); out = Path(sys.argv[1])
cells = cellmap.load(str(Path(__file__).parent.parent / "notebooks" / "f1_podium_prediction.ipynb"))
items = json.load(open(Path(__file__).parent / "oos_items.json"))
O = [dict(id=i["id"], grp=i["sec"], item=i["item"], kind=i["kind"], cells=i["where"], req=i["req"], what=i["what"],
          why=i["why"], alt=i["alt"], pri=i["pri"]) for i in items]
F = []
for f in FINDINGS:
    d = {k: (cellmap.resolve(v, cells) if isinstance(v, str) else v) for k, v in f.items()}
    d["sev"] = f["sev"]; F.append(d)
t = TEMPLATE.read_text()
t = t.replace("/*OOS_JSON*/[]", json.dumps(O, ensure_ascii=False)).replace("/*FND_JSON*/[]", json.dumps(F, ensure_ascii=False))
out.write_text(t); print("board built:", len(O), "items,", len(F), "findings,", len(t), "bytes")
