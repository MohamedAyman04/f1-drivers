"""Entity extraction for the F1 Graph-RAG assistant (Milestone 3, section 1.b).

Gazetteers are built once from the Milestone-1 CSVs in data/raw (the same rows the
Milestone-2 graph was loaded from), so ids returned here are the graph's driverId /
constructorId / circuitId. Matching is greedy longest-span over normalised n-grams,
so "Red Bull Ring" beats "Red Bull" and "Michael Schumacher" beats "Schumacher".

    ex = extract("How many wins did Hamilton have at Silverstone before 2020?")
    ex.entities  -> {'drivers': [{'id': 1, 'name': 'Lewis Hamilton'}], 'circuits': [...],
                     'season_range': [1950, 2019], ...}
    ex.ambiguous -> []          (non-empty => ask the user, see clarification())
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Surname-only match picks one driver without asking only if that driver has at
# least this many times the race entries of every other driver with the surname
# (Lewis 356 vs Duncan Hamilton 5). Max vs Jos Verstappen, Michael vs Ralf
# Schumacher stay ambiguous. The choice is reported in Extraction.assumed.
# ponytail: entry-count prior; a per-session "you meant X" memory would be nicer.
DOMINANCE_RATIO = 5

# Words that appear in questions and must never be read as a name.
STOP = set("""
a about after ahead all an and any are as at average be before best between by
can car championship compare constructor did do does driver drivers during each
fastest finish for from gp grand grid had has have he her him his how in is it
its last least many more most of on or pit points pole podium prix race races
season seasons she since stop stops team teams than that the their them they
this to top was were what when where which who will win wins with won year
ball bell brown cross craft drake duke fair force gene george green grim hall
hart hunt james jean lewis life love lovely march martin may miles pace palm
parker re rose ruby said sharp simon south speed spirit stuck token turner
walker ward wolf ram era shadow duncan
""".split())

ORDINAL = r"(\d{1,2})(?:st|nd|rd|th)?"


def norm(text: str) -> str:
    """Lowercase, strip accents, punctuation to spaces, 'gp' -> 'grand prix'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    text = re.sub(r"\bgp\b|\bgrandprix\b", "grand prix", text)
    return " ".join(text.split())


def _rows(name: str):
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


@dataclass
class Gazetteer:
    alias: dict[str, set[tuple[str, int]]]      # normalised alias -> {(type, id)}
    names: dict[tuple[str, int], str]           # (type, id) -> display name
    years: dict[tuple[str, int], set[int]]      # (type, id) -> seasons active
    entries: dict[int, int]                     # driverId -> race entries
    teams: dict[int, set[int]]                  # driverId -> constructorIds raced for
    gp_names: dict[int, str]                    # fake id -> Grand Prix name
    year_min: int
    year_max: int


# Compact / sponsor aliases the CSV does not contain.
CONSTRUCTOR_EXTRA = {
    "mercedes amg": "mercedes", "merc": "mercedes", "amg": "mercedes",
    "scuderia ferrari": "ferrari", "red bull racing": "red_bull", "rbr": "red_bull",
    "kick sauber": "sauber", "stake": "sauber", "vcarb": "rb", "visa cash app rb": "rb",
    "racing bulls": "rb", "alfa": "alfa", "alpine": "alpine", "haas": "haas",
}
GP_EXTRA = {"us grand prix": "United States Grand Prix", "usa grand prix": "United States Grand Prix",
            "usgp": "United States Grand Prix", "brazil grand prix": "Brazilian Grand Prix",
            "mexico grand prix": "Mexican Grand Prix"}
CIRCUIT_GENERIC = {"circuit", "international", "grand", "prix", "autodromo", "de", "del", "di",
                   "do", "da", "street", "park", "the", "of", "e", "y", "nazionale", "internazionale"}


@lru_cache(maxsize=1)
def gazetteer() -> Gazetteer:
    alias: dict[str, set] = defaultdict(set)
    names, years, entries, teams = {}, defaultdict(set), defaultdict(int), defaultdict(set)

    def add(a: str, key: tuple[str, int]) -> None:
        a = norm(a)
        if a and a not in STOP:
            alias[a].add(key)

    race_year = {}
    gp_ids: dict[str, int] = {}
    for r in _rows("races.csv"):
        rid, y = int(r["raceId"]), int(r["year"])
        race_year[rid] = y
        gid = gp_ids.setdefault(r["name"], len(gp_ids))
        years[("grand_prix", gid)].add(y)
        years[("circuit", int(r["circuitId"]))].add(y)
    gp_names = {gid: n for n, gid in gp_ids.items()}

    for r in _rows("results.csv"):
        y = race_year[int(r["raceId"])]
        did, cid = int(r["driverId"]), int(r["constructorId"])
        years[("driver", did)].add(y)
        years[("constructor", cid)].add(y)
        entries[did] += 1
        teams[did].add(cid)

    constructor_aliases = set()
    for r in _rows("constructors.csv"):
        key = ("constructor", int(r["constructorId"]))
        names[key] = r["name"].replace("&amp;", "&")
        base = re.sub(r"\b(f1 team|f1|team)\b", "", norm(names[key])).strip()
        for a in {names[key], r["constructorRef"], base, base.replace(" ", "")}:
            add(a, key)
            constructor_aliases.add(norm(a))
    ref_to_id = {r["constructorRef"]: int(r["constructorId"]) for r in _rows("constructors.csv")}
    for a, ref in CONSTRUCTOR_EXTRA.items():
        add(a, ("constructor", ref_to_id[ref]))
        constructor_aliases.add(a)

    for r in _rows("drivers.csv"):
        key = ("driver", int(r["driverId"]))
        names[key] = f'{r["forename"]} {r["surname"]}'
        add(names[key], key)
        # Bare surname that is also a team name ("McLaren", "Williams") means the team.
        for a in (r["driverRef"], r["surname"]):
            if norm(a) not in constructor_aliases:
                add(a, key)

    for r in _rows("circuits.csv"):
        key = ("circuit", int(r["circuitId"]))
        names[key] = r["name"]
        short = " ".join(w for w in norm(r["name"]).split() if w not in CIRCUIT_GENERIC)
        # Country too ("Mexico 2024"); a country with several circuits is narrowed by season or asked.
        for a in (r["name"], r["circuitRef"], r["location"], short, r["country"]):
            add(a, key)

    for gid, n in gp_names.items():
        add(n, ("grand_prix", gid))
    for a, n in GP_EXTRA.items():
        add(a, ("grand_prix", gp_ids[n]))

    all_years = set(race_year.values())
    return Gazetteer(dict(alias), names, dict(years), dict(entries), dict(teams), gp_names,
                     min(all_years), max(all_years))


@dataclass
class Extraction:
    entities: dict
    ambiguous: list = field(default_factory=list)   # [{'mention','type','candidates':[{id,name}]}]
    assumed: list = field(default_factory=list)     # human-readable notes on tie-breaks
    spans: list = field(default_factory=list)       # [(start_tok, end_tok, type)] for masking


def _years(text: str, g: Gazetteer) -> tuple[int | None, list[int] | None]:
    """Return (season, season_range). Ranges are inclusive [from, to]."""
    t = text.lower()
    y = r"(1[89]\d\d|20\d\d)"
    m = re.search(rf"(?:between|from)\s+{y}\s+(?:and|to|until|-)\s+{y}", t) or \
        re.search(rf"\b{y}\s*(?:-|–|to)\s*{y}\b", t)
    if m:
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        return None, [a, b]
    m = re.search(rf"\b{y}s\b", t)
    if m:
        d = int(m.group(1))
        return None, [d, d + 9]
    m = re.search(rf"\b(before|prior to|until|up to|through|after|since)\s+(?:the\s+)?(?:end of\s+)?(?:the\s+)?{y}", t)
    if m:
        w, v = m.group(1), int(m.group(2))
        return None, {"before": [g.year_min, v - 1], "prior to": [g.year_min, v - 1],
                      "after": [v + 1, g.year_max], "since": [v, g.year_max]}.get(w, [g.year_min, v])
    found = [int(v) for v in re.findall(rf"\b{y}\b", t)]
    if len(found) == 1:
        return found[0], None
    if len(found) > 1:
        return None, [min(found), max(found)]
    return None, None


def _prediction_fields(text: str) -> dict:
    """Pre-race numbers a user may give for the podium model (grid, standings)."""
    t = text.lower()
    out = {}
    if re.search(r"\bfrom pole\b|\bon pole\b|\bpole position\b", t):
        out["grid"] = 1
    m = re.search(rf"(?:start(?:s|ing)?|grid|qualif(?:ied|ies|ying)|lin(?:es|ing) up)\s+(?:from\s+|in\s+|at\s+|position\s+|slot\s+)*p?{ORDINAL}\b", t) \
        or re.search(rf"\bp{ORDINAL}\s+on the grid\b", t) \
        or re.search(rf"\b{ORDINAL}\s+on the grid\b", t)
    if m:
        out["grid"] = int(m.group(1))
    m = re.search(rf"\bp?{ORDINAL}\s+in the (?:drivers'?\s+)?(?:championship|standings)\b", t)
    if m:
        out["champ_pos"] = int(m.group(1))
    elif re.search(r"championship leader|leading the championship|leads the championship", t):
        out["champ_pos"] = 1
    m = re.search(r"\b(\d{1,2})\s+(?:wins|victories|race wins)\b", t) or re.search(r"\bwon\s+(\d{1,2})\s+races\b", t)
    if m:
        out["wins"] = int(m.group(1))
    m = re.search(r"\b(\d{1,3}(?:\.\d)?)\s+(?:championship\s+)?points\b", t)
    if m:
        out["points"] = float(m.group(1))
    return out


def extract(text: str) -> Extraction:
    g = gazetteer()
    tokens = norm(text).split()
    season, season_range = _years(text, g)
    window = set(range(season_range[0], season_range[1] + 1)) if season_range else ({season} if season else None)

    found: dict[str, list] = defaultdict(list)
    ambiguous, assumed, spans = [], [], []
    i = 0
    while i < len(tokens):
        for n in range(min(6, len(tokens) - i), 0, -1):
            mention = " ".join(tokens[i:i + n])
            keys = g.alias.get(mention)
            if not keys:
                continue
            # Prefer the type that is not a driver when a mention is shared across types
            # only through a full alias ("monaco" circuit vs nothing); otherwise group by type.
            by_type = defaultdict(list)
            for k in keys:
                by_type[k[0]].append(k)
            typ = next(t for t in ("grand_prix", "driver", "constructor", "circuit") if t in by_type)
            cands = by_type[typ]
            if len(cands) > 1 and window:                      # narrow by season
                active = [k for k in cands if g.years.get(k, set()) & window]
                cands = active or cands
            if len(cands) > 1 and typ == "driver":
                cands.sort(key=lambda k: -g.entries.get(k[1], 0))
                top, rest = g.entries.get(cands[0][1], 0), g.entries.get(cands[1][1], 0)
                if top >= DOMINANCE_RATIO * max(rest, 1):
                    assumed.append(f'"{mention}" read as {g.names[cands[0]]} '
                                   f"({top} race entries vs {rest} for the next driver with that name)")
                    cands = cands[:1]
            if len(cands) > 1:
                ambiguous.append({"mention": mention, "type": typ, "candidates": [
                    {"id": k[1], "name": g.names.get(k) or g.gp_names[k[1]]} for k in sorted(cands)]})
            else:
                k = cands[0]
                item = g.gp_names[k[1]] if typ == "grand_prix" else {"id": k[1], "name": g.names[k]}
                if item not in found[typ]:
                    found[typ].append(item)
            spans.append((i, i + n, typ))
            i += n
            break
        else:
            i += 1

    # A team named anywhere in the message narrows an ambiguous driver ("Verstappen at Red Bull" -> Max).
    team_ids = {c["id"] for c in found["constructor"]}
    for a in list(ambiguous):
        if a["type"] == "driver" and team_ids:
            fit = [c for c in a["candidates"] if g.teams.get(c["id"], set()) & team_ids]
            if len(fit) == 1:
                ambiguous.remove(a)
                found["driver"].append(fit[0])

    entities = {
        "drivers": found["driver"],
        "constructors": found["constructor"],
        "circuits": found["circuit"],
        "grand_prix": found["grand_prix"][0] if found["grand_prix"] else None,
        "season": season,
        "season_range": season_range,
        **_prediction_fields(text),
    }
    return Extraction(entities, ambiguous, assumed, spans)


def mask(text: str) -> str:
    """Replace entity mentions and years with type words, so a classifier sees the
    question shape ("how many wins did driver have at circuit in year") and not the names."""
    ex = extract(text)
    tokens = norm(text).split()
    words = {"driver": "driver", "constructor": "team", "circuit": "circuit", "grand_prix": "grand prix"}
    out, i = [], 0
    for s, e, typ in sorted(ex.spans):
        out += tokens[i:s] + [words[typ]]
        i = e
    out += tokens[i:]
    return re.sub(r"\b(1[89]\d\d|20\d\d)s?\b", "year", " ".join(out))


# ---- required slots per intent (what must be present before running a template) ----

def _race(e):
    return bool(e.get("grand_prix") and e.get("season"))


def _time(e):
    return bool(e.get("season") or e.get("season_range"))


REQUIRED = {
    # intent: (check, question to ask when it fails)
    "driver_performance": (lambda e: e["drivers"], "Which driver do you mean?"),
    "constructor_performance": (lambda e: e["constructors"], "Which team (constructor) do you mean?"),
    "head_to_head": (lambda e: len(e["drivers"]) >= 2, "Which two drivers should I compare?"),
    "race_facts": (lambda e: _race(e) or e["circuits"],
                   "Which race (Grand Prix and year) or which circuit do you mean?"),
    "qualifying_grid": (lambda e: e["drivers"] or e["constructors"] or e["circuits"] or _race(e) or _time(e),
                        "Which driver, race, circuit or season do you mean?"),
    "pit_stops": (lambda e: e["drivers"] or _race(e) or _time(e), "Which race (Grand Prix and year), driver or season?"),
    "season_history": (_time, "Which season (or range of seasons)?"),
    "driver_profile": (lambda e: e["drivers"], "Which driver do you mean?"),
    "podium_prediction": (lambda e: e["drivers"], "Which driver should I predict?"),
}


def clarification(intent: str | None, ex: Extraction, entities: dict | None = None) -> str | None:
    """Question to ask the user, or None if the template can run.
    Pass `entities` = merge(cache, ex.entities) so a follow-up's slots count as present."""
    if intent in (None, "out_of_scope"):
        return None
    if ex.ambiguous:
        a = ex.ambiguous[0]
        opts = ", ".join(c["name"] for c in a["candidates"][:8])
        more = f" (and {len(a['candidates']) - 8} more)" if len(a["candidates"]) > 8 else ""
        return f'"{a["mention"].title()}" matches more than one {a["type"].replace("_", " ")}: {opts}{more}. Which one do you mean?'
    e = entities or ex.entities
    if intent == "race_facts" and e.get("grand_prix") and not e.get("season") and not e["circuits"]:
        return f"Which year's {e['grand_prix']}?"
    check = REQUIRED.get(intent)
    if check and not check[0](e):
        return check[1]
    return None


def resolve_choice(reply: str, candidates: list[dict]) -> dict | None:
    """Pick one candidate from a clarification reply ("Michael", "the 2nd", "Mick Schumacher")."""
    r = norm(reply)
    hits = [c for c in candidates if norm(c["name"]) in r or any(w in r.split() for w in norm(c["name"]).split()[:1])]
    if len(hits) == 1:
        return hits[0]
    m = re.search(r"\b(\d)(?:st|nd|rd|th)?\b|\b(first|second|third)\b", r)
    if m:
        idx = int(m.group(1)) - 1 if m.group(1) else ["first", "second", "third"].index(m.group(2))
        return candidates[idx] if 0 <= idx < len(candidates) else None
    return None


def merge(cache: dict | None, new: dict) -> dict:
    """Follow-up merge: keep cached slots, replace only the slots the new message names.
    A season replaces a range and vice versa ("And in 2020?" after "2014-2021")."""
    out = dict(cache or {})
    for k, v in new.items():
        if v in (None, [], {}):
            continue
        out[k] = v
        if k == "season":
            out["season_range"] = None
        elif k == "season_range":
            out["season"] = None
    for k in ("drivers", "constructors", "circuits"):
        out.setdefault(k, [])
    out.setdefault("grand_prix", None)
    out.setdefault("season", None)
    out.setdefault("season_range", None)
    return out


if __name__ == "__main__":
    def ids(ex, k):
        return [d["id"] for d in ex.entities[k]]

    ex = extract("How many wins did Hamilton have at Silverstone before 2020?")
    assert ids(ex, "drivers") == [1] and ids(ex, "circuits") == [9], ex
    assert ex.entities["season_range"] == [1950, 2019] and ex.assumed
    ex = extract("How many points did Schumacher score?")
    assert ex.ambiguous and len(ex.ambiguous[0]["candidates"]) == 3
    assert "Schumacher" in clarification("driver_performance", ex)
    ex = extract("Schumacher wins in 2021")                # only Mick raced in 2021
    assert [d["name"] for d in ex.entities["drivers"]] == ["Mick Schumacher"], ex
    ex = extract("Verstappen podiums in 2022")
    assert ex.entities["drivers"][0]["name"] == "Max Verstappen"
    assert extract("Verstappen career podiums").ambiguous      # Jos vs Max, no season
    ex = extract("How reliable was redbull between 2014 and 2021?")
    assert ids(ex, "constructors") == [9] and ex.entities["season_range"] == [2014, 2021]
    assert ids(extract("mercedes amg points 2020"), "constructors") == [131]
    ex = extract("Who won the 2019 German GP?")
    assert ex.entities["grand_prix"] == "German Grand Prix" and ex.entities["season"] == 2019
    assert clarification("race_facts", extract("Who won the German GP?")) == "Which year's German Grand Prix?"
    ex = extract("Races held at the Red Bull Ring")
    assert ids(ex, "circuits") == [70] and not ex.entities["constructors"]
    assert ids(extract("Who finished ahead more often, Norris or Ricciardo at McLaren?"), "constructors") == [1]
    assert len(extract("Norris vs Ricciardo at McLaren").entities["drivers"]) == 2
    assert ids(extract("perez podiums 2021"), "drivers") == [815]
    assert ids(extract("pit stops in Monte Carlo 2019"), "circuits") == [6]
    assert extract("races in barcelona").ambiguous
    assert [d["name"] for d in extract("How did Albon fare against Verstappen at Red Bull?").entities["drivers"]]         == ["Alexander Albon", "Max Verstappen"]
    assert ids(extract("Sainz top three in Mexico 2024"), "circuits") == [32]
    assert ids(extract("Sainz at Australia 2024"), "circuits") == [1]
    assert extract("races in Italy in 2020").ambiguous      # Monza, Imola, Mugello
    assert extract("Monaco GP winners in the 2010s").entities["season_range"] == [2010, 2019]
    p = extract("Predict Norris at the 2024 Miami GP starting P3, 2nd in the championship with 3 wins and 150 points")
    assert (p.entities["grid"], p.entities["champ_pos"], p.entities["wins"], p.entities["points"]) == (3, 2, 3, 150.0), p
    assert extract("Leclerc from pole at Monza 2024").entities["grid"] == 1
    assert mask("How many wins did Hamilton have at Silverstone in 2019?") == \
        "how many wins did driver have at circuit in year"
    c = {"drivers": [{"id": 1, "name": "Lewis Hamilton"}], "season": 2019}
    assert merge(c, extract("And in 2020?").entities)["drivers"][0]["id"] == 1
    f = extract("And in 2020?")
    assert clarification("driver_performance", f, merge(c, f.entities)) is None
    assert clarification("out_of_scope", extract("capital of France")) is None
    m = merge(c, extract("What about Norris?").entities)
    assert m["season"] == 2019 and m["drivers"][0]["name"] == "Lando Norris"
    cands = extract("Schumacher").ambiguous[0]["candidates"]
    assert resolve_choice("Michael", cands)["name"] == "Michael Schumacher"
    assert resolve_choice("the second one", cands) == cands[1]
    print("entities.py: all checks passed")
