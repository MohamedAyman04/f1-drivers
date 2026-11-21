"""Intent classification for the F1 Graph-RAG assistant (Milestone 3, section 1.a).

Two classifiers over the same 10 labels:
  * RuleClassifier       weighted keyword / paraphrase regexes + entity-count cues
  * EmbeddingClassifier  nearest example by cosine similarity (sentence-transformers),
                         with a minimum score and a margin; below either -> ask to rephrase

Both share:
  * follow-up handling: a message that names no new intent (only entities / years /
    filler, e.g. "And in 2020?", "What about Norris?") keeps the cached intent;
  * the out-of-scope gate: OOS is decided by rules (no F1 signal at all, or an explicit
    unanswerable topic such as salary / tyres / weather), never by similarity alone.

    clf = EmbeddingClassifier()
    r = clf.classify("Who won the 2019 German GP?")      -> Intent(label='race_facts', status='ok', ...)
    r = clf.classify("And in 2018?", prev="race_facts")  -> Intent(label='race_facts', status='follow_up')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from .entities import extract, mask, norm
except ImportError:                       # run as a script: python m3/intents.py
    from entities import extract, mask, norm

# ---------------------------------------------------------------- label set
# Seven graph question types from the brief + driver_profile (the extra type) +
# podium_prediction + out_of_scope = 10 intents. The examples below build the
# embedding index; the evaluation set (eval/intent_testset.csv) is phrased differently.
INTENTS: dict[str, dict] = {
    "driver_performance": {
        "description": "A driver's own results: points, wins, podiums, average finish, best result, "
                       "in a season, a range of seasons, at a circuit, or over a career.",
        "examples": [
            "How many podiums did Leclerc get in 2022?",
            "How many points did Hamilton score in 2019?",
            "How many races did Alonso win in his career?",
            "What is Verstappen's average finishing position at Zandvoort?",
            "How many wins did Hamilton have at Silverstone before 2020?",
            "What was Norris's best result in 2021?",
            "How many times did Vettel finish on the podium between 2010 and 2013?",
            "Total career points for Kimi Raikkonen",
            "How often did Bottas finish in the points in 2018?",
            "Senna victories at Monaco",
            "Give me Piastri's podium count for 2024",
            "How did Webber do in 2010?",
            "What did he score in 2018?",
            "Leclerc win tally in 2019",
        ],
    },
    "constructor_performance": {
        "description": "A team's results: season points, wins, podiums, reliability, retirements "
                       "or finishing rate over a period.",
        "examples": [
            "How reliable was Williams between 2014 and 2021?",
            "How many points did Ferrari score in 2022?",
            "How many races did Mercedes win in 2016?",
            "What was McLaren's finishing rate from 2015 to 2017?",
            "How many retirements did Red Bull have in 2022?",
            "How many podiums did Aston Martin get in 2023?",
            "Team points for Alpine in 2021",
            "How many mechanical failures did Haas suffer in 2019?",
            "How many victories has Ferrari taken at Monza?",
            "Which constructor finished the most races in 2020?",
            "How often did Toro Rosso cars break down in 2015?",
            "Did Ferrari win more than McLaren in 2007?",
        ],
    },
    "head_to_head": {
        "description": "Compare two drivers in the races they both started, usually as teammates: "
                       "who finished or qualified ahead more often.",
        "examples": [
            "Who finished ahead more often, Norris or Ricciardo at McLaren?",
            "Hamilton vs Bottas at Mercedes",
            "Compare Leclerc and Sainz at Ferrari",
            "Did Verstappen beat Perez more often in 2022?",
            "Head to head between Vettel and Webber",
            "Who came out on top between Rosberg and Hamilton as teammates?",
            "How many times did Alonso finish in front of Massa?",
            "Russell versus Hamilton in shared races",
            "Who had the upper hand, Gasly or Ocon?",
            "Button against Perez, who did better?",
            "Who was faster on race day, Massa or Raikkonen?",
        ],
    },
    "race_facts": {
        "description": "Facts about one race or circuit: winner and podium of a Grand Prix, races held "
                       "at a circuit, where a circuit is located.",
        "examples": [
            "Who won the 2019 German Grand Prix?",
            "Who was on the podium at the 2021 Abu Dhabi GP?",
            "How many races have been held at Monza?",
            "Where is the Suzuka circuit located?",
            "Which country is Interlagos in?",
            "Top three finishers of the 2008 Brazilian Grand Prix",
            "What was the result of the 2020 Turkish GP?",
            "When was the first race at Silverstone?",
            "Which Grand Prix races took place at the Hungaroring?",
            "Who came second in the 2016 Spanish Grand Prix?",
        ],
    },
    "qualifying_grid": {
        "description": "Qualifying and starting grid: pole positions, Q3 appearances, front rows, grid "
                       "position versus finish, places gained or lost.",
        "examples": [
            "Which driver gained the most places from the grid at Monza in 2021?",
            "How many pole positions did Senna take?",
            "Who was on pole for the 2023 Singapore Grand Prix?",
            "How many times did Russell reach Q3 in 2022?",
            "How many places did Perez make up in the 2022 Saudi Arabian GP?",
            "Who started on the front row at the 2018 Monaco GP?",
            "What was Leclerc's qualifying position in Baku 2021?",
            "Which driver lost the most positions during the 2020 Austrian GP?",
            "How many poles did Ferrari get in 2022?",
            "Who started first at the 2019 Austrian GP?",
            "Who climbed the most positions from the start in Baku 2017?",
            "How often did Ocon reach the last qualifying session in 2021?",
        ],
    },
    "pit_stops": {
        "description": "Pit stops and strategy: fastest stop in a race, number of stops, average "
                       "stops per driver, stop durations.",
        "examples": [
            "What was the fastest pit stop at the 2023 Qatar Grand Prix?",
            "How many pit stops did Hamilton make in the 2021 Abu Dhabi GP?",
            "Average number of stops per driver in the 2022 season",
            "Which driver spent the least time in the pits at Monza 2020?",
            "How long was Verstappen's longest pit stop in 2021?",
            "On which lap did Leclerc pit in the 2022 Monaco Grand Prix?",
            "Who made the most pit stops in the 2019 German GP?",
            "Was it a one-stop or two-stop race for Norris at Silverstone 2023?",
            "Average number of stops at Monza in 2019",
        ],
    },
    "season_history": {
        "description": "Whole-season or multi-season facts: championship winner or points leader, "
                       "how many drivers or races in a season, drivers who changed teams.",
        "examples": [
            "Which drivers changed teams between 2020 and 2021?",
            "Who was the world champion in 2016?",
            "Who led the championship points in 2021?",
            "How many drivers competed in the 2019 season?",
            "How many races were on the 2023 calendar?",
            "Who scored the most points across 2014 to 2020?",
            "Which rookies made their debut in 2019?",
            "Final drivers' standings for 2012",
            "Who switched constructors before the 2022 season?",
            "Who claimed the title in 2005?",
            "How many rounds were there in 2010?",
            "Which drivers joined a different team for 2018?",
            "Who was top of the table in 2017?",
        ],
    },
    "driver_profile": {
        "description": "Who a driver is: nationality, date of birth, racing number, driver code, the "
                       "teams they drove for, their debut and the span of their career.",
        "examples": [
            "What nationality is Valtteri Bottas?",
            "When was Charles Leclerc born?",
            "Which teams has Fernando Alonso driven for?",
            "When did Lando Norris make his debut?",
            "What is Max Verstappen's car number?",
            "How old was Kimi Raikkonen at his first race?",
            "Tell me about Sebastian Vettel",
            "How many seasons did Jenson Button race in F1?",
            "Where is Daniel Ricciardo from?",
            "What is Hamilton's three-letter code?",
        ],
    },
    "podium_prediction": {
        "description": "Run the podium model: will a driver finish in the top three at a given race, "
                       "possibly with pre-race information such as grid slot or standings.",
        "examples": [
            "Predict Norris at the 2024 Miami Grand Prix",
            "Will Leclerc finish on the podium at Monza if he starts third?",
            "What are Verstappen's chances of a podium at the 2023 Japanese GP?",
            "Podium probability for Russell starting P5 in Las Vegas 2023",
            "Can Sainz make the top three from pole in Singapore 2023?",
            "How likely is Piastri to get a podium at the 2024 Hungarian GP?",
            "Predict whether Alonso finishes top 3 at Bahrain 2023 starting 5th, 3rd in the championship",
            "Run the podium model for Hamilton at Silverstone 2024",
        ],
    },
    "out_of_scope": {
        "description": "Anything the Formula 1 knowledge graph cannot answer: non-F1 topics, and F1 "
                       "topics outside the data (salaries, tyres, weather, sprint races, lap times, "
                       "penalties, car technology, personal life).",
        "examples": [
            "What is Hamilton's salary at Ferrari?",
            "Which tyre compound did Verstappen use at Monza?",
            "Was it raining during the 2021 Belgian GP?",
            "Who won the sprint race in Brazil 2023?",
            "What is the lap record at Spa?",
            "How much horsepower does the Mercedes engine produce?",
            "Who is Charles Leclerc dating?",
            "Who won the football World Cup in 2022?",
            "Give me a recipe for lasagna",
            "What is the capital of Australia?",
            "Who won the 2023 MotoGP title?",
            "Why did the stewards penalise Perez?",
        ],
    },
}
LABELS = list(INTENTS)

# ---------------------------------------------------------------- shared helpers

# Explicit unanswerable topics: the graph has no salary, tyre, weather, sprint, lap-time,
# penalty or car-technology data, and nothing outside F1.
OOS_PATTERNS = [
    (r"\b(salary|salaries|paid|earns?|earnings|net worth|contract|sponsors?\w*|money|budget)\b", 4),
    (r"\b(weather|rain\w*|wet|temperature|sunny)\b", 4),
    (r"\b(tyres?|tires?|compounds?|softs?|mediums?|hards?)\b", 4),
    (r"\bsprints?\b", 4),
    (r"\b(lap times?|lap record|fastest lap|top speed|speed trap)\b", 4),
    (r"\b(horsepower|hp|downforce|car setup|wind tunnel)\b", 3),
    (r"\b(girlfriend|wife|married|dating|instagram|twitter|social media|favou?rite|hobby|hobbies|pets?)\b", 4),
    (r"\b(tickets?|tv|stream\w*|broadcast|watch)\b", 3),
    (r"\b(penalt\w*|penalis\w*|penaliz\w*|stewards?|safety car|red flag|yellow flag|crash\w*|collision)\b", 3),
    (r"\b(football|soccer|nba|nfl|tennis|cricket|golf|motogp|nascar|indycar|formula e|wrc|rally|world cup|olympics?|le mans 24|tour de france|cycling|boxing|ufc|baseball|basketball|hockey|rugby)\b", 5),
    (r"\b(recipe|cook\w*|capital of|stocks?|bitcoin|crypto|movie|film|song|python|homework|translate|joke|poem)\b", 5),
]
F1_WORDS = re.compile(r"\b(f1|formula (1|one)|grand prix|poles?|podiums?|pit\w*|constructors?|qualif\w*|grid|"
                      r"champion\w*|races?|racing|drivers?|teams?|teammates?|seasons?|circuits?|track|laps?|"
                      r"wins?|won|points|standings|retire\w*|dnfs?|q3)\b")
FOLLOW_FILLER = set("""and what about how in the for at then same with of now ok okay please him her he
she they them his their season year instead also too there that this one again it is was did do does
so then same but driver team circuit grand prix year""".split())


@dataclass
class Intent:
    label: str | None          # one of LABELS, or None when we ask the user to rephrase
    status: str                # ok | follow_up | clarify | out_of_scope
    scores: dict = field(default_factory=dict)
    reason: str = ""


def oos_keyword(text: str) -> bool:
    t = norm(text)
    return any(re.search(p, t) for p, _ in OOS_PATTERNS)


def has_f1_signal(text: str, ex=None) -> bool:
    ex = ex or extract(text)
    e = ex.entities
    return bool(e["drivers"] or e["constructors"] or e["circuits"] or e["grand_prix"] or ex.ambiguous
                or F1_WORDS.search(norm(text)))


def is_follow_up(text: str) -> bool:
    """True if, after masking names and years, nothing but filler is left ("And in 2020?")."""
    rest = [w for w in mask(text).split() if w not in FOLLOW_FILLER]
    return not rest


def _gate(text: str, prev: str | None) -> Intent | None:
    """Decisions both classifiers make the same way, before any scoring."""
    ex = extract(text)
    if prev and prev != "out_of_scope" and is_follow_up(text):
        return Intent(prev, "follow_up", reason="no new intent named; kept cached intent")
    if not has_f1_signal(text, ex):
        return Intent("out_of_scope", "out_of_scope", reason="no Formula 1 entity or term")
    if oos_keyword(text):
        return Intent("out_of_scope", "out_of_scope", reason="topic the graph has no data for")
    return None


# ---------------------------------------------------------------- rule-based

RULES: dict[str, list[tuple[str, float]]] = {
    "podium_prediction": [
        (r"\bpredict\w*", 4), (r"\b(chances?|probability|odds|likely|likelihood)\b", 3),
        # Future / hypothetical tense outweighs the fact keywords it co-occurs with ("podium at the ... GP").
        (r"\b(will|would|can|could)\b.*\b(podium|top (3|three))\b", 4), (r"\bpodium model\b", 4),
        (r"\bif (he|she|they) starts?\b", 2), (r"\bgoing to\b.*\b(podium|top (3|three)|win)\b", 4),
        (r"\b(forecast|estimate)\w*", 3), (r"\bthe model\b", 3),
    ],
    "head_to_head": [
        (r"\b(vs|versus)\b", 3), (r"\bcompar\w*", 2), (r"\bhead to head\b", 3),
        (r"\bteam ?mates?\b", 2), (r"\bahead (of|more often)\b|\bin front of\b", 2.5),
        (r"\bout ?(qualif|scor|perform|rac)\w*", 2.5), (r"\b(beat|beaten|upper hand|on top between)\b", 2),
        (r"\bwho (did|was) better\b", 2), (r"\bagainst\b", 2), (r"\bfare[ds]?\b", 1),
        (r"\bfinish(ed)? (higher|lower)\b", 2.5), (r"\bbetter record\b", 2.5),
        (r"\b(quicker|faster)\b", 1.5), (r"\bmore (often|times)\b", 1),
    ],
    "pit_stops": [
        (r"\bpit ?stops?\b|\bpit(ted|ting|s)?\b", 3.5), (r"\b(one|two|three|1|2|3) stop\w*", 2.5),
        (r"\bstops?\b", 1.5), (r"\b(stationary|boxed|box)\b", 2), (r"\bstrategy\b", 1.5),
    ],
    "qualifying_grid": [
        (r"\bpoles?\b|\bpole positions?\b", 3), (r"\bqualif\w*", 3), (r"\bgrid\b", 2), (r"\bq[123]\b", 3),
        (r"\bfront row\b", 3), (r"\bstart(ed|ing)? (on|from|position)\b|\bstarting position\b", 2),
        (r"\b(gain\w*|lost|lose|made up|make up|climb\w*|dropp?\w*)\b.*\b(places|positions|spots)\b", 3),
        (r"\bstart(ed|s)? (first|second|third|last|\d+(st|nd|rd|th)?)\b", 2.5),
        (r"\b(starting|grid) (slot|spot|place)\b", 2.5), (r"\bclimbers?\b", 2.5),
        (r"\b(places|positions) (gained|lost|made up)\b", 3),
    ],
    "race_facts": [
        (r"\bwho won the\b.*\bgrand prix\b|\bwinner of the\b", 3), (r"\bwho won\b", 1.5),
        (r"\b(top (3|three)|podium|result|results|classification) (finishers )?(of|at|in) the\b", 2.5),
        (r"\bwho (came|finished|was) (first|second|third|1st|2nd|3rd|on the podium)\b", 2.5),
        (r"\bwhere is\b|\blocat\w*|\bwhich (country|city)\b|\bwhat (country|city)\b", 3),
        (r"\b(held|hosted|staged|took place)\b", 2.5),
        (r"\b(took|take|takes) (the )?victory\b|\bvictory (at|in) the\b|\bwon at the\b", 2.5), (r"\b(first|last|latest) (race|grand prix) (at|in|held)\b", 2),
    ],
    "season_history": [
        (r"\b(world )?champions?\b|\btitles?\b", 2.5), (r"\bwon the (drivers )?(world )?championship\b", 3),
        (r"\b(led|leader|leading) (the )?(championship|standings|points)\b|\bpoints leader\b", 3),
        (r"\bstandings\b", 2), (r"\bhow many (drivers|races|rounds|teams)\b", 2.5),
        (r"\b(drivers|teams) (competed|took part|raced|entered)\b", 2),
        (r"\b(changed|switched|moved|swapped|change|switch)\b.*\b(teams?|constructors?)\b|\bteam changes?\b", 3),
        (r"\brookies?\b|\bdebutants?\b", 2.5), (r"\bcalendar\b", 2.5), (r"\bacross\b", 1),
        (r"\bmost points\b", 1.5), (r"\btopped\b|\btop of the\b|\b(points|championship) table\b", 2.5),
    ],
    "constructor_performance": [
        (r"\breliab\w*|\bfinishing rate\b|\bfinish rate\b|\bclassified\b", 3),
        (r"\b(dnfs?|retirements?|retired|mechanical|failures?|breakdowns?)\b", 2),
        (r"\b(team|constructor)s?\b.*\b(points|wins|podiums|record|results)\b|\bteam points\b", 2.5),
        (r"\bconstructors?\b", 1.5), (r"\b(points|wins?|won|victories|podiums?)\b", 1),
        (r"\bdependab\w*|\bretire\w*", 2.5), (r"\b(cars?|machines?)\b", 1),
    ],
    "driver_performance": [
        (r"\b(points|wins?|won|victor(y|ies)|podiums?|top (3|three) finishes)\b", 1.5),
        (r"\bon average\b", 2), (r"\bhow did\b.*\b(do|perform|get on)\b", 1.5), (r"\b(tally|haul|count|total)\b", 1),
        (r"\baverage (finish\w*|result|position|place)\b", 3), (r"\bbest (result|finish)\b", 2.5),
        (r"\bscor\w*\b", 1), (r"\bcareer\b", 1), (r"\bin the points\b|\bpoints finish\w*\b", 2),
        (r"\b(dnfs?|retirements?|retired)\b", 1),
    ],
    "driver_profile": [
        (r"\bborn\b|\bbirth\w*|\bhow old\b|\bage\b", 3.5), (r"\bnationality\b|\bwhere is \w+( \w+)? from\b"
         r"|\bwhat country is\b|\b(come|comes) from\b|\brepresent\w*|\bwhat country does\b", 3.5),
        (r"\bs first\b|\bfirst (in|ever) (f1|formula)", 2.5),
        (r"\b(driven|drove|raced|race|driving|drive) for\b|\bteams? (has|did)\b", 3),
        (r"\bdebut\w*\b|\bfirst (f1 |formula 1 )?(race|grand prix|start)\b", 2.5),
        (r"\b(car|racing|race) number\b|\bnumber\b|\bcode\b|\babbreviation\b", 2.5),
        (r"\btell me about\b|\bwho is\b|\bbiography\b|\bprofile\b", 3),
        (r"\bhow many seasons\b|\bcareer (span|length)\b|\byears active\b|\bstill racing\b", 2.5),
    ],
}


class RuleClassifier:
    name = "rule-based"

    def scores(self, text: str) -> dict[str, float]:
        t = norm(text)
        s = {lab: sum(w for p, w in pats if re.search(p, t)) for lab, pats in RULES.items()}
        s["out_of_scope"] = sum(w for p, w in OOS_PATTERNS if re.search(p, t))
        return s

    def classify(self, text: str, prev: str | None = None) -> Intent:
        g = _gate(text, prev)
        if g:
            return g
        s = self.scores(text)
        if not any(s.values()):
            if prev:
                return Intent(prev, "follow_up", s, "no intent keyword; kept cached intent")
            return Intent(None, "clarify", s, "no intent keyword matched")
        # Entity-count cues break ties between families that share words.
        e = extract(text).entities
        if len(e["drivers"]) >= 2:
            s["head_to_head"] += 2
        if e["constructors"] and not e["drivers"]:
            s["constructor_performance"] += 2
        if e["drivers"]:
            s["driver_performance"] += 1
            s["driver_profile"] += 0.5
        if e["grand_prix"]:
            s["race_facts"] += 1
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        (best, b), (_, b2) = ranked[0], ranked[1]
        if b == b2:
            return Intent(None, "clarify", s, f"tie between {ranked[0][0]} and {ranked[1][0]}")
        return Intent(best, "out_of_scope" if best == "out_of_scope" else "ok", s)


# ---------------------------------------------------------------- embedding similarity

class EmbeddingClassifier:
    """all-MiniLM-L6-v2: 22M parameters, 384-d, runs on CPU in milliseconds, trained for
    sentence similarity, so short paraphrased questions land near their examples.
    Texts are masked first (names/years -> type words) so similarity reflects the question
    shape, not who it is about."""

    name = "embedding"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, min_score: float = 0.45, margin: float = 0.03, use_mask: bool = True):
        from sentence_transformers import SentenceTransformer
        self.min_score, self.margin, self.use_mask = min_score, margin, use_mask
        self.model = SentenceTransformer(self.MODEL)
        self.ex_labels = [lab for lab, d in INTENTS.items() for _ in d["examples"]]
        texts = [x for d in INTENTS.values() for x in d["examples"]]
        self.ex_vecs = self._embed(texts)          # embedded once, kept in memory

    def _embed(self, texts):
        if self.use_mask:
            texts = [mask(t) for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def scores(self, text: str) -> dict[str, float]:
        sims = self.ex_vecs @ self._embed([text])[0]
        out = {lab: -1.0 for lab in LABELS}
        for lab, sim in zip(self.ex_labels, sims):
            out[lab] = max(out[lab], float(sim))
        return out

    def classify(self, text: str, prev: str | None = None) -> Intent:
        g = _gate(text, prev)
        if g:
            return g
        s = self.scores(text)
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        (best, b), (second, b2) = ranked[0], ranked[1]
        if b < self.min_score:
            return Intent(None, "clarify", s, f"best score {b:.2f} < {self.min_score}")
        if b - b2 < self.margin:
            return Intent(None, "clarify", s, f"{best} {b:.2f} too close to {second} {b2:.2f}")
        if best == "out_of_scope":
            # Similarity alone never marks OOS: an explicit unanswerable topic must confirm it.
            if oos_keyword(text):
                return Intent("out_of_scope", "out_of_scope", s, "nearest OOS + unanswerable topic")
            return Intent(None, "clarify", s, "nearest example is OOS but no rule confirms it")
        return Intent(best, "ok", s)


class HybridClassifier:
    """Rules first; when the rules cannot decide (no keyword, or a tie) fall back to the
    embedding classifier, which still asks to rephrase below its min score / margin."""

    name = "hybrid"

    def __init__(self, embedding: EmbeddingClassifier | None = None):
        self.rules, self.emb = RuleClassifier(), embedding or EmbeddingClassifier()

    def classify(self, text: str, prev: str | None = None) -> Intent:
        r = self.rules.classify(text, prev)
        if r.status != "clarify":
            return r
        e = self.emb.classify(text, prev)
        e.reason = f"rules undecided ({r.reason}); embedding: {e.reason or 'top score'}"
        return e


# Deployed classifier (see eval/results/intent_report.md for the comparison).
DEPLOYED = HybridClassifier


if __name__ == "__main__":
    r = RuleClassifier()
    assert r.classify("How many wins did Hamilton have at Silverstone before 2020?").label == "driver_performance"
    assert r.classify("And in 2020?", prev="driver_performance").status == "follow_up"
    assert r.classify("What about Norris?", prev="head_to_head").label == "head_to_head"
    assert r.classify("What's the capital of Peru?", prev="race_facts").label == "out_of_scope"
    assert r.classify("Hamilton's salary in 2020").label == "out_of_scope"
    assert r.classify("Predict Norris at the 2024 Miami Grand Prix").label == "podium_prediction"
    assert is_follow_up("2021?") and is_follow_up("What about Norris?") and not is_follow_up("How many points did he score in 2019?")
    print("intents.py rule checks passed")
