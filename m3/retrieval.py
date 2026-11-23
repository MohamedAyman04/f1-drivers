"""Section 2: Graph Retrieval Layer for the Formula 1 Graph-RAG Assistant (Member 2),
integrated with Member 1 (`m3/intents.py` and `m3/entities.py`).

Supports:
1. All 10 canonical intent labels from `m3.intents.LABELS`:
   - `driver_performance`
   - `constructor_performance`
   - `head_to_head`
   - `race_facts`
   - `qualifying_grid`
   - `pit_stops`
   - `season_history`
   - `driver_profile`
   - `podium_prediction`
   - `out_of_scope`
   Plus granular sub-intent routing across 22 parameterized Cypher templates (`CYPHER_TEMPLATES`).
2. Direct compatibility with Member 1's `Extraction.entities` and `merge(cache, ex.entities)`
   dictionary format (`drivers`, `constructors`, `circuits`, `grand_prix`, `season`,
   `season_range`, `grid`, `champ_pos`, `wins`, `points`) as well as flat dictionaries.
3. Strict adherence to the Member 1 + Member 2 Entity-Cache Contract (`m3/README.md`):
   each template reads only the slots its intent needs, and leftover optional slots from earlier
   turns are not applied to new non-follow-up questions unless referenced.
4. Empty-result detection (both 0-row Cypher matches and 0-count aggregation rows), returning
   `"The knowledge graph has no data for that question."` without asking the LLM to guess.
5. Subgraph extraction (`nodes`, `edges`, `relationship_statements`) for the Section 5.3 UI trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import neo4j

try:
    from .entities import Extraction, clarification, extract, merge, norm, resolve_choice
    from .intents import DEPLOYED, INTENTS, LABELS, Intent, is_follow_up
except ImportError:
    from entities import Extraction, clarification, extract, merge, norm, resolve_choice  # type: ignore
    from intents import DEPLOYED, INTENTS, LABELS, Intent, is_follow_up  # type: ignore


EMPTY_RESULT_MESSAGE = "The knowledge graph has no data for that question."
OUT_OF_SCOPE_MESSAGE = (
    "I do not have the knowledge in the Formula 1 Knowledge Graph to answer that question. "
    "The graph covers Grand Prix drivers, constructors, circuits, seasons, races, results, "
    "qualifying, and pit stops (1950–2024), excluding sprint results, telemetry, salaries, "
    "weather, tyres, or non-F1 topics."
)
REPHRASE_MESSAGE = (
    "I could not confidently determine what you are asking. Could you please rephrase your question?"
)


# ---------------------------------------------------------------------------
# Section 2.a: Intent Catalog (Aligned with m3.intents.LABELS + Sub-Intents)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    m1_label: str
    category: str
    description: str
    required_entities: tuple[str, ...]
    optional_entities: tuple[str, ...]
    min_hops: int
    multi_entity: bool
    new_relative_to_m2: bool
    templates: tuple[str, ...]
    example_questions: tuple[str, ...]


INTENT_CATALOG: dict[str, IntentSpec] = {
    "driver_performance": IntentSpec(
        intent_id="driver_performance",
        m1_label="driver_performance",
        category="Driver performance",
        description=(
            "A driver's results: points, wins, podiums, points finishes, retirements, best finish, "
            "and average finishing position in a season, a range of seasons, at a circuit, or over a career."
        ),
        required_entities=("drivers",),
        optional_entities=("circuits", "grand_prix", "season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=True,
        templates=("driver_season_or_career_stats", "driver_at_circuit_stats", "season_driver_leaderboard", "circuit_top_drivers_by_avg_finish"),
        example_questions=(
            "How many podiums did Leclerc get in 2022?",
            "How many wins did Hamilton have at Silverstone before 2020?",
            "What is Verstappen's average finishing position at Zandvoort?",
            "How many races did Alonso win in his career?",
        ),
    ),
    "constructor_performance": IntentSpec(
        intent_id="constructor_performance",
        m1_label="constructor_performance",
        category="Constructor performance",
        description=(
            "A constructor's results: season points, wins, podiums, classified finishes, retirements, "
            "and reliability rate over a season, a range of seasons, at a circuit, or across seasons."
        ),
        required_entities=("constructors",),
        optional_entities=("circuits", "grand_prix", "season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=False,
        templates=("constructor_period_performance", "constructor_by_season_performance", "all_constructors_reliability_ranking"),
        example_questions=(
            "How reliable was Williams between 2014 and 2021?",
            "How many points did Ferrari score in 2022?",
            "Did redbull win more races than mercedes amg in 2021?",
            "Which season brought Alpine the most points?",
        ),
    ),
    "head_to_head": IntentSpec(
        intent_id="head_to_head",
        m1_label="head_to_head",
        category="Head-to-head",
        description=(
            "Compare two drivers in shared races (as teammates or in the same season): shared races, "
            "how many times each finished ahead (by positionOrder), points, wins, and podiums."
        ),
        required_entities=("drivers",),
        optional_entities=("constructors", "season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=False,
        templates=("teammate_race_head_to_head", "any_shared_race_head_to_head"),
        example_questions=(
            "Who finished ahead more often, Norris or Ricciardo at McLaren?",
            "Compare Lewis Hamilton and Valtteri Bottas at Mercedes.",
            "How did Charles Leclerc and Carlos Sainz compare at Ferrari?",
            "How did Albon fare against Verstappen at Red Bull?",
        ),
    ),
    "race_facts": IntentSpec(
        intent_id="race_facts",
        m1_label="race_facts",
        category="Race and circuit facts",
        description=(
            "Facts about a specific Grand Prix or circuit: race winner and podium finishers, "
            "circuit location/city/country/altitude, total races held at a circuit, and historical winners."
        ),
        required_entities=("grand_prix_or_circuit",),
        optional_entities=("season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=True,
        templates=("race_podium_and_results", "circuit_profile_and_history", "circuit_winners_in_period", "circuits_by_country"),
        example_questions=(
            "Who won the 2019 German Grand Prix?",
            "Who was on the podium at the 2021 Abu Dhabi GP?",
            "Where is the Suzuka circuit located?",
            "How many Grands Prix has Spa hosted over the years?",
        ),
    ),
    "qualifying_grid": IntentSpec(
        intent_id="qualifying_grid",
        m1_label="qualifying_grid",
        category="Qualifying and grid",
        description=(
            "Qualifying and starting grid facts: pole positions, Q3 appearances, front-row starts, "
            "qualifying classification for a race, and grid-to-finish positions gained or lost."
        ),
        required_entities=("any_entity_or_time",),
        optional_entities=("drivers", "constructors", "circuits", "grand_prix", "season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=True,
        templates=(
            "race_grid_to_finish_gains",
            "race_grid_to_finish_losses",
            "qualifying_race_results",
            "driver_qualifying_stats",
            "constructor_qualifying_stats",
            "circuit_front_row_to_podium_conversion",
        ),
        example_questions=(
            "Which driver gained the most places from the grid at Monza in 2021?",
            "How many times did Charles Leclerc qualify on pole in 2022?",
            "How often did Norris make it into the final qualifying session in 2023?",
            "Who started first at the 2022 Belgian Grand Prix?",
        ),
    ),
    "pit_stops": IntentSpec(
        intent_id="pit_stops",
        m1_label="pit_stops",
        category="Pit stops and strategy",
        description=(
            "Pit stops and strategy: fastest or slowest pit stop in a race, pit stop counts and "
            "laps for a driver in a race, and average pit stops per driver in a race or season."
        ),
        required_entities=("race_or_driver_or_season",),
        optional_entities=("drivers", "circuits", "grand_prix", "season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=True,
        templates=(
            "race_fastest_pit_stops",
            "race_slowest_pit_stops",
            "race_pit_stop_averages",
            "driver_season_pit_stop_stats",
            "season_pit_stop_summary",
        ),
        example_questions=(
            "What was the fastest pit stop at the 2023 Qatar Grand Prix?",
            "How many times did Gasly come into the pits in the 2023 Dutch GP?",
            "What's the typical number of pit visits per driver in 2019?",
            "Slowest service in the pit lane at the 2022 Monaco GP?",
        ),
    ),
    "season_history": IntentSpec(
        intent_id="season_history",
        m1_label="season_history",
        category="Season and history",
        description=(
            "Whole-season or multi-season history: championship Grand Prix points leader, number of "
            "drivers and rounds in a season, multi-season points leader, and drivers who changed teams."
        ),
        required_entities=("season_or_range",),
        optional_entities=("season", "season_range"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=False,
        templates=("driver_team_changes_between_seasons", "season_overview_and_leader", "period_points_leaders"),
        example_questions=(
            "Which drivers changed teams between 2020 and 2021?",
            "Who topped the points table in 2010?",
            "How many different drivers took part in 2022?",
            "Who moved to a new team ahead of 2019?",
        ),
    ),
    "driver_profile": IntentSpec(
        intent_id="driver_profile",
        m1_label="driver_profile",
        category="Driver profile (extra type)",
        description=(
            "Biographical and career profile of a driver: nationality, date of birth, permanent car "
            "number, 3-letter code, all constructors driven for, debut race, and career span."
        ),
        required_entities=("drivers",),
        optional_entities=(),
        min_hops=3,
        multi_entity=False,
        new_relative_to_m2=True,
        templates=("driver_profile_and_career_summary",),
        example_questions=(
            "What country does Oscar Piastri represent?",
            "What is Fernando Alonso's date of birth?",
            "List every team Kimi Raikkonen has raced for.",
            "Which race was Max Verstappen's first in Formula 1?",
        ),
    ),
    "podium_prediction": IntentSpec(
        intent_id="podium_prediction",
        m1_label="podium_prediction",
        category="Podium prediction explainer",
        description=(
            "Routes pre-race podium prediction requests to Section 4 (saved Milestone 1 model, "
            "live SHAP/LIME local explanation, and pre-race Knowledge Graph facts strictly before race date)."
        ),
        required_entities=("drivers",),
        optional_entities=("grand_prix", "circuits", "season", "grid", "champ_pos", "wins", "points"),
        min_hops=3,
        multi_entity=True,
        new_relative_to_m2=True,
        templates=("pre_race_driver_circuit_context", "pre_race_constructor_context"),
        example_questions=(
            "Predict Norris at the 2024 Miami Grand Prix.",
            "Is Lewis Hamilton going to make the podium at the 2024 British Grand Prix?",
            "Leclerc starts P2 at Monaco 2024 and is 3rd in the standings - will he be on the podium?",
            "What would the model say about Russell in Las Vegas 2024 from pole?",
        ),
    ),
    "out_of_scope": IntentSpec(
        intent_id="out_of_scope",
        m1_label="out_of_scope",
        category="Out of scope",
        description=(
            "Refuses questions outside the Formula 1 Knowledge Graph (salaries, tyres, weather, "
            "sprint races, lap times, penalties, car technology, personal lives, or non-F1 topics)."
        ),
        required_entities=(),
        optional_entities=(),
        min_hops=0,
        multi_entity=False,
        new_relative_to_m2=True,
        templates=(),
        example_questions=(
            "What is Hamilton's salary at Ferrari?",
            "Which tyre compound did Verstappen use at Monza?",
            "Who won the sprint race in Brazil 2023?",
            "Who won the Tour de France in 2020?",
        ),
    ),
}

# Map granular sub-intent aliases (from earlier test sets or direct calls) to canonical specs
SUB_INTENT_TO_M1_LABEL: dict[str, str] = {
    "driver_performance": "driver_performance",
    "driver_season_career_performance": "driver_performance",
    "driver_circuit_performance": "driver_performance",
    "constructor_performance": "constructor_performance",
    "constructor_performance_and_reliability": "constructor_performance",
    "head_to_head": "head_to_head",
    "teammate_head_to_head": "head_to_head",
    "race_facts": "race_facts",
    "race_results_and_podium": "race_facts",
    "circuit_info_and_history": "race_facts",
    "qualifying_grid": "qualifying_grid",
    "qualifying_poles_and_results": "qualifying_grid",
    "grid_vs_finish_gains": "qualifying_grid",
    "pit_stops": "pit_stops",
    "pit_stop_strategy": "pit_stops",
    "season_history": "season_history",
    "season_history_and_transfers": "season_history",
    "driver_profile": "driver_profile",
    "podium_prediction": "podium_prediction",
    "out_of_scope": "out_of_scope",
}


# ---------------------------------------------------------------------------
# Section 2.b: Parameterized Cypher Templates (22 Templates)
# ---------------------------------------------------------------------------

CYPHER_TEMPLATES: dict[str, str] = {
    # 1a. Driver performance in a season, year range, or full career
    "driver_season_or_career_stats": """
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE ($year IS NULL OR s.year = $year)
  AND ($start_year IS NULL OR s.year >= $start_year)
  AND ($end_year IS NULL OR s.year <= $end_year)
WITH d,
     min(s.year) AS First_Season,
     max(s.year) AS Last_Season,
     count(res) AS Entries,
     count(DISTINCT CASE WHEN NOT st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew'] THEN r END) AS Starts,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums,
     count(CASE WHEN res.points > 0 THEN 1 END) AS Points_Finishes,
     count(CASE WHEN NOT (st.status = 'Finished' OR st.status STARTS WITH '+') THEN 1 END) AS Retirements,
     min(res.positionOrder) AS Best_Finish,
     round(avg(res.positionOrder), 2) AS Avg_Finish
RETURN d.forename + ' ' + d.surname AS Driver,
       First_Season,
       Last_Season,
       Entries,
       Starts,
       Points,
       Wins,
       Podiums,
       Points_Finishes,
       Retirements,
       Best_Finish,
       Avg_Finish
LIMIT 10
""",

    # 1b. Driver performance at a specific circuit (with optional season or range)
    "driver_at_circuit_stats": """
MATCH (c:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver {driverId: $driverId}),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($before_year IS NULL OR r.date < date(toString($before_year) + '-01-01'))
  AND ($before_date IS NULL OR r.date < date($before_date))
  AND NOT st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew']
WITH d, c,
     count(DISTINCT r) AS Starts,
     count(CASE WHEN res.positionOrder = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums,
     min(res.positionOrder) AS Best_Finish,
     round(avg(res.positionOrder), 2) AS Avg_Finish,
     sum(res.points) AS Points
RETURN d.forename + ' ' + d.surname AS Driver,
       c.circuit_name AS Circuit,
       Starts,
       Wins,
       Podiums,
       Best_Finish,
       Avg_Finish,
       Points
LIMIT 10
""",

    # 1c. Season driver leaderboard (covers M2 Q1)
    "season_driver_leaderboard": """
MATCH (s:Season {year: $year})<-[:IN_SEASON]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
WITH d,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums,
     count(DISTINCT r) AS Races_Entered
RETURN d.forename + ' ' + d.surname AS Driver,
       Points,
       Wins,
       Podiums,
       Races_Entered
ORDER BY Points DESC, Wins DESC, Driver ASC
LIMIT 25
""",

    # 1d. Top drivers at a circuit with minimum starts (covers M2 Q4)
    "circuit_top_drivers_by_avg_finish": """
MATCH (c:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE ($before_year IS NULL OR r.date < date(toString($before_year) + '-01-01'))
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($year IS NULL OR r.year = $year)
  AND NOT st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew']
WITH d.forename + ' ' + d.surname AS Driver,
     c.circuit_name AS Circuit,
     count(DISTINCT r) AS Starts,
     avg(res.positionOrder) AS Avg_Finish,
     count(CASE WHEN res.positionOrder = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums
WHERE Starts >= coalesce($min_starts, 5)
RETURN Driver,
       Circuit,
       Starts,
       round(Avg_Finish, 4) AS Avg_Finish,
       Wins,
       Podiums
ORDER BY Avg_Finish ASC, Wins DESC, Driver ASC
LIMIT 15
""",

    # 2a. Constructor performance & reliability over a season, period, or circuit (supports 1 or 2+ constructors)
    "constructor_period_performance": """
MATCH (c:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season),
      (res)-[:HAS_STATUS]->(st:Status),
      (r)-[:AT_CIRCUIT]->(cir:Circuit)
WHERE c.constructorId IN $constructorIds
  AND ($circuitId IS NULL OR cir.circuitId = $circuitId)
  AND ($year IS NULL OR s.year = $year)
  AND ($start_year IS NULL OR s.year >= $start_year)
  AND ($end_year IS NULL OR s.year <= $end_year)
WITH c,
     min(s.year) AS Start_Season,
     max(s.year) AS End_Season,
     count(res) AS Entries,
     count(CASE WHEN NOT st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew'] THEN 1 END) AS Starts,
     count(CASE WHEN st.status = 'Finished' OR st.status STARTS WITH '+' THEN 1 END) AS Classified,
     count(CASE WHEN NOT (st.status = 'Finished' OR st.status STARTS WITH '+' OR st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew']) THEN 1 END) AS Retirements,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums
RETURN c.constructor_name AS Constructor,
       Start_Season,
       End_Season,
       Entries,
       Starts,
       Classified,
       Retirements,
       round(toFloat(Classified) / Entries, 4) AS Reliability,
       round(toFloat(Classified) / CASE WHEN Starts > 0 THEN Starts ELSE 1 END, 4) AS Classified_Start_Share,
       Points,
       Wins,
       Podiums
ORDER BY Wins DESC, Points DESC, Constructor ASC
LIMIT 10
""",

    # 2b. Constructor season-by-season points & wins breakdown (e.g. "Which season brought Alpine the most points?")
    "constructor_by_season_performance": """
MATCH (c:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE c.constructorId IN $constructorIds
WITH c.constructor_name AS Constructor,
     s.year AS Season,
     count(res) AS Entries,
     count(CASE WHEN st.status = 'Finished' OR st.status STARTS WITH '+' THEN 1 END) AS Classified,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums
RETURN Constructor,
       Season,
       Entries,
       Classified,
       round(toFloat(Classified) / Entries, 4) AS Reliability,
       Points,
       Wins,
       Podiums
ORDER BY Points DESC, Wins DESC, Season DESC
LIMIT 10
""",

    # 2c. All constructors reliability ranking over a period (covers M2 Q2)
    "all_constructors_reliability_ranking": """
MATCH (c:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE ($start_year IS NULL OR s.year >= $start_year)
  AND ($end_year IS NULL OR s.year <= $end_year)
  AND ($year IS NULL OR s.year = $year)
WITH c.constructor_name AS Constructor,
     count(res) AS Entries,
     count(CASE WHEN st.status = 'Finished' OR st.status STARTS WITH '+' THEN 1 END) AS Classified,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins
WHERE Entries >= coalesce($min_entries, 100)
RETURN Constructor,
       Entries,
       Classified,
       round(toFloat(Classified) / Entries, 6) AS Reliability,
       Points,
       Wins
ORDER BY Reliability DESC, Entries DESC, Constructor ASC
LIMIT 20
""",

    # 3a. Teammate head-to-head comparison in shared races for the same constructor (covers M2 Q3)
    "teammate_race_head_to_head": """
MATCH (d1:Driver {driverId: $driver1Id})<-[:BY_DRIVER]-(res1:Result)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(res2:Result)-[:BY_DRIVER]->(d2:Driver {driverId: $driver2Id}),
      (res1)-[:FOR_CONSTRUCTOR]->(c:Constructor)<-[:FOR_CONSTRUCTOR]-(res2)
WHERE ($constructorId IS NULL OR c.constructorId = $constructorId)
  AND ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
RETURN c.constructor_name AS Constructor,
       d1.forename + ' ' + d1.surname AS Driver_1,
       d2.forename + ' ' + d2.surname AS Driver_2,
       count(DISTINCT r) AS Shared_Races,
       count(CASE WHEN res1.positionOrder < res2.positionOrder THEN 1 END) AS Driver_1_Ahead,
       count(CASE WHEN res2.positionOrder < res1.positionOrder THEN 1 END) AS Driver_2_Ahead,
       sum(res1.points) AS Driver_1_Points,
       sum(res2.points) AS Driver_2_Points,
       count(CASE WHEN res1.positionOrder = 1 THEN 1 END) AS Driver_1_Wins,
       count(CASE WHEN res2.positionOrder = 1 THEN 1 END) AS Driver_2_Wins,
       count(CASE WHEN res1.positionOrder <= 3 THEN 1 END) AS Driver_1_Podiums,
       count(CASE WHEN res2.positionOrder <= 3 THEN 1 END) AS Driver_2_Podiums
ORDER BY Shared_Races DESC
LIMIT 10
""",

    # 3b. Head-to-head comparison across any shared races (when two drivers raced in the same season/period on different teams)
    "any_shared_race_head_to_head": """
MATCH (d1:Driver {driverId: $driver1Id})<-[:BY_DRIVER]-(res1:Result)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(res2:Result)-[:BY_DRIVER]->(d2:Driver {driverId: $driver2Id})
WHERE ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
RETURN d1.forename + ' ' + d1.surname AS Driver_1,
       d2.forename + ' ' + d2.surname AS Driver_2,
       count(DISTINCT r) AS Shared_Races,
       count(CASE WHEN res1.positionOrder < res2.positionOrder THEN 1 END) AS Driver_1_Ahead,
       count(CASE WHEN res2.positionOrder < res1.positionOrder THEN 1 END) AS Driver_2_Ahead,
       sum(res1.points) AS Driver_1_Points,
       sum(res2.points) AS Driver_2_Points
LIMIT 5
""",

    # 4a. Specific race winner, podium, and top classified finishers
    "race_podium_and_results": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE res.positionOrder <= coalesce($top_n, 3)
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       cir.circuit_loc AS Location,
       cir.circuit_country AS Country,
       res.positionOrder AS Position,
       d.forename + ' ' + d.surname AS Driver,
       c.constructor_name AS Constructor,
       res.grid AS Grid,
       res.points AS Points,
       coalesce(res.time, st.status) AS Time_Or_Status
ORDER BY Position ASC
LIMIT 10
""",

    # 4b. Circuit profile, geographical metadata, and hosting history
    "circuit_profile_and_history": """
MATCH (cir:Circuit {circuitId: $circuitId})
OPTIONAL MATCH (cir)<-[:AT_CIRCUIT]-(r:Race)
WHERE ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
WITH cir,
     count(DISTINCT r) AS Races_Hosted,
     min(r.year) AS First_Held,
     max(r.year) AS Last_Held,
     collect(DISTINCT r.name) AS Events_Staged
RETURN cir.circuit_name AS Circuit,
       cir.circuit_loc AS Location,
       cir.circuit_country AS Country,
       cir.lat AS Latitude,
       cir.lng AS Longitude,
       cir.alt AS Altitude_m,
       Races_Hosted,
       First_Held,
       Last_Held,
       Events_Staged
LIMIT 5
""",

    # 4c. Race winners at a circuit or Grand Prix across a year range (e.g. "Monaco GP winners in the 2010s")
    "circuit_winners_in_period": """
MATCH (cir:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WHERE res.positionOrder = 1
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Winner,
       c.constructor_name AS Constructor,
       res.grid AS Grid
ORDER BY r.year ASC
LIMIT 25
""",

    # 4d. All circuits in a country
    "circuits_by_country": """
MATCH (cir:Circuit)
WHERE toLower(cir.circuit_country) = toLower($country)
OPTIONAL MATCH (cir)<-[:AT_CIRCUIT]-(r:Race)
WITH cir,
     count(DISTINCT r) AS Races_Hosted,
     min(r.year) AS First_Held,
     max(r.year) AS Last_Held
RETURN cir.circuit_name AS Circuit,
       cir.circuit_loc AS Location,
       cir.circuit_country AS Country,
       cir.alt AS Altitude_m,
       Races_Hosted,
       First_Held,
       Last_Held
ORDER BY Races_Hosted DESC, Circuit ASC
LIMIT 15
""",

    # 5a. Grid vs. finish position gains in a race ("Which driver gained the most places from the grid...")
    "race_grid_to_finish_gains": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE res.grid > 0
  AND ($driverId IS NULL OR d.driverId = $driverId)
WITH r, cir, d, c, res, st,
     (res.grid - res.positionOrder) AS Places_Gained
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Driver,
       c.constructor_name AS Constructor,
       res.grid AS Grid_Position,
       res.positionOrder AS Finish_Position,
       Places_Gained,
       st.status AS Status
ORDER BY Places_Gained DESC, Finish_Position ASC
LIMIT 10
""",

    # 5b. Grid vs. finish position losses in a race ("Which drivers dropped the most spots relative to their grid slot...")
    "race_grid_to_finish_losses": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE res.grid > 0
  AND ($driverId IS NULL OR d.driverId = $driverId)
WITH r, cir, d, c, res, st,
     (res.positionOrder - res.grid) AS Places_Lost
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Driver,
       c.constructor_name AS Constructor,
       res.grid AS Grid_Position,
       res.positionOrder AS Finish_Position,
       Places_Lost,
       st.status AS Status
ORDER BY Places_Lost DESC, Finish_Position DESC
LIMIT 10
""",

    # 5c. Qualifying and starting grid order for a specific race (or specific driver in that race)
    "qualifying_race_results": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
OPTIONAL MATCH (d)<-[:BY_DRIVER]-(q:Qualifying)-[:FOR_RACE]->(r)
WHERE ($driverId IS NULL OR d.driverId = $driverId)
WITH r, cir, d, c, res, q,
     coalesce(q.position, CASE WHEN res.grid > 0 THEN res.grid ELSE 99 END) AS Order_Key
WHERE $driverId IS NOT NULL OR Order_Key <= coalesce($top_n, 10)
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Driver,
       c.constructor_name AS Constructor,
       q.position AS Qualifying_Position,
       res.grid AS Grid_Position,
       res.positionOrder AS Finish_Position,
       q.q1 AS Q1_Time,
       q.q2 AS Q2_Time,
       q.q3 AS Q3_Time
ORDER BY Order_Key ASC, res.positionOrder ASC
LIMIT 10
""",

    # 5d. Driver qualifying and starting grid stats (both Qualifying node and Result.grid for pre-1994 drivers like Senna)
    "driver_qualifying_stats": """
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(cir:Circuit)
WHERE ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($circuitId IS NULL OR cir.circuitId = $circuitId)
OPTIONAL MATCH (d)<-[:BY_DRIVER]-(q:Qualifying)-[:FOR_RACE]->(r)
WITH d,
     count(DISTINCT r) AS Races_Entered,
     count(CASE WHEN res.grid = 1 THEN 1 END) AS Grid_Poles,
     count(CASE WHEN res.grid IN [1, 2] THEN 1 END) AS Front_Row_Starts,
     count(q) AS Qualifying_Sessions,
     count(CASE WHEN q.position = 1 THEN 1 END) AS Qualifying_Poles,
     count(CASE WHEN q.position <= 2 THEN 1 END) AS Front_Row_Qualifying,
     count(CASE WHEN q.q3 IS NOT NULL THEN 1 END) AS Q3_Appearances,
     round(avg(q.position), 2) AS Avg_Qualifying_Position
RETURN d.forename + ' ' + d.surname AS Driver,
       Races_Entered,
       Grid_Poles,
       Front_Row_Starts,
       Qualifying_Sessions,
       Qualifying_Poles,
       Front_Row_Qualifying,
       Q3_Appearances,
       Avg_Qualifying_Position
LIMIT 10
""",

    # 5e. Constructor qualifying & pole position stats ("How many poles did Ferrari get in 2022?")
    "constructor_qualifying_stats": """
MATCH (c:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(cir:Circuit)
WHERE ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($circuitId IS NULL OR cir.circuitId = $circuitId)
OPTIONAL MATCH (c)<-[:FOR_CONSTRUCTOR]-(q:Qualifying)-[:FOR_RACE]->(r)
WHERE q IS NULL OR EXISTS {
    MATCH (d:Driver)<-[:BY_DRIVER]-(res)
    MATCH (d)<-[:BY_DRIVER]-(q)
}
WITH c,
     count(DISTINCT r) AS Races_Entered,
     count(DISTINCT CASE WHEN res.grid = 1 THEN r END) AS Grid_Poles,
     count(CASE WHEN res.grid IN [1, 2] THEN 1 END) AS Front_Row_Grid_Slots,
     count(DISTINCT CASE WHEN q.position = 1 THEN r END) AS Qualifying_Poles,
     count(CASE WHEN q.q3 IS NOT NULL THEN 1 END) AS Q3_Appearances
RETURN c.constructor_name AS Constructor,
       Races_Entered,
       Grid_Poles,
       Front_Row_Grid_Slots,
       Qualifying_Poles,
       Q3_Appearances
LIMIT 10
""",

    # 5f. Front-row (grid 1 or 2) to podium conversion rate at a circuit (M1 DE Q1)
    "circuit_front_row_to_podium_conversion": """
MATCH (cir:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
WHERE res.grid IN [1, 2]
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($year IS NULL OR r.year = $year)
WITH cir,
     count(res) AS Front_Row_Starts,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podium_Finishes,
     count(CASE WHEN res.positionOrder = 1 THEN 1 END) AS Wins_From_Front_Row
RETURN cir.circuit_name AS Circuit,
       Front_Row_Starts,
       Podium_Finishes,
       Wins_From_Front_Row,
       round(toFloat(Podium_Finishes) / Front_Row_Starts, 4) AS Front_Row_Podium_Rate
LIMIT 5
""",

    # 6a. Fastest pit stops in a specific race (or all stops for a specific driver in that race)
    "race_fastest_pit_stops": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(p:PitStop)-[:FOR_RACE]->(r)
WHERE ($driverId IS NULL OR d.driverId = $driverId)
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Driver,
       p.stop AS Stop_Number,
       p.lap AS Lap,
       p.duration AS Duration_Text,
       p.milliseconds AS Duration_ms,
       round(toFloat(p.milliseconds) / 1000.0, 3) AS Duration_Seconds
ORDER BY p.milliseconds ASC
LIMIT 10
""",

    # 6b. Slowest / longest pit stops in a specific race
    "race_slowest_pit_stops": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(p:PitStop)-[:FOR_RACE]->(r)
WHERE ($driverId IS NULL OR d.driverId = $driverId)
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       d.forename + ' ' + d.surname AS Driver,
       p.stop AS Stop_Number,
       p.lap AS Lap,
       p.duration AS Duration_Text,
       p.milliseconds AS Duration_ms,
       round(toFloat(p.milliseconds) / 1000.0, 3) AS Duration_Seconds
ORDER BY p.milliseconds DESC
LIMIT 10
""",

    # 6c. Average pit stops per driver and race pit-stop overview for a specific Grand Prix
    "race_pit_stop_averages": """
MATCH (r:Race {raceId: $raceId})-[:AT_CIRCUIT]->(cir:Circuit),
      (d:Driver)<-[:BY_DRIVER]-(p:PitStop)-[:FOR_RACE]->(r)
WITH r, cir, d, count(p) AS Driver_Stops, min(p.milliseconds) AS Driver_Fastest_ms, max(p.milliseconds) AS Driver_Slowest_ms
WITH r, cir,
     count(DISTINCT d) AS Drivers_Who_Pitted,
     sum(Driver_Stops) AS Total_Pit_Stops,
     round(avg(Driver_Stops), 2) AS Avg_Stops_Per_Driver,
     min(Driver_Fastest_ms) AS Fastest_Stop_ms,
     max(Driver_Slowest_ms) AS Slowest_Stop_ms
RETURN r.year AS Season,
       r.name AS Grand_Prix,
       cir.circuit_name AS Circuit,
       Drivers_Who_Pitted,
       Total_Pit_Stops,
       Avg_Stops_Per_Driver,
       round(toFloat(Fastest_Stop_ms) / 1000.0, 3) AS Fastest_Stop_Seconds,
       round(toFloat(Slowest_Stop_ms) / 1000.0, 3) AS Slowest_Stop_Seconds
LIMIT 5
""",

    # 6d. Driver pit stop summary in a season or career/circuit
    "driver_season_pit_stop_stats": """
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(p:PitStop)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(cir:Circuit)
WHERE ($year IS NULL OR r.year = $year)
  AND ($start_year IS NULL OR r.year >= $start_year)
  AND ($end_year IS NULL OR r.year <= $end_year)
  AND ($circuitId IS NULL OR cir.circuitId = $circuitId)
WITH d,
     count(p) AS Total_Pit_Stops,
     count(DISTINCT r) AS Races_With_Pit_Stops,
     min(p.milliseconds) AS Fastest_Stop_ms,
     max(p.milliseconds) AS Longest_Stop_ms,
     avg(p.milliseconds) AS Avg_Stop_ms
RETURN d.forename + ' ' + d.surname AS Driver,
       Total_Pit_Stops,
       Races_With_Pit_Stops,
       round(toFloat(Total_Pit_Stops) / Races_With_Pit_Stops, 2) AS Avg_Stops_Per_Race,
       round(toFloat(Fastest_Stop_ms) / 1000.0, 3) AS Fastest_Stop_Seconds,
       round(toFloat(Longest_Stop_ms) / 1000.0, 3) AS Longest_Stop_Seconds,
       round(toFloat(Avg_Stop_ms) / 1000.0, 3) AS Avg_Stop_Seconds
LIMIT 10
""",

    # 6e. Whole-season pit stop summary ("What's the typical number of pit visits per driver in 2019?")
    "season_pit_stop_summary": """
MATCH (s:Season {year: $year})<-[:IN_SEASON]-(r:Race)<-[:FOR_RACE]-(p:PitStop)-[:BY_DRIVER]->(d:Driver)
WITH s.year AS Season, r, d, count(p) AS Stops_In_Race, min(p.milliseconds) AS Min_ms
WITH Season,
     count(DISTINCT r) AS Races_With_Data,
     count(DISTINCT d) AS Drivers_With_Stops,
     sum(Stops_In_Race) AS Total_Pit_Stops,
     round(avg(Stops_In_Race), 2) AS Avg_Stops_Per_Driver_Per_Race,
     round(toFloat(min(Min_ms)) / 1000.0, 3) AS Season_Fastest_Stop_Seconds
RETURN Season,
       Races_With_Data,
       Drivers_With_Stops,
       Total_Pit_Stops,
       Avg_Stops_Per_Driver_Per_Race,
       Season_Fastest_Stop_Seconds
LIMIT 5
""",

    # 7a. Drivers who changed primary constructor between two seasons (covers M2 Q5)
    "driver_team_changes_between_seasons": """
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE s.year IN [$year1, $year2]
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WITH d, s.year AS yr, c, count(res) AS entries
ORDER BY entries DESC, c.constructorId ASC
WITH d, yr, head(collect(c)) AS primaryCon
WITH d,
     head([x IN collect(CASE WHEN yr = $year1 THEN primaryCon END) WHERE x IS NOT NULL]) AS con1,
     head([x IN collect(CASE WHEN yr = $year2 THEN primaryCon END) WHERE x IS NOT NULL]) AS con2
WHERE con1 IS NOT NULL AND con2 IS NOT NULL AND con1.constructorId <> con2.constructorId
WITH d.forename + ' ' + d.surname AS Driver,
     con1.constructor_name AS Constructor_Season_1,
     con2.constructor_name AS Constructor_Season_2
ORDER BY Driver ASC
RETURN $year1 AS Season_1,
       $year2 AS Season_2,
       count(Driver) AS Drivers_Changed,
       collect(Driver) AS Drivers,
       collect(Driver + ' (' + Constructor_Season_1 + ' -> ' + Constructor_Season_2 + ')') AS Transfers
LIMIT 1
""",

    # 7b. Single-season championship overview (Grand Prix points leader, total races, drivers, constructors)
    "season_overview_and_leader": """
MATCH (s:Season {year: $year})<-[:IN_SEASON]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver),
      (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WITH s.year AS Season,
     count(DISTINCT r) AS Total_Races,
     count(DISTINCT d) AS Total_Drivers,
     count(DISTINCT c) AS Total_Constructors
MATCH (s2:Season {year: Season})<-[:IN_SEASON]-(r2:Race)<-[:FOR_RACE]-(res2:Result)-[:BY_DRIVER]->(leader:Driver)
WITH Season, Total_Races, Total_Drivers, Total_Constructors,
     leader.forename + ' ' + leader.surname AS Driver,
     sum(res2.points) AS Points,
     count(CASE WHEN res2.position = 1 THEN 1 END) AS Wins
ORDER BY Points DESC, Wins DESC, Driver ASC
WITH Season, Total_Races, Total_Drivers, Total_Constructors,
     head(collect(Driver)) AS Points_Leader,
     head(collect(Points)) AS Leader_Points,
     head(collect(Wins)) AS Leader_Wins,
     collect(Driver + ' (' + toString(Points) + ' pts, ' + toString(Wins) + ' wins)')[0..5] AS Top_5_Standings
RETURN Season,
       Total_Races,
       Total_Drivers,
       Total_Constructors,
       Points_Leader,
       Leader_Points,
       Leader_Wins,
       Top_5_Standings
LIMIT 1
""",

    # 7c. Multi-season points leaders & summary ("Who scored the most points across 2014 to 2020?")
    "period_points_leaders": """
MATCH (s:Season)<-[:IN_SEASON]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
WHERE s.year >= $start_year AND s.year <= $end_year
WITH d.forename + ' ' + d.surname AS Driver,
     min(s.year) AS First_Season_In_Range,
     max(s.year) AS Last_Season_In_Range,
     count(DISTINCT r) AS Races_Entered,
     sum(res.points) AS Points,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Podiums
RETURN Driver,
       First_Season_In_Range,
       Last_Season_In_Range,
       Races_Entered,
       Points,
       Wins,
       Podiums
ORDER BY Points DESC, Wins DESC, Driver ASC
LIMIT 10
""",

    # 8. Driver profile & career summary (Member 1's `driver_profile` extra intent)
    "driver_profile_and_career_summary": """
MATCH (d:Driver {driverId: $driverId})
OPTIONAL MATCH (d)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)
OPTIONAL MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WITH d, res, r, c
ORDER BY r.date ASC, r.round ASC
WITH d,
     collect(DISTINCT c.constructor_name) AS Teams_Driven_For,
     min(r.year) AS Debut_Season,
     max(r.year) AS Latest_Season,
     count(DISTINCT r.year) AS Seasons_Competed,
     head(collect(CASE WHEN r IS NOT NULL THEN toString(r.year) + ' ' + r.name END)) AS First_Grand_Prix,
     head(collect(CASE WHEN r IS NOT NULL THEN toString(r.date) END)) AS First_Race_Date,
     count(res) AS Total_Entries,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Career_Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Career_Podiums,
     sum(res.points) AS Career_Points
RETURN d.forename + ' ' + d.surname AS Driver,
       d.nationality AS Nationality,
       toString(d.dob) AS Date_Of_Birth,
       d.number AS Car_Number,
       d.code AS Code,
       d.driverRef AS Driver_Ref,
       Teams_Driven_For,
       Debut_Season,
       Latest_Season,
       Seasons_Competed,
       First_Grand_Prix,
       First_Race_Date,
       Total_Entries,
       Career_Wins,
       Career_Podiums,
       Career_Points
LIMIT 1
""",

    # 9a. Section 4 Pre-Race Context: Driver's prior record at the circuit strictly before the race date
    "pre_race_driver_circuit_context": """
MATCH (cir:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver {driverId: $driverId}),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE r.date < date($before_date)
  AND NOT st.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew']
WITH d, cir,
     count(DISTINCT r) AS Prior_Circuit_Starts,
     count(CASE WHEN res.positionOrder = 1 THEN 1 END) AS Prior_Circuit_Wins,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Prior_Circuit_Podiums,
     min(res.positionOrder) AS Prior_Circuit_Best_Finish,
     round(avg(res.positionOrder), 2) AS Prior_Circuit_Avg_Finish
RETURN d.forename + ' ' + d.surname AS Driver,
       cir.circuit_name AS Circuit,
       Prior_Circuit_Starts,
       Prior_Circuit_Wins,
       Prior_Circuit_Podiums,
       Prior_Circuit_Best_Finish,
       Prior_Circuit_Avg_Finish
LIMIT 1
""",

    # 9b. Section 4 Pre-Race Context: Constructor's prior reliability & podium record strictly before the race date
    "pre_race_constructor_context": """
MATCH (c:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race),
      (res)-[:HAS_STATUS]->(st:Status)
WHERE r.date < date($before_date)
  AND r.year >= ($year - 2)
WITH c,
     count(res) AS Prior_Constructor_Entries_Recent,
     count(CASE WHEN st.status = 'Finished' OR st.status STARTS WITH '+' THEN 1 END) AS Prior_Constructor_Classified,
     count(CASE WHEN res.positionOrder <= 3 THEN 1 END) AS Prior_Constructor_Podiums,
     count(CASE WHEN res.position = 1 THEN 1 END) AS Prior_Constructor_Wins
RETURN c.constructor_name AS Constructor,
       Prior_Constructor_Entries_Recent,
       Prior_Constructor_Classified,
       round(toFloat(Prior_Constructor_Classified) / Prior_Constructor_Entries_Recent, 4) AS Prior_Constructor_Reliability,
       Prior_Constructor_Podiums,
       Prior_Constructor_Wins
LIMIT 1
""",
}


# ---------------------------------------------------------------------------
# Entity Resolution & Normalization Helpers
# ---------------------------------------------------------------------------

CONSTRUCTOR_ALIASES: dict[str, str] = {
    "redbull": "Red Bull",
    "red bull": "Red Bull",
    "red bull racing": "Red Bull",
    "rbr": "Red Bull",
    "mercedes": "Mercedes",
    "mercedes amg": "Mercedes",
    "mercedes-amg": "Mercedes",
    "merc": "Mercedes",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "mclaren": "McLaren",
    "williams": "Williams",
    "alpine": "Alpine F1 Team",
    "alpine f1": "Alpine F1 Team",
    "renault": "Renault",
    "aston martin": "Aston Martin",
    "aston": "Aston Martin",
    "haas": "Haas F1 Team",
    "haas f1": "Haas F1 Team",
    "alfa romeo": "Alfa Romeo",
    "sauber": "Sauber",
    "kick sauber": "Sauber",
    "toro rosso": "Toro Rosso",
    "alphatauri": "AlphaTauri",
    "alpha tauri": "AlphaTauri",
    "rb": "RB F1 Team",
    "vcarb": "RB F1 Team",
    "racing bulls": "RB F1 Team",
    "force india": "Force India",
    "racing point": "Racing Point",
    "lotus": "Lotus F1",
    "brawngp": "Brawn",
    "brawn": "Brawn",
    "brawn gp": "Brawn",
    "benetton": "Benetton",
    "brabham": "Brabham",
    "tyrrell": "Tyrrell",
}

CIRCUIT_ALIASES: dict[str, int] = {
    "silverstone": 9,
    "british grand prix": 9,
    "monza": 14,
    "italian grand prix": 14,
    "monaco": 6,
    "monte carlo": 6,
    "monte-carlo": 6,
    "circuit de monaco": 6,
    "monaco grand prix": 6,
    "spa": 13,
    "spa-francorchamps": 13,
    "belgian grand prix": 13,
    "suzuka": 22,
    "japanese grand prix": 22,
    "interlagos": 18,
    "sao paulo": 18,
    "são paulo": 18,
    "brazilian grand prix": 18,
    "yas marina": 24,
    "abu dhabi": 24,
    "bahrain": 3,
    "sakhir": 3,
    "jeddah": 77,
    "saudi arabian grand prix": 77,
    "miami": 79,
    "lusail": 78,
    "losail": 78,
    "qatar": 78,
    "zandvoort": 39,
    "imola": 21,
    "red bull ring": 70,
    "spielberg": 70,
    "hungaroring": 11,
    "budapest": 11,
    "barcelona": 4,
    "catalunya": 4,
    "montreal": 7,
    "gilles villeneuve": 7,
    "austin": 69,
    "cota": 69,
    "circuit of the americas": 69,
    "mexico city": 32,
    "hermanos rodriguez": 32,
    "hermanos rodríguez": 32,
    "autódromo hermanos rodríguez": 32,
    "autodromo hermanos rodriguez": 32,
    "melbourne": 1,
    "albert park": 1,
    "baku": 73,
    "singapore": 15,
    "marina bay": 15,
    "hockenheim": 10,
    "hockenheimring": 10,
    "nurburgring": 20,
    "nürburgring": 20,
    "pescara": 65,
    "las vegas": 80,
}


@dataclass
class RetrievalResult:
    intent: str
    template_id: str
    cypher: str
    params: dict[str, Any]
    rows: list[dict[str, Any]]
    is_empty: bool
    message: str | None = None
    ambiguity_prompt: str | None = None
    ambiguous_candidates: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    resolved_entities: dict[str, Any] = field(default_factory=dict)
    is_graphical: bool = False
    subgraph: dict[str, Any] = field(default_factory=lambda: {
        "nodes": [],
        "edges": [],
        "relationship_statements": [],
    })


def load_neo4j_config(config_path: Path | None = None) -> dict[str, str]:
    """Read URI, USERNAME, PASSWORD, and optional DATABASE dynamically by key."""
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        candidates = [root / "config.txt", Path.cwd() / "config.txt"]
        for c in candidates:
            if c.is_file():
                config_path = c
                break
        if config_path is None:
            raise FileNotFoundError("Could not find config.txt")

    config: dict[str, str] = {}
    with config_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            config[key.strip().upper()] = val.strip()
    return config


def filter_active_entities(
    intent_label: str,
    merged_entities: dict[str, Any],
    current_ex: Extraction | None = None,
    is_follow_up_turn: bool = False,
    text: str | None = None,
) -> dict[str, Any]:
    """Apply the Member 1 + Member 2 Cache Contract (`m3/README.md`):
    - Each intent only reads the slots it actually uses.
    - When `is_follow_up_turn` is True (e.g. "And in 2020?", "What about Norris?"), all merged
      slots relevant to the intent stay active.
    - When `is_follow_up_turn` is False (a full new question) and `current_ex` is provided,
      required slots not mentioned in `current_ex` (such as a pronoun "he" or "they") are filled
      from `merged_entities`, while leftover optional slots from earlier turns (such as an old
      Silverstone circuit when asking "How many times did Leclerc qualify on pole in 2022?") are
      only kept if mentioned in `current_ex` or deictically referenced ("there", "that circuit",
      "same period", "that season").
    """
    if current_ex is None or is_follow_up_turn:
        return dict(merged_entities)

    cur = current_ex.entities
    t_low = (text or "").lower()
    out = dict(merged_entities)

    has_venue_ref = bool(re.search(r"\b(there|that circuit|that track|that race|same circuit|same track)\b", t_low))
    has_time_ref = bool(re.search(r"\b(that season|that year|same season|same year|same period|that period)\b", t_low))
    has_team_ref = bool(re.search(r"\b(that team|same team|that constructor|their)\b", t_low))
    has_drv_ref = bool(re.search(r"\b(he|him|his|she|her|they|them)\b", t_low))

    canon = SUB_INTENT_TO_M1_LABEL.get(intent_label, intent_label)

    # Clear unreferenced optional venue slots on new full questions
    if canon in ("driver_performance", "constructor_performance", "qualifying_grid", "pit_stops", "driver_profile", "season_history", "head_to_head"):
        if not cur.get("circuits") and not has_venue_ref:
            out["circuits"] = []
        if not cur.get("grand_prix") and not has_venue_ref:
            out["grand_prix"] = None

    # Clear unreferenced optional time slots on new full questions where time is optional
    if canon in ("driver_performance", "constructor_performance", "head_to_head", "race_facts", "qualifying_grid", "pit_stops", "driver_profile"):
        if cur.get("season") is None and cur.get("season_range") is None and not has_time_ref:
            # Keep season only if required by race_facts with grand_prix
            if not (canon == "race_facts" and out.get("grand_prix") and not out.get("circuits")):
                out["season"] = None
                out["season_range"] = None

    # Clear unreferenced optional constructor slot on head_to_head or qualifying_grid
    if canon in ("head_to_head", "qualifying_grid"):
        if not cur.get("constructors") and not has_team_ref:
            if not (canon == "qualifying_grid" and not cur.get("drivers") and not cur.get("circuits") and not cur.get("grand_prix")):
                out["constructors"] = []

    # Clear unreferenced driver slot on race-level qualifying_grid or pit_stops questions
    if canon in ("qualifying_grid", "pit_stops"):
        if not cur.get("drivers") and not has_drv_ref and (cur.get("grand_prix") or cur.get("circuits") or cur.get("season") or cur.get("constructors")):
            out["drivers"] = []

    return out


class GraphRetriever:
    """Deterministic Neo4j retrieval engine for Milestone 3 (Member 2), integrated with Member 1."""

    def __init__(self, driver: neo4j.Driver | None = None, database: str | None = None) -> None:
        if driver is None:
            cfg = load_neo4j_config()
            self._driver = neo4j.GraphDatabase.driver(
                cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"])
            )
            self._owns_driver = True
            self._database = database or cfg.get("DATABASE")
        else:
            self._driver = driver
            self._owns_driver = False
            self._database = database

    def close(self) -> None:
        if self._owns_driver and self._driver is not None:
            self._driver.close()

    def _session(self) -> neo4j.Session:
        opts = {"database": self._database} if self._database else {}
        return self._driver.session(**opts)

    # ------------------------------------------------------------------
    # Entity Resolution Methods (supports both Member 1 {id, name} dicts and raw strings)
    # ------------------------------------------------------------------

    def resolve_driver(self, query_val: Any) -> dict[str, Any]:
        """Resolve a driver from Member 1's `{'id': int, 'name': str}` dict, integer ID, or string."""
        if isinstance(query_val, dict) and "id" in query_val:
            return {
                "status": "resolved",
                "driverId": int(query_val["id"]),
                "name": str(query_val.get("name", query_val["id"])),
            }
        if isinstance(query_val, int) or (isinstance(query_val, str) and query_val.isdigit()):
            did = int(query_val)
            with self._session() as s:
                rec = s.run(
                    "MATCH (d:Driver {driverId: $did}) "
                    "RETURN d.driverId AS id, d.forename + ' ' + d.surname AS name LIMIT 1",
                    did=did,
                ).single()
                if rec:
                    return {"status": "resolved", "driverId": rec["id"], "name": rec["name"]}
                return {"status": "not_found", "query": str(query_val)}

        q = str(query_val or "").strip()
        if not q:
            return {"status": "not_found", "query": ""}

        with self._session() as s:
            exact_full = s.run(
                "MATCH (d:Driver) "
                "WHERE toLower(d.forename + ' ' + d.surname) = toLower($q) "
                "RETURN d.driverId AS id, d.forename + ' ' + d.surname AS name "
                "ORDER BY d.driverId ASC LIMIT 5",
                q=q,
            ).data()
            if len(exact_full) == 1:
                return {"status": "resolved", "driverId": exact_full[0]["id"], "name": exact_full[0]["name"]}

            surname_matches = s.run(
                "MATCH (d:Driver) "
                "WHERE toLower(d.surname) = toLower($q) "
                "RETURN d.driverId AS id, d.forename + ' ' + d.surname AS name "
                "ORDER BY d.driverId ASC LIMIT 10",
                q=q,
            ).data()
            if len(surname_matches) == 1:
                return {"status": "resolved", "driverId": surname_matches[0]["id"], "name": surname_matches[0]["name"]}
            if len(surname_matches) > 1:
                return {
                    "status": "ambiguous",
                    "query": q,
                    "candidates": [m["name"] for m in surname_matches],
                }

            ref_matches = s.run(
                "MATCH (d:Driver) "
                "WHERE toLower(d.driverRef) = toLower($q) "
                "RETURN d.driverId AS id, d.forename + ' ' + d.surname AS name "
                "ORDER BY d.driverId ASC LIMIT 5",
                q=q,
            ).data()
            if len(ref_matches) == 1:
                return {"status": "resolved", "driverId": ref_matches[0]["id"], "name": ref_matches[0]["name"]}

            all_drivers = s.run(
                "MATCH (d:Driver) "
                "RETURN d.driverId AS id, d.forename AS forename, d.surname AS surname, d.code AS code"
            ).data()
            norm_q = norm(q)
            full_matches = [
                d for d in all_drivers if norm(f"{d['forename']} {d['surname']}") == norm_q
            ]
            if len(full_matches) == 1:
                d = full_matches[0]
                return {"status": "resolved", "driverId": d["id"], "name": f"{d['forename']} {d['surname']}"}

            sur_matches = [d for d in all_drivers if norm(d["surname"]) == norm_q]
            if len(sur_matches) == 1:
                d = sur_matches[0]
                return {"status": "resolved", "driverId": d["id"], "name": f"{d['forename']} {d['surname']}"}
            if len(sur_matches) > 1:
                return {
                    "status": "ambiguous",
                    "query": q,
                    "candidates": [f"{d['forename']} {d['surname']}" for d in sur_matches],
                }

            if len(q) == 3 and q.isalpha():
                code_matches = [d for d in all_drivers if (d.get("code") or "").upper() == q.upper()]
                if len(code_matches) == 1:
                    d = code_matches[0]
                    return {"status": "resolved", "driverId": d["id"], "name": f"{d['forename']} {d['surname']}"}

        return {"status": "not_found", "query": q}

    def resolve_constructor(self, query_val: Any) -> dict[str, Any]:
        """Resolve constructor from Member 1's `{'id': int, 'name': str}` dict, ID, or alias string."""
        if isinstance(query_val, dict) and "id" in query_val:
            return {
                "status": "resolved",
                "constructorId": int(query_val["id"]),
                "name": str(query_val.get("name", query_val["id"])),
            }
        if isinstance(query_val, int) or (isinstance(query_val, str) and query_val.isdigit()):
            cid = int(query_val)
            with self._session() as s:
                rec = s.run(
                    "MATCH (c:Constructor {constructorId: $cid}) "
                    "RETURN c.constructorId AS id, c.constructor_name AS name LIMIT 1",
                    cid=cid,
                ).single()
                if rec:
                    return {"status": "resolved", "constructorId": rec["id"], "name": rec["name"]}
                return {"status": "not_found", "query": str(query_val)}

        raw_q = str(query_val or "").strip()
        q_key = re.sub(r"[^a-z0-9 ]+", " ", raw_q.lower()).strip()
        canonical = CONSTRUCTOR_ALIASES.get(q_key, raw_q)

        with self._session() as s:
            exact = s.run(
                "MATCH (c:Constructor) "
                "WHERE toLower(c.constructor_name) = toLower($q) "
                "   OR toLower(c.constructorRef) = toLower($q) "
                "RETURN c.constructorId AS id, c.constructor_name AS name "
                "ORDER BY c.constructorId ASC LIMIT 5",
                q=canonical,
            ).data()
            if exact:
                return {"status": "resolved", "constructorId": exact[0]["id"], "name": exact[0]["name"]}

            partial = s.run(
                "MATCH (c:Constructor) "
                "WHERE toLower(c.constructor_name) CONTAINS toLower($q) "
                "RETURN c.constructorId AS id, c.constructor_name AS name "
                "ORDER BY size(c.constructor_name) ASC, c.constructorId ASC LIMIT 5",
                q=canonical,
            ).data()
            if partial:
                return {"status": "resolved", "constructorId": partial[0]["id"], "name": partial[0]["name"]}

        return {"status": "not_found", "query": raw_q}

    def resolve_circuit(self, query_val: Any) -> dict[str, Any]:
        """Resolve circuit from Member 1's `{'id': int, 'name': str}` dict, ID, or alias string."""
        if isinstance(query_val, dict) and "id" in query_val:
            return {
                "status": "resolved",
                "circuitId": int(query_val["id"]),
                "name": str(query_val.get("name", query_val["id"])),
            }
        if isinstance(query_val, int) or (isinstance(query_val, str) and query_val.isdigit()):
            cid = int(query_val)
            with self._session() as s:
                rec = s.run(
                    "MATCH (c:Circuit {circuitId: $cid}) "
                    "RETURN c.circuitId AS id, c.circuit_name AS name LIMIT 1",
                    cid=cid,
                ).single()
                if rec:
                    return {"status": "resolved", "circuitId": rec["id"], "name": rec["name"]}
                return {"status": "not_found", "query": str(query_val)}

        raw_q = str(query_val or "").strip()
        q_low = raw_q.lower()
        if q_low in CIRCUIT_ALIASES:
            return self.resolve_circuit(CIRCUIT_ALIASES[q_low])

        with self._session() as s:
            matches = s.run(
                "MATCH (c:Circuit) "
                "WHERE toLower(c.circuit_name) = toLower($q) "
                "   OR toLower(c.circuitRef) = toLower($q) "
                "   OR toLower(c.circuit_loc) = toLower($q) "
                "   OR toLower(c.circuit_name) CONTAINS toLower($q) "
                "   OR toLower(c.circuit_loc) CONTAINS toLower($q) "
                "RETURN c.circuitId AS id, c.circuit_name AS name "
                "ORDER BY CASE WHEN toLower(c.circuit_name) = toLower($q) THEN 0 "
                "              WHEN toLower(c.circuit_loc) = toLower($q) THEN 1 "
                "              ELSE 2 END, c.circuitId ASC LIMIT 5",
                q=raw_q,
            ).data()
            if matches:
                return {"status": "resolved", "circuitId": matches[0]["id"], "name": matches[0]["name"]}

        return {"status": "not_found", "query": raw_q}

    def resolve_race(self, year: int, race_or_circuit: Any) -> dict[str, Any]:
        """Resolve a specific Race node by (year, Grand Prix name or circuit dict/ID/name)."""
        if isinstance(race_or_circuit, dict) and "id" in race_or_circuit:
            race_or_circuit = int(race_or_circuit["id"])

        if isinstance(race_or_circuit, int):
            with self._session() as s:
                rec = s.run(
                    "MATCH (r:Race {year: $year})-[:AT_CIRCUIT]->(c:Circuit {circuitId: $cid}) "
                    "RETURN r.raceId AS raceId, r.name AS name, r.date AS date, "
                    "       c.circuitId AS circuitId, c.circuit_name AS circuit_name "
                    "ORDER BY r.round DESC LIMIT 1",
                    year=int(year),
                    cid=int(race_or_circuit),
                ).single()
                if rec:
                    return {
                        "status": "resolved",
                        "raceId": rec["raceId"],
                        "name": rec["name"],
                        "date": str(rec["date"]),
                        "circuitId": rec["circuitId"],
                        "circuit_name": rec["circuit_name"],
                    }
                return {"status": "not_found", "query": f"{year} circuitId={race_or_circuit}"}

        q = str(race_or_circuit or "").strip()
        with self._session() as s:
            rec = s.run(
                "MATCH (r:Race {year: $year})-[:AT_CIRCUIT]->(c:Circuit) "
                "WHERE toLower(r.name) = toLower($q) "
                "   OR toLower(r.name) CONTAINS toLower($q) "
                "   OR toLower(c.circuit_name) CONTAINS toLower($q) "
                "   OR toLower(c.circuit_loc) CONTAINS toLower($q) "
                "   OR toLower(c.circuit_country) = toLower($q) "
                "RETURN r.raceId AS raceId, r.name AS name, r.date AS date, "
                "       c.circuitId AS circuitId, c.circuit_name AS circuit_name "
                "ORDER BY CASE WHEN toLower(r.name) = toLower($q) THEN 0 ELSE 1 END, r.round ASC LIMIT 1",
                year=int(year),
                q=q,
            ).single()
            if rec:
                return {
                    "status": "resolved",
                    "raceId": rec["raceId"],
                    "name": rec["name"],
                    "date": str(rec["date"]),
                    "circuitId": rec["circuitId"],
                    "circuit_name": rec["circuit_name"],
                }

        cir_res = self.resolve_circuit(q)
        if cir_res["status"] == "resolved":
            return self.resolve_race(year, int(cir_res["circuitId"]))

        return {"status": "not_found", "query": f"{year} {q}"}

    # ------------------------------------------------------------------
    # Normalize Member 1 Entities OR Flat Entities into Unified Slots
    # ------------------------------------------------------------------

    def _normalize_entity_slots(self, raw_entities: dict[str, Any]) -> dict[str, Any]:
        """Support both Member 1's `entities` dict (`drivers`, `constructors`, `circuits`,
        `grand_prix`, `season`, `season_range`) and flat keys (`driver`, `driver1`, `driver2`,
        `constructor`, `circuit`, `race_name`, `year`, `start_year`, `end_year`, `year1`, `year2`).
        """
        e = dict(raw_entities or {})
        drivers_list = list(e.get("drivers") or [])
        if not drivers_list:
            if e.get("driver") is not None:
                drivers_list.append(e["driver"])
            if e.get("driver1") is not None:
                drivers_list.append(e["driver1"])
            if e.get("driver2") is not None:
                drivers_list.append(e["driver2"])

        constructors_list = list(e.get("constructors") or [])
        if not constructors_list and e.get("constructor") is not None:
            constructors_list.append(e["constructor"])

        circuits_list = list(e.get("circuits") or [])
        if not circuits_list and e.get("circuit") is not None:
            circuits_list.append(e["circuit"])

        gp = e.get("grand_prix") or e.get("race_name")
        season = e.get("season") if e.get("season") is not None else e.get("year")
        s_range = e.get("season_range")
        start_year = e.get("start_year") if e.get("start_year") is not None else e.get("year1")
        end_year = e.get("end_year") if e.get("end_year") is not None else e.get("year2")
        if s_range and len(s_range) == 2:
            if start_year is None:
                start_year = int(s_range[0])
            if end_year is None:
                end_year = int(s_range[1])

        return {
            "drivers": drivers_list,
            "constructors": constructors_list,
            "circuits": circuits_list,
            "grand_prix": gp,
            "country": e.get("country"),
            "year": int(season) if season is not None else None,
            "start_year": int(start_year) if start_year is not None else None,
            "end_year": int(end_year) if end_year is not None else None,
            "before_year": int(e["before_year"]) if e.get("before_year") is not None else None,
            "before_date": str(e["before_date"]) if e.get("before_date") is not None else None,
            "min_starts": int(e["min_starts"]) if e.get("min_starts") is not None else None,
            "min_entries": int(e["min_entries"]) if e.get("min_entries") is not None else None,
            "top_n": int(e["top_n"]) if e.get("top_n") is not None else None,
        }

    # ------------------------------------------------------------------
    # Main Retrieval Execution Entry Point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        intent: str | None,
        entities: dict[str, Any] | None = None,
        question_text: str | None = None,
        current_ex: Extraction | None = None,
        is_follow_up_turn: bool = False,
    ) -> RetrievalResult:
        """Route an intent and extracted entities to the matching parameterized Cypher template,
        execute against Neo4j, and handle empty results or ambiguous/missing entities.
        """
        if intent is None or intent == "clarify":
            return RetrievalResult(
                intent="clarify",
                template_id="none",
                cypher="",
                params={},
                rows=[],
                is_empty=True,
                message=REPHRASE_MESSAGE,
            )

        canon_intent = SUB_INTENT_TO_M1_LABEL.get(intent, intent)
        if canon_intent == "out_of_scope" or canon_intent not in INTENT_CATALOG:
            return RetrievalResult(
                intent="out_of_scope",
                template_id="none",
                cypher="",
                params={},
                rows=[],
                is_empty=True,
                message=OUT_OF_SCOPE_MESSAGE,
            )

        filtered_raw = filter_active_entities(
            intent_label=intent,
            merged_entities=dict(entities or {}),
            current_ex=current_ex,
            is_follow_up_turn=is_follow_up_turn,
            text=question_text,
        )
        slots = self._normalize_entity_slots(filtered_raw)
        q_low = (question_text or "").lower()

        resolved: dict[str, Any] = {}
        ambiguous_candidates: list[str] = []
        ambiguity_query: str | None = None

        # Resolve driver list
        resolved_drivers: list[dict[str, Any]] = []
        for d_item in slots["drivers"]:
            d_res = self.resolve_driver(d_item)
            if d_res["status"] == "resolved":
                resolved_drivers.append(d_res)
            elif d_res["status"] == "ambiguous":
                ambiguity_query = str(d_item)
                ambiguous_candidates = d_res["candidates"]
                return self._ambiguity_response(intent, ambiguity_query, ambiguous_candidates)
            else:
                return self._empty_response(intent, "driver_season_or_career_stats", {"driver": str(d_item)})

        if len(resolved_drivers) == 1:
            resolved["driver"] = resolved_drivers[0]
        elif len(resolved_drivers) >= 2:
            resolved["driver1"] = resolved_drivers[0]
            resolved["driver2"] = resolved_drivers[1]

        # Resolve constructor list
        resolved_constructors: list[dict[str, Any]] = []
        for c_item in slots["constructors"]:
            c_res = self.resolve_constructor(c_item)
            if c_res["status"] == "resolved":
                resolved_constructors.append(c_res)
            else:
                return self._empty_response(intent, "constructor_period_performance", {"constructor": str(c_item)})
        if resolved_constructors:
            resolved["constructor"] = resolved_constructors[0]

        # Resolve circuit if present
        circuit_id: int | None = None
        if slots["circuits"]:
            cir_res = self.resolve_circuit(slots["circuits"][0])
            if cir_res["status"] == "resolved":
                resolved["circuit"] = cir_res
                circuit_id = int(cir_res["circuitId"])
            elif intent != "circuit_info_and_history":
                return self._empty_response(intent, "circuit_profile_and_history", {"circuit": str(slots["circuits"][0])})
        elif slots["grand_prix"] and slots["year"] is None and canon_intent in ("driver_performance", "race_facts", "qualifying_grid"):
            cir_res = self.resolve_circuit(slots["grand_prix"])
            if cir_res["status"] == "resolved":
                resolved["circuit"] = cir_res
                circuit_id = int(cir_res["circuitId"])

        year = slots["year"]
        start_year = slots["start_year"]
        end_year = slots["end_year"]
        before_year = slots["before_year"]
        before_date = slots["before_date"]
        if year is not None:
            resolved["year"] = year

        # ------------------------------------------------------------------
        # 1. driver_performance (includes driver_season_career_performance & driver_circuit_performance)
        # ------------------------------------------------------------------
        if canon_intent == "driver_performance":
            if intent == "driver_circuit_performance" or circuit_id is not None:
                if circuit_id is None:
                    return self._missing_entity_response(intent, ["circuit"])
                if resolved_drivers:
                    params = {
                        "driverId": int(resolved_drivers[0]["driverId"]),
                        "circuitId": circuit_id,
                        "year": year,
                        "start_year": start_year,
                        "end_year": end_year,
                        "before_year": before_year,
                        "before_date": before_date,
                    }
                    return self._run_template(intent, "driver_at_circuit_stats", params, resolved)
                params = {
                    "circuitId": circuit_id,
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                    "before_year": before_year,
                    "min_starts": slots["min_starts"] or 5,
                }
                return self._run_template(intent, "circuit_top_drivers_by_avg_finish", params, resolved)

            if resolved_drivers:
                params = {
                    "driverId": int(resolved_drivers[0]["driverId"]),
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                }
                return self._run_template(intent, "driver_season_or_career_stats", params, resolved)

            if year is not None:
                return self._run_template(intent, "season_driver_leaderboard", {"year": year}, resolved)

            return self._missing_entity_response(intent, ["driver"])

        # ------------------------------------------------------------------
        # 2. constructor_performance
        # ------------------------------------------------------------------
        if canon_intent == "constructor_performance":
            if resolved_constructors:
                c_ids = [int(c["constructorId"]) for c in resolved_constructors]
                # Check if asking "which season brought <team> the most points?"
                if year is None and start_year is None and re.search(r"\b(which|what|best|most)\b.*\bseason\b", q_low):
                    return self._run_template(
                        intent,
                        "constructor_by_season_performance",
                        {"constructorIds": c_ids},
                        resolved,
                    )
                params = {
                    "constructorIds": c_ids,
                    "circuitId": circuit_id,
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                }
                return self._run_template(intent, "constructor_period_performance", params, resolved)

            if any(x is not None for x in (year, start_year, end_year)):
                min_ent = slots["min_entries"] or (100 if (start_year and end_year) else 10)
                params = {
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                    "min_entries": min_ent,
                }
                return self._run_template(intent, "all_constructors_reliability_ranking", params, resolved)

            return self._missing_entity_response(intent, ["constructor"])

        # ------------------------------------------------------------------
        # 3. head_to_head
        # ------------------------------------------------------------------
        if canon_intent == "head_to_head":
            if len(resolved_drivers) < 2:
                return self._missing_entity_response(intent, ["driver1", "driver2"])

            d1_id = int(resolved_drivers[0]["driverId"])
            d2_id = int(resolved_drivers[1]["driverId"])
            con_id = int(resolved_constructors[0]["constructorId"]) if resolved_constructors else None

            params = {
                "driver1Id": d1_id,
                "driver2Id": d2_id,
                "constructorId": con_id,
                "year": year,
                "start_year": start_year,
                "end_year": end_year,
            }
            return self._run_template(intent, "teammate_race_head_to_head", params, resolved)

        # ------------------------------------------------------------------
        # 4. race_facts (includes race_results_and_podium & circuit_info_and_history)
        # ------------------------------------------------------------------
        if canon_intent == "race_facts":
            race_target = slots["grand_prix"] or (resolved.get("circuit") or {}).get("circuitId") or (slots["circuits"][0] if slots["circuits"] else None)
            if year is not None and race_target is not None and intent != "circuit_info_and_history":
                race_res = self.resolve_race(year, race_target)
                if race_res["status"] != "resolved":
                    return self._empty_response(intent, "race_podium_and_results", {"year": year, "race": str(race_target)})
                resolved["race"] = race_res
                params = {
                    "raceId": int(race_res["raceId"]),
                    "top_n": slots["top_n"] or 3,
                }
                return self._run_template(intent, "race_podium_and_results", params, resolved)

            if circuit_id is not None:
                if start_year is not None and end_year is not None and re.search(r"\b(winners?|won|victories|champions?)\b", q_low):
                    params = {
                        "circuitId": circuit_id,
                        "start_year": start_year,
                        "end_year": end_year,
                    }
                    return self._run_template(intent, "circuit_winners_in_period", params, resolved)
                params = {
                    "circuitId": circuit_id,
                    "start_year": start_year,
                    "end_year": end_year,
                }
                return self._run_template(intent, "circuit_profile_and_history", params, resolved)

            country_val = slots["country"] or (str(slots["circuits"][0]) if slots["circuits"] else None)
            if country_val:
                return self._run_template(intent, "circuits_by_country", {"country": country_val}, resolved)

            return self._missing_entity_response(intent, ["grand_prix", "year"])

        # ------------------------------------------------------------------
        # 5. qualifying_grid (includes qualifying_poles_and_results & grid_vs_finish_gains)
        # ------------------------------------------------------------------
        if canon_intent == "qualifying_grid":
            race_target = slots["grand_prix"] or circuit_id
            is_gain_q = (
                intent == "grid_vs_finish_gains"
                or bool(re.search(r"\b(gain\w*|made up|make up|climber\w*|places from the grid)\b", q_low))
            )
            is_loss_q = bool(re.search(r"\b(lost|lose|dropp?\w*)\b.*\b(places|positions|spots)\b", q_low))
            is_front_row_conv = bool(re.search(r"\bfront[- ]row\b.*\b(conversion|podium)\b", q_low))

            if is_front_row_conv and circuit_id is not None:
                params = {
                    "circuitId": circuit_id,
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                }
                return self._run_template(intent, "circuit_front_row_to_podium_conversion", params, resolved)

            if year is not None and race_target is not None:
                race_res = self.resolve_race(year, race_target)
                if race_res["status"] != "resolved":
                    return self._empty_response(intent, "qualifying_race_results", {"year": year, "race": str(race_target)})
                resolved["race"] = race_res
                drv_id = int(resolved_drivers[0]["driverId"]) if resolved_drivers else None

                if is_loss_q:
                    return self._run_template(
                        intent,
                        "race_grid_to_finish_losses",
                        {"raceId": int(race_res["raceId"]), "driverId": drv_id},
                        resolved,
                    )
                if is_gain_q:
                    return self._run_template(
                        intent,
                        "race_grid_to_finish_gains",
                        {"raceId": int(race_res["raceId"]), "driverId": drv_id},
                        resolved,
                    )
                return self._run_template(
                    intent,
                    "qualifying_race_results",
                    {"raceId": int(race_res["raceId"]), "driverId": drv_id, "top_n": slots["top_n"] or 10},
                    resolved,
                )

            if resolved_drivers:
                params = {
                    "driverId": int(resolved_drivers[0]["driverId"]),
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                    "circuitId": circuit_id,
                }
                return self._run_template(intent, "driver_qualifying_stats", params, resolved)

            if resolved_constructors:
                params = {
                    "constructorId": int(resolved_constructors[0]["constructorId"]),
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                    "circuitId": circuit_id,
                }
                return self._run_template(intent, "constructor_qualifying_stats", params, resolved)

            if circuit_id is not None:
                params = {
                    "circuitId": circuit_id,
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                }
                return self._run_template(intent, "circuit_front_row_to_podium_conversion", params, resolved)

            return self._missing_entity_response(intent, ["driver", "race_name", "year"])

        # ------------------------------------------------------------------
        # 6. pit_stops (includes pit_stop_strategy)
        # ------------------------------------------------------------------
        if canon_intent == "pit_stops":
            race_target = slots["grand_prix"] or circuit_id
            is_slowest = bool(re.search(r"\b(slowest|longest)\b", q_low))
            is_avg_race = bool(re.search(r"\b(on average|typical|average number of stops)\b", q_low)) and not resolved_drivers

            if year is not None and race_target is not None:
                race_res = self.resolve_race(year, race_target)
                if race_res["status"] != "resolved":
                    return self._empty_response(intent, "race_fastest_pit_stops", {"year": year, "race": str(race_target)})
                resolved["race"] = race_res
                drv_id = int(resolved_drivers[0]["driverId"]) if resolved_drivers else None

                if is_avg_race:
                    return self._run_template(
                        intent,
                        "race_pit_stop_averages",
                        {"raceId": int(race_res["raceId"])},
                        resolved,
                    )
                if is_slowest:
                    return self._run_template(
                        intent,
                        "race_slowest_pit_stops",
                        {"raceId": int(race_res["raceId"]), "driverId": drv_id},
                        resolved,
                    )
                return self._run_template(
                    intent,
                    "race_fastest_pit_stops",
                    {"raceId": int(race_res["raceId"]), "driverId": drv_id},
                    resolved,
                )

            if resolved_drivers:
                params = {
                    "driverId": int(resolved_drivers[0]["driverId"]),
                    "year": year,
                    "start_year": start_year,
                    "end_year": end_year,
                    "circuitId": circuit_id,
                }
                return self._run_template(intent, "driver_season_pit_stop_stats", params, resolved)

            if year is not None:
                return self._run_template(intent, "season_pit_stop_summary", {"year": year}, resolved)

            return self._missing_entity_response(intent, ["race_name", "year", "driver"])

        # ------------------------------------------------------------------
        # 7. season_history (includes season_history_and_transfers)
        # ------------------------------------------------------------------
        if canon_intent == "season_history":
            is_transfer_q = (
                intent == "season_history_and_transfers" and start_year is not None and end_year is not None
            ) or bool(re.search(r"\b(changed|switched|moved|swapped|transfers?|new team)\b", q_low))

            if is_transfer_q:
                if start_year is not None and end_year is not None:
                    y1, y2 = start_year, end_year
                elif year is not None:
                    y1, y2 = year - 1, year
                else:
                    return self._missing_entity_response(intent, ["year1", "year2"])
                resolved["year1"], resolved["year2"] = y1, y2
                return self._run_template(
                    intent,
                    "driver_team_changes_between_seasons",
                    {"year1": y1, "year2": y2},
                    resolved,
                )

            if start_year is not None and end_year is not None:
                if end_year - start_year == 1 and intent == "season_history_and_transfers":
                    return self._run_template(
                        intent,
                        "driver_team_changes_between_seasons",
                        {"year1": start_year, "year2": end_year},
                        resolved,
                    )
                return self._run_template(
                    intent,
                    "period_points_leaders",
                    {"start_year": start_year, "end_year": end_year},
                    resolved,
                )

            if year is not None:
                return self._run_template(intent, "season_overview_and_leader", {"year": year}, resolved)

            return self._missing_entity_response(intent, ["year"])

        # ------------------------------------------------------------------
        # 8. driver_profile (Member 1's extra intent)
        # ------------------------------------------------------------------
        if canon_intent == "driver_profile":
            if not resolved_drivers:
                return self._missing_entity_response(intent, ["driver"])
            params = {"driverId": int(resolved_drivers[0]["driverId"])}
            return self._run_template(intent, "driver_profile_and_career_summary", params, resolved)

        # ------------------------------------------------------------------
        # 9. podium_prediction (Pre-race graph fact retrieval)
        # ------------------------------------------------------------------
        if canon_intent == "podium_prediction":
            if not resolved_drivers:
                return self._missing_entity_response(intent, ["driver"])
            yr = year or 2024
            race_target = slots["grand_prix"] or circuit_id or "Bahrain Grand Prix"
            race_res = self.resolve_race(yr, race_target)
            if race_res["status"] == "resolved":
                resolved["race"] = race_res
                params = {
                    "driverId": int(resolved_drivers[0]["driverId"]),
                    "circuitId": int(race_res["circuitId"]),
                    "before_date": race_res["date"],
                }
                return self._run_template(intent, "pre_race_driver_circuit_context", params, resolved)
            return self._empty_response(
                intent,
                "pre_race_driver_circuit_context",
                {"driverId": int(resolved_drivers[0]["driverId"]), "year": yr, "race": str(race_target)},
            )

        return self._empty_response(intent, "none", dict(entities or {}))

    def fetch_pre_race_graph_facts(
        self,
        driver_id: int,
        circuit_id: int,
        constructor_id: int | None,
        before_date: str,
        year: int,
    ) -> dict[str, Any]:
        """Execute parameterized Cypher queries to retrieve pre-race Knowledge Graph facts
        strictly before `before_date` (`r.date < date($before_date)`) for Section 4.c explanations.
        """
        driver_circuit_params = {
            "driverId": int(driver_id),
            "circuitId": int(circuit_id),
            "before_date": str(before_date),
        }
        driver_circuit_q = CYPHER_TEMPLATES["pre_race_driver_circuit_context"].strip()

        con_q = CYPHER_TEMPLATES["pre_race_constructor_context"].strip()
        con_params = {
            "constructorId": int(constructor_id) if constructor_id is not None else -1,
            "before_date": str(before_date),
            "year": int(year),
        }

        with self._session() as s:
            drv_rows = s.run(driver_circuit_q, **driver_circuit_params).data()
            con_rows = s.run(con_q, **con_params).data() if constructor_id is not None else []

        if drv_rows and drv_rows[0].get("Prior_Circuit_Starts", 0) == 0:
            drv_rows = []
        if con_rows and con_rows[0].get("Prior_Constructor_Entries_Recent", 0) == 0:
            con_rows = []

        return {
            "before_date_cutoff": str(before_date),
            "driver_circuit_cypher": driver_circuit_q,
            "driver_circuit_params": driver_circuit_params,
            "driver_circuit_facts": drv_rows[0] if drv_rows else None,
            "constructor_cypher": con_q if constructor_id is not None else None,
            "constructor_params": con_params if constructor_id is not None else None,
            "constructor_facts": con_rows[0] if con_rows else None,
        }

    # ------------------------------------------------------------------
    # Subgraph Extraction for UI Trace (Section 5.3)
    # ------------------------------------------------------------------

    def _build_subgraph_trace(
        self,
        intent: str,
        resolved: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            return {"nodes": [], "edges": [], "relationship_statements": []}

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        statements: list[str] = []
        seen_nodes: set[str] = set()

        def _add_node(node_id: str, label: str, display_name: str, props: dict[str, Any] | None = None) -> None:
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "name": display_name,
                    "properties": props or {},
                })

        def _add_edge(src_id: str, src_label: str, src_name: str, rel: str, dst_id: str, dst_label: str, dst_name: str) -> None:
            edges.append({
                "source": src_id,
                "target": dst_id,
                "relationship": rel,
                "source_label": src_label,
                "target_label": dst_label,
            })
            stmt = f"({src_label} {{name: '{src_name}'}})-[:{rel}]->({dst_label} {{name: '{dst_name}'}})"
            if stmt not in statements:
                statements.append(stmt)

        if "driver" in resolved:
            d = resolved["driver"]
            _add_node(f"Driver:{d['driverId']}", "Driver", d["name"], {"driverId": d["driverId"]})
        if "driver1" in resolved and "driver2" in resolved:
            d1, d2 = resolved["driver1"], resolved["driver2"]
            _add_node(f"Driver:{d1['driverId']}", "Driver", d1["name"], {"driverId": d1["driverId"]})
            _add_node(f"Driver:{d2['driverId']}", "Driver", d2["name"], {"driverId": d2["driverId"]})
        if "constructor" in resolved:
            c = resolved["constructor"]
            _add_node(f"Constructor:{c['constructorId']}", "Constructor", c["name"], {"constructorId": c["constructorId"]})
        if "circuit" in resolved:
            cir = resolved["circuit"]
            _add_node(f"Circuit:{cir['circuitId']}", "Circuit", cir["name"], {"circuitId": cir["circuitId"]})
        if "race" in resolved:
            rc = resolved["race"]
            r_node = f"Race:{rc['raceId']}"
            _add_node(r_node, "Race", f"{rc.get('year', '')} {rc['name']}".strip(), {"raceId": rc["raceId"]})
            if "circuitId" in rc:
                c_node = f"Circuit:{rc['circuitId']}"
                _add_node(c_node, "Circuit", rc["circuit_name"], {"circuitId": rc["circuitId"]})
                _add_edge(r_node, "Race", rc["name"], "AT_CIRCUIT", c_node, "Circuit", rc["circuit_name"])

        for idx, row in enumerate(rows[:5]):
            fact_label = "Result"
            if "Qualifying_Position" in row or "Qualifying_Poles" in row:
                fact_label = "Qualifying"
            elif "Stop_Number" in row or "Total_Pit_Stops" in row:
                fact_label = "PitStop"

            fact_id = f"{fact_label}:row_{idx + 1}"
            _add_node(fact_id, fact_label, f"{fact_label} #{idx + 1}", row)

            drv_name = row.get("Driver") or row.get("Driver_1") or row.get("Winner") or (resolved.get("driver") or {}).get("name")
            if drv_name:
                d_id = f"Driver:{drv_name}"
                _add_node(d_id, "Driver", str(drv_name))
                _add_edge(fact_id, fact_label, f"{fact_label} #{idx + 1}", "BY_DRIVER", d_id, "Driver", str(drv_name))

            if "Driver_2" in row:
                d2_name = str(row["Driver_2"])
                d2_id = f"Driver:{d2_name}"
                _add_node(d2_id, "Driver", d2_name)
                _add_edge(fact_id, fact_label, f"{fact_label} #{idx + 1}", "BY_DRIVER", d2_id, "Driver", d2_name)

            con_name = row.get("Constructor") or (resolved.get("constructor") or {}).get("name")
            if con_name and fact_label in ("Result", "Qualifying"):
                c_id = f"Constructor:{con_name}"
                _add_node(c_id, "Constructor", str(con_name))
                _add_edge(fact_id, fact_label, f"{fact_label} #{idx + 1}", "FOR_CONSTRUCTOR", c_id, "Constructor", str(con_name))

            gp_name = row.get("Grand_Prix") or (resolved.get("race") or {}).get("name")
            yr = row.get("Season") or resolved.get("year")
            if gp_name:
                r_display = f"{yr} {gp_name}".strip() if yr else str(gp_name)
                r_id = f"Race:{r_display}"
                _add_node(r_id, "Race", r_display)
                _add_edge(fact_id, fact_label, f"{fact_label} #{idx + 1}", "FOR_RACE", r_id, "Race", r_display)
                if yr:
                    s_id = f"Season:{yr}"
                    _add_node(s_id, "Season", str(yr))
                    _add_edge(r_id, "Race", r_display, "IN_SEASON", s_id, "Season", str(yr))

            cir_name = row.get("Circuit") or (resolved.get("circuit") or {}).get("name")
            if cir_name:
                cir_id = f"Circuit:{cir_name}"
                _add_node(cir_id, "Circuit", str(cir_name))
                if gp_name:
                    r_display = f"{yr} {gp_name}".strip() if yr else str(gp_name)
                    _add_edge(f"Race:{r_display}", "Race", r_display, "AT_CIRCUIT", cir_id, "Circuit", str(cir_name))

        return {
            "nodes": nodes,
            "edges": edges,
            "relationship_statements": statements,
        }

    # ------------------------------------------------------------------
    # Internal Execution & Empty-Result / Ambiguity Helpers
    # ------------------------------------------------------------------

    def _run_template(
        self,
        intent: str,
        template_id: str,
        params: dict[str, Any],
        resolved: dict[str, Any],
    ) -> RetrievalResult:
        cypher = CYPHER_TEMPLATES[template_id].strip()
        with self._session() as s:
            rows = s.run(cypher, **params).data()

        is_empty = len(rows) == 0
        if not is_empty and len(rows) == 1:
            row0 = rows[0]
            for zero_key in (
                "Entries",
                "Starts",
                "Shared_Races",
                "Races_Entered",
                "Total_Pit_Stops",
                "Front_Row_Starts",
                "Prior_Circuit_Starts",
                "Races_With_Data",
                "Drivers_Who_Pitted",
            ):
                if zero_key in row0 and (row0[zero_key] is None or row0[zero_key] == 0):
                    is_empty = True
                    rows = []
                    break
            if "Drivers_Changed" in row0 and row0["Drivers_Changed"] == 0:
                is_empty = True
                rows = []

        if is_empty:
            return RetrievalResult(
                intent=intent,
                template_id=template_id,
                cypher=cypher,
                params=params,
                rows=[],
                is_empty=True,
                message=EMPTY_RESULT_MESSAGE,
                resolved_entities=resolved,
                is_graphical=False,
            )

        subgraph = self._build_subgraph_trace(intent, resolved, rows)
        return RetrievalResult(
            intent=intent,
            template_id=template_id,
            cypher=cypher,
            params=params,
            rows=rows,
            is_empty=False,
            message=None,
            resolved_entities=resolved,
            is_graphical=len(subgraph["nodes"]) > 0,
            subgraph=subgraph,
        )

    def _empty_response(self, intent: str, template_id: str, params: dict[str, Any]) -> RetrievalResult:
        cypher = CYPHER_TEMPLATES.get(template_id, "").strip()
        return RetrievalResult(
            intent=intent,
            template_id=template_id,
            cypher=cypher,
            params=params,
            rows=[],
            is_empty=True,
            message=EMPTY_RESULT_MESSAGE,
        )

    def _ambiguity_response(self, intent: str, query_name: str | None, candidates: list[str]) -> RetrievalResult:
        cand_list = ", ".join(candidates)
        return RetrievalResult(
            intent=intent,
            template_id="none",
            cypher="",
            params={},
            rows=[],
            is_empty=True,
            ambiguity_prompt=(
                f'"{query_name}" matches more than one driver: '
                f"{cand_list}. Which one do you mean?"
            ),
            ambiguous_candidates=candidates,
        )

    def _missing_entity_response(self, intent: str, missing: list[str]) -> RetrievalResult:
        needed = ", ".join(missing)
        return RetrievalResult(
            intent=intent,
            template_id="none",
            cypher="",
            params={},
            rows=[],
            is_empty=True,
            missing_entities=missing,
            message=f"Could you please specify the missing information ({needed}) for your question?",
        )


# ---------------------------------------------------------------------------
# End-to-End Member 1 + Member 2 Stateful Session Integration
# ---------------------------------------------------------------------------

class PipelineSession:
    """Connects Member 1 (`DEPLOYED` intent classifier + `extract`, `merge`, `clarification`,
    `resolve_choice`) directly to Member 2 (`GraphRetriever` and `PodiumPredictor`).
    """

    def __init__(
        self,
        classifier: Any | None = None,
        retriever: GraphRetriever | None = None,
        predictor: Any | None = None,
    ) -> None:
        self.clf = classifier or DEPLOYED()
        self.retriever = retriever or GraphRetriever()
        self.predictor = predictor
        self.cache_intent: str | None = None
        self.cache_entities: dict[str, Any] = {}
        self.pending_ambiguity: dict[str, Any] | None = None
        self.pending_question: str | None = None

    def step(self, message: str) -> dict[str, Any]:
        """Process one user message through Member 1 and Member 2, maintaining session cache."""
        # 1. Handle reply to a pending disambiguation question (e.g. "Michael")
        if self.pending_ambiguity is not None:
            chosen = resolve_choice(message, self.pending_ambiguity["candidates"])
            if chosen is not None:
                typ = self.pending_ambiguity["type"]
                slot = f"{typ}s" if typ in ("driver", "constructor", "circuit") else typ
                update_dict = {slot: [chosen] if slot.endswith("s") else chosen}
                self.cache_entities = merge(self.cache_entities, update_dict)
                intent_label = self.cache_intent
                orig_q = self.pending_question or message
                self.pending_ambiguity = None
                self.pending_question = None
                retrieval_res = self.retriever.retrieve(
                    intent_label,
                    self.cache_entities,
                    question_text=orig_q,
                    current_ex=None,
                    is_follow_up_turn=True,
                )
                return {
                    "status": "resolved_ambiguity",
                    "intent": Intent(intent_label, "follow_up", reason="disambiguation resolved"),
                    "entities": dict(self.cache_entities),
                    "clarification": None,
                    "retrieval": retrieval_res,
                }
            # User asked a brand new question instead of answering the disambiguation
            self.pending_ambiguity = None
            self.pending_question = None

        # 2. Classify intent via Member 1
        intent_obj: Intent = self.clf.classify(message, prev=self.cache_intent)
        ex: Extraction = extract(message)

        # Out-of-scope does not clear the session cache (Brief Section 1.b)
        if intent_obj.status == "out_of_scope" or intent_obj.label == "out_of_scope":
            return {
                "status": "out_of_scope",
                "intent": intent_obj,
                "entities": dict(self.cache_entities),
                "clarification": None,
                "retrieval": RetrievalResult(
                    intent="out_of_scope",
                    template_id="none",
                    cypher="",
                    params={},
                    rows=[],
                    is_empty=True,
                    message=OUT_OF_SCOPE_MESSAGE,
                ),
            }

        if intent_obj.status == "clarify" or intent_obj.label is None:
            return {
                "status": "clarify_intent",
                "intent": intent_obj,
                "entities": dict(self.cache_entities),
                "clarification": REPHRASE_MESSAGE,
                "retrieval": None,
            }

        # 3. Update entity cache and check Member 1's clarification gate
        merged = merge(self.cache_entities, ex.entities)
        active = filter_active_entities(
            intent_label=intent_obj.label,
            merged_entities=merged,
            current_ex=ex,
            is_follow_up_turn=(intent_obj.status == "follow_up"),
            text=message,
        )
        ask = clarification(intent_obj.label, ex, active)
        self.cache_intent = intent_obj.label
        self.cache_entities = active

        if ask is not None:
            if ex.ambiguous:
                self.pending_ambiguity = ex.ambiguous[0]
                self.pending_question = message
            return {
                "status": "clarify_entity",
                "intent": intent_obj,
                "entities": dict(self.cache_entities),
                "clarification": ask,
                "ambiguous": ex.ambiguous,
                "retrieval": RetrievalResult(
                    intent=intent_obj.label,
                    template_id="none",
                    cypher="",
                    params={},
                    rows=[],
                    is_empty=True,
                    message=ask,
                    ambiguity_prompt=ask if ex.ambiguous else None,
                    ambiguous_candidates=[c["name"] for c in ex.ambiguous[0]["candidates"]] if ex.ambiguous else [],
                ),
            }

        # 4. Execute Member 2 Graph Retrieval (and Podium Prediction if intent == 'podium_prediction')
        retrieval_res = self.retriever.retrieve(
            intent_obj.label,
            self.cache_entities,
            question_text=message,
            current_ex=ex,
            is_follow_up_turn=(intent_obj.status == "follow_up"),
        )

        podium_res = None
        if intent_obj.label == "podium_prediction" and self.predictor is not None:
            podium_res = self.predictor.predict_and_explain(entities=self.cache_entities)

        return {
            "status": "empty" if retrieval_res.is_empty else "ok",
            "intent": intent_obj,
            "entities": dict(self.cache_entities),
            "clarification": None,
            "retrieval": retrieval_res,
            "podium": podium_res,
        }


if __name__ == "__main__":
    try:
        from .intents import RuleClassifier
    except ImportError:
        from intents import RuleClassifier  # type: ignore

    sess = PipelineSession(classifier=RuleClassifier())
    try:
        # 1. Grounded multi-hop query with circuit + cutoff
        t1 = sess.step("How many wins did Lewis Hamilton have at Silverstone before 2020?")
        assert t1["status"] == "ok" and t1["retrieval"].rows[0]["Wins"] == 6, t1

        # 2. Pronoun follow-up that names a new season (must keep cached driver Hamilton, must NOT filter by Silverstone)
        t2 = sess.step("How many points did he score in 2019?")
        assert t2["status"] == "ok" and t2["retrieval"].rows[0]["Points"] == 413.0, t2

        # 3. Elliptical follow-up ("And in 2020?") keeps driver Hamilton and replaces season with 2020
        t3 = sess.step("And in 2020?")
        assert t3["status"] == "ok" and t3["retrieval"].rows[0]["Points"] == 347.0, t3

        # 4. Out-of-scope question does NOT clear the cache
        t4 = sess.step("What is Hamilton's salary?")
        assert t4["status"] == "out_of_scope" and sess.cache_entities["drivers"][0]["id"] == 1, t4

        # 5. Ambiguous driver surname -> clarification turn -> resolve_choice("Michael")
        t5 = sess.step("How many wins and podiums did Schumacher get in his career?")
        assert t5["status"] == "clarify_entity" and "Schumacher" in t5["clarification"], t5
        t6 = sess.step("Michael")
        assert t6["status"] == "resolved_ambiguity" and t6["retrieval"].rows[0]["Wins"] == 91, t6

        # 6. Empty result
        t7 = sess.step("How many points did Lewis Hamilton score in 1955?")
        assert t7["status"] == "empty" and t7["retrieval"].message == EMPTY_RESULT_MESSAGE, t7

        # 7. Member 1's extra intent: driver_profile
        t8 = sess.step("Which teams has Fernando Alonso driven for?")
        assert t8["status"] == "ok" and "Ferrari" in t8["retrieval"].rows[0]["Teams_Driven_For"], t8
    finally:
        sess.retriever.close()
    print("retrieval.py: all checks passed")
