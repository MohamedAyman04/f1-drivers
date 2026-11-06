"""Create the Formula 1 knowledge graph in Neo4j from the nine source CSVs."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

import neo4j

CSV_FILES = (
    "drivers.csv",
    "constructors.csv",
    "circuits.csv",
    "seasons.csv",
    "status.csv",
    "races.csv",
    "results.csv",
    "qualifying.csv",
    "pit_stops.csv",
)

BATCH_SIZE = 1000

CONSTRAINTS = (
    ("driver_id_unique", "Driver", "driverId"),
    ("constructor_id_unique", "Constructor", "constructorId"),
    ("circuit_id_unique", "Circuit", "circuitId"),
    ("season_year_unique", "Season", "year"),
    ("status_id_unique", "Status", "statusId"),
    ("race_id_unique", "Race", "raceId"),
    ("result_id_unique", "Result", "resultId"),
    ("qualifying_id_unique", "Qualifying", "qualifyId"),
    ("pitstop_event_id_unique", "PitStop", "eventId"),
)

DRIVER_QUERY = """
UNWIND $rows AS row
MERGE (n:Driver {driverId: row.props.driverId})
SET n += row.props
"""

CONSTRUCTOR_QUERY = """
UNWIND $rows AS row
MERGE (n:Constructor {constructorId: row.props.constructorId})
SET n += row.props
"""

CIRCUIT_QUERY = """
UNWIND $rows AS row
MERGE (n:Circuit {circuitId: row.props.circuitId})
SET n += row.props
"""

SEASON_QUERY = """
UNWIND $rows AS row
MERGE (n:Season {year: row.props.year})
SET n += row.props
"""

STATUS_QUERY = """
UNWIND $rows AS row
MERGE (n:Status {statusId: row.props.statusId})
SET n += row.props
"""

RACE_QUERY = """
UNWIND $rows AS row
MERGE (n:Race {raceId: row.props.raceId})
SET n += row.props
WITH n, row
MATCH (c:Circuit {circuitId: row.circuitId})
MERGE (n)-[:AT_CIRCUIT]->(c)
WITH n, row
MATCH (s:Season {year: row.year})
MERGE (n)-[:IN_SEASON]->(s)
"""

RESULT_QUERY = """
UNWIND $rows AS row
MERGE (n:Result {resultId: row.props.resultId})
SET n += row.props
WITH n, row
MATCH (r:Race {raceId: row.raceId})
MERGE (n)-[:FOR_RACE]->(r)
WITH n, row
MATCH (d:Driver {driverId: row.driverId})
MERGE (n)-[:BY_DRIVER]->(d)
WITH n, row
MATCH (c:Constructor {constructorId: row.constructorId})
MERGE (n)-[:FOR_CONSTRUCTOR]->(c)
WITH n, row
MATCH (s:Status {statusId: row.statusId})
MERGE (n)-[:HAS_STATUS]->(s)
"""

QUALIFYING_QUERY = """
UNWIND $rows AS row
MERGE (n:Qualifying {qualifyId: row.props.qualifyId})
SET n += row.props
WITH n, row
MATCH (r:Race {raceId: row.raceId})
MERGE (n)-[:FOR_RACE]->(r)
WITH n, row
MATCH (d:Driver {driverId: row.driverId})
MERGE (n)-[:BY_DRIVER]->(d)
WITH n, row
MATCH (c:Constructor {constructorId: row.constructorId})
MERGE (n)-[:FOR_CONSTRUCTOR]->(c)
"""

PITSTOP_QUERY = """
UNWIND $rows AS row
MERGE (n:PitStop {eventId: row.props.eventId})
SET n += row.props
WITH n, row
MATCH (r:Race {raceId: row.raceId})
MERGE (n)-[:FOR_RACE]->(r)
WITH n, row
MATCH (d:Driver {driverId: row.driverId})
MERGE (n)-[:BY_DRIVER]->(d)
"""


def text_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value == "\\N":
        return None
    return value


def int_value(value: str | None) -> int | None:
    text = text_value(value)
    return None if text is None else int(text)


def float_value(value: str | None) -> float | None:
    text = text_value(value)
    return None if text is None else float(text)


def date_value(value: str | None) -> date | None:
    text = text_value(value)
    return None if text is None else date.fromisoformat(text)


def put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def required_int(value: str | None, field: str) -> int:
    result = int_value(value)
    if result is None:
        raise ValueError(f"missing required value for {field}")
    return result


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def driver_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "driverId", required_int(row.get("driverId"), "driverId"))
        put(props, "driverRef", text_value(row.get("driverRef")))
        put(props, "number", int_value(row.get("number")))
        put(props, "code", text_value(row.get("code")))
        put(props, "forename", text_value(row.get("forename")))
        put(props, "surname", text_value(row.get("surname")))
        put(props, "dob", date_value(row.get("dob")))
        put(props, "nationality", text_value(row.get("nationality")))
        yield {"props": props}


def constructor_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "constructorId", required_int(row.get("constructorId"), "constructorId"))
        put(props, "constructorRef", text_value(row.get("constructorRef")))
        put(props, "constructor_name", text_value(row.get("name")))
        put(props, "constructor_nationality", text_value(row.get("nationality")))
        yield {"props": props}


def circuit_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "circuitId", required_int(row.get("circuitId"), "circuitId"))
        put(props, "circuitRef", text_value(row.get("circuitRef")))
        put(props, "circuit_name", text_value(row.get("name")))
        put(props, "circuit_loc", text_value(row.get("location")))
        put(props, "circuit_country", text_value(row.get("country")))
        put(props, "lat", float_value(row.get("lat")))
        put(props, "lng", float_value(row.get("lng")))
        put(props, "alt", int_value(row.get("alt")))
        yield {"props": props}


def season_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "year", required_int(row.get("year"), "year"))
        yield {"props": props}


def status_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "statusId", required_int(row.get("statusId"), "statusId"))
        put(props, "status", text_value(row.get("status")))
        yield {"props": props}


def race_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        year = required_int(row.get("year"), "year")
        props: dict[str, Any] = {}
        put(props, "raceId", required_int(row.get("raceId"), "raceId"))
        put(props, "year", year)
        put(props, "round", int_value(row.get("round")))
        put(props, "name", text_value(row.get("name")))
        put(props, "date", date_value(row.get("date")))
        put(props, "time", text_value(row.get("time")))
        yield {
            "props": props,
            "circuitId": required_int(row.get("circuitId"), "circuitId"),
            "year": year,
        }


def result_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "resultId", required_int(row.get("resultId"), "resultId"))
        put(props, "number", int_value(row.get("number")))
        put(props, "grid", int_value(row.get("grid")))
        put(props, "position", int_value(row.get("position")))
        put(props, "positionText", text_value(row.get("positionText")))
        put(props, "positionOrder", int_value(row.get("positionOrder")))
        put(props, "points", float_value(row.get("points")))
        put(props, "laps", int_value(row.get("laps")))
        put(props, "time", text_value(row.get("time")))
        put(props, "milliseconds", int_value(row.get("milliseconds")))
        put(props, "fastestLap", int_value(row.get("fastestLap")))
        put(props, "rank", int_value(row.get("rank")))
        put(props, "fastestLapTime", text_value(row.get("fastestLapTime")))
        put(props, "fastestLapSpeed", float_value(row.get("fastestLapSpeed")))
        yield {
            "props": props,
            "raceId": required_int(row.get("raceId"), "raceId"),
            "driverId": required_int(row.get("driverId"), "driverId"),
            "constructorId": required_int(row.get("constructorId"), "constructorId"),
            "statusId": required_int(row.get("statusId"), "statusId"),
        }


def qualifying_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        props: dict[str, Any] = {}
        put(props, "qualifyId", required_int(row.get("qualifyId"), "qualifyId"))
        put(props, "number", int_value(row.get("number")))
        put(props, "position", int_value(row.get("position")))
        put(props, "q1", text_value(row.get("q1")))
        put(props, "q2", text_value(row.get("q2")))
        put(props, "q3", text_value(row.get("q3")))
        yield {
            "props": props,
            "raceId": required_int(row.get("raceId"), "raceId"),
            "driverId": required_int(row.get("driverId"), "driverId"),
            "constructorId": required_int(row.get("constructorId"), "constructorId"),
        }


def pitstop_records(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_rows(path):
        race_id = required_int(row.get("raceId"), "raceId")
        driver_id = required_int(row.get("driverId"), "driverId")
        stop = required_int(row.get("stop"), "stop")
        props: dict[str, Any] = {}
        put(props, "eventId", f"{race_id}:{driver_id}:{stop}")
        put(props, "stop", stop)
        put(props, "lap", int_value(row.get("lap")))
        put(props, "time", text_value(row.get("time")))
        put(props, "duration", text_value(row.get("duration")))
        put(props, "milliseconds", int_value(row.get("milliseconds")))
        yield {
            "props": props,
            "raceId": race_id,
            "driverId": driver_id,
        }


def find_data_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = (
        script_dir,
        script_dir / "data" / "raw",
        Path.cwd(),
        Path.cwd() / "data" / "raw",
    )
    checked: list[Path] = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if all((candidate / name).is_file() for name in CSV_FILES):
            return candidate
    listing = "\n".join(f"  - {path}" for path in checked)
    raise SystemExit(f"Could not find all nine CSV files. Checked:\n{listing}")


def find_config() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = (script_dir / "config.txt", Path.cwd() / "config.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Could not find config.txt next to the script or in the current directory.")


def read_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip().upper()] = value.strip()

    for key in ("URI", "USERNAME", "PASSWORD"):
        if not config.get(key):
            raise SystemExit(f"Missing {key} in {path}")
    return config


def create_constraints(session: neo4j.Session) -> None:
    for name, label, property_name in CONSTRAINTS:
        session.run(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        ).consume()


def write_batches(session: neo4j.Session, query: str, rows: Iterable[dict[str, Any]]) -> int:
    batch: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            session.run(query, rows=batch).consume()
            total += len(batch)
            batch.clear()
    if batch:
        session.run(query, rows=batch).consume()
        total += len(batch)
    return total


def build_graph(session: neo4j.Session, data_dir: Path) -> None:
    create_constraints(session)

    loaders = (
        ("Driver", DRIVER_QUERY, driver_records(data_dir / "drivers.csv")),
        ("Constructor", CONSTRUCTOR_QUERY, constructor_records(data_dir / "constructors.csv")),
        ("Circuit", CIRCUIT_QUERY, circuit_records(data_dir / "circuits.csv")),
        ("Season", SEASON_QUERY, season_records(data_dir / "seasons.csv")),
        ("Status", STATUS_QUERY, status_records(data_dir / "status.csv")),
        ("Race", RACE_QUERY, race_records(data_dir / "races.csv")),
        ("Result", RESULT_QUERY, result_records(data_dir / "results.csv")),
        ("Qualifying", QUALIFYING_QUERY, qualifying_records(data_dir / "qualifying.csv")),
        ("PitStop", PITSTOP_QUERY, pitstop_records(data_dir / "pit_stops.csv")),
    )

    for label, query, rows in loaders:
        count = write_batches(session, query, rows)
        print(f"Loaded {count} {label} nodes")


def main() -> None:
    data_dir = find_data_dir()
    config = read_config(find_config())
    database = config.get("DATABASE")
    session_options = {"database": database} if database else {}

    driver = neo4j.GraphDatabase.driver(
        config["URI"],
        auth=(config["USERNAME"], config["PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        with driver.session(**session_options) as session:
            build_graph(session, data_dir)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
