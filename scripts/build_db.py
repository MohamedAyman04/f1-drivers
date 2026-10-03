r"""Load the 14 raw F1 CSVs into SQLite and run the Milestone-1 integrity checks.

Usage:
    python scripts/build_db.py

Raw CSVs in data/raw/ are never modified; they are mirrored into data/f1.db.
Sentinel values (\N) are stored as-is so the database is a faithful copy of the
download; cleaning happens later on copies, inside the notebook.
"""

from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "f1.db"

TABLES = [
    "circuits", "constructors", "constructor_results", "constructor_standings",
    "drivers", "driver_standings", "races", "results", "qualifying",
    "lap_times", "pit_stops", "seasons", "sprint_results", "status",
]

PRIMARY_KEYS = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "constructor_results": ["constructorResultsId"],
    "constructor_standings": ["constructorStandingsId"],
    "drivers": ["driverId"],
    "driver_standings": ["driverStandingsId"],
    "races": ["raceId"],
    "results": ["resultId"],
    "qualifying": ["qualifyId"],
    "lap_times": ["raceId", "driverId", "lap"],
    "pit_stops": ["raceId", "driverId", "stop"],
    "seasons": ["year"],
    "sprint_results": ["resultId"],
    "status": ["statusId"],
}

FOREIGN_KEYS = {
    "constructor_results": [("raceId", "races", "raceId"), ("constructorId", "constructors", "constructorId")],
    "constructor_standings": [("raceId", "races", "raceId"), ("constructorId", "constructors", "constructorId")],
    "driver_standings": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId")],
    "races": [("circuitId", "circuits", "circuitId")],
    "results": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId"),
                ("constructorId", "constructors", "constructorId"), ("statusId", "status", "statusId")],
    "qualifying": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId"),
                   ("constructorId", "constructors", "constructorId")],
    "lap_times": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId")],
    "pit_stops": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId")],
    "sprint_results": [("raceId", "races", "raceId"), ("driverId", "drivers", "driverId"),
                       ("constructorId", "constructors", "constructorId"), ("statusId", "status", "statusId")],
}


def load() -> dict[str, int]:
    if not RAW_DIR.is_dir():
        sys.exit(f"missing {RAW_DIR}; unzip archive.zip into data/raw/ first")
    if DB_PATH.exists():
        DB_PATH.unlink()
    counts = {}
    con = sqlite3.connect(DB_PATH)
    try:
        for name in TABLES:
            csv = RAW_DIR / f"{name}.csv"
            if not csv.exists():
                sys.exit(f"missing {csv}")
            df = pd.read_csv(csv, keep_default_na=False, na_values=[], low_memory=False)
            df.to_sql(name, con, index=False, if_exists="replace")
            counts[name] = len(df)
    finally:
        con.close()
    return counts


def checks(counts: dict[str, int]) -> list[str]:
    problems = []
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print("\n== Table row counts (CSV vs SQLite) ==")
    for name in TABLES:
        db_rows = cur.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        flag = "ok" if db_rows == counts[name] else "MISMATCH"
        if flag != "ok":
            problems.append(f"{name}: {counts[name]} csv rows vs {db_rows} db rows")
        print(f"  {name:<24} {db_rows:>7}  {flag}")

    print("\n== Primary key uniqueness / nulls ==")
    for table, cols in PRIMARY_KEYS.items():
        expr = ", ".join(f'"{c}"' for c in cols)
        n = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        distinct = cur.execute(f'SELECT COUNT(*) FROM (SELECT DISTINCT {expr} FROM "{table}")').fetchone()[0]
        nulls = cur.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {' OR '.join(f'"{c}" IS NULL' for c in cols)}"
        ).fetchone()[0]
        status = "ok" if (distinct == n and nulls == 0) else "PROBLEM"
        if status != "ok":
            problems.append(f"{table}: pk distinct={distinct}/{n}, nulls={nulls}")
        print(f"  {table:<24} unique={distinct == n}  nulls={nulls}  {status}")

    print("\n== Foreign key reconciliation (orphans must be 0) ==")
    if not FOREIGN_KEYS:
        print("  (none defined)")
    for table, fks in FOREIGN_KEYS.items():
        for col, ref_table, ref_col in fks:
            orphans = cur.execute(
                f'SELECT COUNT(*) FROM "{table}" t LEFT JOIN "{ref_table}" r '
                f'ON t."{col}" = r."{ref_col}" WHERE r."{ref_col}" IS NULL'
            ).fetchone()[0]
            status = "ok" if orphans == 0 else "PROBLEM"
            if orphans:
                problems.append(f"{table}.{col} -> {ref_table}.{ref_col}: {orphans} orphans")
            print(f"  {table}.{col:<16} -> {ref_table}.{ref_col:<14} orphans={orphans}  {status}")

    print("\n== Modeling grain: one row per (raceId, driverId) in results ==")
    n = cur.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    dups = cur.execute(
        "SELECT COUNT(*) FROM (SELECT raceId, driverId FROM results "
        "GROUP BY raceId, driverId HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    dup_rows = cur.execute(
        "SELECT COUNT(*) FROM results WHERE (raceId, driverId) IN "
        "(SELECT raceId, driverId FROM results GROUP BY raceId, driverId HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    print(f"  results rows={n}, duplicated (raceId, driverId) pairs={dups} affecting {dup_rows} rows")
    print("  -> shared drives in early F1; dedup rule is a notebook decision")

    print("\n== Sentinel (\\N) presence ==")
    total = 0
    for table in TABLES:
        cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')]
        hits = []
        for col in cols:
            c = cur.execute(
                f"SELECT COUNT(*) FROM \"{table}\" WHERE CAST(\"{col}\" AS TEXT) = '\\N'"
            ).fetchone()[0]
            if c:
                hits.append(f"{col}={c}")
                total += c
        if hits:
            print(f"  {table:<24} {', '.join(hits)}")
    print(f"  total sentinel cells: {total}")

    con.close()
    return problems


def demo() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    print("\n== Sanity queries (multi-hop joins) ==")
    print("  races per season (last 6 seasons):")
    for year, n in cur.execute(
        "SELECT year, COUNT(*) FROM races GROUP BY year ORDER BY year DESC LIMIT 6"
    ):
        print(f"    {year}: {n} races")
    print("  podium rate by grid bucket (all eras):")
    for bucket, starts, podiums in cur.execute(
        "SELECT CASE WHEN grid <= 3 THEN 'grid 1-3' WHEN grid <= 10 THEN 'grid 4-10' "
        "ELSE 'grid 11+' END AS b, COUNT(*), SUM(positionOrder <= 3) "
        "FROM results WHERE grid > 0 GROUP BY b ORDER BY MIN(grid)"
    ):
        print(f"    {bucket:<9} starts={starts:<6} podiums={podiums:<5} rate={podiums / starts:.1%}")
    print("  most recent race with winner and circuit:")
    for row in cur.execute(
        "SELECT r.year, r.name, c.name, d.forename || ' ' || d.surname "
        "FROM results res JOIN races r ON res.raceId = r.raceId "
        "JOIN circuits c ON r.circuitId = c.circuitId JOIN drivers d ON res.driverId = d.driverId "
        "WHERE res.positionOrder = 1 ORDER BY r.date DESC LIMIT 3"
    ):
        print(f"    {row[0]} {row[1]} @ {row[2]}: {row[3]}")
    con.close()


if __name__ == "__main__":
    print(f"Loading {RAW_DIR} -> {DB_PATH}")
    counts = load()
    problems = checks(counts)
    demo()
    if problems:
        print("\nPROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAll integrity checks passed.")
