# Leakage spec (moved out of the notebook §11)

> **Old working document (kept for history).** Written during the project; numbers, cell references and some decisions (e.g. a "final model") are out of date. The final work is `notebooks/f1_podium_prediction.ipynb` and `report/main.pdf`.

Prediction cutoff: **after qualifying, before race start**. Anything from the current race
known only once the race is running (or over) is leakage. Enforced by the `RESULTS_ROLES` /
`FORBIDDEN` audit cell in §11.

| Table | Rule |
|---|---|
| `results` | Only `raceId`, `driverId`, `constructorId`, `number`, `grid` usable; everything else in that row is post-race |
| `qualifying` | Fully allowed — cutoff is *after* qualifying |
| `driver_standings` / `constructor_standings` | Only rows of **previous** rounds (join on `round - 1` or earlier) |
| Form features | Races strictly before the current one (`groupby.shift(1)` / rolling windows) |
| `lap_times`, `pit_stops` | Forbidden — they happen during the race |
| `sprint_results` | **Excluded (Gate 4.C closed, M3 2026-10-06)** — exists only from 2021 and is off-schedule historically |
| Identity encodings (driver / constructor / circuit) | Target encoding only from races strictly before the current one; never fit on the full table (M3) |
| `status`, `results.position/points/laps/time/fastestLap` | Forbidden — these *are* the outcome |

Target: `podium` derived from `positionOrder <= 3`. Twelve post-race `results` columns are
excluded from the feature set; the audit cell asserts the allow/deny mapping is exhaustive.
