# Leakage spec (moved out of the notebook §11)

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
| `sprint_results` | **Open (Gate 4.C)** — runs before Sunday's race on sprint weekends, but off-schedule historically; excluded by default |
| `status`, `results.position/points/laps/time/fastestLap` | Forbidden — these *are* the outcome |

Target: `podium` derived from `positionOrder <= 3`. Twelve post-race `results` columns are
excluded from the feature set; the audit cell asserts the allow/deny mapping is exhaustive.
