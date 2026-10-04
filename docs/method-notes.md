# Method notes (moved out of the notebook)

## Sentinel registry (Gates 1.B / 2.1)

CSVs are loaded with `dtype=str, keep_default_na=False` so pandas never silently coerces:
`\N` stays a visible literal until cleaning maps it to NaN on copies (plus `""` → NaN for
quali q2/q3, text strip). Registry: `\N` → standard NaN; `grid = 0` / `position = 0` /
zero lat-lng / `""` → per-column rules in §12 (D4). Zero `\N` may remain after cleaning
(asserted).

## Typing (§7 → §12)

Calendar dates → `datetime64` (+ `race_datetime` where date+time exist); lap/race/pit
durations (`m:ss.mmm` / `h:mm:ss.mmm` / `ss.mmm`) → float seconds via `time_to_seconds`;
numerics → nullable `Int64`/`Float64`; circuit aliases unified (`United States` → `USA`).
Imputation policy (Lab 1 `Cabin` rule): drop only dead weight (`url`, practice/quali
wall-clock); keep informative NaN (`position` for DNFs, q2/q3 knockouts); model-feature
imputation happens inside the train split only (median skewed numerics, mode categoricals).

## Raw-audit reading (§§3–4)

- `results.position/time/fastestLap*` gaps = did-not-finish/not-classified (outcome, not a
  defect) → typed NaN, never imputed.
- `qualifying.q2/q3` gaps = structural (no knockout format pre-2003) → era/coverage effect,
  main input to the 2003+ modeling-window decision.
- `grid` spike at 0 = no starting slot → Gate 2.D split.
- `pit_stops.duration` has no `\N`; its story is coverage (since 2011), not missingness.

## Coverage reading (§8)

q2/q3 100% missing pre-2003; `results.position` missingness *is* retirement (post-race, Q3
input only); `fastestLap*` only from ~2004; `grid == 0` excluded from grid-based analyses.

## Target reading (§9)

Base rate low and era-dependent (field size, competitive balance) → PR-AUC primary and a
time-ordered (season) split, not a random shuffle.

## Grain background (§6)

Brief requires one row per `(raceId, driverId)`; early-F1 shared drives break it. Rule: D3
in `docs/decisions.md`.

## Q1–Q3 backgrounds (§10)

- **Q1**: `qualifying.position` (rank in qualifying) ≠ `results.grid` (slot after penalties);
  quantify the gap per era, carry both columns.
- **Q2**: `drivers.nationality` adjectives → `circuits.country`; mapping sketch in §10.2,
  full rules D6.
- **Q3**: 139 `status` labels → mechanical vs accident/driver/regulatory (+ ambiguous);
  canonical sets in the §10.3 cell, final strict mapping in the §12 taxonomy cell, rules D7.
  Denominator = starts (non-starters excluded).

## Roadmap

Next phases (features, splits, models, XAI, inference, report) follow `PLAN.md`.
