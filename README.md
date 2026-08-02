# Surf Height Prediction

An exercise in predictive modeling, this project is all about forecasting wave height off the sunny coast in Queensland, Australia. I chose to revisit this subject to see if I could beat [my previous model](legacy/) performance, but found that the data source had revised some of its measurements, yeilding a much rougher dataset and making direct model performance comparison unhelpful. Even so, this expansion was an enjoyable learning experience.

> **Previous version.** The original project from AMLI summer 2019 is archived unchanged under [`legacy/`](legacy/). Its RMSE figures predate the upstream data revision described below and aren't comparable to the results here.

## Data source

All data in this project comes from the Queensland Government [open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), which provides us with several wind and wave monitoring stations in the region. Since all raw records are AEST we don't have to worry about time changes, and every output unified CSV carries a gap-free Brisbane `datetime` index.

### Upstream revisions note

The QLD portal publishes these as derived, delayed-mode wave parameters, and it periodically re-derives and republishes whole yearly resource files. Comparing an October 2025 snapshot against a re-download confirmed this: of ~178k shared timestamps, **26.9% changed**, with a clear signature rather than random drift, but no revision notice published.

- `hmax_m`, `tz_s`, `tp_s`, and `peak_dir_deg` *all* change on the same ~27% of rows (`sst_c` on ~26%).
- New values are higher 50.3% / lower 49.7% of the time (mean Δ ≈ 0, -0.011 m) and the maximum is unchanged (5.204 m), so it is not a clipping, units, or one-sided shift.
- Magnitudes are large (median |Δ| = 0.42 m).
- They form 195 contiguous blocks (median ~5 days, max 18), never scattered single points, with no time-of-day pattern.
- The largest blocks all fall in the Dec–Mar storm/cyclone season (e.g. 2025-01-27→02-15, 2023-12-01→12-11).
- Waves >3 m were revised 40% of the time (mean |Δ| 0.63 m) vs ~25% / 0.11 m for 0.5-1.5 m waves.
- 2017, 2018, and 2021 are untouched; 2015/16/19/20/22/23/24/25 were republished.

The net effect is that the revised data is rougher:
- 12h autocorrelation dropped 0.85 → 0.74
- persistence RMSE rose from 26.5 cm on the old snapshot to ~40 cm now.

I assume the revision is a data-quality improvement (more accurate storm-period measurements), but it means absolute RMSE is not comparable across snapshots or against my old project on a smaller and older subset of this data.


### Wave buoy network

30-minute cadence. Mooloolaba is the prediction target; Brisbane, Caloundra, Gold Coast, North Moreton Bay, Palm Beach, Tweed Heads, and Wide Bay feed in as neighbour-buoy features where their histories overlap.

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (meters) |
| `hmax_m` | Maximum wave height (meters) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

![Wave coverage](notebooks/figures/wave_coverage.png)

### Wind (air-quality monitoring network)

Hourly cadence (reindexed onto the 30-minute wave grid by forward-fill), 10 m ultrasonic wind sensors on the QLD air-quality monitoring stations. Mountain Creek pairs with the Mooloolaba buoy, Deception Bay sits ~50 km south on Moreton Bay, Lytton is at the mouth of the Brisbane River, and Southport sits on the Gold Coast.

| Column | Description |
|--------|-------------|
| `wind_dir_deg` | Wind direction (degrees true north) |
| `wind_speed_ms` | Wind speed (meters/second) |
| `wind_sigma_theta_deg` | Wind direction standard deviation (degrees) |
| `wind_speed_std_ms` | Wind speed standard deviation (meters/second) |

![Wind coverage](notebooks/figures/wind_coverage.png)

> The coverage grids define the breadth-vs-depth trade for any experiment: how far back to train, and how many neighbour sources to include. Palm Beach (deployed 2017), Southport wind (mid-2018), and Wide Bay (2019, the only buoy upstream of northerly swells) only appear later.


## Data preparation

### Feature engineering

The feature matrix is assembled in three layers, each a single call:

1. **Base primary-buoy matrix** (`fc.build_buoy_features`) — circular encoding, hour/doy time features, lags, rolling stats, momentum. Lag/rolling/delta grids are tunable via `FeatureConfig`. For sequence models, `fc.build_seq_features` swaps in raw channels with no pre-built lags (the model windows its own input).
2. **Neighbour buoys** (`fc.add_neighbour_features`) — raw value, lag copies, and rolling mean/std per neighbour column, reusing the same `FeatureConfig`.
3. **Wind stations** — same `add_neighbour_features` call on each wind station's columns. `fc.load_wind` sin/cos-encodes `wind_dir_deg` and station-prefixes every column so cross-station features stay distinguishable.

### Preprocessing pipeline

`forecast.preprocess.Preprocessor` bundles the three steps the playgrounds run between `chronological_split` and `model.fit`:

1. **Drop sparse columns** — any column whose **train-set** NaN fraction exceeds `max_nan_frac` (default 0.5) is removed. Mean-imputing a near-empty column gives a near-constant feature that silently corrodes gradient-based sequence models. The `wave_column_coverage.png` and `wind_column_coverage.png` EDA figures surface candidates ahead of time.
2. **Mean impute** — column-wise mean from training data fills remaining NaNs.
3. **Scale** (optional) — `"robust"` (median/IQR) or `"standard"`. Linear models default to robust because wave data is heavy-tailed and storm spikes would inflate a standard-deviation scale. Trees (HGB) take the raw matrix. Sequence models scale internally (`scaler="robust"` or `"standard"`, fit on train) and consume the unscaled `build_seq_features` frame. `*_sin`/`*_cos` columns pass through untouched in all cases.


## Non-model baselines

Two no-model references frame every result in this project:

- **Persistence** — predict ŷ(t+h) = y(t). Strong at short horizons because Mooloolaba's `hsig_m` is highly autocorrelated on the order of hours, so "looks like now" is hard to beat in the first half-day.
- **Climatology hour** — predict the train-set mean of `hsig_m` conditioned on hour-of-day(t+h). Horizon-independent: it ignores `t` entirely, so its RMSE is flat at ~48 cm across every horizon.

The two crossover between **h=12 and h=24** on the pinned 2023-01-01 → 2024-12-31 test window, so persistence wins for h≤12, climatology from h≥24 onward:

![Baseline residuals at h=24](notebooks/figures/baseline_residuals.png)

## Model selection and tuning

Three model families are compared head-to-head: **Ridge** (linear, robust-scaled) and a regularised **GRU** (sequence model over the raw circular-encoded channels), both predicting the wave-height *level* `y(t+h)` directly, plus **HGB-on-residual** — gradient-boosted trees that instead predict the *persistence residual* `y(t+h) − y(t)` (the change from the current height) and add it back. Each is tuned on the pinned 2023-01-01 → 2024-12-31 test window and scored across six forecast horizons. Each line is that family's **lowest-RMSE run at each horizon**, pooled across every feature combo and config logged. Taking the minimum over many runs flatters all three a little — the noise section below bounds by how much.

![Best model per family vs forecast horizon](notebooks/figures/horizon_sweep.png)

| h | Best baseline RMSE (cm) | Ridge (cm) | HGB (cm) | GRU (cm) | Best skill vs baseline |
|---|---|---|---|---|---|
| 6h  | 29.1 (persistence) | 26.0 | **25.4** | 25.9 | +12.7% |
| 12h | 40.0 (persistence) | **34.7** | 34.8 | 34.8 | +13.2% |
| 24h | 47.9 (climatology) | 44.2 | 43.7 | **43.4** | +9.3% |
| 36h | 47.9 (climatology) | **45.3** | 46.3 | 45.8 | +5.3% |
| 48h | 47.9 (climatology) | **46.0** | 48.4 | 46.2 | +4.1% |
| 72h | 47.9 (climatology) | 47.7 | 50.3 | **47.4** | +1.1% |

Three conclusions:

- **Neighbour and wind stations only earn their keep at short lead.** At h=6 the best models lean on a wider feature set, but the edge is only ~0.5–1 cm. By h=24 the best Ridge and HGB are the primary-buoy-only models, and adding stations from there just hands the model noise to regularise away.

- **The regularised GRU does not beat the simpler models.** It tracks them closely everywhere and posts two sub-centimetre nominal "wins" (h=24, h=72), but those margins are ≤ 0.3 cm. Gaps this small sit inside the test-window noise band measured below, so the sequence model's extra weight and training cost don't pay off.

- **HGB for the first few hours, Ridge from a day out.** HGB-on-residual is best at h=6, but the two are a tie through h=36, and from there HGB falls behind as its residual target loses the structure that makes it learnable.

All three clear the no-model baseline comfortably at short range (skill ≈ +13% at h=6/12) and then decay toward it: by h=72 even the best model is barely 1% under flat climatology. To beat it at that horizon would require much more distant leading wave observations and/or a spectral wave model.

### How much of this is signal? (test-window noise)

With only one fixed test window, a one-centimetre gap could be nothing more than which storms happened to fall after 2023-01-01. There's no cross-validation here to average that out, so `notebooks/test_window_noise.py` measures the single-window uncertainty directly with a moving-block bootstrap (two-week blocks, preserving storm-scale autocorrelation).

Taking the shipped Ridge-on-primary model as representative (the band is set by the test window, not the model, so it's similar for all three), the **absolute** RMSE is pinned to roughly ±2 cm at short lead, widening to ±4 cm at long lead:

| h | Ridge / primary RMSE (cm) | 95% CI (cm) | resolution |
|---|---|---|---|
| 6h  | 27.0 | 24.9–29.1 | ±2.1 cm |
| 12h | 35.4 | 32.7–38.2 | ±2.8 cm |
| 24h | 44.2 | 40.6–47.6 | ±3.5 cm |
| 36h | 45.3 | 41.9–48.9 | ±3.5 cm |
| 48h | 46.0 | 42.4–49.9 | ±3.7 cm |
| 72h | 47.7 | 43.6–51.9 | ±4.1 cm |

Most gaps in the table above are smaller than that, so on absolute RMSE alone the families look tied. But every model is scored on the *same* storms, so the **paired** difference is resolved far more tightly than those overlapping bands suggest — and that is the right test for "is A actually better than B here?" Bootstrapping the paired RMSE difference (HGB − Ridge, primary buoy) shows what's real:

- **h=6 — HGB beats Ridge by 0.9 cm**, 95% CI [−1.4, −0.6]: real, not luck.
- **h=12 / 24 / 36 — genuine ties** (CI straddles zero).
- **h=48 / 72 — Ridge beats HGB by 2.2 / 2.6 cm**, CIs [+0.8, +3.8] / [+1.5, +4.0]: HGB's long-lead collapse is real.

So the structural story survives a significance test — HGB for the first few hours, Ridge from a day or two out — while everything in between is a coin-flip and the GRU's sub-centimetre wins are noise.


## Real world performance

Each new year that passes can be scored as a true blind set against our best models. This way we evaluate them against brand new data that wasn't implicitly leaked through the train/test iterative process. Awaiting the QLD wind 2025 release expected September 2026.

**Pre-committed candidates for 2025**:

- **Ridge (α=1)** — primary-buoy engineered features only (lags, rolling stats, momentum; no neighbours or wind), mean-imputed and robust-scaled by the fitted `Preprocessor`.
- **HGB-on-residual** — `HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_depth=6, min_samples_leaf=50, l2_regularization=1.0)` fit on the persistence residual `y(t+h) − y(t)` of the `tweed_mc` set: primary buoy + Tweed Heads buoy + Mountain Creek wind (native NaN handling, no scaling).

Scoring a new year against these committed candidates is a re-fit of the same recipe on the same training data, not a load of a serialised model.

| Year | h | Model | RMSE (cm) | Skill |
|------|---|-------|-----------|-------|
| 2025 | 12h | HGB | _TBD_ | _TBD_ |
| 2025 | 24h | Ridge | _TBD_ | _TBD_ |
| 2025 | 48h | Ridge | _TBD_ | _TBD_ |


## Reproducibility

### Setup

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --all-extras
```

Run tests after changes:

```bash
./.venv/bin/pytest src/tests/ -v
```

### Packages

- **`qld_ckan`**: Downloads yearly records from the QLD CKAN Datastore API, unifies the schema, and writes a cleaned CSV per source.
    - `qld_ckan.wave` (wave buoys) 
    - `qld_ckan.wind` (air-quality-station 10 m wind)
- **`viz`**: Source-agnostic plotting, organised by pipeline stage: shared time-series primitives, post-download EDA heatmaps, and post-experiment result charts that consume `forecast.find_runs` output.
- **`forecast`**: Target construction, chronological splits, feature engineering, baselines, metrics, and an evaluation harness. See *Available forecasters* below for the model list.

Experiment scripts in `notebooks/` run on top of these packages. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full source tree and a per-module breakdown.

### Running the pipeline

Generate `data/` with these commands (one CSV per source):

```bash
# Wave - default Mooloolaba 2015-2025
./.venv/bin/python -m qld_ckan wave [--buoy brisbane|caloundra|gold-coast|north-moreton-bay|palm-beach|tweed-heads|wide-bay]

# Wind - default Mountain Creek 2010-2024
./.venv/bin/python -m qld_ckan wind [--station deception-bay|lytton|southport]
```

Both subcommands accept `--year-min` / `--year-max` (inclusive) to clip the registry before download.

```bash
./.venv/bin/python -m qld_ckan wave --buoy brisbane --year-min 2018 --year-max 2020
```

The standalone helpers (`fc.drop_sparse_columns`, `fc.mean_impute`, `fc.scale_features`) still exist for one-shot use, but the playgrounds use the class so the fitted state can be inspected, asserted, and pickled:

```python
preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_train)
X_train_p = preproc.transform(X_train)
X_test_p  = preproc.transform(X_test)

# Held-out year scoring: pair the model with its preprocessor.
preproc.save("models/ridge_preproc.pkl")
# Later, with a new year's raw inputs:
preproc = fc.Preprocessor.load("models/ridge_preproc.pkl")
X_2025_p = preproc.transform(X_2025)  # raises if any fit-time column is missing
```

`transform()` enforces the schema the preprocessor was fitted on: any missing required column raises `ValueError`; extra columns (e.g. a new wind station appearing later) are silently dropped. This catches the failure mode where a held-out year's feature matrix doesn't line up with the training-time decisions — without the class, that mismatch would surface as a silently wrong prediction.

`forecast` exposes a flat import surface (`import forecast as fc`): target construction (`make_target` shifts `hsig_m` 24 steps ahead), chronological 80/20 split, feature builders, and an `evaluate_and_log` harness that scores `MAE / RMSE / Bias / SkillVsBaseline` against persistence and appends to `experiments.jsonl`. A typical call:

```python
result = fc.evaluate_and_log(
    fc.Ridge(alpha=1.0), X_tr, y_tr, X_te, y_te,
    name="ridge", data_sources=["mooloolaba"],
)
print(result.metrics)
```

### Available forecasters

| family          | classes                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (Ridge, Lasso, HGB, …)                               |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`       |

### Logging experiments

`fc.evaluate_and_log(...)` is a drop-in for `fc.evaluate(...)` that appends a record to `experiments.jsonl` (committed at repo root); `fc.log_run(result, ...)` covers results computed outside the harness. Read the log back as a DataFrame with `fc.read_log()`.

### Experiment scripts

All scripts are plain `.py` files — run directly:

```bash
./.venv/bin/python notebooks/<script>.py
```


## Roadmap / Expansions

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90, or conformalised intervals over Ridge. Also the structural fix for the tail underfit — residuals-vs-predicted shows bias at high wave heights that pinball loss addresses and a target transform won't.

2. **Multi-output forecasts.** 2 m at 90°/14 s breaks differently from 2 m at 150°/8 s. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`) so downstream code can apply break-specific transforms.

3. **Long-cadence historical bundles.** Deeper wave history for swell-upstream buoys: Mooloolaba 2000-2014 (1h), Brisbane 1976-2011 (12h), Gold Coast 1987-2014 (6h), Tweed Heads 1995-2011 (1h). Excluded from `qld_ckan.wave.constants.BUOYS` because the pipeline assumes a 30-min axis and these have drifting minute offsets (e.g. 08:55, 14:56). Needs: a cadence parameter on the wave pipeline, snap-to-grid (floor + dedup) before reindex, and a join strategy mixing coarse history with the 30-min grid. Resource IDs at `coastal-data-system-waves-{slug}` on `data.qld.gov.au`. QLD's 1989-1992 DST window forces choosing fixed UTC+10 (`Etc/GMT-10`) or per-row DST.

4. **Ensemble the families.** A flat nanmean of Ridge, HGB, and GRU edged the best single model by ~0.5 cm at short lead in earlier runs (e.g. h=12), where the three make partly uncorrelated errors.