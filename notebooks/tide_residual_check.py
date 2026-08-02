"""Does Mooloolaba tide phase modulate hsig_m residuals?

Pulls 2023-2024 tide gauge readings (10-min, CKAN Datastore) at the buoy
location, aligns them to the 30-min wave grid, and asks whether residuals
from a persistence forecast at h=12h show structure vs the tide. Informed 
project roadmap and now leaving tide out of this project.

Checks (all on 2023-01-01 -> 2024-12-31 AEST):
  1. corr(persistence h=12 residual, tide level)
  2. corr(persistence h=12 residual, dtide/dt)
  3. corr(persistence h=12 residual, tide phase via Hilbert transform)
  4. RMSE of residuals split into tide-level quintiles (any monotone trend?)

A "passing" signal would be |r| > ~0.05 on any of (1)-(3), or a >2 cm spread
in RMSE across quintiles. Smaller than that and tide is below the noise floor
of the rest of the feature matrix.
"""
import numpy as np
import pandas as pd
import requests
from scipy.signal import hilbert

import forecast as fc

TIDE_RESOURCES = {
    2023: "2a16884a-74c3-4b7c-a012-4c13b8414bdc",
    2024: "67cad1e1-5c54-4399-a08b-6c60d868d654",
}
CKAN_BASE = "https://www.data.qld.gov.au/api/3/action/datastore_search"
BATCH = 32000


def fetch_tide_year(resource_id: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        r = requests.get(
            CKAN_BASE,
            params={"resource_id": resource_id, "limit": BATCH, "offset": offset},
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()["result"]
        batch = result["records"]
        rows.extend(batch)
        if len(rows) >= result["total"] or not batch:
            break
        offset += len(batch)
    df = pd.DataFrame(rows)[["Date", "Time", "Reading"]]
    stamp = (df["Date"].str.strip() + " " + df["Time"].str.strip()).str.strip()
    df["datetime"] = pd.to_datetime(stamp, format="mixed", dayfirst=True)
    df["tide_m"] = pd.to_numeric(df["Reading"], errors="coerce")
    return df.set_index("datetime")[["tide_m"]].sort_index()


def main():
    print("Fetching tide 2023-2024...")
    tide = pd.concat(fetch_tide_year(rid) for rid in TIDE_RESOURCES.values())
    tide.index = tide.index.tz_localize("Australia/Brisbane")
    print(f"  {len(tide):,} 10-min readings, "
          f"{tide.index.min()} -> {tide.index.max()}")
    print(f"  range {tide['tide_m'].min():.2f} - {tide['tide_m'].max():.2f} m, "
          f"NaN frac {tide['tide_m'].isna().mean():.3f}")

    # 30-min wave grid (forecast origins), 2023-2024 AEST
    print("\nLoading mooloolaba waves...")
    wave = fc.load_data("mooloolaba")
    wave = fc.restrict_to_years(wave, 2023, 2024)
    print(f"  {len(wave):,} 30-min rows on hsig_m")

    # Reindex tide onto wave grid (nearest 30-min sample, max 15-min gap).
    tide_30m = tide["tide_m"].reindex(
        wave.index, method="nearest", tolerance=pd.Timedelta("15min")
    )
    print(f"  tide on wave grid: NaN frac {tide_30m.isna().mean():.3f}")

    # Persistence residual at h=12h (24 30-min steps): yhat(t) = hsig(t).
    h_steps = 24
    hsig = wave["hsig_m"]
    y_true = hsig.shift(-h_steps)
    resid = y_true - hsig  # truth - forecast

    # Tide features at forecast origin t
    dtide = tide_30m.diff()
    # Phase via Hilbert on a detrended tide series. Fill small gaps for the
    # transform (linear interp), then mask back to original NaN locations.
    tide_filled = tide_30m.interpolate(limit=6)
    tide_dt = tide_filled - tide_filled.rolling("25h", min_periods=10).mean()
    valid = tide_dt.notna()
    phase = pd.Series(index=tide_30m.index, dtype=float)
    analytic = hilbert(tide_dt[valid].to_numpy())
    phase.loc[valid] = np.angle(analytic)
    phase[tide_30m.isna()] = np.nan
    # Drift correction: also try phase of the *raw* tide (Hilbert on detrended is
    # the textbook way, but on a 24h-window detrend we may have edge artefacts).

    df = pd.DataFrame({
        "resid": resid,
        "tide": tide_30m,
        "dtide": dtide,
        "phase": phase,
    }).dropna()
    print(f"\nAligned rows for analysis: {len(df):,}")
    print(f"Persistence residual stats: "
          f"mean {df['resid'].mean():+.3f} m, "
          f"std {df['resid'].std():.3f} m, "
          f"RMSE {np.sqrt((df['resid']**2).mean()):.3f} m")

    print("\n--- correlations with persistence h=12 residual ---")
    for col in ["tide", "dtide", "phase"]:
        r = df["resid"].corr(df[col])
        r_sin = df["resid"].corr(np.sin(df[col])) if col == "phase" else None
        r_cos = df["resid"].corr(np.cos(df[col])) if col == "phase" else None
        print(f"  {col:<8} r = {r:+.4f}"
              + (f"   sin r = {r_sin:+.4f}   cos r = {r_cos:+.4f}"
                 if col == "phase" else ""))

    # RMSE by tide-level quintile
    print("\n--- residual RMSE by tide-level quintile ---")
    df["tide_q"] = pd.qcut(df["tide"], 5, labels=False)
    by_q = df.groupby("tide_q")["resid"].agg(
        n="count",
        rmse=lambda s: np.sqrt((s**2).mean()),
        bias="mean",
    )
    by_q["tide_lo"] = df.groupby("tide_q")["tide"].min()
    by_q["tide_hi"] = df.groupby("tide_q")["tide"].max()
    print(by_q.round(3).to_string())
    spread = by_q["rmse"].max() - by_q["rmse"].min()
    print(f"\nRMSE spread across quintiles: {spread*100:.2f} cm")

    # Same split, by abs(resid) just to sanity-check the metric direction
    print("\n--- residual RMSE by |dtide/dt| quintile (rising vs slack) ---")
    df["dtide_abs_q"] = pd.qcut(df["dtide"].abs(), 5, labels=False)
    by_dq = df.groupby("dtide_abs_q")["resid"].agg(
        n="count",
        rmse=lambda s: np.sqrt((s**2).mean()),
    )
    print(by_dq.round(3).to_string())

    print("\n--- verdict ---")
    max_r = max(
        abs(df["resid"].corr(df["tide"])),
        abs(df["resid"].corr(df["dtide"])),
        abs(df["resid"].corr(np.sin(df["phase"]))),
        abs(df["resid"].corr(np.cos(df["phase"]))),
    )
    # Monotonicity check: if tide truly modulates hsig, RMSE-vs-tide should
    # trend one way, not bounce around. Spearman of (quintile rank, RMSE).
    rmse_by_q = by_q["rmse"].to_numpy()
    rank_r = pd.Series(rmse_by_q).corr(
        pd.Series(np.arange(len(rmse_by_q))), method="spearman"
    )
    print(f"max |r| across tide features:  {max_r:.4f}")
    print(f"quintile RMSE spread:          {spread*100:.2f} cm")
    print(f"quintile RMSE monotonicity ρ:  {rank_r:+.3f}  (0 = no trend)")
    if max_r < 0.05 and (spread < 0.02 or abs(rank_r) < 0.6):
        print("=> No detectable tide modulation of h=12 persistence residual.")
        print("   Roadmap item #4 should be killed.")
    else:
        print("=> Some structure present — worth a proper experiment.")


if __name__ == "__main__":
    main()
