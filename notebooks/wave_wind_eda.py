"""Joint wave + wind EDA: alignment overview, feature-horizon screening, joint distributions.

Run:  ./.venv/bin/python notebooks/wave_wind_eda.py

Saves three PNGs to notebooks/figures/ (all prefixed ``wave_wind_``):
  wave_wind_timeseries.png        — 3-panel 2020 overview: hsig_m, wind speed (per station), tp_s
  wave_wind_feature_horizon.png   — corr(wave + wind features at t, hsig_m at t+h)
  wave_wind_joint.png             — hexbin: primary-station wind speed vs hsig_m and tp_s
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from forecast.data import load_data
from forecast.features import encode_circular

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Stations to load. The first is treated as the primary for any single-station
# panels (e.g. the wind-speed-vs-wave-height hexbin).
STATIONS = ["mountain-creek", "deception-bay"]

STATION_COLORS = {
    "mountain-creek": "#e07b39",
    "deception-bay":  "#3b8db4",
}


def load_wind(station: str) -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob(f"{station}_wind_data_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No wind CSV for station={station!r} in {DATA_DIR}. "
            f"Run `python -m qld_ckan wind --station {station}` to generate it."
        )
    return pd.read_csv(matches[-1], parse_dates=["datetime"], index_col="datetime")


def aligned_hourly(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resample 30-min wave data to 1h and inner-join with each station's wind.

    Wind columns are prefixed with the station slug (e.g.
    ``mountain-creek_wind_speed_ms``) so multi-station merges keep them
    distinct. The join is inner across all sources.
    """
    wave_h = wave.resample("1h").mean()
    prefixed = [w.add_prefix(f"{name}_") for name, w in winds.items()]
    return pd.concat([wave_h, *prefixed], axis=1, join="inner")


# --------------------------------------------------------------------------- #
# Figure 1 — 3-panel overview time series (a representative year)
# --------------------------------------------------------------------------- #
def plot_overview_timeseries(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    year_wave = wave.loc["2020"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    viz.plot_series(
        year_wave["hsig_m"], ylabel="hsig_m (m)",
        title="Significant wave height — Mooloolaba 2020", ax=axes[0],
    )
    for name, wind in winds.items():
        year_wind = wind.loc["2020"]
        # Match the pandas date converter used by the wave panels — mixing
        # ax.plot here puts wind points outside the shared xlim (pandas
        # encodes x as minutes-since-epoch, mpl as days-since-epoch).
        year_wind["wind_speed_ms"].plot(
            ax=axes[1], label=name, color=STATION_COLORS.get(name),
            alpha=0.75, linewidth=0.8,
        )
    axes[1].set(ylabel="Wind speed (m/s)", title="Wind speed — 2020")
    axes[1].legend(loc="upper right", fontsize=9)
    viz.plot_series(
        year_wave["tp_s"], ylabel="tp_s (s)",
        title="Peak wave period — Mooloolaba 2020", ax=axes[2], color="#2ca02c",
    )
    fig.tight_layout()
    out = FIG_DIR / "wave_wind_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 2 — Feature × horizon heatmap (wave + wind features → hsig_m)
# --------------------------------------------------------------------------- #
def plot_feature_horizon(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    df = aligned_hourly(wave, winds)

    df = encode_circular(df, periods={"peak_dir_deg": 360.0})
    for name in winds:
        col = f"{name}_wind_dir_deg"
        df[f"{name}_wind_dir_sin"] = np.sin(2 * np.pi * df[col] / 360.0)
        df[f"{name}_wind_dir_cos"] = np.cos(2 * np.pi * df[col] / 360.0)
        df = df.drop(columns=[col])

    feature_cols = [
        "hmax_m", "tz_s", "tp_s",
        "peak_dir_deg_sin", "peak_dir_deg_cos",
        "sst_c",
    ]
    for name in winds:
        feature_cols += [
            f"{name}_wind_speed_ms",
            f"{name}_wind_speed_std_ms",
            f"{name}_wind_sigma_theta_deg",
            f"{name}_wind_dir_sin",
            f"{name}_wind_dir_cos",
        ]

    fig, ax = plt.subplots(figsize=(13, 0.45 * len(feature_cols) + 3))
    _, grid = viz.feature_horizon_heatmap(
        df, target_col="hsig_m", feature_cols=feature_cols,
        horizons_h=(1, 3, 6, 12, 24, 48, 72), sampling_freq_min=60,
        source_label="Mooloolaba hsig_m target", ax=ax,
    )
    fig.tight_layout()
    out = FIG_DIR / "wave_wind_feature_horizon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")
    print("  Feature-horizon grid:")
    print(grid.round(3).to_string())


# --------------------------------------------------------------------------- #
# Figure 3 — Wind-wave joint distributions (primary station only)
# --------------------------------------------------------------------------- #
def plot_joint_distributions(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    primary = next(iter(winds))
    df = aligned_hourly(wave, {primary: winds[primary]}).dropna(
        subset=["hsig_m", f"{primary}_wind_speed_ms", "tp_s"]
    )
    speed = df[f"{primary}_wind_speed_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hb = axes[0].hexbin(speed, df["hsig_m"], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=axes[0], label="Count")
    axes[0].set(xlabel="Wind speed (m/s)", ylabel="hsig_m (m)",
                title="Wind speed vs significant wave height")

    hb2 = axes[1].hexbin(speed, df["tp_s"], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb2, ax=axes[1], label="Count")
    axes[1].set(xlabel="Wind speed (m/s)", ylabel="tp_s (s)",
                title="Wind speed vs peak wave period")

    fig.suptitle(
        f"Wave–wind joint distributions  (Mooloolaba + {primary}, aligned to hourly)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "wave_wind_joint.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


def main() -> None:
    wave = load_data(buoy="mooloolaba")
    print(f"Wave: {len(wave):,} rows  ({wave.index.min().date()} → {wave.index.max().date()})")
    winds: dict[str, pd.DataFrame] = {}
    for name in STATIONS:
        winds[name] = load_wind(name)
        w = winds[name]
        print(f"Wind ({name}): {len(w):,} rows  "
              f"({w.index.min().date()} → {w.index.max().date()})")

    print("\n--- Figure 1: overview time series (2020) ---"); plot_overview_timeseries(wave, winds)
    print("\n--- Figure 2: feature x horizon heatmap ---");   plot_feature_horizon(wave, winds)
    print("\n--- Figure 3: joint distributions ---");          plot_joint_distributions(wave, winds)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
