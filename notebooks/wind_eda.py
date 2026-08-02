"""Wind-only EDA across the available stations.

Run:  ./.venv/bin/python notebooks/wind_eda.py

Saves six PNGs to notebooks/figures/ (all prefixed ``wind_``):
  wind_coverage.png            — % valid wind_speed_ms per station x year
  wind_column_coverage.png     — % valid per channel per station (all years pooled)
  wind_timeseries.png          — overlaid wind speed per station, 2020
  wind_autocorrelation.png     — ACF (72h window, 12h horizon marked)
  wind_direction_roses.png     — polar bar charts per station
  wind_station_comparison.png  — overlaid trace + hexbin agreement
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import viz

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Stations to load. The first is treated as the primary for any single-station
# panels. Each station must already have a CSV under ``data/`` produced by
# ``python -m qld_ckan wind --station <slug>``.
STATIONS = ["mountain-creek", "deception-bay", "lytton", "southport"]

STATION_COLORS = {
    "mountain-creek": "#e07b39",
    "deception-bay":  "#3b8db4",
    "lytton":         "#2ca02c",
    "southport":      "#9467bd",
}


def load_wind(station: str) -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob(f"{station}_wind_data_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No wind CSV for station={station!r} in {DATA_DIR}. "
            f"Run `python -m qld_ckan wind --station {station}` to generate it."
        )
    return pd.read_csv(matches[-1], parse_dates=["datetime"], index_col="datetime")


def load_all() -> dict[str, pd.DataFrame]:
    winds: dict[str, pd.DataFrame] = {}
    for name in STATIONS:
        winds[name] = load_wind(name)
        w = winds[name]
        print(f"  {name:16s}  rows={len(w):>7,}  "
              f"({w.index.min().date()} → {w.index.max().date()})  "
              f"mean speed={w['wind_speed_ms'].mean():.2f} m/s")
    return winds


# --------------------------------------------------------------------------- #
# Figure 1 — Data coverage heatmap
# --------------------------------------------------------------------------- #
def plot_coverage(winds: dict[str, pd.DataFrame]) -> None:
    years = list(range(2010, 2026))
    rows = []
    for name, wind in winds.items():
        row: dict[int, float] = {}
        for yr in years:
            mask = wind.index.year == yr
            total = int(mask.sum())
            row[yr] = float(wind.loc[mask, "wind_speed_ms"].notna().sum()) / total * 100 if total else np.nan
        rows.append(pd.Series(row, name=name))
    coverage = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 2.6 + 0.5 * len(winds)))
    ax.set_facecolor("#cccccc")  # gray = not deployed
    sns.heatmap(
        coverage, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=70, vmax=100, mask=coverage.isna(),
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% valid wind_speed_ms", "shrink": 0.8}, ax=ax,
    )
    ax.set(
        title="Data completeness — % valid wind_speed_ms rows per station per year  (grey = not deployed)",
        xlabel="Year", ylabel="",
    )
    fig.tight_layout()
    out = FIG_DIR / "wind_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    print("  Years with < 80 % valid wind_speed_ms:")
    for name in coverage.index:
        for yr in years:
            v = coverage.loc[name, yr]
            if not np.isnan(v) and v < 80:
                print(f"    {name} {yr}: {v:.0f}%")


# --------------------------------------------------------------------------- #
# Figure 1b — Per-column coverage (all years pooled)
# --------------------------------------------------------------------------- #
def plot_column_coverage(winds: dict[str, pd.DataFrame]) -> None:
    """% valid per channel per station — surfaces single-column gaps the
    year-level chart hides. ``wind_speed_ms`` may look fully populated while
    a sibling column (e.g. ``wind_speed_std_ms`` at Lytton) is mostly empty;
    mean-imputation of a near-constant feature then quietly poisons any
    gradient-based model that ingests it.
    """
    rows = []
    for name, wind in winds.items():
        rows.append(pd.Series(
            {col: float(wind[col].notna().mean()) * 100 for col in wind.columns},
            name=name,
        ))
    coverage = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(1.4 * len(coverage.columns) + 4, 0.55 * len(coverage) + 1.8))
    sns.heatmap(
        coverage, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=100, linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% valid (all years pooled)", "shrink": 0.8}, ax=ax,
    )
    ax.set(
        title="Wind: per-column completeness (all years pooled, lower = more imputation pressure)",
        xlabel="Column", ylabel="",
    )
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    out = FIG_DIR / "wind_column_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")

    print("  Columns with < 50 % valid:")
    for name in coverage.index:
        for col in coverage.columns:
            v = coverage.loc[name, col]
            if v < 50:
                print(f"    {name}.{col}: {v:.0f}%  (mean-imputation will turn this near-constant)")


# --------------------------------------------------------------------------- #
# Figure 2 — Wind speed time series (one representative year, all stations)
# --------------------------------------------------------------------------- #
def plot_timeseries(winds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for name, wind in winds.items():
        year = wind.loc["2020"]
        ax.plot(year.index, year["wind_speed_ms"],
                label=name, color=STATION_COLORS.get(name),
                alpha=0.75, linewidth=0.8)
    ax.set(xlabel="Time", ylabel="Wind speed (m/s)",
           title="Wind speed — 2020")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "wind_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 3 — Autocorrelation curves (one panel per station)
# --------------------------------------------------------------------------- #
def plot_autocorrelation(winds: dict[str, pd.DataFrame]) -> None:
    n = len(winds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, (name, wind) in zip(axes[0], winds.items()):
        viz.autocorrelation_curve(
            wind["wind_speed_ms"].dropna(), max_hours=72, step_hours=1.0,
            sampling_freq_min=60, highlight_hours=[12], threshold=0.5,
            source_label=f"wind_speed_ms ({name})", ax=ax,
        )
    fig.suptitle("Wind-speed autocorrelation — 72-h window, 12-h forecast horizon marked",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "wind_autocorrelation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 4 — Direction roses (one panel per station)
# --------------------------------------------------------------------------- #
def plot_direction_roses(winds: dict[str, pd.DataFrame]) -> None:
    bin_edges = np.linspace(0, 2 * np.pi, 37)  # 10° bins
    bin_width = bin_edges[1] - bin_edges[0]

    n = len(winds)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (name, wind) in enumerate(winds.items(), start=1):
        ax = fig.add_subplot(1, n, i, projection="polar")
        dirs_rad = np.deg2rad(wind["wind_dir_deg"].dropna().values)
        counts, _ = np.histogram(dirs_rad, bins=bin_edges)
        fracs = counts / counts.sum()
        ax.bar(bin_edges[:-1], fracs, width=bin_width, alpha=0.75,
               color=STATION_COLORS.get(name, "#888"))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Wind direction\n{name}", pad=14)
        ax.yaxis.set_visible(False)

    fig.suptitle("Wind direction distributions  (all years pooled)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "wind_direction_roses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 5 — Cross-station comparison (overlaid trace + hexbin agreement)
# --------------------------------------------------------------------------- #
def plot_station_comparison(winds: dict[str, pd.DataFrame]) -> None:
    if len(winds) < 2:
        print("Skipping station comparison: only one station loaded.")
        return

    names = list(winds)
    a, b = names[0], names[1]
    speed_a = winds[a]["wind_speed_ms"]
    speed_b = winds[b]["wind_speed_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    year_a = speed_a.loc["2020"]
    year_b = speed_b.loc["2020"]
    axes[0].plot(year_a.index, year_a, label=a,
                 color=STATION_COLORS.get(a), alpha=0.75, linewidth=0.8)
    axes[0].plot(year_b.index, year_b, label=b,
                 color=STATION_COLORS.get(b), alpha=0.75, linewidth=0.8)
    axes[0].set(xlabel="Time", ylabel="Wind speed (m/s)", title="Wind speed — 2020")
    axes[0].legend(loc="upper right")

    joint = pd.concat([speed_a, speed_b], axis=1, join="inner",
                      keys=[a, b]).dropna()
    r = joint[a].corr(joint[b])
    hb = axes[1].hexbin(joint[a], joint[b], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=axes[1], label="Count")
    lim = max(joint[a].max(), joint[b].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    axes[1].set(
        xlabel=f"{a} wind speed (m/s)", ylabel=f"{b} wind speed (m/s)",
        title=f"Hourly speed agreement  (Pearson r = {r:.3f})",
        xlim=(0, lim), ylim=(0, lim),
    )

    fig.suptitle("Wind station comparison", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "wind_station_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


def main() -> None:
    print("Wind summary:")
    winds = load_all()

    print("\n--- Figure 1: coverage ---");             plot_coverage(winds)
    print("\n--- Figure 1b: per-column coverage ---"); plot_column_coverage(winds)
    print("\n--- Figure 2: time series (2020) ---");   plot_timeseries(winds)
    print("\n--- Figure 3: autocorrelation ---");      plot_autocorrelation(winds)
    print("\n--- Figure 4: direction roses ---");      plot_direction_roses(winds)
    print("\n--- Figure 5: station comparison ---");   plot_station_comparison(winds)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
