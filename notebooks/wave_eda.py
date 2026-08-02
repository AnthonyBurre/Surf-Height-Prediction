"""Wave-only EDA across all eight buoys.

Run:  ./.venv/bin/python notebooks/wave_eda.py

Saves nine PNGs to notebooks/figures/ (all prefixed ``wave_``):
  wave_coverage.png                  — % valid hsig_m per buoy x year
  wave_column_coverage.png           — % valid per channel per buoy (all years pooled)
  wave_distributions.png             — violins (full history) + annual mean lines
  wave_target_yearly_stats.png       — Mooloolaba target: per-year boxplot + summary stat lines
  wave_seasonality.png               — monthly climatology + peak swell direction rose
  wave_timeseries.png                — Mooloolaba 2020 trace: hsig_m, tp_s, hmax_m
  wave_autocorrelation.png           — ACF (72h window, 12h horizon marked)
  wave_cross_source_correlation.png  — zero-lag matrix + lagged ±24h curves
  wave_neighbour_predictive.png      — corr(neighbour@t, mooloolaba@t+h) heatmap
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import viz
from forecast.data import load_data

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

BUOYS = ["wide-bay", "mooloolaba", "caloundra", "north-moreton-bay", "brisbane", "gold-coast",
         "palm-beach", "tweed-heads"]
NEIGHBOURS = ["wide-bay", "caloundra", "north-moreton-bay", "brisbane", "gold-coast",
              "palm-beach", "tweed-heads"]

COLORS: dict[str, str] = {
    "mooloolaba":        "#1f77b4",
    "caloundra":         "#ff7f0e",
    "brisbane":          "#2ca02c",
    "gold-coast":        "#d62728",
    "north-moreton-bay": "#9467bd",
    "palm-beach":        "#e377c2",
    "tweed-heads":       "#8c564b",
    "wide-bay":          "#17becf",
}


def load_all() -> dict[str, pd.DataFrame]:
    buoys: dict[str, pd.DataFrame] = {}
    print("Buoy summary (full history):")
    for name in BUOYS:
        df = load_data(buoy=name)
        buoys[name] = df
        hs = df["hsig_m"]
        print(
            f"  {name:20s}  {df.index.year.min()}-{df.index.year.max()}"
            f"  rows={len(df):>8,}  NaN={hs.isna().mean()*100:5.1f}%"
            f"  mean={hs.mean():.2f} m  p95={hs.quantile(0.95):.2f} m"
            f"  max={hs.max():.2f} m"
        )
    return buoys


# --------------------------------------------------------------------------- #
# Figure 1 — Data coverage heatmap
# --------------------------------------------------------------------------- #
def plot_coverage(buoys: dict[str, pd.DataFrame]) -> None:
    years = list(range(2010, 2026))
    rows = []
    for name, df in buoys.items():
        row: dict[int, float] = {}
        for yr in years:
            mask = df.index.year == yr
            total = int(mask.sum())
            row[yr] = float(df.loc[mask, "hsig_m"].notna().sum()) / total * 100 if total else np.nan
        rows.append(pd.Series(row, name=name))
    coverage = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_facecolor("#cccccc")  # gray = not deployed
    sns.heatmap(
        coverage, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=70, vmax=100, mask=coverage.isna(),
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% valid hsig_m", "shrink": 0.8}, ax=ax,
    )
    ax.set(
        title="Data completeness — % valid hsig_m rows per buoy per year  (grey = not deployed)",
        xlabel="Year", ylabel="",
    )
    plt.tight_layout()
    out = FIG_DIR / "wave_coverage.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    print("  Years with < 80 % valid hsig_m:")
    for name in coverage.index:
        for yr in years:
            v = coverage.loc[name, yr]
            if not np.isnan(v) and v < 80:
                print(f"    {name} {yr}: {v:.0f}%")


# --------------------------------------------------------------------------- #
# Figure 1b — Per-column coverage (all years pooled)
# --------------------------------------------------------------------------- #
def plot_column_coverage(buoys: dict[str, pd.DataFrame]) -> None:
    """% valid per channel per buoy — surfaces sst/peak-direction gaps the
    hsig_m year-level chart hides. Same intent as the wind equivalent.
    """
    rows = []
    for name, df in buoys.items():
        rows.append(pd.Series(
            {col: float(df[col].notna().mean()) * 100 for col in df.columns},
            name=name,
        ))
    coverage = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(1.4 * len(coverage.columns) + 4, 0.55 * len(coverage) + 1.8))
    sns.heatmap(
        coverage, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=50, vmax=100, linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% valid (all years pooled)", "shrink": 0.8}, ax=ax,
    )
    ax.set(
        title="Wave: per-column completeness (all years pooled)",
        xlabel="Column", ylabel="",
    )
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    out = FIG_DIR / "wave_column_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 2 — Distributions (violin) + annual mean trend
# --------------------------------------------------------------------------- #
def plot_distributions(buoys: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    data = [buoys[n]["hsig_m"].dropna().values for n in BUOYS]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[BUOYS[i]])
        body.set_alpha(0.72)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
    ax.set_xticks(range(1, len(BUOYS) + 1))
    ax.set_xticklabels([n.replace("-", "-\n") for n in BUOYS], fontsize=8)
    ax.set(ylabel="hsig_m (m)", title="Wave height distributions — full history")
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for name, df in buoys.items():
        hs = df["hsig_m"]
        annual_mean  = hs.groupby(df.index.year).mean()
        annual_valid = hs.groupby(df.index.year).apply(lambda s: s.notna().mean())
        annual_mean  = annual_mean[annual_valid > 0.5]
        ax.plot(
            annual_mean.index, annual_mean.values,
            marker="o", linewidth=1.5, label=name, color=COLORS[name],
        )
    ax.set(
        xlabel="Year", ylabel="Annual mean hsig_m (m)",
        title="Inter-annual variability (years > 50 % valid)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "wave_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 2b — Target (Mooloolaba hsig_m) statistics by year
# --------------------------------------------------------------------------- #
def plot_target_yearly_stats(mool: pd.DataFrame) -> None:
    """Per-year hsig_m stats for the Mooloolaba target — boxplot + summary lines.

    The annual-mean panel in ``plot_distributions`` only shows the central
    tendency across buoys; here we focus on the *target* and expose the full
    distribution shape per year (IQR, p95/p99, max) so storm-season variability
    is visible. Years with <50 % valid hsig_m are dropped to avoid misleading
    partial-year stats.
    """
    hs = mool["hsig_m"]
    grouped = hs.groupby(mool.index.year)
    valid_frac = grouped.apply(lambda s: s.notna().mean())
    years = valid_frac[valid_frac > 0.5].index.tolist()

    stats = pd.DataFrame({
        "min":   [grouped.get_group(y).min()           for y in years],
        "p25":   [grouped.get_group(y).quantile(0.25)  for y in years],
        "mean":  [grouped.get_group(y).mean()          for y in years],
        "p50":   [grouped.get_group(y).quantile(0.50)  for y in years],
        "p75":   [grouped.get_group(y).quantile(0.75)  for y in years],
        "p95":   [grouped.get_group(y).quantile(0.95)  for y in years],
        "p99":   [grouped.get_group(y).quantile(0.99)  for y in years],
        "max":   [grouped.get_group(y).max()           for y in years],
        "count": [int(grouped.get_group(y).notna().sum()) for y in years],
    }, index=years)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: per-year boxplot showing median, IQR, whiskers, outliers
    ax = axes[0]
    data = [grouped.get_group(y).dropna().to_numpy() for y in years]
    bp = ax.boxplot(
        data, positions=years, widths=0.65,
        showfliers=True, flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
        patch_artist=True, medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["mooloolaba"])
        patch.set_alpha(0.55)
    ax.set(
        xlabel="Year", ylabel="hsig_m (m)",
        title="Per-year distribution — Mooloolaba hsig_m",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=0)

    # Right: layered summary statistics — IQR band, mean/median, tail percentiles, max
    ax = axes[1]
    ax.fill_between(years, stats["p25"], stats["p75"],
                    alpha=0.20, color=COLORS["mooloolaba"], label="IQR (p25–p75)")
    ax.plot(years, stats["mean"], marker="o", color=COLORS["mooloolaba"],
            linewidth=2.0, label="mean")
    ax.plot(years, stats["p50"],  marker="s", color=COLORS["mooloolaba"],
            linewidth=1.3, linestyle="--", alpha=0.75, label="median")
    ax.plot(years, stats["p95"],  marker="^", color="#ff7f0e", linewidth=1.4, label="p95")
    ax.plot(years, stats["p99"],  marker="^", color="#d62728", linewidth=1.4, label="p99")
    ax.plot(years, stats["max"],  marker="*", color="black", linewidth=0.9,
            linestyle=":", markersize=11, label="max")
    ax.set(
        xlabel="Year", ylabel="hsig_m (m)",
        title="Yearly summary statistics — Mooloolaba hsig_m",
    )
    ax.legend(fontsize=8, loc="upper left", ncols=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=0)

    plt.tight_layout()
    out = FIG_DIR / "wave_target_yearly_stats.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    print("\n  Per-year stats (Mooloolaba hsig_m, m):")
    print(stats.round(2).to_string())
    calm  = stats["mean"].idxmin()
    stormy = stats["mean"].idxmax()
    tail  = stats["max"].idxmax()
    print(f"\n  Calmest year   : {calm}  (mean {stats.loc[calm, 'mean']:.2f} m)")
    print(f"  Stormiest year : {stormy}  (mean {stats.loc[stormy, 'mean']:.2f} m)")
    print(f"  Largest peak   : {tail}  (max {stats.loc[tail, 'max']:.2f} m)")


# --------------------------------------------------------------------------- #
# Figure 3 — Seasonal climatology + peak direction roses
# --------------------------------------------------------------------------- #
def plot_seasonality(buoys: dict[str, pd.DataFrame]) -> None:
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = plt.figure(figsize=(14, 5))
    ax_season = fig.add_subplot(1, 2, 1)
    ax_dir    = fig.add_subplot(1, 2, 2, projection="polar")

    for name, df in buoys.items():
        monthly_median = df["hsig_m"].groupby(df.index.month).median()
        ax_season.plot(
            monthly_median.index, monthly_median.values,
            marker="o", linewidth=1.5, label=name, color=COLORS[name],
        )
    ax_season.set_xticks(range(1, 13))
    ax_season.set_xticklabels(month_labels, fontsize=8)
    ax_season.set(ylabel="Median hsig_m (m)", title="Seasonal climatology (all years pooled)")
    ax_season.legend(fontsize=8)
    ax_season.grid(alpha=0.3)

    bin_edges = np.linspace(0, 2 * np.pi, 37)
    bin_width = bin_edges[1] - bin_edges[0]
    for name, df in buoys.items():
        dirs_rad = np.deg2rad(df["peak_dir_deg"].dropna().values)
        counts, _ = np.histogram(dirs_rad, bins=bin_edges)
        fracs = counts / counts.sum()
        ax_dir.bar(bin_edges[:-1], fracs, width=bin_width, alpha=0.45,
                   label=name, color=COLORS[name])
    ax_dir.set_theta_zero_location("N")
    ax_dir.set_theta_direction(-1)
    ax_dir.set_title("Peak swell direction distribution", pad=14)
    ax_dir.legend(loc="lower left", bbox_to_anchor=(-0.18, -0.12), fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "wave_seasonality.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    print("  Seasonal peak (month of highest median hsig_m):")
    for name, df in buoys.items():
        monthly = df["hsig_m"].groupby(df.index.month).median()
        peak_m  = int(monthly.idxmax())
        print(f"    {name:20s}  peak month = {month_labels[peak_m - 1]}"
              f"  ({monthly.max():.2f} m median)")


# --------------------------------------------------------------------------- #
# Figure 4 — Time series (single representative year, three wave channels)
# --------------------------------------------------------------------------- #
def plot_timeseries(mooloolaba: pd.DataFrame) -> None:
    year = mooloolaba.loc["2020"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    viz.plot_series(year["hsig_m"], ylabel="hsig_m (m)",
                    title="Significant wave height — Mooloolaba 2020", ax=axes[0])
    viz.plot_series(year["tp_s"], ylabel="tp_s (s)",
                    title="Peak wave period — Mooloolaba 2020", ax=axes[1], color="#2ca02c")
    viz.plot_series(year["hmax_m"], ylabel="hmax_m (m)",
                    title="Maximum wave height — Mooloolaba 2020", ax=axes[2], color="#d62728")
    fig.tight_layout()
    out = FIG_DIR / "wave_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 5 — Autocorrelation curves (three wave channels)
# --------------------------------------------------------------------------- #
def plot_autocorrelation(mooloolaba: pd.DataFrame) -> None:
    channels = [
        (mooloolaba["hsig_m"].dropna(), "hsig_m"),
        (mooloolaba["tp_s"].dropna(),   "tp_s"),
        (mooloolaba["hmax_m"].dropna(), "hmax_m"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (series, label) in zip(axes, channels):
        viz.autocorrelation_curve(
            series, max_hours=72, step_hours=1.0, sampling_freq_min=30,
            highlight_hours=[12], threshold=0.5,
            source_label=f"Mooloolaba {label}", ax=ax,
        )
    fig.suptitle("Autocorrelation vs lag — 72-h window, 12-h forecast horizon marked", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "wave_autocorrelation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 6 — Cross-source: zero-lag matrix + lagged ±24 h curves
# --------------------------------------------------------------------------- #
def plot_cross_source(buoys: dict[str, pd.DataFrame]) -> None:
    start = max(df.index.min() for df in buoys.values())
    end   = min(df.index.max() for df in buoys.values())
    print(f"\n  Overlap window: {start.date()} → {end.date()}")

    sources = {name: buoys[name].loc[start:end, "hsig_m"] for name in BUOYS}
    mool = sources["mooloolaba"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _, corr = viz.cross_source_correlation(sources, ax=axes[0])
    print("\n  Zero-lag Pearson r (hsig_m, overlap window):")
    print(corr.round(3).to_string())

    lags_h  = np.arange(-24, 25, 1)
    ax = axes[1]
    print("\n  Peak lagged correlation (neighbour leads Mooloolaba):")
    for name in NEIGHBOURS:
        s  = sources[name]
        rs = [s.shift(int(round(h * 2))).corr(mool) for h in lags_h]
        ax.plot(lags_h, rs, linewidth=1.5, label=name, color=COLORS[name])
        peak_lag = lags_h[int(np.nanargmax(rs))]
        print(f"    {name:20s}  peak r={max(rs):.3f} at lag={peak_lag:+d} h")

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set(
        xlabel="Lag h  (positive = neighbour leads Mooloolaba)",
        ylabel="Pearson r vs Mooloolaba hsig_m",
        title="Lagged cross-correlation ±24 h (overlap window)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "wave_cross_source_correlation.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 7 — Predictive horizon heatmap (neighbour@t vs mooloolaba@t+h)
# --------------------------------------------------------------------------- #
def plot_neighbour_predictive(buoys: dict[str, pd.DataFrame]) -> None:
    start = max(df.index.min() for df in buoys.values())
    end   = min(df.index.max() for df in buoys.values())

    df_all = pd.DataFrame({
        name: buoys[name].loc[start:end, "hsig_m"] for name in BUOYS
    })

    fig, ax = plt.subplots(figsize=(12, 3.8))
    _, grid = viz.feature_horizon_heatmap(
        df_all, target_col="mooloolaba", feature_cols=NEIGHBOURS,
        horizons_h=(0, 1, 3, 6, 12, 24, 48), sampling_freq_min=30, ax=ax,
    )
    plt.tight_layout()
    out = FIG_DIR / "wave_neighbour_predictive.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)

    print("\n  Pearson r (neighbour@t vs Mooloolaba@t+h):")
    print(grid.round(3).to_string())
    print(f"Saved {out.name}")


def main() -> None:
    buoys = load_all()
    mool = buoys["mooloolaba"]

    print("\n--- Figure 1: coverage ---");                plot_coverage(buoys)
    print("\n--- Figure 1b: per-column coverage ---");    plot_column_coverage(buoys)
    print("\n--- Figure 2: distributions & trends ---");  plot_distributions(buoys)
    print("\n--- Figure 2b: target yearly stats ---");     plot_target_yearly_stats(mool)
    print("\n--- Figure 3: seasonality & direction ---"); plot_seasonality(buoys)
    print("\n--- Figure 4: time series (2020) ---");      plot_timeseries(mool)
    print("\n--- Figure 5: autocorrelation ---");         plot_autocorrelation(mool)
    print("\n--- Figure 6: cross-source correlation ---"); plot_cross_source(buoys)
    print("\n--- Figure 7: neighbour predictive ---");    plot_neighbour_predictive(buoys)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
