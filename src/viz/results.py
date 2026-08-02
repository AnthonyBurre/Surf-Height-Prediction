"""Result-comparison plots driven by long-format run tables.

These helpers consume DataFrames the caller has already pulled out of
``experiments.jsonl`` (typically via :func:`forecast.find_runs` plus a
small reshape). The viz layer stays domain-agnostic: it knows nothing
about which name patterns mean what — the notebook is responsible for
labelling rows in a way that reads well in the chart legend.

Long-format input convention
----------------------------
Every plot here expects a "tall" DataFrame with one row per measurement:

    horizon_col   metric_col   label_col
    ----------    ----------   ---------
    6             0.2541       baseline / hgb
    6             0.2900       persistence
    12            0.3444       wide / ensemble
    ...

The ``baseline_label`` row(s) are drawn as a black dashed reference line
and excluded from winner-finding.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


# Colour palette cycled by index of distinct winning labels in the order
# they first appear on the x-axis — gives stable colours across reruns
# as long as the winning labels don't change.
WINNER_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]


DEFAULT_BASELINE_STYLE = {
    "color": "#000000", "linestyle": "--", "linewidth": 1.6,
    "marker": "o", "markersize": 5, "alpha": 0.7,
}


COLLAPSED_BASELINE_LABEL = "best baseline"


def plot_horizon_winners(
    runs: pd.DataFrame,
    *,
    horizon_col: str = "horizon_h",
    metric_col: str = "RMSE",
    label_col: str = "label",
    baseline_label: str | dict[str, dict] = "persistence",
    collapse_baselines: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Single panel: metric-vs-horizon for the per-horizon winners.

    For every horizon, the row with the lowest ``metric_col`` (excluding
    any baseline row) is the winner. For each distinct winning label, the
    function then draws that label's full trajectory across every horizon
    so the reader can see how it compares everywhere — not just the one
    horizon it won. A star marker highlights each label's winning
    horizon(s).

    ``baseline_label`` accepts either:

    - a single string — that label is drawn with the default black-dashed
      reference style (``DEFAULT_BASELINE_STYLE``);
    - a dict mapping ``{label: matplotlib_line_kwargs}`` — each baseline
      drawn with its own style. Use this to overlay multiple reference
      lines (e.g. persistence solid + climatology dotted).

    When ``collapse_baselines=True``, all baseline rows are merged into a
    single per-horizon-minimum line labelled ``"best baseline"``; the
    rendering style is taken from the *first* entry of ``baseline_label``
    (so callers can still control the look). Useful when the individual
    baselines have very different magnitudes and only the lower envelope
    is interesting for setting the y-axis floor.

    All baseline labels are excluded from winner-finding.

    Args:
        runs: long-format DataFrame; one row per (label, horizon).
        horizon_col: column with integer/numeric forecast horizons.
        metric_col:  column to minimise (default RMSE).
        label_col:   column with the line label (e.g. "wide / ensemble").
        baseline_label: single label or {label: style_kwargs} for the
            reference line(s); drawn dashed by default, ineligible to win.
        collapse_baselines: collapse all baseline rows to one min-per-
            horizon line; style taken from the first baseline_label entry.
        title: figure title.
        ax: existing axes to draw into; created if None.

    Returns the matplotlib Axes for further customisation.
    """
    if runs.empty:
        raise ValueError("plot_horizon_winners: runs is empty")
    missing = {horizon_col, metric_col, label_col} - set(runs.columns)
    if missing:
        raise KeyError(f"runs is missing required columns: {sorted(missing)}")

    # Normalise baseline_label into {label: style_kwargs}.
    if isinstance(baseline_label, str):
        baseline_styles = {baseline_label: dict(DEFAULT_BASELINE_STYLE)}
    else:
        baseline_styles = {k: {**DEFAULT_BASELINE_STYLE, **v}
                           for k, v in baseline_label.items()}

    horizons = sorted(runs[horizon_col].unique())
    is_baseline   = runs[label_col].isin(baseline_styles.keys())
    baseline_rows = runs[is_baseline]
    contenders    = runs[~is_baseline]

    # --- Per-horizon winners ----------------------------------------------
    per_h_winner: dict[int, tuple[str, float]] = {}
    for h in horizons:
        sub = contenders[contenders[horizon_col] == h]
        if sub.empty:
            continue
        winning = sub.loc[sub[metric_col].idxmin()]
        per_h_winner[h] = (winning[label_col], float(winning[metric_col]))

    # Distinct winning labels in the order they first appear (left-to-right
    # along the x-axis) — keeps line colours stable across reruns.
    distinct: list[str] = []
    for h in horizons:
        if h in per_h_winner:
            lab = per_h_winner[h][0]
            if lab not in distinct:
                distinct.append(lab)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6.5))

    # Baseline reference line(s). When collapse_baselines is True, all
    # baseline rows reduce to a single min-per-horizon envelope; style is
    # borrowed from the first entry in baseline_label.
    if collapse_baselines and not baseline_rows.empty:
        style = next(iter(baseline_styles.values()))
        ref = (
            baseline_rows.groupby(horizon_col)[metric_col]
            .min()
            .reindex(horizons)
        )
        ax.plot(horizons, ref.values,
                label=f"{COLLAPSED_BASELINE_LABEL} (no-model)", **style)
    else:
        for blabel, style in baseline_styles.items():
            sub = baseline_rows[baseline_rows[label_col] == blabel]
            if sub.empty:
                continue
            ref = sub.groupby(horizon_col)[metric_col].mean().reindex(horizons)
            ax.plot(horizons, ref.values, label=f"{blabel} (no-model)", **style)

    # One trajectory per distinct winner, star-marking its winning horizon(s).
    for i, lab in enumerate(distinct):
        sub = contenders[contenders[label_col] == lab].set_index(horizon_col)[metric_col]
        sub = sub.reindex(horizons)
        color = WINNER_COLORS[i % len(WINNER_COLORS)]
        ax.plot(horizons, sub.values, color=color, linewidth=2.2,
                marker="o", markersize=7, label=lab)
        for h, val in sub.items():
            if pd.notna(val) and per_h_winner.get(h, (None,))[0] == lab:
                ax.plot(h, val, marker="*", color=color, markersize=18,
                        markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5)

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_xlabel(f"Forecast horizon")
    ax.set_ylabel(metric_col)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              title=f"winning {label_col}", title_fontsize=9)
    return ax
