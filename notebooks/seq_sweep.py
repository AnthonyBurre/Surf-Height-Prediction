"""Hyperparameter sweep for the RNN / GRU / LSTM / TCN sequence forecasters.

Run all four on both feature sets:
    ./.venv/bin/python notebooks/seq_sweep.py
Run on one set:
    ./.venv/bin/python notebooks/seq_sweep.py narrow
    ./.venv/bin/python notebooks/seq_sweep.py wide
Filter by model too:
    ./.venv/bin/python notebooks/seq_sweep.py narrow rnn gru
    ./.venv/bin/python notebooks/seq_sweep.py wide tcn

Reuses seq_playground's data-loading / feature-building helpers so the sweep
sees exactly the same inputs a single playground run would. Builds the data +
persistence baseline once per feature set, then loops the per-model grid over
each selected model class. Every run is logged to experiments.jsonl under the
``seqsweep_<set>`` prefix so results stay distinguishable from manual runs
and from each other.

Feature sets
------------
* ``narrow`` — 5 neighbour buoys + 3 wind stations on the full available wave
  history (year_max=2024). Matches the README headline-table inputs.
* ``wide``   — 7 neighbour buoys + 4 wind stations restricted to 2019-2024,
  the window where Palm Beach / Wide Bay / Southport all have coverage.
  Matches the README "Wider source set" table.

The wide set is **anchored on narrow's natural 80/20 split timestamp** so
both feature sets are scored against the same held-out window — only the
feature set (and the training-history depth) differ. Narrow's split point
falls inside the wide span, so wide trains on ~4 years (2019 → that
timestamp) and tests on the same ~2-year slice narrow does.

The per-model grid keeps the original anchor configs (so existing
experiments.jsonl rows remain reproducible) AND adds wider-hidden / more-epoch
/ deeper-TCN probes. The README notes RNN/LSTM regressed on the wider input
and "want re-tuning (e.g. wider hidden, more epochs)" — these added rows
test exactly that hypothesis.
"""

import sys
import time
import warnings

import pandas as pd

import forecast as fc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

# --- fixed data config (mirrors seq_playground CONFIG) ---
PRIMARY_BUOY = "mooloolaba"
FEATURE_MODE = "raw"  # raw = circular-encoded channels; fastest + best historically

# Feature sets: each entry defines a complete data window.
# Keep these aligned with linear_playground.CONFIG and the README tables.
FEATURE_SETS: dict[str, dict] = {
    "narrow": {
        "neighbours":    ["caloundra", "brisbane", "gold-coast",
                          "north-moreton-bay", "tweed-heads"],
        "wind_stations": ["mountain-creek", "deception-bay", "lytton"],
        "year_min":      None,
        "year_max":      2024,
    },
    "wide": {
        "neighbours":    ["caloundra", "brisbane", "gold-coast",
                          "north-moreton-bay", "tweed-heads",
                          "palm-beach", "wide-bay"],
        "wind_stations": ["mountain-creek", "deception-bay", "lytton",
                          "southport"],
        "year_min":      2019,
        "year_max":      2024,
    },
}
ALL_SETS = list(FEATURE_SETS)
ALL_MODELS = ["rnn", "gru", "lstm", "tcn"]

# Per-model grid. Each row is
# (seq_len, hidden, num_layers, epochs, weight_decay, rnn_dropout).
# rnn_dropout is repurposed for TCN as its native conv dropout.
#
# Anchor rows reproduce the previous best-per-model from the legacy 4nb+2w
# sweep (so existing experiments.jsonl entries stay comparable). The added
# rows probe wider hidden / more epochs / deeper TCN stacks to give the
# wider input space a fair shot at outperforming.
GRIDS: dict[str, list[tuple]] = {
    "rnn": [
        # README anchor: sl48 h128 L2 ep3 (RMSE 0.232, +23.4%)
        (48, 128, 2, 3, 0.0,    0.0),   # baseline (sanity)
        (48, 128, 2, 3, 1e-5,   0.0),
        (48, 128, 2, 3, 1e-4,   0.0),
        (48, 128, 2, 3, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.2),
        (48, 256, 2, 5, 1e-4,   0.2),
        (48,  64, 2, 5, 1e-4,   0.1),
        # --- broadened: wider hidden / more epochs for the larger inputs ---
        (48, 256, 2, 3, 1e-4,   0.1),
        (48, 256, 2, 7, 1e-4,   0.2),
        (48, 128, 2, 7, 1e-4,   0.2),
        (48, 256, 1, 3, 1e-4,   0.0),
    ],
    "gru": [
        # README anchor: sl48 h64 L1 ep2 (RMSE 0.236, +20.9%)
        (48,  64, 1, 2, 0.0,    0.0),
        (48,  64, 1, 3, 1e-4,   0.0),
        (48,  64, 1, 5, 1e-4,   0.0),
        (48, 128, 2, 3, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.2),
        (48,  64, 2, 5, 1e-4,   0.1),
        # --- broadened ---
        (48, 256, 2, 3, 1e-4,   0.1),
        (48, 256, 2, 5, 1e-4,   0.2),
        (48, 128, 1, 3, 1e-4,   0.0),
        (48, 128, 2, 7, 1e-4,   0.2),
    ],
    "lstm": [
        # README anchor: sl48 h64 L1 ep3 (RMSE 0.259, +5.3%) — easy bar
        (48,  64, 1, 3, 0.0,    0.0),
        (48,  64, 1, 3, 1e-4,   0.0),
        (48,  64, 1, 5, 1e-4,   0.0),
        (48, 128, 2, 3, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.1),
        (48, 128, 2, 5, 1e-4,   0.2),
        (48,  64, 2, 5, 1e-4,   0.1),
        # --- broadened ---
        (48, 256, 2, 3, 1e-4,   0.1),
        (48, 256, 2, 5, 1e-4,   0.2),
        (48, 128, 1, 3, 1e-4,   0.0),
        (48, 128, 2, 7, 1e-4,   0.2),
    ],
    "tcn": [
        # README anchor: sl48 channels=(64,) ep2 (RMSE 0.230, +24.9%)
        (48,  64, 1, 2, 0.0,    0.10),   # baseline, native dropout=0.1
        (48,  64, 1, 3, 1e-4,   0.10),
        (48,  64, 1, 3, 1e-4,   0.20),
        (48,  64, 2, 3, 1e-4,   0.10),
        (48,  64, 2, 3, 1e-4,   0.20),
        (48,  64, 4, 3, 1e-4,   0.20),   # full receptive field
        (48, 128, 1, 3, 1e-4,   0.20),
        # --- broadened: deeper / wider stacks for the larger inputs ---
        (48, 128, 2, 3, 1e-4,   0.20),
        (48, 128, 4, 3, 1e-4,   0.20),
        (48,  64, 4, 5, 1e-4,   0.20),
    ],
}

BATCH_SIZE = 512
LR = 1e-3
SEED = 42
SCALER = "robust"  # in-forecaster input/target scaling: "standard" or "robust"


def build_data(feature_set: str, test_start_ts: pd.Timestamp | None = None):
    """Build X / y / persistence once; return everything the sweep needs.

    ``test_start_ts`` overrides the natural 80/20 chronological split: rows
    with index strictly before the timestamp go to train, rows at or after
    go to test. Used to anchor wide on narrow's split so both feature sets
    score against the same held-out window.
    """
    cfg = FEATURE_SETS[feature_set]
    wave, neighbours, wind = fc.load_sources(
        buoy=PRIMARY_BUOY,
        neighbours=cfg["neighbours"], wind_stations=cfg["wind_stations"],
        year_min=cfg["year_min"], year_max=cfg["year_max"],
    )
    X = fc.build_design(wave, neighbours, wind, kind=FEATURE_MODE)
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]

    if test_start_ts is None:
        X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)
        X_p_tr, X_p_te, _, _ = fc.chronological_split(X_p, y)
    else:
        pos = X.index.searchsorted(test_start_ts)
        X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
        y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
        pos_p = X_p.index.searchsorted(test_start_ts)
        X_p_tr, X_p_te = X_p.iloc[:pos_p], X_p.iloc[pos_p:]
    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)

    persist = fc.evaluate(fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te)
    return {
        "X_tr": X_tr_imp, "X_te": X_te_imp, "y_tr": y_tr, "y_te": y_te,
        "train_index": X_tr.index, "test_index": X_te.index,
        "n_features": X.shape[1],
        "persist_preds": persist.predictions,
        "persist_rmse": persist.metrics["RMSE"],
        "neighbours": cfg["neighbours"],
        "wind_stations": cfg["wind_stations"],
    }


def make_model(model: str, seq_len: int, hidden: int, num_layers: int,
               epochs: int, weight_decay: float, rnn_dropout: float):
    common = dict(
        seq_len=seq_len, epochs=epochs,
        batch_size=BATCH_SIZE, lr=LR, seed=SEED, device=fc.auto_device(), verbose=False,
        scaler=SCALER, weight_decay=weight_decay,
    )
    if model == "tcn":
        return fc.TCNForecaster(
            channels=(hidden,) * num_layers, dropout=rnn_dropout, **common,
        )
    return {
        "rnn": fc.SimpleRNNForecaster,
        "gru": fc.GRUForecaster,
        "lstm": fc.LSTMForecaster,
    }[model](hidden=hidden, num_layers=num_layers, rnn_dropout=rnn_dropout, **common)


def run_one(d: dict, feature_set: str, model: str, seq_len: int, hidden: int,
            num_layers: int, epochs: int, weight_decay: float,
            rnn_dropout: float) -> dict:
    wd_tag = f"_wd{weight_decay:.0e}".replace("e-0", "e-") if weight_decay else ""
    do_tag = f"_do{rnn_dropout:g}" if rnn_dropout else ""
    name = (f"seqsweep_{feature_set}_{model}_{FEATURE_MODE}_sl{seq_len}_h{hidden}"
            f"_L{num_layers}_ep{epochs}{wd_tag}{do_tag}")
    m = make_model(model, seq_len, hidden, num_layers, epochs,
                   weight_decay, rnn_dropout)
    t0 = time.time()
    m.fit(d["X_tr"], d["y_tr"])
    preds = m.predict(d["X_te"])
    elapsed = time.time() - t0

    metrics = fc.summarise(
        d["y_te"].to_numpy(), preds, y_pred_baseline=d["persist_preds"]
    )
    result = fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=m)
    fc.log_run(
        result,
        data_sources=[PRIMARY_BUOY] + d["neighbours"] + d["wind_stations"],
        train_index=d["train_index"],
        test_index=d["test_index"],
        n_features=d["n_features"],
        extra={
            "feature_set": feature_set,
            "feature_mode": FEATURE_MODE, "seq_len": seq_len, "hidden": hidden,
            "num_layers": num_layers, "epochs": epochs, "lr": LR,
            "batch_size": BATCH_SIZE, "scaler": SCALER, "device": fc.auto_device(),
            "imputation": "mean",
            "weight_decay": weight_decay, "rnn_dropout": rnn_dropout,
            "elapsed_min": round(elapsed / 60, 2), "sweep": True,
        },
    )
    row = {
        "set": feature_set,
        "model": model, "seq_len": seq_len, "hidden": hidden,
        "num_layers": num_layers, "epochs": epochs,
        "weight_decay": weight_decay, "rnn_dropout": rnn_dropout,
        "RMSE": metrics["RMSE"], "Skill": metrics["SkillVsBaseline"],
        "MAE": metrics["MAE"], "Bias": metrics["Bias"], "secs": round(elapsed, 1),
    }
    flag = "  <-- beats persistence" if metrics["SkillVsBaseline"] > 0 else ""
    print(f"  {name:78s}  RMSE {metrics['RMSE']:.4f}  "
          f"Skill {metrics['SkillVsBaseline']:+.4f}  ({elapsed:5.1f}s){flag}",
          flush=True)
    return row


def main(sets: list[str], models: list[str]) -> None:
    # Anchor wide on narrow's natural 80/20 split so both feature sets are
    # scored against the same held-out window. Build narrow even if not
    # selected (cheap) when wide is requested, so we can extract the anchor.
    builds: dict[str, dict] = {}
    if "narrow" in sets:
        print(f"\n###### feature set: NARROW ######", flush=True)
        print(f"building data ({FEATURE_MODE} mode)...", flush=True)
        builds["narrow"] = build_data("narrow")
        d = builds["narrow"]
        print(f"persistence RMSE: {d['persist_rmse']:.4f}   "
              f"(features={d['n_features']}, train={len(d['y_tr']):,}, "
              f"test={len(d['y_te']):,}, "
              f"test={d['test_index'][0]} → {d['test_index'][-1]})\n", flush=True)

    if "wide" in sets:
        if "narrow" in builds:
            anchor_ts = builds["narrow"]["test_index"][0]
        else:
            print("(deriving wide's anchor from narrow's natural 80/20 split)",
                  flush=True)
            anchor_ts = build_data("narrow")["test_index"][0]
        print(f"\n###### feature set: WIDE (anchored on narrow split @ "
              f"{anchor_ts}) ######", flush=True)
        print(f"building data ({FEATURE_MODE} mode)...", flush=True)
        builds["wide"] = build_data("wide", test_start_ts=anchor_ts)
        d = builds["wide"]
        print(f"persistence RMSE: {d['persist_rmse']:.4f}   "
              f"(features={d['n_features']}, train={len(d['y_tr']):,}, "
              f"test={len(d['y_te']):,}, "
              f"test={d['test_index'][0]} → {d['test_index'][-1]})\n", flush=True)

    rows: list[dict] = []
    for feature_set in sets:
        d = builds[feature_set]

        for model in models:
            print(f"=== {feature_set.upper()} / {model.upper()} ===", flush=True)
            for cfg in GRIDS[model]:
                rows.append(run_one(d, feature_set, model, *cfg))
            print(flush=True)

    summary = pd.DataFrame(rows).sort_values(["set", "Skill"],
                                             ascending=[True, False])
    print("=== sweep, best skill first within each set ===")
    print(summary.to_string(index=False))

    print("\n=== best config per (set, model) ===")
    for feature_set in sets:
        for model in models:
            sub = summary[(summary["set"] == feature_set) &
                          (summary["model"] == model)]
            if sub.empty:
                continue
            best = sub.iloc[0]
            beat = "BEATS baseline" if best["Skill"] > 0 else "below baseline"
            print(f"  {feature_set:6s} {model:5s}: "
                  f"sl{int(best['seq_len'])} h{int(best['hidden'])} "
                  f"L{int(best['num_layers'])} ep{int(best['epochs'])} "
                  f"wd{best['weight_decay']:g} do{best['rnn_dropout']:g}  "
                  f"RMSE {best['RMSE']:.4f}  Skill {best['Skill']:+.4f}  ({beat})")

    print("\n=== narrow vs wide delta (best per model) ===")
    for model in models:
        best_per_set = {}
        for feature_set in sets:
            sub = summary[(summary["set"] == feature_set) &
                          (summary["model"] == model)]
            if not sub.empty:
                best_per_set[feature_set] = sub.iloc[0]
        if len(best_per_set) == 2:
            n = best_per_set["narrow"]; w = best_per_set["wide"]
            d_skill = w["Skill"] - n["Skill"]
            print(f"  {model:5s}: narrow Skill {n['Skill']:+.4f}  "
                  f"wide Skill {w['Skill']:+.4f}  delta {d_skill:+.4f}")


def _parse_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv into (sets, models). First arg may be a set name; remaining
    are model names. Defaults: all sets, all models."""
    sets, models = ALL_SETS, ALL_MODELS
    if not argv:
        return sets, models
    first = argv[0].lower()
    if first in ALL_SETS or first == "both":
        sets = ALL_SETS if first == "both" else [first]
        rest = [a.lower() for a in argv[1:]]
    else:
        rest = [a.lower() for a in argv]
    if rest:
        bad = [m for m in rest if m not in ALL_MODELS]
        if bad:
            sys.exit(f"unknown model(s) {bad!r}; choose from {ALL_MODELS}")
        models = rest
    return sets, models


if __name__ == "__main__":
    sets, models = _parse_argv(sys.argv[1:])
    main(sets, models)
