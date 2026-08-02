import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from forecast import (
    EvaluationResult,
    PersistenceForecaster,
    best_metric,
    best_run,
    evaluate,
    evaluate_and_log,
    find_runs,
    latest_metric,
    latest_run,
    log_run,
    read_log,
)
from forecast.runlog import _jsonable, _model_hyperparams


# ---------------------------------------------------------------------------
# _jsonable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(7), 7),
        (np.float32(1.5), pytest.approx(1.5)),
        (np.float64("nan"), None),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (Path("/tmp/x"), "/tmp/x"),
        ({"a": np.int64(1), "b": [np.float64(2.0), "x"]}, {"a": 1, "b": [2.0, "x"]}),
    ],
    ids=["np_int", "np_float32", "np_nan", "np_array", "path", "nested"],
)
def test_jsonable(value, expected):
    assert _jsonable(value) == expected


def test_jsonable_handles_timestamp():
    ts = pd.Timestamp("2020-01-01", tz="Australia/Brisbane")
    assert _jsonable(ts).startswith("2020-01-01")


def test_jsonable_falls_back_to_str_for_unknown_types():
    class Weird:
        def __repr__(self):
            return "<Weird>"
    assert _jsonable(Weird()) == "<Weird>"


# ---------------------------------------------------------------------------
# _model_hyperparams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected_subset",
    [
        (Ridge(alpha=2.5, fit_intercept=False), {"alpha": 2.5, "fit_intercept": False}),
        (PersistenceForecaster(target_col="custom"), {"target_col": "custom"}),
        (None, {}),
    ],
    ids=["sklearn_get_params", "custom_vars", "none"],
)
def test_model_hyperparams(model, expected_subset):
    params = _model_hyperparams(model)
    if not expected_subset:
        assert params == {}
    else:
        for key, val in expected_subset.items():
            assert params[key] == val


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


def test_log_run_appends_one_record(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="persistence")
    log_path = tmp_path / "exp.jsonl"

    log_run(
        result,
        data_sources=["mooloolaba"],
        train_index=Xtr.index,
        test_index=Xte.index,
        n_features=Xtr.shape[1],
        path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "persistence"
    assert parsed["model_class"] == "PersistenceForecaster"
    assert parsed["data_sources"] == ["mooloolaba"]
    assert parsed["train"]["n"] == len(Xtr)
    assert parsed["test"]["n"] == len(Xte)
    assert parsed["n_features"] == Xtr.shape[1]
    assert "MAE" in parsed["metrics"]


def test_log_run_appends_does_not_overwrite(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    for n in ("a", "b", "c"):
        result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name=n)
        log_run(result, data_sources=["s"],
                train_index=Xtr.index, test_index=Xte.index, path=log_path)
    assert len(log_path.read_text().splitlines()) == 3


def test_log_run_records_git_sha_when_in_repo(tmp_path, split):
    """Test suite runs from a git repo, so sha should be a 40-char hex string
    (possibly with -dirty suffix). Off-repo runs should yield None."""
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="r")
    record = log_run(result, data_sources=["s"],
                     train_index=Xtr.index, test_index=Xte.index, path=tmp_path / "x.jsonl")
    sha = record["git_sha"]
    assert sha is None or (len(sha.split("-")[0]) == 40)


def test_log_run_serialises_sklearn_hyperparams(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    result = evaluate(Ridge(alpha=0.7), Xtr, ytr, Xte, yte, name="ridge")
    log_run(result, data_sources=["s"],
            train_index=Xtr.index, test_index=Xte.index, path=log_path)
    parsed = json.loads(log_path.read_text())
    assert parsed["model_class"] == "Ridge"
    assert parsed["hyperparams"]["alpha"] == 0.7


def test_log_run_extra_field_passes_through(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="r")
    record = log_run(
        result, data_sources=["s"],
        train_index=Xtr.index, test_index=Xte.index,
        extra={"feature_set": "raw_v1", "notes": "smoke test"},
        path=tmp_path / "x.jsonl",
    )
    assert record["extra"]["feature_set"] == "raw_v1"


# ---------------------------------------------------------------------------
# evaluate_and_log
# ---------------------------------------------------------------------------


def test_evaluate_and_log_writes_and_returns_result(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    result = evaluate_and_log(
        PersistenceForecaster(), Xtr, ytr, Xte, yte,
        data_sources=["x"], name="run1", path=log_path,
    )
    assert isinstance(result, EvaluationResult)
    parsed = json.loads(log_path.read_text())
    assert parsed["name"] == "run1"
    assert parsed["metrics"]["RMSE"] == pytest.approx(result.metrics["RMSE"])


# ---------------------------------------------------------------------------
# read_log
# ---------------------------------------------------------------------------


def test_read_log_returns_dataframe(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    for name in ["a", "b", "c"]:
        evaluate_and_log(PersistenceForecaster(), Xtr, ytr, Xte, yte,
                         data_sources=["s"], name=name, path=log_path)
    df_log = read_log(log_path)
    assert len(df_log) == 3
    assert {"timestamp", "name", "metrics"} <= set(df_log.columns)
    assert set(df_log["name"]) == {"a", "b", "c"}


def test_read_log_missing_file_returns_empty(tmp_path):
    assert read_log(tmp_path / "no-such-file.jsonl").empty


def test_read_log_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.touch()
    assert read_log(p).empty


# ---------------------------------------------------------------------------
# find_runs / latest_run / latest_metric
# ---------------------------------------------------------------------------


def _seed_log(tmp_path, split, rows):
    """Append rows={name, model, extra, test_extra?} to a fresh JSONL."""
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    for r in rows:
        result = evaluate(r["model"], Xtr, ytr, Xte, yte, name=r["name"])
        # Optionally override test window so we can filter on a date.
        train_idx = r.get("train_index", Xtr.index)
        test_idx  = r.get("test_index",  Xte.index)
        log_run(
            result, data_sources=["s"],
            train_index=train_idx, test_index=test_idx,
            extra=r.get("extra"),
            path=log_path,
            model_class=r.get("model_class"),
        )
    return log_path


def test_find_runs_empty_log_returns_empty(tmp_path):
    assert find_runs(path=tmp_path / "x.jsonl", model_class="Ridge").empty


def test_find_runs_filters_by_model_class(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "p1", "model": PersistenceForecaster()},
        {"name": "r1", "model": Ridge(alpha=0.5)},
        {"name": "r2", "model": Ridge(alpha=1.0)},
    ])
    hits = find_runs(model_class="Ridge", path=log_path)
    assert sorted(hits["name"].tolist()) == ["r1", "r2"]


def test_find_runs_filters_by_name_prefix(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "hsweep_a", "model": PersistenceForecaster()},
        {"name": "hsweep_b", "model": PersistenceForecaster()},
        {"name": "other",    "model": PersistenceForecaster()},
    ])
    hits = find_runs(name_prefix="hsweep_", path=log_path)
    assert sorted(hits["name"].tolist()) == ["hsweep_a", "hsweep_b"]


def test_find_runs_filters_by_extra_kwargs(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "a", "model": PersistenceForecaster(), "extra": {"horizon_h": 12}},
        {"name": "b", "model": PersistenceForecaster(), "extra": {"horizon_h": 24}},
        {"name": "c", "model": PersistenceForecaster(), "extra": {"horizon_h": 12, "feature_set": "wide"}},
    ])
    hits = find_runs(horizon_h=12, path=log_path)
    assert sorted(hits["name"].tolist()) == ["a", "c"]
    hits = find_runs(horizon_h=12, feature_set="wide", path=log_path)
    assert hits["name"].tolist() == ["c"]


def test_find_runs_filters_by_test_window_prefix(tmp_path, split):
    _df, Xtr, Xte, _, _ = split
    idx_2023 = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=len(Xte), freq="30min", tz="Australia/Brisbane"))
    idx_2024 = pd.DatetimeIndex(pd.date_range("2024-06-01", periods=len(Xte), freq="30min", tz="Australia/Brisbane"))
    log_path = _seed_log(tmp_path, split, [
        {"name": "a", "model": PersistenceForecaster(), "test_index": idx_2023},
        {"name": "b", "model": PersistenceForecaster(), "test_index": idx_2024},
    ])
    hits = find_runs(test_start="2023-", path=log_path)
    assert hits["name"].tolist() == ["a"]


def test_latest_run_returns_most_recent(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "tcn_1", "model": PersistenceForecaster(), "model_class": "TCNForecaster",
         "extra": {"horizon_h": 12}},
        {"name": "tcn_2", "model": PersistenceForecaster(), "model_class": "TCNForecaster",
         "extra": {"horizon_h": 12}},
    ])
    row = latest_run(model_class="TCNForecaster", horizon_h=12, path=log_path)
    assert row is not None
    assert row["name"] == "tcn_2"


def test_latest_run_returns_none_when_no_match(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "r", "model": PersistenceForecaster()},
    ])
    assert latest_run(model_class="TCNForecaster", path=log_path) is None


def test_latest_metric_returns_value(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "tcn", "model": PersistenceForecaster(), "model_class": "TCNForecaster",
         "extra": {"horizon_h": 12}},
    ])
    rmse = latest_metric("RMSE", model_class="TCNForecaster", horizon_h=12, path=log_path)
    assert rmse is not None
    assert rmse > 0


def test_latest_metric_returns_none_for_unknown_metric(tmp_path, split):
    log_path = _seed_log(tmp_path, split, [
        {"name": "r", "model": PersistenceForecaster()},
    ])
    assert latest_metric("not_a_metric", name="r", path=log_path) is None


# ---------------------------------------------------------------------------
# best_run / best_metric
# ---------------------------------------------------------------------------


def _append_raw(path: Path, **record_overrides) -> None:
    """Append one hand-crafted JSONL record — lets tests control metric values
    without round-tripping a real model fit."""
    base = {
        "timestamp": "2025-01-01T00:00:00+00:00",
        "git_sha": None,
        "name": "r",
        "model_class": "Ridge",
        "hyperparams": {},
        "data_sources": ["s"],
        "n_features": 1,
        "train": {"start": None, "end": None, "n": 0},
        "test":  {"start": None, "end": None, "n": 0},
        "metrics": {"RMSE": 1.0, "MAE": 0.5},
        "extra": {},
    }
    base.update(record_overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(base) + "\n")


def test_best_run_minimises_by_default(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    _append_raw(log_path, name="a", model_class="TCNForecaster", metrics={"RMSE": 0.50})
    _append_raw(log_path, name="b", model_class="TCNForecaster", metrics={"RMSE": 0.20})
    _append_raw(log_path, name="c", model_class="TCNForecaster", metrics={"RMSE": 0.40})
    row = best_run("RMSE", model_class="TCNForecaster", path=log_path)
    assert row is not None and row["name"] == "b"


def test_best_run_maximises_when_asked(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    _append_raw(log_path, name="a", metrics={"SkillVsBaseline": 0.10})
    _append_raw(log_path, name="b", metrics={"SkillVsBaseline": 0.30})
    _append_raw(log_path, name="c", metrics={"SkillVsBaseline": 0.25})
    row = best_run("SkillVsBaseline", maximise=True, path=log_path)
    assert row is not None and row["name"] == "b"


def test_best_run_respects_filters(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    # Globally lowest RMSE is on a non-TCN row; best_run should ignore it.
    _append_raw(log_path, name="ridge_low", model_class="Ridge", metrics={"RMSE": 0.10})
    _append_raw(log_path, name="tcn_a",     model_class="TCNForecaster", metrics={"RMSE": 0.50},
                extra={"horizon_h": 12})
    _append_raw(log_path, name="tcn_b",     model_class="TCNForecaster", metrics={"RMSE": 0.30},
                extra={"horizon_h": 24})
    _append_raw(log_path, name="tcn_c",     model_class="TCNForecaster", metrics={"RMSE": 0.40},
                extra={"horizon_h": 12})
    row = best_run("RMSE", model_class="TCNForecaster", horizon_h=12, path=log_path)
    assert row is not None and row["name"] == "tcn_c"  # min RMSE among h=12 TCNs


def test_best_run_skips_rows_missing_the_metric(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    _append_raw(log_path, name="a", metrics={"MAE": 0.1})   # no RMSE
    _append_raw(log_path, name="b", metrics={"RMSE": 0.5})
    _append_raw(log_path, name="c", metrics={"RMSE": None})  # null → skipped
    row = best_run("RMSE", path=log_path)
    assert row is not None and row["name"] == "b"


def test_best_run_returns_none_when_no_match(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    _append_raw(log_path, name="r", model_class="Ridge")
    assert best_run("RMSE", model_class="TCNForecaster", path=log_path) is None


def test_best_metric_returns_value(tmp_path):
    log_path = tmp_path / "exp.jsonl"
    _append_raw(log_path, name="a", model_class="TCNForecaster", metrics={"RMSE": 0.50})
    _append_raw(log_path, name="b", model_class="TCNForecaster", metrics={"RMSE": 0.20})
    val = best_metric("RMSE", model_class="TCNForecaster", path=log_path)
    assert val == 0.20
