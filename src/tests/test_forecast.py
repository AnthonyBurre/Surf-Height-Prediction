import numpy as np
import pandas as pd
import pytest

from forecast import (
    ClimatologyHourForecaster,
    HORIZON_STEPS,
    PersistenceForecaster,
    Preprocessor,
    SOURCE_TZ,
    add_lag_features,
    add_momentum,
    add_neighbour_features,
    add_rolling_features,
    bias,
    build_buoy_features,
    build_seq_features,
    chronological_split,
    compare,
    drop_sparse_columns,
    encode_circular,
    evaluate,
    mae,
    make_target,
    mean_impute,
    restrict_to_overlap,
    restrict_to_years,
    rmse,
    scale_features,
    skill_score,
    summarise,
)


# ---------------------------------------------------------------------------
# data.make_target
# ---------------------------------------------------------------------------


def test_make_target_shifts_by_horizon(synthetic_df):
    df = synthetic_df(100)
    y = make_target(df, horizon_steps=HORIZON_STEPS)
    # y at t should equal hsig_m at t + HORIZON_STEPS
    assert y.iloc[0] == pytest.approx(df["hsig_m"].iloc[HORIZON_STEPS])
    assert y.iloc[50] == pytest.approx(df["hsig_m"].iloc[50 + HORIZON_STEPS])


def test_make_target_tail_is_nan(synthetic_df):
    df = synthetic_df(100)
    y = make_target(df, horizon_steps=HORIZON_STEPS)
    # The last HORIZON_STEPS values have no future observation to target.
    assert y.iloc[-HORIZON_STEPS:].isna().all()
    assert not y.iloc[:-HORIZON_STEPS].isna().any()


def test_make_target_name_reflects_horizon(synthetic_df):
    y = make_target(synthetic_df(50), horizon_steps=6, target_col="hsig_m")
    assert y.name == "hsig_m_plus_6"


def test_make_target_custom_column(synthetic_df):
    df = synthetic_df(50)
    y = make_target(df, horizon_steps=2, target_col="sst_c")
    assert y.iloc[0] == pytest.approx(df["sst_c"].iloc[2])


# ---------------------------------------------------------------------------
# data.chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_sizes(synthetic_df):
    df = synthetic_df(100)
    y = make_target(df, horizon_steps=2)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.2)
    assert len(Xtr) == 80
    assert len(Xte) == 20
    assert len(ytr) == 80
    assert len(yte) == 20


def test_chronological_split_preserves_order(synthetic_df):
    df = synthetic_df(100)
    y = make_target(df, horizon_steps=2)
    Xtr, Xte, _, _ = chronological_split(df, y, test_frac=0.2)
    # All training timestamps must precede all test timestamps.
    assert Xtr.index.max() < Xte.index.min()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_chronological_split_rejects_invalid_frac(bad, synthetic_df):
    df = synthetic_df(50)
    y = make_target(df, horizon_steps=2)
    with pytest.raises(ValueError):
        chronological_split(df, y, test_frac=bad)


# ---------------------------------------------------------------------------
# features — leakage + correctness
# ---------------------------------------------------------------------------


def test_add_lag_features_values_match_shift(synthetic_df):
    df = synthetic_df(20)
    out = add_lag_features(df, columns=["hsig_m"], lags=[1, 3])
    # lag_k at row i should equal hsig_m at row i-k
    assert out["hsig_m_lag_1"].iloc[1] == pytest.approx(df["hsig_m"].iloc[0])
    assert out["hsig_m_lag_3"].iloc[5] == pytest.approx(df["hsig_m"].iloc[2])
    # first k rows are NaN
    assert out["hsig_m_lag_3"].iloc[:3].isna().all()


def test_add_rolling_features_are_shifted_by_one(synthetic_df):
    """Rolling features must summarise strictly-past values (see the
    ``add_rolling_features`` docstring for why this convention is enforced
    even though the current 12h horizon doesn't strictly require it)."""
    df = synthetic_df(30)
    out = add_rolling_features(df, columns=["hsig_m"], windows=[4], stats=("mean",))
    # shift(1) then rolling(4) at row 10 aggregates the shifted series'
    # positions 7..10, which are the unshifted series' positions 6..9.
    expected = df["hsig_m"].iloc[6:10].mean()
    assert out["hsig_m_roll4_mean"].iloc[10] == pytest.approx(expected)


def test_add_rolling_features_rejects_unknown_stat(synthetic_df):
    with pytest.raises(ValueError, match="Unknown stats"):
        add_rolling_features(synthetic_df(10), columns=["hsig_m"], windows=[4], stats=("zscore",))


def test_add_momentum_computes_value_minus_lag(synthetic_df):
    df = synthetic_df(20)
    out = add_momentum(df, columns=["hsig_m"], deltas=[1, 3])
    # delta_k at row i should equal hsig_m[i] - hsig_m[i-k]
    assert out["hsig_m_delta_1"].iloc[5] == pytest.approx(
        df["hsig_m"].iloc[5] - df["hsig_m"].iloc[4]
    )
    assert out["hsig_m_delta_3"].iloc[7] == pytest.approx(
        df["hsig_m"].iloc[7] - df["hsig_m"].iloc[4]
    )
    # first k rows are NaN (no prior reference)
    assert out["hsig_m_delta_3"].iloc[:3].isna().all()


def test_encode_circular_hour_virtual_loops_after_24h(synthetic_df):
    # 49 rows of 30-min data covers 24h plus the next 00:00 timestamp.
    df = synthetic_df(49)
    out = encode_circular(df, periods={"hour": 24.0})
    assert ((out["hour_sin"] >= -1.0) & (out["hour_sin"] <= 1.0)).all()
    assert ((out["hour_cos"] >= -1.0) & (out["hour_cos"] <= 1.0)).all()
    # Over a full day the sin/cos trace should loop back to the starting value.
    assert out["hour_sin"].iloc[0] == pytest.approx(out["hour_sin"].iloc[48], abs=1e-9)
    assert out["hour_cos"].iloc[0] == pytest.approx(out["hour_cos"].iloc[48], abs=1e-9)


def test_encode_circular_doy_virtual_does_not_drop_data_columns(synthetic_df):
    df = synthetic_df(10)
    out = encode_circular(df, periods={"doy": 365.25})
    assert "doy_sin" in out.columns
    assert "doy_cos" in out.columns
    assert set(df.columns) <= set(out.columns)


def test_encode_circular_replaces_original_column(synthetic_df):
    df = synthetic_df(20)
    out = encode_circular(df)
    assert "peak_dir_deg" not in out.columns
    assert "peak_dir_deg_sin" in out.columns
    assert "peak_dir_deg_cos" in out.columns
    # sin^2 + cos^2 == 1 for all rows
    ss = out["peak_dir_deg_sin"] ** 2 + out["peak_dir_deg_cos"] ** 2
    assert np.allclose(ss, 1.0)


def test_encode_circular_handles_360_equals_0():
    df = pd.DataFrame({"peak_dir_deg": [0.0, 360.0]}, index=pd.date_range("2020-01-01", periods=2, freq="30min"))
    out = encode_circular(df)
    assert out["peak_dir_deg_sin"].iloc[0] == pytest.approx(out["peak_dir_deg_sin"].iloc[1])
    assert out["peak_dir_deg_cos"].iloc[0] == pytest.approx(out["peak_dir_deg_cos"].iloc[1])


def test_encode_circular_rejects_unknown_name(synthetic_df):
    with pytest.raises(ValueError, match="not in df.columns"):
        encode_circular(synthetic_df(5), periods={"not_a_column": 1.0})


# ---------------------------------------------------------------------------
# features — high-level pipelines
# ---------------------------------------------------------------------------


def test_build_buoy_features_includes_each_feature_family(synthetic_df):
    """build_buoy_features composes circular + lag + rolling + momentum.
    Assert by family rather than exact column list so the test survives
    FeatureConfig defaults shifting."""
    out = build_buoy_features(synthetic_df(200))
    cols = set(out.columns)
    # circular: peak_dir_deg replaced; hour/doy added
    assert "peak_dir_deg" not in cols
    assert {"peak_dir_deg_sin", "peak_dir_deg_cos",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"} <= cols
    # at least one of each derived family
    assert any(c.startswith("hsig_m_lag_") for c in cols)
    assert any(c.startswith("hsig_m_roll") for c in cols)
    assert any(c.startswith("hsig_m_delta_") for c in cols)


def test_build_seq_features_omits_lag_and_rolling(synthetic_df):
    """Sequence models window their own input, so build_seq_features must
    only encode circular features — no lag, rolling, or momentum columns."""
    out = build_seq_features(synthetic_df(50))
    cols = set(out.columns)
    assert "peak_dir_deg_sin" in cols and "peak_dir_deg_cos" in cols
    assert "hour_sin" in cols and "doy_sin" in cols
    assert not any("_lag_" in c for c in cols)
    assert not any("_roll" in c for c in cols)
    assert not any("_delta_" in c for c in cols)


def test_add_neighbour_features_appends_without_dropping_rows(synthetic_df):
    main = synthetic_df(100)
    neighbour = synthetic_df(100, seed=1).rename(columns={"hsig_m": "nb_hsig_m"})
    out = add_neighbour_features(main, neighbour, columns=["nb_hsig_m"])
    # Original columns and row count survive.
    assert set(main.columns) <= set(out.columns)
    assert len(out) == len(main)
    # Raw neighbour value at lag 0 equals the source.
    assert out["nb_hsig_m"].equals(neighbour["nb_hsig_m"])
    # At least one lag and one rolling-mean column appeared.
    assert any(c.startswith("nb_hsig_m_lag") for c in out.columns)
    assert any(c.startswith("nb_hsig_m_roll") and c.endswith("_mean") for c in out.columns)


def test_restrict_to_overlap_clips_to_neighbour_intersection():
    """With no wind, the window is the intersection of neighbour-buoy
    valid ranges."""
    idx = pd.date_range("2020-01-01", periods=10, freq="30min", tz="Australia/Brisbane")
    wave = pd.DataFrame({"hsig_m": np.arange(10, dtype=float)}, index=idx)
    # Neighbour A is valid 1..8; Neighbour B is valid 3..7. Intersection: 3..7.
    a = pd.Series([np.nan, *range(1, 9), np.nan], index=idx)
    b = pd.Series([np.nan]*3 + list(range(3, 8)) + [np.nan]*2, index=idx)
    neighbours = {"a": a, "b": b}

    wave_out, nb_out, wind_out = restrict_to_overlap(wave, neighbours, wind=None)

    assert wind_out is None
    assert wave_out.index.min() == idx[3]
    assert wave_out.index.max() == idx[7]
    for s in nb_out.values():
        assert s.index.min() == idx[3]
        assert s.index.max() == idx[7]


def test_restrict_to_overlap_wind_window_wins_over_neighbours():
    """When wind is present its valid range defines the window even if it
    is tighter than the neighbour intersection. A regression here would
    silently drop most of the training set."""
    idx = pd.date_range("2020-01-01", periods=10, freq="30min", tz="Australia/Brisbane")
    wave = pd.DataFrame({"hsig_m": np.arange(10, dtype=float)}, index=idx)
    # Neighbour valid 1..8 — would normally clip to 1..8.
    nb = pd.Series([np.nan, *range(1, 9), np.nan], index=idx)
    # Wind only has data in rows 4..6 — tighter; should win.
    wind = pd.DataFrame({"wind_speed_ms": [np.nan]*4 + [3.0, 3.1, 3.2] + [np.nan]*3}, index=idx)

    wave_out, nb_out, wind_out = restrict_to_overlap(wave, {"a": nb}, wind=wind)

    assert wave_out.index.min() == idx[4]
    assert wave_out.index.max() == idx[6]
    assert nb_out["a"].index.min() == idx[4]
    assert wind_out.index.min() == idx[4]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_persistence_predict_equals_current_target(synthetic_df):
    df = synthetic_df(50)
    y = make_target(df, horizon_steps=2)
    preds = PersistenceForecaster().fit(df, y).predict(df)
    np.testing.assert_allclose(preds, df["hsig_m"].to_numpy())


def test_persistence_perfect_on_constant_series():
    """A flat series is exactly forecastable by persistence."""
    idx = pd.date_range("2020-01-01", periods=100, freq="30min", tz="Australia/Brisbane")
    df = pd.DataFrame({"hsig_m": 1.5}, index=idx)
    y = make_target(df, horizon_steps=24)
    preds = PersistenceForecaster().fit(df, y).predict(df)
    mask = ~y.isna()
    np.testing.assert_allclose(preds[mask], y[mask])


def test_climatology_hour_predicts_training_hourly_mean():
    # Build a series whose hsig_m depends only on hour-of-day.
    n = 48 * 10  # 10 days of 30-min data
    idx = pd.date_range("2020-01-01", periods=n, freq="30min", tz="Australia/Brisbane")
    hsig = idx.hour.to_numpy(dtype=float)  # equals the hour-of-day
    df = pd.DataFrame({"hsig_m": hsig}, index=idx)
    y = make_target(df, horizon_steps=24)  # 12h ahead
    model = ClimatologyHourForecaster(horizon_steps=24).fit(df, y)
    preds = model.predict(df.iloc[:48])
    # For the first row (t=00:00), forecast time is 12:00, so prediction = 12.
    assert preds[0] == pytest.approx(12.0)
    # For row at t=06:00 (index 12), forecast time is 18:00, so prediction = 18.
    assert preds[12] == pytest.approx(18.0)


def test_climatology_predict_before_fit_raises(synthetic_df):
    with pytest.raises(RuntimeError):
        ClimatologyHourForecaster().predict(synthetic_df(50))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_mae_rmse_bias_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx((0.5 + 0 + 1.0) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((0.25 + 0 + 1.0) / 3))
    assert bias(y_true, y_pred) == pytest.approx((0.5 + 0 - 1.0) / 3)


def test_metrics_ignore_nan():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.nan, 5.0])
    # Only rows 0 and 3 are finite in both.
    assert mae(y_true, y_pred) == pytest.approx((0 + 1.0) / 2)


@pytest.mark.parametrize(
    "y_true, y_pred, baseline, predicate",
    [
        # model == baseline → skill 0
        ([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], lambda s: s == pytest.approx(0.0)),
        # model perfect, baseline far off → skill 1
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], lambda s: s == pytest.approx(1.0)),
        # model worse than baseline → skill < 0
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.1, 2.0, 2.9], lambda s: s < 0),
    ],
    ids=["zero", "perfect", "negative"],
)
def test_skill_score(y_true, y_pred, baseline, predicate):
    score = skill_score(np.array(y_true), np.array(y_pred), np.array(baseline))
    assert predicate(score)


def test_summarise_includes_skill_only_when_baseline_given():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    out = summarise(y_true, y_pred)
    assert "SkillVsBaseline" not in out
    out2 = summarise(y_true, y_pred, y_pred_baseline=np.array([0.0, 0.0, 0.0]))
    assert "SkillVsBaseline" in out2


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_metrics_and_predictions(synthetic_df):
    df = synthetic_df(200)
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.25)
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="persistence")
    assert result.name == "persistence"
    assert {"MAE", "RMSE", "Bias"} <= set(result.metrics)
    assert len(result.predictions) == len(Xte)


def test_evaluate_masks_nan_training_rows(synthetic_df):
    """A lag feature with NaNs in the first rows shouldn't crash model.fit."""
    df = synthetic_df(200)
    X = add_lag_features(df, columns=["hsig_m"], lags=[10])
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(X, y, test_frac=0.25)
    from sklearn.linear_model import LinearRegression
    result = evaluate(LinearRegression(), Xtr, ytr, Xte, yte, name="lr")
    assert not np.isnan(result.metrics["MAE"])


def test_compare_sorts_by_rmse(synthetic_df):
    df = synthetic_df(200)
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.25)
    r1 = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="a")
    r2 = evaluate(ClimatologyHourForecaster(horizon_steps=4), Xtr, ytr, Xte, yte, name="b")
    table = compare([r1, r2])
    assert list(table.index) == sorted(table.index, key=lambda m: table.loc[m, "RMSE"])


# ---------------------------------------------------------------------------
# Documented-baseline anchor
# ---------------------------------------------------------------------------


def test_persistence_baseline_matches_documented_values():
    """Lock the canonical persistence baseline to the current evaluator.

    The README's headline table and `experiments.jsonl` quote persistence
    metrics for the standard config (mooloolaba primary, +12h horizon,
    80/20 chronological split, year_max=2024, no overlap-restriction). This
    test asserts those numbers reproduce from the current code path so
    that any change to the evaluator, metric definitions, target horizon,
    split, or data file is caught the moment it diverges from the doc.

    If this test fails:
      - Intentional change? Update the expected values here AND the
        README's headline-table row in the same commit.
      - Unintentional? You've drifted from what the project documents.

    Skipped when the data CSV isn't present (e.g. CI without bulk data).
    """
    from pathlib import Path

    import forecast as fc

    data_dir = Path(fc.__file__).parents[2] / "data"
    if not list(data_dir.glob("mooloolaba_wave_data_*.csv")):
        pytest.skip("mooloolaba wave CSV not present; baseline anchor needs it.")

    wave = fc.restrict_to_years(fc.load_data(buoy="mooloolaba"), None, 2024)
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]
    X_tr, X_te, y_tr, y_te = fc.chronological_split(X_p, y)
    metrics = fc.evaluate(
        fc.PersistenceForecaster(), X_tr, y_tr, X_te, y_te
    ).metrics

    # Locked 2026-05-21 against the current data file + evaluator. A 3 mm
    # tolerance absorbs minor data refreshes (single-year QC updates) but
    # would still catch the MAE-vs-RMSE confusion that prompted this test
    # (gap is ~133 mm).
    assert metrics["MAE"] == pytest.approx(0.2660, abs=0.003)
    assert metrics["RMSE"] == pytest.approx(0.3996, abs=0.003)


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------


def _scale_frames():
    idx_tr = pd.date_range("2020-01-01", periods=50, freq="30min", name="datetime")
    idx_te = pd.date_range("2020-02-01", periods=20, freq="30min", name="datetime")
    cols = ["hsig_m", "tp_s", "peak_dir_deg_sin", "peak_dir_deg_cos"]
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(
        {
            "hsig_m": rng.normal(2.0, 0.5, 50),
            "tp_s": rng.normal(9.0, 2.0, 50),
            "peak_dir_deg_sin": rng.uniform(-1, 1, 50),
            "peak_dir_deg_cos": rng.uniform(-1, 1, 50),
        },
        index=idx_tr,
    )[cols]
    # Test frame deliberately shifted to a different distribution.
    Xte = pd.DataFrame(
        {
            "hsig_m": rng.normal(5.0, 0.5, 20),
            "tp_s": rng.normal(15.0, 2.0, 20),
            "peak_dir_deg_sin": rng.uniform(-1, 1, 20),
            "peak_dir_deg_cos": rng.uniform(-1, 1, 20),
        },
        index=idx_te,
    )[cols]
    return Xtr, Xte


def test_scale_features_leaves_circular_columns_untouched():
    Xtr, Xte = _scale_frames()
    Str, Ste = scale_features(Xtr, Xte, method="robust")
    for col in ("peak_dir_deg_sin", "peak_dir_deg_cos"):
        pd.testing.assert_series_equal(Str[col], Xtr[col])
        pd.testing.assert_series_equal(Ste[col], Xte[col])


def test_scale_features_centres_non_circular_train_columns():
    Xtr, _ = _scale_frames()
    Str, _ = scale_features(Xtr, Xtr, method="robust")
    # RobustScaler centres on the median -> scaled train median ~ 0.
    assert Str["hsig_m"].median() == pytest.approx(0.0, abs=1e-9)
    assert Str["tp_s"].median() == pytest.approx(0.0, abs=1e-9)


def test_scale_features_preserves_columns_order_and_index():
    Xtr, Xte = _scale_frames()
    Str, Ste = scale_features(Xtr, Xte, method="standard")
    assert list(Str.columns) == list(Xtr.columns)
    assert list(Ste.columns) == list(Xte.columns)
    pd.testing.assert_index_equal(Str.index, Xtr.index)
    pd.testing.assert_index_equal(Ste.index, Xte.index)


def test_scale_features_applies_train_stats_to_test():
    """Test frame is transformed with train-fitted stats, not its own."""
    Xtr, Xte = _scale_frames()
    Str, Ste = scale_features(Xtr, Xte, method="robust")
    # Test distribution sits well above train -> scaled test stays far from 0.
    assert Ste["hsig_m"].median() > 1.0
    assert Ste["tp_s"].median() > 1.0


# ---------------------------------------------------------------------------
# drop_sparse_columns + mean_impute (standalone helpers)
# ---------------------------------------------------------------------------


def _sparse_frames():
    """Train: dense_col is 100% valid; sparse_col is 80% NaN. Test mirrors schema."""
    idx_tr = pd.date_range("2020-01-01", periods=100, freq="30min", name="datetime")
    idx_te = pd.date_range("2020-03-01", periods=40, freq="30min", name="datetime")
    rng = np.random.default_rng(0)
    sparse = np.full(100, np.nan)
    sparse[:20] = rng.normal(0, 1, 20)  # 80 of 100 missing → above the 0.5 default
    Xtr = pd.DataFrame({"dense_col": rng.normal(0, 1, 100), "sparse_col": sparse}, index=idx_tr)
    Xte = pd.DataFrame(
        {"dense_col": rng.normal(0, 1, 40), "sparse_col": rng.normal(0, 1, 40)},
        index=idx_te,
    )
    return Xtr, Xte


def test_drop_sparse_columns_removes_columns_above_threshold(capsys):
    Xtr, Xte = _sparse_frames()
    Xtr_d, Xte_d = drop_sparse_columns(Xtr, Xte, max_nan_frac=0.5)
    assert "sparse_col" not in Xtr_d.columns
    assert "sparse_col" not in Xte_d.columns
    assert "dense_col" in Xtr_d.columns
    # Surface the drop in stdout for notebook runs.
    out = capsys.readouterr().out
    assert "sparse_col" in out


def test_drop_sparse_columns_returns_inputs_unchanged_when_nothing_drops():
    Xtr, _ = _scale_frames()  # all four columns are fully populated
    Xte, _ = _scale_frames()
    Xtr_d, Xte_d = drop_sparse_columns(Xtr, Xte, max_nan_frac=0.5)
    pd.testing.assert_frame_equal(Xtr_d, Xtr)
    pd.testing.assert_frame_equal(Xte_d, Xte)


def test_mean_impute_uses_train_means_for_test():
    Xtr, Xte = _sparse_frames()
    train_mean = Xtr["sparse_col"].mean()
    Xtr_imp, Xte_imp = mean_impute(Xtr, Xte)
    assert not Xtr_imp.isna().any().any()
    assert not Xte_imp.isna().any().any()
    # NaN cells in train are filled with the train mean.
    nan_positions = Xtr["sparse_col"].isna()
    assert Xtr_imp.loc[nan_positions, "sparse_col"].eq(train_mean).all()


# ---------------------------------------------------------------------------
# Preprocessor — stateful drop / impute / scale bundle
# ---------------------------------------------------------------------------


def test_preprocessor_fit_records_drop_and_kept_columns():
    Xtr, _ = _sparse_frames()
    prep = Preprocessor(max_nan_frac=0.5).fit(Xtr)
    assert prep.dropped_columns_ == ["sparse_col"]
    assert prep.kept_columns_ == ["dense_col"]
    assert prep.is_fitted


def test_preprocessor_transform_drops_same_columns_at_inference():
    Xtr, Xte = _sparse_frames()
    prep = Preprocessor(max_nan_frac=0.5).fit(Xtr)
    Xte_out = prep.transform(Xte)
    # Inference-time frame has no NaN in sparse_col, but the column is still
    # dropped because the *training-time* decision is what governs.
    assert list(Xte_out.columns) == ["dense_col"]


def test_preprocessor_transform_raises_on_missing_kept_column():
    Xtr, _ = _sparse_frames()
    prep = Preprocessor(max_nan_frac=0.5).fit(Xtr)
    # Inference input missing a column the preprocessor was fitted on.
    X_new = pd.DataFrame({"sparse_col": [1.0]}, index=pd.date_range("2021", periods=1, freq="30min", name="datetime"))
    with pytest.raises(ValueError, match="missing 1 columns"):
        prep.transform(X_new)


def test_preprocessor_transform_ignores_extra_columns():
    Xtr, Xte = _sparse_frames()
    prep = Preprocessor(max_nan_frac=0.5).fit(Xtr)
    Xte_with_extra = Xte.assign(bonus_col=1.0)  # new column not seen at fit time
    Xte_out = prep.transform(Xte_with_extra)
    # Extras are silently dropped (source-set drift is recoverable).
    assert list(Xte_out.columns) == ["dense_col"]


def test_preprocessor_transform_raises_before_fit():
    prep = Preprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        prep.transform(pd.DataFrame({"a": [1.0]}))


def test_preprocessor_applies_scaler_when_configured():
    Xtr, _ = _scale_frames()
    prep = Preprocessor(scaling="robust").fit(Xtr)
    Xtr_out = prep.transform(Xtr)
    assert Xtr_out["hsig_m"].median() == pytest.approx(0.0, abs=1e-9)
    # Circular columns pass through untouched.
    pd.testing.assert_series_equal(Xtr_out["peak_dir_deg_sin"], Xtr["peak_dir_deg_sin"])


def test_preprocessor_save_and_load_round_trips(tmp_path):
    Xtr, Xte = _sparse_frames()
    prep = Preprocessor(max_nan_frac=0.5, scaling="robust").fit(Xtr)
    expected = prep.transform(Xte)

    path = tmp_path / "preproc.pkl"
    prep.save(path)
    loaded = Preprocessor.load(path)

    assert loaded.dropped_columns_ == prep.dropped_columns_
    assert loaded.kept_columns_ == prep.kept_columns_
    pd.testing.assert_frame_equal(loaded.transform(Xte), expected)


def test_preprocessor_rejects_unknown_scaling():
    with pytest.raises(ValueError, match="scaling must be one of"):
        Preprocessor(scaling="minmax")  # type: ignore[arg-type]


def test_preprocessor_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError, match="nothing to save"):
        Preprocessor().save(tmp_path / "x.pkl")


# ---------------------------------------------------------------------------
# data.restrict_to_years (AEST-aware year slice)
# ---------------------------------------------------------------------------


def _aest_frame_around_year_boundary() -> pd.DataFrame:
    """Six AEST-tagged rows straddling 2024→2025.

    Project convention is AEST-tagged indexes throughout (set by
    ``qld_ckan.wave.pipeline.clean``), so ``df.index.year`` already returns
    the AEST year. These rows verify ``restrict_to_years`` slices on the
    Brisbane boundary as expected — the first two are 2024, the last four
    are 2025.
    """
    idx = pd.DatetimeIndex(
        [
            "2024-12-31 23:00",
            "2024-12-31 23:30",
            "2025-01-01 00:00",
            "2025-01-01 00:30",
            "2025-01-01 01:00",
            "2025-01-01 01:30",
        ],
        tz="Australia/Brisbane",
        name="datetime",
    )
    return pd.DataFrame({"hsig_m": np.arange(6.0)}, index=idx)


def test_restrict_to_years_year_max_inclusive():
    df = _aest_frame_around_year_boundary()
    kept = restrict_to_years(df, year_min=None, year_max=2024)
    assert kept["hsig_m"].tolist() == [0.0, 1.0]


def test_restrict_to_years_year_min_inclusive():
    df = _aest_frame_around_year_boundary()
    kept = restrict_to_years(df, year_min=2025, year_max=None)
    assert kept["hsig_m"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_restrict_to_years_both_bounds_slice_inclusively():
    df = _aest_frame_around_year_boundary()
    assert restrict_to_years(df, 2024, 2024)["hsig_m"].tolist() == [0.0, 1.0]
    assert restrict_to_years(df, 2025, 2025)["hsig_m"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_restrict_to_years_no_bounds_returns_input_unchanged():
    df = _aest_frame_around_year_boundary()
    out = restrict_to_years(df, year_min=None, year_max=None)
    # Identity-equal — no rebuild when there's nothing to do.
    assert out is df


def test_source_tz_constant_matches_qld_ckan_clean():
    # The string must match qld_ckan.{wave,wind}.pipeline._SOURCE_TZ —
    # both refer to the same canonical project timezone. Hard-coded here
    # so a stray rename in either place would surface a test failure
    # rather than a silent semantic drift.
    assert SOURCE_TZ == "Australia/Brisbane"
