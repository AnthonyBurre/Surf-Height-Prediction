"""Feature preprocessing — sparse-column drop, mean imputation, feature scaling.

Two layers:

- **Standalone helpers** (``drop_sparse_columns``, ``mean_impute``,
  ``scale_features``) — pure ``(X_train, X_test) → (X_train', X_test')``
  transforms. Cheap to call ad-hoc in a notebook; no fit-time state escapes.

- **``Preprocessor`` class** — the same three steps bundled into a stateful
  fit/transform object. ``fit(X_train)`` learns the drop list, imputer means,
  and scaler stats; ``transform(X)`` applies them deterministically to any
  future frame and asserts the schema matches. ``save()``/``load()``
  round-trips the fitted object via ``pickle`` so a model + its preprocessor
  can ship together. Use this when you care about the train→serve path
  (e.g. scoring a held-out year against a pre-committed configuration).
"""
import logging
import pickle
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

_SCALERS = {"robust": RobustScaler, "standard": StandardScaler}
ScalingMethod = Literal["robust", "standard"]


# --------------------------------------------------------------------------- #
# Standalone helpers — stateless transforms over a (train, test) pair.
# --------------------------------------------------------------------------- #


def drop_sparse_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    max_nan_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns whose **train-set** NaN fraction exceeds ``max_nan_frac``.

    Imputation turns such a column near-constant, which is silently corrosive
    for gradient-based sequence models: every window contains the imputed
    value, the gradient is dominated by it, but it carries no real signal
    (e.g. Lytton's ``wind_speed_std_ms`` at 90 % NaN — the column is
    essentially absent from that station's feed). Dropping these columns
    before ``mean_impute`` keeps the feature matrix honest.

    Returns ``(X_train, X_test)`` with the dropped columns removed from both,
    and logs the names and NaN fractions of anything dropped so the deletion
    is never silent.
    """
    nan_frac = X_train.isna().mean()
    drop = nan_frac[nan_frac > max_nan_frac].sort_values(ascending=False)
    if drop.empty:
        return X_train, X_test
    for col, frac in drop.items():
        logger.info("drop_sparse_columns: dropping %s (%.1f%% NaN)", col, frac * 100)
    msg = (
        f"drop_sparse_columns: dropped {len(drop)}/{len(nan_frac)} columns "
        f"with > {max_nan_frac * 100:.0f}% train-set NaN"
    )
    logger.info(msg)
    print(msg)
    for col, frac in drop.items():
        print(f"  - {col}  ({frac * 100:.1f}% NaN)")
    return X_train.drop(columns=drop.index), X_test.drop(columns=drop.index)


def mean_impute(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a column-wise mean imputer on train, apply to both frames.

    Sequence models in particular need NaN-free inputs: with seq_len=48 and
    any column carrying a few % NaN, almost every window contains a NaN and
    training collapses. Linear/tree models also benefit when chained with
    sklearn estimators that reject NaN.
    """
    imp = SimpleImputer(strategy="mean")
    return (
        pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index),
        pd.DataFrame(imp.transform(X_test),      columns=X_test.columns,  index=X_test.index),
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    method: ScalingMethod = "robust",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a feature scaler on train, apply to both frames.

    ``method`` is "robust" (RobustScaler — median/IQR, resists storm-spike
    outliers in wave data) or "standard" (StandardScaler).

    Circular columns (suffix ``_sin``/``_cos``) are passed through untouched:
    they are already in [-1, 1], and scaling sin/cos independently would
    distort the unit-circle relationship. Penalised linear models (Ridge,
    Lasso) need this so ``alpha`` shrinks every coefficient on a comparable
    scale; tree models are scale-invariant and don't need it.
    """
    scale_cols = [c for c in X_train.columns if not c.endswith(("_sin", "_cos"))]
    scaler = _SCALERS[method]()
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    Xte[scale_cols] = scaler.transform(X_test[scale_cols])
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Stateful Preprocessor — the bundle that knows what it learned.
# --------------------------------------------------------------------------- #


class Preprocessor:
    """Stateful drop → impute → (optional) scale pipeline.

    Why this exists (vs the three standalone helpers above): the helpers are
    transient — they fit the imputer/scaler on the spot and the choices are
    lost the moment the function returns. That's fine for one-shot notebook
    runs where train and test live in the same Python session, but it makes
    the train→serve path implicit: a held-out year scored against a
    "pre-committed" model has no record of which columns to drop, which means
    to impute with, or what scaling to apply.

    ``Preprocessor.fit(X_train)`` learns the drop list, imputer means, and
    scaler stats once. ``transform(X)`` applies them deterministically. The
    fitted object pickles cleanly via ``save()``/``load()``, so a trained
    model and its preprocessor can ship together.

    Parameters
    ----------
    max_nan_frac
        Columns whose **train-set** NaN fraction exceeds this are dropped at
        ``fit()`` time. Default 0.5. Set to 1.0 to disable.
    scaling
        ``None``, ``"robust"`` (default for linear models), or ``"standard"``.
        Sequence models scale internally — pass ``None``.

    Examples
    --------
    >>> prep = Preprocessor(scaling="robust").fit(X_train)
    >>> X_train_p = prep.transform(X_train)
    >>> X_test_p  = prep.transform(X_test)
    >>> prep.save("models/ridge_preproc.pkl")
    >>> # Later, in a serving environment:
    >>> prep = Preprocessor.load("models/ridge_preproc.pkl")
    >>> X_new_p = prep.transform(X_new)  # raises if X_new lacks any kept column
    """

    def __init__(
        self,
        *,
        max_nan_frac: float = 0.5,
        scaling: ScalingMethod | None = None,
    ) -> None:
        if scaling is not None and scaling not in _SCALERS:
            raise ValueError(f"scaling must be one of {list(_SCALERS)} or None, got {scaling!r}")
        self.max_nan_frac = max_nan_frac
        self.scaling = scaling
        # Fit-time state — populated by fit(); None until then.
        self.dropped_columns_: list[str] | None = None
        self.kept_columns_: list[str] | None = None
        self.scale_columns_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: RobustScaler | StandardScaler | None = None

    @property
    def is_fitted(self) -> bool:
        return self.kept_columns_ is not None

    def fit(self, X_train: pd.DataFrame) -> Self:
        """Learn drop list, imputer means, and scaler stats from training data."""
        nan_frac = X_train.isna().mean()
        drop_mask = nan_frac > self.max_nan_frac
        self.dropped_columns_ = nan_frac.index[drop_mask].tolist()
        self.kept_columns_ = [c for c in X_train.columns if c not in self.dropped_columns_]

        if self.dropped_columns_:
            for col in self.dropped_columns_:
                logger.info(
                    "Preprocessor.fit: dropping %s (%.1f%% NaN)",
                    col, nan_frac[col] * 100,
                )
            msg = (
                f"Preprocessor.fit: dropped {len(self.dropped_columns_)}/{len(nan_frac)} "
                f"columns with > {self.max_nan_frac * 100:.0f}% train-set NaN"
            )
            logger.info(msg)
            # Mirror the standalone helper: surface drops in stdout so notebook
            # runs don't silently shed columns.
            print(msg)
            for col in self.dropped_columns_:
                print(f"  - {col}  ({nan_frac[col] * 100:.1f}% NaN)")

        X_kept = X_train[self.kept_columns_]
        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp_arr = self.imputer_.fit_transform(X_kept)

        if self.scaling is not None:
            self.scale_columns_ = [c for c in self.kept_columns_ if not c.endswith(("_sin", "_cos"))]
            X_imp = pd.DataFrame(X_imp_arr, columns=self.kept_columns_, index=X_kept.index)
            self.scaler_ = _SCALERS[self.scaling]()
            self.scaler_.fit(X_imp[self.scale_columns_])
        else:
            self.scale_columns_ = None
            self.scaler_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned drop/impute/scale steps to a new frame.

        Raises ``RuntimeError`` if ``fit()`` hasn't been called and
        ``ValueError`` if any column the preprocessor was fitted on is
        missing from ``X``. Extra columns in ``X`` not seen at fit time are
        silently ignored — drift in the *source set* is recoverable, but a
        missing required column would corrupt downstream predictions and so
        is treated as a hard error.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted; call fit(X_train) first.")
        missing = [c for c in self.kept_columns_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} columns the preprocessor was fitted on: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        X_kept = X[self.kept_columns_]
        X_arr = self.imputer_.transform(X_kept)
        X_out = pd.DataFrame(X_arr, columns=self.kept_columns_, index=X_kept.index)

        if self.scaler_ is not None:
            X_out[self.scale_columns_] = self.scaler_.transform(X_out[self.scale_columns_])
        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    def save(self, path: str | Path) -> None:
        """Pickle the fitted preprocessor to disk.

        Plain ``pickle`` is intentional — no new dependency, and the object
        only holds scikit-learn primitives plus a few lists. Be aware that
        pickle is **not** stable across major sklearn versions; pair the
        artifact with its training environment.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted; nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
