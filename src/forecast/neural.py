"""Sequence-model forecasters in PyTorch: RNN, GRU, LSTM, and a TCN.

All four share the same windowing / scaling / training harness
(``_TorchSeqForecaster``); only the encoder architecture differs.

The models window their own input: for a row at time *t* they consume
the previous ``seq_len`` feature vectors ending at *t*, and predict
``hsig_m`` at *t + HORIZON_STEPS* (the shifted target from
``data.make_target``). No lag/rolling features are needed — the
sequence model is expected to learn temporal structure itself.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TARGET_COL


def auto_device(preferred: str | None = None) -> str:
    """Pick the best available torch device, honouring an explicit preference.

    Order: explicit ``preferred`` → cuda → mps → cpu. Lives here (next to the
    forecasters that consume it) so notebooks can do ``device =
    fc.auto_device()`` without re-implementing the cascade.
    """
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(preferred: str | None) -> torch.device:
    """Resolve ``preferred`` to a real torch.device, warning on a fallback.

    A user passing ``device="cuda"`` on a CPU-only box would otherwise
    silently train on CPU and wonder why fit time blew up. Warn once.
    """
    if preferred is None:
        return torch.device(auto_device())
    if preferred == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "device='cuda' requested but CUDA is unavailable; falling back to CPU.",
            stacklevel=3,
        )
        return torch.device("cpu")
    if preferred == "mps" and not torch.backends.mps.is_available():
        warnings.warn(
            "device='mps' requested but MPS is unavailable; falling back to CPU.",
            stacklevel=3,
        )
        return torch.device("cpu")
    return torch.device(preferred)


class _WindowDataset(Dataset):
    """Emit (seq_len, n_features) slices on demand — avoids materialising the full (N, L, F) tensor."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None, seq_len: int) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, i: int):
        window = self.X[i : i + self.seq_len]
        if self.y is None:
            return window
        return window, self.y[i + self.seq_len - 1]


class _TorchSeqForecaster:
    """Shared fit/predict scaffold. Subclasses override ``_build_encoder``."""

    def __init__(
        self,
        seq_len: int = 48,
        hidden: int = 32,
        num_layers: int = 1,
        epochs: int = 5,
        batch_size: int = 512,
        lr: float = 1e-3,
        feature_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
        device: str | None = None,
        seed: int = 42,
        verbose: bool = False,
        residual: bool = True,
        scaler: str = "standard",
        weight_decay: float = 0.0,
        rnn_dropout: float = 0.0,
    ) -> None:
        if scaler not in ("standard", "robust"):
            raise ValueError(
                f"scaler must be 'standard' or 'robust', got {scaler!r}"
            )
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = _resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.residual = residual
        self.scaler = scaler
        self.weight_decay = weight_decay
        self.rnn_dropout = rnn_dropout
        self._model: nn.Module | None = None
        self._x_center: np.ndarray | None = None
        self._x_scale: np.ndarray | None = None
        self._y_center: float | None = None
        self._y_scale: float | None = None
        self._train_tail: np.ndarray | None = None

    def _build_encoder(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _select(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        return X[cols].to_numpy(dtype=np.float32)

    def _fit_scaler(self, arr: np.ndarray, axis: int | None):
        """Return (center, spread) for ``arr`` per the configured scaler.

        "standard" → mean / std; "robust" → median / IQR (resists the
        storm-spike outliers in wave data). Both are NaN-aware: a single
        NaN in any column would otherwise poison the statistic, after
        which the per-batch valid-row mask discards almost every window
        and training collapses.
        """
        if self.scaler == "robust":
            center = np.nanmedian(arr, axis=axis)
            q75, q25 = np.nanpercentile(arr, [75, 25], axis=axis)
            spread = q75 - q25
        else:  # "standard"
            center = np.nanmean(arr, axis=axis)
            spread = np.nanstd(arr, axis=axis)
        return center, spread

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchSeqForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xa = self._select(X)
        ya = y.to_numpy(dtype=np.float32)
        if self.residual:
            ya = ya - X[self.target_col].to_numpy(dtype=np.float32)

        self._x_center, x_scale = self._fit_scaler(Xa, axis=0)
        self._x_scale = x_scale + 1e-8
        y_center, y_scale = self._fit_scaler(ya, axis=None)
        self._y_center = float(y_center)
        y_scale = float(y_scale)
        if y_scale == 0:
            raise ValueError(
                f"{type(self).__name__}.fit: target has zero variance — "
                "predictions would be ill-defined. Check that y_train is not constant."
            )
        self._y_scale = y_scale + 1e-8

        # nanmedian/nanpercentile upcast to float64; keep tensors float32
        # so they match the model weights.
        Xs = ((Xa - self._x_center) / self._x_scale).astype(np.float32)
        ys = ((ya - self._y_center) / self._y_scale).astype(np.float32)

        dataset = _WindowDataset(Xs, ys, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model = self._build_encoder(Xa.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                # Skip any batch with NaN in inputs or targets (safer than
                # assuming the caller imputed everything).
                valid = (~torch.isnan(bx).any(dim=(1, 2))) & (~torch.isnan(by))
                if not valid.any():
                    continue
                bx, by = bx[valid], by[valid]
                opt.zero_grad()
                pred = model(bx).squeeze(-1)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            if self.verbose and epoch_batches:
                rmse = (epoch_loss / epoch_batches) ** 0.5 * self._y_scale
                label = "residual RMSE" if self.residual else "train RMSE"
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  {label} ≈ {rmse:.4f}")

        self._model = model
        if self.seq_len > 1:
            # Keep the last (seq_len - 1) train rows so the first test row
            # can still be given a full-history window at predict time.
            self._train_tail = Xs[-(self.seq_len - 1):].copy()
        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        Xa = self._select(X)
        Xs = ((Xa - self._x_center) / self._x_scale).astype(np.float32)
        if self._train_tail is not None:
            Xs_ext = np.concatenate([self._train_tail, Xs], axis=0)
        else:
            Xs_ext = Xs

        dataset = _WindowDataset(Xs_ext, None, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self._model.eval()
        outs: list[np.ndarray] = []
        for bx in loader:
            bx = bx.to(self.device)
            bad = torch.isnan(bx).any(dim=(1, 2))
            bx_clean = torch.nan_to_num(bx, nan=0.0)
            pred = self._model(bx_clean).squeeze(-1).cpu().numpy()
            pred = np.where(bad.cpu().numpy(), np.nan, pred)
            outs.append(pred)
        preds = np.concatenate(outs) if outs else np.empty(0, dtype=np.float32)
        preds = preds * self._y_scale + self._y_center

        # If the caller did not supply enough history (e.g. X shorter than
        # seq_len and no train tail), pad the front with NaN.
        if len(preds) < len(X):
            pad = np.full(len(X) - len(preds), np.nan, dtype=preds.dtype)
            preds = np.concatenate([pad, preds])
        preds = preds[-len(X):]
        if self.residual:
            preds = preds + X[self.target_col].to_numpy(dtype=np.float32)
        return preds


class _RNNHead(nn.Module):
    def __init__(
        self,
        cell_cls: type,
        n_features: int,
        hidden: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # PyTorch ignores dropout when num_layers == 1 and warns; suppress by zeroing.
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = cell_cls(
            n_features, hidden, num_layers=num_layers, batch_first=True,
            dropout=rnn_dropout,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class SimpleRNNForecaster(_TorchSeqForecaster):
    """Vanilla tanh RNN — the simplest recurrent baseline."""

    def _build_encoder(self, n_features: int) -> nn.Module:
        return _RNNHead(nn.RNN, n_features, self.hidden, self.num_layers, self.rnn_dropout)


class GRUForecaster(_TorchSeqForecaster):
    """Gated Recurrent Unit — fewer parameters than LSTM, often comparable in skill."""

    def _build_encoder(self, n_features: int) -> nn.Module:
        return _RNNHead(nn.GRU, n_features, self.hidden, self.num_layers, self.rnn_dropout)


class LSTMForecaster(_TorchSeqForecaster):
    """Long Short-Term Memory — the default sequence model for a reason."""

    def _build_encoder(self, n_features: int) -> nn.Module:
        return _RNNHead(nn.LSTM, n_features, self.hidden, self.num_layers, self.rnn_dropout)


class _TCNBlock(nn.Module):
    """Dilated causal conv block with residual connection.

    The trailing ``padding`` samples are trimmed so the output at position t
    depends only on inputs at positions ≤ t (causality).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.residual: nn.Module = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)[:, :, : -self.padding]
        y = self.drop(self.relu(y))
        y = self.conv2(y)[:, :, : -self.padding]
        y = self.drop(self.relu(y))
        return self.relu(y + self.residual(x))


class _TCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: tuple[int, ...] = (32, 32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            blocks.append(
                _TCNBlock(in_ch, ch, dilation=2**i, kernel_size=kernel_size, dropout=dropout)
            )
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, F) → (B, F, L) for Conv1d
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return self.fc(y[:, :, -1])


class TCNForecaster(_TorchSeqForecaster):
    """Temporal Convolutional Network — dilated causal 1D convs.

    Receptive field with default ``channels=(32,)*4`` and ``kernel_size=3``
    is ``1 + 2*(k-1)*sum(2^i for i in range(n))`` = 61 steps, more than
    enough for a 48-step context window.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _build_encoder(self, n_features: int) -> nn.Module:
        return _TCN(
            n_features,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
