"""
LensNN.py
=========

Neural-network inference for lens-pair scoring using the ResTCN NNET
trained in `resTCN_NNET_train_on_BIG_same_DGDR_inference_FULL.ipynb`.

Main function
-------------
    run_lensnn_inference(input_csv, model_path, output_csv, ...)

It:
1. reads a light-curve CSV;
2. builds every available component pair inside each source_id;
3. rebuilds the same 9-channel local derivative features used at training:
   time_feature, flux_z, highpass, d1, d2, dt_prev, dt_next, local_amp, mask;
4. loads `resTCN.pt` or another exported NNET bundle;
5. writes a local CSV with exactly:
       sourceID, compA, compB, Proba

The function call is intentionally notebook-friendly and similar to the RF helper.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - import error is clearer at runtime
    raise ImportError(
        "LensNN.py requires PyTorch. Install torch in the same environment used by the notebook."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration copied from the NNET notebook
# -----------------------------------------------------------------------------

@dataclass
class CFG:
    # Real light-curve columns
    source_col: str = "source_id"
    id_col: str = "lensComponentSourceId"
    time_col: str = "epoch_obs_jd"
    flux_col: str = "flux_obs"
    err_col: str = "flux_obs_error"
    outlier_col: str = "flag_outlier"

    # Dataset geometry
    T_days: float = 2000.0
    max_abs_delay: float = 760.0
    max_len: int = 80
    min_points_per_image: int = 5
    augment_swap_train: bool = True

    # Local derivative features
    input_dim: int = 9
    latent_deriv_dim: int = 56
    use_absolute_time_feature: bool = False

    # Twin ResTCN encoder
    hidden: int = 72
    tcn_blocks: int = 8
    kernel_size: int = 5
    dropout: float = 0.12
    n_filter_heads: int = 4

    delay_fusion_hidden: int = 32
    delay_fusion_kernel: int = 5

    # Delay grid + soft alignment
    n_delay_grid: int = 201
    align_sigma_days: float = 30.0
    delay_target_sigma_days: float = 28.0
    peak_exclusion_days: float = 60.0
    peak_margin: float = 0.35

    # Training values retained for bundle compatibility
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1.2e-3
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.20
    num_workers: int = 0

    lambda_cls: float = 0.25
    lambda_delay: float = 1.25
    lambda_peak: float = 0.25
    lambda_neg_entropy: float = 0.02
    lambda_symmetry: float = 0.10


def cfg_from_dict(d: Optional[Dict[str, Any]]) -> CFG:
    """Create CFG while ignoring unknown keys from older/newer bundles."""
    cfg = CFG()
    if not isinstance(d, dict):
        return cfg
    valid = {f.name for f in fields(CFG)}
    for key, value in d.items():
        if key in valid:
            try:
                setattr(cfg, key, value)
            except Exception:
                pass
    return cfg


# -----------------------------------------------------------------------------
# Same derivative feature extraction as the training notebook
# -----------------------------------------------------------------------------

def robust_zscore(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) + eps
    z = (y - med) / (1.4826 * mad + eps)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32)


def moving_average_np(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 3:
        return x.copy()
    k = min(int(k), len(x))
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k, dtype=np.float32) / k, mode="valid").astype(np.float32)


def safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient used by the notebook; fixes duplicate/non-increasing x locally."""
    if len(y) <= 1:
        return np.zeros_like(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x2 = x.copy()
    for i in range(1, len(x2)):
        if x2[i] <= x2[i - 1]:
            x2[i] = x2[i - 1] + 1e-3
    return np.gradient(y, x2).astype(np.float32)


def _downsample_sorted_arrays(t: np.ndarray, y: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the full temporal support when a curve has more than max_len observations."""
    n = len(t)
    if n <= max_len:
        return t, y
    idx = np.linspace(0, n - 1, max_len)
    idx = np.unique(np.round(idx).astype(int))
    if len(idx) < max_len:
        used = set(int(i) for i in idx)
        missing = [i for i in range(n) if i not in used]
        idx = np.sort(np.r_[idx, missing[: max_len - len(idx)]])
    return t[idx[:max_len]], y[idx[:max_len]]


def make_local_derivative_features(t: np.ndarray, y: np.ndarray, cfg: CFG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return padded features [max_len, 9], padded times [max_len], mask [max_len].

    Feature channels are exactly the training channels:
    0 time_feature
    1 robust-z flux
    2 highpass flux
    3 d1 first derivative
    4 d2 second derivative
    5 dt_prev
    6 dt_next
    7 local_amp
    8 valid mask
    """
    t = np.asarray(t, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]

    if len(t) == 0:
        X = np.zeros((cfg.max_len, cfg.input_dim), dtype=np.float32)
        Tpad = np.zeros((cfg.max_len,), dtype=np.float32)
        M = np.zeros((cfg.max_len,), dtype=np.float32)
        return X, Tpad, M

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    t, y = _downsample_sorted_arrays(t, y, int(cfg.max_len))
    n = len(t)

    y_z = robust_zscore(y)
    smooth = moving_average_np(y_z, k=5)
    highpass = y_z - smooth

    t_norm01 = t / max(float(cfg.T_days), 1.0)

    # Derivatives: these are required by the trained NNET.
    d1 = safe_gradient(y_z, t_norm01)
    d1 = np.tanh(d1 / 8.0).astype(np.float32)
    d2 = safe_gradient(d1, t_norm01)
    d2 = np.tanh(d2 / 8.0).astype(np.float32)

    dt_prev = np.zeros(n, dtype=np.float32)
    dt_next = np.zeros(n, dtype=np.float32)
    if n > 1:
        diffs = np.diff(t)
        dt_prev[1:] = diffs
        dt_next[:-1] = diffs
        dt_prev[0] = dt_prev[1]
        dt_next[-1] = dt_next[-2]
    denom = np.log1p(max(float(cfg.T_days), 1.0))
    dt_prev = np.log1p(np.maximum(dt_prev, 0.0)) / denom
    dt_next = np.log1p(np.maximum(dt_next, 0.0)) / denom

    local_amp = np.abs(highpass).astype(np.float32)
    if bool(cfg.use_absolute_time_feature):
        time_feature = ((t / float(cfg.T_days)) - 0.5) * 2.0
    else:
        time_feature = np.zeros(n, dtype=np.float32)

    mask_valid = np.ones(n, dtype=np.float32)
    feats = np.stack(
        [
            time_feature.astype(np.float32),
            y_z.astype(np.float32),
            highpass.astype(np.float32),
            d1.astype(np.float32),
            d2.astype(np.float32),
            dt_prev.astype(np.float32),
            dt_next.astype(np.float32),
            local_amp.astype(np.float32),
            mask_valid,
        ],
        axis=-1,
    ).astype(np.float32)

    X = np.zeros((cfg.max_len, cfg.input_dim), dtype=np.float32)
    Tpad = np.zeros((cfg.max_len,), dtype=np.float32)
    M = np.zeros((cfg.max_len,), dtype=np.float32)

    n_keep = min(n, int(cfg.max_len))
    X[:n_keep] = feats[:n_keep]
    Tpad[:n_keep] = t[:n_keep]
    M[:n_keep] = 1.0
    return X, Tpad, M


# -----------------------------------------------------------------------------
# Model architecture copied from the training notebook
# -----------------------------------------------------------------------------

class MaskedLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ln(x)


class LatentDerivativeMLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden: int = 96, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.norm(z)
        return z * mask[..., None]


class ResTCNBlock(nn.Module):
    """ResNet-style temporal convolution block over sparse point index order."""

    def __init__(self, hidden: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=8 if hidden % 8 == 0 else 4, num_channels=hidden)
        self.norm2 = nn.GroupNorm(num_groups=8 if hidden % 8 == 0 else 4, num_channels=hidden)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x * mask)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h * mask)
        h = self.norm2(h)
        g = self.gate(residual)
        out = residual + g * h
        out = F.gelu(out)
        return out * mask


class SharedResTCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_blocks: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                ResTCNBlock(
                    hidden=hidden,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) * mask[..., None]
        h = h.transpose(1, 2)
        m = mask[:, None, :]
        for block in self.blocks:
            h = block(h, m)
        h = h.transpose(1, 2)
        h = self.out_norm(h)
        return h * mask[..., None]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask[..., None]).sum(dim=dim) / (mask.sum(dim=dim, keepdim=True) + eps)


def masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    mu = masked_mean(x, mask, dim=dim, eps=eps).unsqueeze(dim)
    var = ((x - mu) ** 2 * mask[..., None]).sum(dim=dim) / (mask.sum(dim=dim, keepdim=True) + eps)
    return torch.sqrt(var + eps)


class DelayFusion(nn.Module):
    """NGCC/SiamFC spirit: fuse multi-head latent correlation maps over the delay axis."""

    def __init__(self, n_heads: int, hidden: int = 32, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_heads, hidden, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, score_map: torch.Tensor) -> torch.Tensor:
        return self.net(score_map).squeeze(1)


class SyncNGCCResTCNModel(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.delay_grid = nn.Parameter(
            torch.linspace(-float(cfg.max_abs_delay), float(cfg.max_abs_delay), int(cfg.n_delay_grid)),
            requires_grad=False,
        )
        self.latent_deriv = LatentDerivativeMLP(
            input_dim=int(cfg.input_dim),
            out_dim=int(cfg.latent_deriv_dim),
            hidden=96,
            dropout=float(cfg.dropout) * 0.5,
        )
        self.encoder = SharedResTCNEncoder(
            in_dim=int(cfg.latent_deriv_dim),
            hidden=int(cfg.hidden),
            n_blocks=int(cfg.tcn_blocks),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
        )
        self.delay_fusion = DelayFusion(
            n_heads=int(cfg.n_filter_heads),
            hidden=int(cfg.delay_fusion_hidden),
            kernel=int(cfg.delay_fusion_kernel),
        )

        pair_dim = 4 * int(cfg.hidden) + 8
        self.cls_head = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(64, 1),
        )

    def split_heads_norm(self, z: torch.Tensor) -> torch.Tensor:
        B, N, C = z.shape
        H = int(self.cfg.n_filter_heads)
        Dh = C // H
        z = z.view(B, N, H, Dh)
        z = F.normalize(z, dim=-1, eps=1e-6)
        return z

    def soft_align_score(
        self,
        z_src: torch.Tensor,
        t_src: torch.Tensor,
        m_src: torch.Tensor,
        z_tgt: torch.Tensor,
        t_tgt: torch.Tensor,
        m_tgt: torch.Tensor,
        direction: int = +1,
    ) -> torch.Tensor:
        B, Ns, C = z_src.shape
        H = int(self.cfg.n_filter_heads)
        sigma = float(self.cfg.align_sigma_days)

        z_src_h = self.split_heads_norm(z_src)
        z_tgt_h = self.split_heads_norm(z_tgt)

        grid = self.delay_grid.to(z_src.device)
        target_times = t_src[:, :, None] + int(direction) * grid[None, None, :]

        dist = t_tgt[:, None, :, None] - target_times[:, :, None, :]
        weights = torch.exp(-0.5 * (dist / sigma) ** 2) * m_tgt[:, None, :, None]
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        weights = weights / denom

        z_interp = torch.einsum("bijd,bjhc->bidhc", weights, z_tgt_h)
        sim = (z_src_h[:, :, None, :, :] * z_interp).sum(dim=-1)

        valid = (denom.squeeze(2) > 1e-5).float()
        point_w = m_src[:, :, None] * valid
        score = (sim * point_w[:, :, :, None]).sum(dim=1) / (
            point_w.sum(dim=1, keepdim=False).clamp_min(1e-6)[:, :, None]
        )
        return score.permute(0, 2, 1).contiguous()

    def forward(
        self,
        xa: torch.Tensor,
        ta: torch.Tensor,
        ma: torch.Tensor,
        xb: torch.Tensor,
        tb: torch.Tensor,
        mb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        za0 = self.latent_deriv(xa, ma)
        zb0 = self.latent_deriv(xb, mb)

        za = self.encoder(za0, ma)
        zb = self.encoder(zb0, mb)

        map_ab = self.soft_align_score(za, ta, ma, zb, tb, mb, direction=+1)
        map_ba = self.soft_align_score(zb, tb, mb, za, ta, ma, direction=-1)
        score_map = 0.5 * (map_ab + map_ba)
        delay_logits = self.delay_fusion(score_map)

        pa = masked_mean(za, ma)
        pb = masked_mean(zb, mb)
        sa = masked_std(za, ma)
        sb = masked_std(zb, mb)

        probs_delay = F.softmax(delay_logits, dim=-1)
        entropy = -(probs_delay * probs_delay.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        max_logit = delay_logits.max(dim=-1, keepdim=True).values
        mean_logit = delay_logits.mean(dim=-1, keepdim=True)
        std_logit = delay_logits.std(dim=-1, keepdim=True)
        tau_hat = (probs_delay * self.delay_grid[None, :].to(delay_logits.device)).sum(dim=-1, keepdim=True)
        peakiness = max_logit - mean_logit
        delay_stats = torch.cat(
            [
                max_logit,
                mean_logit,
                std_logit,
                entropy,
                tau_hat / float(self.cfg.max_abs_delay),
                peakiness,
                probs_delay.max(dim=-1, keepdim=True).values,
                probs_delay.var(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        pair_feat = torch.cat([pa, pb, torch.abs(pa - pb), pa * pb, delay_stats], dim=-1)
        cls_logit = self.cls_head(pair_feat).squeeze(-1)

        return {
            "cls_logit": cls_logit,
            "delay_logits": delay_logits,
            "score_map": score_map,
            "map_ab": map_ab,
            "map_ba": map_ba,
            "za": za,
            "zb": zb,
        }


# -----------------------------------------------------------------------------
# Loading and inference utilities
# -----------------------------------------------------------------------------

def _torch_load_compat(path: str | Path, device: str) -> Any:
    """torch.load wrapper compatible with older and newer PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(bundle: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], float]:
    """Return state_dict, cfg_dict, threshold from a notebook bundle or raw state_dict."""
    if isinstance(bundle, dict):
        if "model_state_dict" in bundle:
            return bundle["model_state_dict"], bundle.get("cfg", {}), float(bundle.get("best_thr", 0.5))
        if "state_dict" in bundle:
            return bundle["state_dict"], bundle.get("cfg", {}), float(bundle.get("best_thr", 0.5))
        # It may already be a raw state_dict.
        if bundle and all(isinstance(k, str) for k in bundle.keys()):
            tensorish = [v for v in bundle.values() if torch.is_tensor(v)]
            if len(tensorish) > 0:
                return bundle, {}, 0.5
    raise ValueError(
        "Could not find a model state_dict. Expected a bundle with 'model_state_dict' "
        "or a raw PyTorch state_dict."
    )


def load_lensnn_model(
    model_path: str | Path,
    *,
    device: Optional[str] = None,
    override_cfg: Optional[Dict[str, Any]] = None,
    eval_mode: bool = True,
) -> Tuple[SyncNGCCResTCNModel, CFG, float, str]:
    """Load exported `resTCN.pt` bundle and return model, cfg, best threshold, device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    bundle = _torch_load_compat(model_path, device=device)
    state_dict, cfg_dict, best_thr = _extract_state_dict(bundle)
    if override_cfg:
        cfg_dict = {**dict(cfg_dict), **dict(override_cfg)}
    cfg = cfg_from_dict(cfg_dict)

    if int(cfg.hidden) % int(cfg.n_filter_heads) != 0:
        raise ValueError("cfg.hidden must be divisible by cfg.n_filter_heads")

    model = SyncNGCCResTCNModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) or len(unexpected):
        # strict=False allows small wrapper differences, but architecture mismatches should be visible.
        msg = []
        if len(missing):
            msg.append(f"missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if len(unexpected):
            msg.append(f"unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        print("Warning: loaded model with non-strict key match (" + "; ".join(msg) + ")")
    if eval_mode:
        model.eval()
    return model, cfg, float(best_thr), device


def _guess_col(df: pd.DataFrame, preferred: str, candidates: Iterable[str]) -> str:
    if preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"Could not find column {preferred!r}. Available columns: {list(df.columns)}")


def _prepare_input_df(
    input_csv: str | Path | pd.DataFrame,
    cfg: CFG,
    *,
    source_col: Optional[str] = None,
    component_col: Optional[str] = None,
    time_col: Optional[str] = None,
    flux_col: Optional[str] = None,
    filter_outliers: bool = True,
) -> pd.DataFrame:
    if isinstance(input_csv, pd.DataFrame):
        df = input_csv.copy()
    else:
        df = pd.read_csv(input_csv, low_memory=False)

    source_col = source_col or _guess_col(df, cfg.source_col, ["source_id", "SourceID", "sourceID", "sourceId"])
    component_col = component_col or _guess_col(
        df,
        cfg.id_col,
        ["lensComponentSourceId", "component_id", "componentId", "comp", "comp_id", "image_id"],
    )
    time_col = time_col or _guess_col(
        df,
        cfg.time_col,
        ["epoch_obs_jd", "jd", "JD", "mjd", "MJD", "time", "time_days", "date"],
    )
    flux_col = flux_col or _guess_col(df, cfg.flux_col, ["flux_obs", "flux", "magnitude", "mag", "y"])

    rename_map = {
        source_col: cfg.source_col,
        component_col: cfg.id_col,
        time_col: cfg.time_col,
        flux_col: cfg.flux_col,
    }
    df = df.rename(columns=rename_map)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[cfg.source_col, cfg.id_col, cfg.time_col, cfg.flux_col]).copy()
    df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
    df[cfg.flux_col] = pd.to_numeric(df[cfg.flux_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col, cfg.flux_col]).copy()

    if filter_outliers and cfg.outlier_col in df.columns:
        df = df[~df[cfg.outlier_col].astype(bool)].copy()

    df = df.sort_values([cfg.source_col, cfg.id_col, cfg.time_col]).reset_index(drop=True)
    return df


def _raw_component_cache(df: pd.DataFrame, cfg: CFG) -> Dict[Any, Dict[Any, Tuple[np.ndarray, np.ndarray]]]:
    cache: Dict[Any, Dict[Any, Tuple[np.ndarray, np.ndarray]]] = {}
    for sid, gsrc in df.groupby(cfg.source_col, sort=False):
        comps: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        for comp, g in gsrc.groupby(cfg.id_col, sort=False):
            gg = g.sort_values(cfg.time_col)
            t = gg[cfg.time_col].to_numpy(dtype=np.float64)
            y = gg[cfg.flux_col].to_numpy(dtype=np.float64)
            ok = np.isfinite(t) & np.isfinite(y)
            comps[comp] = (t[ok], y[ok])
        cache[sid] = comps
    return cache


def make_real_pair_tensors_from_arrays(
    tA_raw: np.ndarray,
    yA: np.ndarray,
    tB_raw: np.ndarray,
    yB: np.ndarray,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(tA_raw) < int(cfg.min_points_per_image) or len(tB_raw) < int(cfg.min_points_per_image):
        raise ValueError(f"Not enough points: A={len(tA_raw)}, B={len(tB_raw)}")

    t0 = min(float(np.nanmin(tA_raw)), float(np.nanmin(tB_raw)))
    t1 = max(float(np.nanmax(tA_raw)), float(np.nanmax(tB_raw)))
    span_real = max(float(t1 - t0), 1.0)

    # Same inference rescaling as notebook: observed pair span -> [0, cfg.T_days].
    tA = (tA_raw - t0) / span_real * float(cfg.T_days)
    tB = (tB_raw - t0) / span_real * float(cfg.T_days)

    xa, ta, ma = make_local_derivative_features(tA, yA, cfg)
    xb, tb, mb = make_local_derivative_features(tB, yB, cfg)
    return xa, ta, ma, xb, tb, mb


@torch.no_grad()
def _score_tensor_batch(
    model: SyncNGCCResTCNModel,
    batch_tensors: Dict[str, np.ndarray],
    device: str,
) -> np.ndarray:
    model.eval()
    xa = torch.as_tensor(batch_tensors["xa"], dtype=torch.float32, device=device)
    ta = torch.as_tensor(batch_tensors["ta"], dtype=torch.float32, device=device)
    ma = torch.as_tensor(batch_tensors["ma"], dtype=torch.float32, device=device)
    xb = torch.as_tensor(batch_tensors["xb"], dtype=torch.float32, device=device)
    tb = torch.as_tensor(batch_tensors["tb"], dtype=torch.float32, device=device)
    mb = torch.as_tensor(batch_tensors["mb"], dtype=torch.float32, device=device)
    out = model(xa, ta, ma, xb, tb, mb)
    proba = torch.sigmoid(out["cls_logit"]).detach().cpu().numpy().astype(np.float64)
    return proba


def _flush_batch(
    *,
    model: SyncNGCCResTCNModel,
    device: str,
    batch_meta: List[Tuple[Any, Any, Any]],
    batch_arrays: Dict[str, List[np.ndarray]],
    rows: List[Dict[str, Any]],
) -> None:
    if not batch_meta:
        return
    stacked = {k: np.stack(v, axis=0).astype(np.float32) for k, v in batch_arrays.items()}
    probas = _score_tensor_batch(model, stacked, device)
    for (sid, comp_a, comp_b), p in zip(batch_meta, probas):
        rows.append({"sourceID": sid, "compA": comp_a, "compB": comp_b, "Proba": float(p)})
    batch_meta.clear()
    for v in batch_arrays.values():
        v.clear()


def run_lensnn_inference(
    input_csv: str | Path | pd.DataFrame,
    model_path: str | Path,
    output_csv: str | Path = "nnet_pair_predictions.csv",
    *,
    batch_size: int = 64,
    keep_failed_pairs: bool = True,
    verbose: bool = True,
    device: Optional[str] = None,
    source_col: Optional[str] = None,
    component_col: Optional[str] = None,
    time_col: Optional[str] = None,
    flux_col: Optional[str] = None,
    filter_outliers: bool = True,
    return_errors: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every component pair per source with the trained ResTCN NNET.

    Parameters
    ----------
    input_csv:
        CSV path or DataFrame containing light curves.
    model_path:
        Path to exported weights/bundle, e.g. `models/resTCN.pt`.
    output_csv:
        Local output CSV path. The saved file contains exactly:
        sourceID, compA, compB, Proba
    batch_size:
        Number of pairs scored at once.
    keep_failed_pairs:
        If True, pairs that cannot be scored are kept with Proba = NaN.
    return_errors:
        If True, returns `(pred_df, err_df)` instead of only `pred_df`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model, cfg, best_thr, used_device = load_lensnn_model(model_path, device=device)
    df = _prepare_input_df(
        input_csv,
        cfg,
        source_col=source_col,
        component_col=component_col,
        time_col=time_col,
        flux_col=flux_col,
        filter_outliers=filter_outliers,
    )

    cache = _raw_component_cache(df, cfg)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    batch_meta: List[Tuple[Any, Any, Any]] = []
    batch_arrays: Dict[str, List[np.ndarray]] = {"xa": [], "ta": [], "ma": [], "xb": [], "tb": [], "mb": []}

    total_pairs = 0
    scored_attempts = 0

    if verbose:
        n_sources = df[cfg.source_col].nunique()
        n_components = df[[cfg.source_col, cfg.id_col]].drop_duplicates().shape[0]
        print("========== LensNN inference ==========")
        print(f"Input rows: {len(df)}")
        print(f"Sources: {n_sources}")
        print(f"Source/component groups: {n_components}")
        print(f"Model: {model_path}")
        print(f"Device: {used_device}")
        print(f"Derivative features: enabled (d1 and d2 channels)")

    for sid, comps in cache.items():
        comp_ids = list(comps.keys())
        if len(comp_ids) < 2:
            continue

        for comp_a, comp_b in combinations(comp_ids, 2):
            total_pairs += 1
            try:
                tA, yA = comps[comp_a]
                tB, yB = comps[comp_b]
                xa, ta, ma, xb, tb, mb = make_real_pair_tensors_from_arrays(tA, yA, tB, yB, cfg)

                batch_meta.append((sid, comp_a, comp_b))
                batch_arrays["xa"].append(xa)
                batch_arrays["ta"].append(ta)
                batch_arrays["ma"].append(ma)
                batch_arrays["xb"].append(xb)
                batch_arrays["tb"].append(tb)
                batch_arrays["mb"].append(mb)
                scored_attempts += 1

                if len(batch_meta) >= int(batch_size):
                    _flush_batch(
                        model=model,
                        device=used_device,
                        batch_meta=batch_meta,
                        batch_arrays=batch_arrays,
                        rows=rows,
                    )

            except Exception as exc:
                errors.append({"sourceID": sid, "compA": comp_a, "compB": comp_b, "error": repr(exc)})
                if keep_failed_pairs:
                    rows.append({"sourceID": sid, "compA": comp_a, "compB": comp_b, "Proba": np.nan})

        if verbose and total_pairs > 0 and total_pairs % 1000 == 0:
            print(f"Pairs visited: {total_pairs:,} | scored/queued: {scored_attempts:,} | errors: {len(errors):,}")

    _flush_batch(model=model, device=used_device, batch_meta=batch_meta, batch_arrays=batch_arrays, rows=rows)

    pred_df = pd.DataFrame(rows, columns=["sourceID", "compA", "compB", "Proba"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)

    if verbose:
        print("\n========== LensNN summary ==========")
        print(f"Total available pairs: {total_pairs}")
        print(f"Rows written: {len(pred_df)}")
        print(f"Failed pairs: {len(errors)}")
        if len(pred_df) and pred_df["Proba"].notna().any():
            print(f"Max Proba: {pred_df['Proba'].max():.6f}")
        print(f"Saved: {output_csv}")
        print("Output columns: sourceID, compA, compB, Proba")
        print("====================================\n")

    if return_errors:
        return pred_df, pd.DataFrame(errors)
    return pred_df


# Backward-compatible aliases / convenient names.
run_nn_inference = run_lensnn_inference
run_nnet_inference = run_lensnn_inference
run_inference = run_lensnn_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResTCN LensNN inference on every component pair.")
    parser.add_argument("input_csv", help="Input light-curve CSV")
    parser.add_argument("--model", default="models/resTCN.pt", help="Path to exported resTCN.pt weights/bundle")
    parser.add_argument("--output", default="nnet_pair_predictions.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=64, help="Pairs per inference batch")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    parser.add_argument("--drop-failed", action="store_true", help="Skip failed pairs instead of writing NaN Proba")
    args = parser.parse_args()

    run_lensnn_inference(
        input_csv=args.input_csv,
        model_path=args.model,
        output_csv=args.output,
        batch_size=args.batch_size,
        keep_failed_pairs=not args.drop_failed,
        device=args.device,
        verbose=True,
    )
