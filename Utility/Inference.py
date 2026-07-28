"""
Inference par composante avec RF/Catch22 et EDSM-Lite.

Ce script est dérivé de ``inference(1).ipynb``. Il n'utilise pas les modèles
``NNlens`` ou ``RFlens``, qui servent à classer des paires de composantes.

Modèles pris en charge
----------------------
Réels :
    models/random_forest_catch22.joblib
    models/edsm_lite.pt

Synthétiques :
    models/random_forest_catch22_synth.joblib
    models/edsm_lite_synth.pt

Exemples
--------
Depuis la racine du projet :

    python Utility/Inference.py data/NewCleanDGDR.csv --domain real

    python Utility/Inference.py data/NewCleanDGDR.csv \
        --domain synthetic \
        --output results/inference_synthetic.csv

    python Utility/Inference.py data/NewCleanDGDR.csv \
        --domain both \
        --include-probabilities \
        --output results/inference_all.csv

Entrée minimale
---------------
Les alias usuels sont acceptés, mais les noms canoniques sont :

    epoch_obs_jd, flux_obs, source_id, lensComponentSourceId

Sortie
------
Pour un seul domaine :

    SourceID, componentID, RFpred, EDSMpred

Avec ``--domain both`` :

    SourceID, componentID,
    RFpred_real, EDSMpred_real,
    RFpred_synth, EDSMpred_synth

L'option ``--include-probabilities`` ajoute les probabilités correspondantes.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sktime.transformations.panel.catch22 import Catch22
except ImportError as exc:  # pragma: no cover - dépend de l'environnement utilisateur
    raise ImportError(
        "sktime est requis pour l'inférence RF/Catch22. "
        "Installe les dépendances avec : pip install sktime"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration et compatibilité avec les anciens bundles joblib
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Compatibilité avec les bundles RF créés depuis un notebook."""

    flq_path: str = "FLQ.csv"
    fls_path: str = "FLS.csv"
    object_id_col: str = "lensComponentSourceId"
    time_col: str = "time"
    flux_col: str = "flux_obs"
    min_measures: int = 20
    merge_window_days: float = 2.0
    test_size: float = 0.15
    val_size: float = 0.15
    rf_n_estimators: int = 500
    rf_max_depth: Any = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_class_weight: str = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1


MODEL_FILENAMES = {
    "real": {
        "rf": "random_forest_catch22.joblib",
        "edsm": "edsm_lite.pt",
    },
    "synthetic": {
        "rf": "random_forest_catch22_synth.joblib",
        "edsm": "edsm_lite_synth.pt",
    },
}

CANONICAL_COLUMNS = [
    "epoch_obs_jd",
    "flux_obs",
    "source_id",
    "lensComponentSourceId",
]

COLUMN_ALIASES = {
    "epoch_obs_jd": ("epoch_obs_jd", "time", "jd", "JD", "mjd", "MJD"),
    "flux_obs": ("flux_obs", "flux", "Flux"),
    "source_id": ("source_id", "SourceID", "sourceID", "sourceId"),
    "lensComponentSourceId": (
        "lensComponentSourceId",
        "component_source_id",
        "componentID",
        "component_id",
        "componentId",
    ),
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def project_root() -> Path:
    """Trouve la racine lorsque le script est placé dans ``Utility/``."""
    script_path = Path(__file__).resolve()
    parent_candidate = script_path.parent.parent
    if (parent_candidate / "models").is_dir():
        return parent_candidate
    if (Path.cwd() / "models").is_dir():
        return Path.cwd()
    return parent_candidate


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def canonicalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accepte les alias du notebook et retourne les quatre colonnes canoniques."""
    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in COLUMN_ALIASES.items():
        found = first_existing(df.columns, aliases)
        if found is None:
            missing.append(canonical)
        elif found != canonical:
            rename_map[found] = canonical

    if missing:
        raise ValueError(
            f"Colonnes requises absentes : {missing}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    out = df.rename(columns=rename_map)[CANONICAL_COLUMNS].copy()
    out["epoch_obs_jd"] = pd.to_numeric(out["epoch_obs_jd"], errors="coerce")
    out["flux_obs"] = pd.to_numeric(out["flux_obs"], errors="coerce")
    out = out.dropna(subset=CANONICAL_COLUMNS).copy()

    out["epoch_obs_jd"] = out["epoch_obs_jd"].astype(np.float32)
    out["flux_obs"] = out["flux_obs"].astype(np.float32)
    return out.reset_index(drop=True)


def load_input_dataframe(path: str | Path) -> pd.DataFrame:
    return canonicalize_input_dataframe(pd.read_csv(path, low_memory=False))


def build_component_to_source_map(df: pd.DataFrame) -> pd.DataFrame:
    """Associe chaque composante au premier ``source_id`` observé."""
    return (
        df.groupby("lensComponentSourceId", as_index=False, sort=False)["source_id"]
        .first()
        .rename(
            columns={
                "lensComponentSourceId": "componentID",
                "source_id": "SourceID",
            }
        )
    )


# ---------------------------------------------------------------------------
# Prétraitement partagé, identique au notebook d'inférence
# ---------------------------------------------------------------------------


def aggregate_in_time_windows(
    group: pd.DataFrame,
    object_id_col: str,
    time_col: str,
    flux_col: str,
    window_days: float = 2.0,
) -> pd.DataFrame:
    group = group.sort_values(time_col).reset_index(drop=True)
    times = group[time_col].to_numpy()
    fluxes = group[flux_col].to_numpy()
    object_id = group[object_id_col].iloc[0]

    rows: list[dict[str, Any]] = []
    start_idx = 0
    while start_idx < len(group):
        block_start_time = times[start_idx]
        end_idx = start_idx + 1
        while (
            end_idx < len(group)
            and (times[end_idx] - block_start_time) <= window_days
        ):
            end_idx += 1

        block = slice(start_idx, end_idx)
        rows.append(
            {
                object_id_col: object_id,
                time_col: float(np.mean(times[block])),
                flux_col: float(np.mean(fluxes[block])),
            }
        )
        start_idx = end_idx

    return pd.DataFrame(rows)


def preprocess_inference_dataframe(
    df: pd.DataFrame,
    object_id_col: str,
    time_col: str,
    flux_col: str,
    min_measures: int = 20,
    window_days: float = 2.0,
) -> pd.DataFrame:
    """
    Reproduit les deux filtrages du notebook.

    Le notebook utilise strictement ``count > min_measures`` avant et après
    l'agrégation. Cette convention est conservée pour rester compatible avec
    l'entraînement sauvegardé dans chaque modèle.
    """
    counts = df.groupby(object_id_col).size()
    valid_ids = counts[counts > min_measures].index
    filtered = df[df[object_id_col].isin(valid_ids)].copy()

    if filtered.empty:
        return filtered[[object_id_col, time_col, flux_col]].copy()

    groups = [
        aggregate_in_time_windows(
            group=g,
            object_id_col=object_id_col,
            time_col=time_col,
            flux_col=flux_col,
            window_days=window_days,
        )
        for _, g in filtered.groupby(object_id_col, sort=False)
    ]
    processed = pd.concat(groups, ignore_index=True)

    counts_after = processed.groupby(object_id_col).size()
    valid_ids_after = counts_after[counts_after > min_measures].index
    processed = processed[processed[object_id_col].isin(valid_ids_after)].copy()

    return processed.sort_values([object_id_col, time_col]).reset_index(drop=True)


def model_dataframe(
    raw_df: pd.DataFrame,
    object_id_col: str,
    time_col: str,
    flux_col: str,
) -> pd.DataFrame:
    """Construit les noms de colonnes attendus par la configuration sauvegardée."""
    return pd.DataFrame(
        {
            time_col: raw_df["epoch_obs_jd"].to_numpy(),
            flux_col: raw_df["flux_obs"].to_numpy(),
            object_id_col: raw_df["lensComponentSourceId"].to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# Random Forest + Catch22
# ---------------------------------------------------------------------------


def install_legacy_config_shim() -> None:
    """Rend ``Config`` visible pour les bundles picklés depuis ``__main__``."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "Config"):
        setattr(main_module, "Config", Config)


def safe_load_rf_bundle(path: str | Path) -> Any:
    install_legacy_config_shim()
    return joblib.load(path)


def rf_apply_normalization(
    df: pd.DataFrame,
    flux_col: str,
    stats: Mapping[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    mean = float(stats["flux"]["mean"])
    std = float(stats["flux"]["std"])
    if not np.isfinite(std) or abs(std) < 1e-12:
        raise ValueError("Écart-type RF invalide dans normalization_stats.")
    out[flux_col] = (out[flux_col] - mean) / std
    return out


def rf_build_sktime_panel(
    df: pd.DataFrame,
    object_id_col: str,
    time_col: str,
    flux_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for object_id, group in df.groupby(object_id_col, sort=False):
        group = group.sort_values(time_col)
        rows.append(
            {
                object_id_col: object_id,
                "flux": pd.Series(
                    group[flux_col].to_numpy(dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
    return pd.DataFrame(rows)


def positive_class_index(classifier: Any) -> int:
    classes = np.asarray(getattr(classifier, "classes_", [0, 1]))
    matches = np.flatnonzero(classes == 1)
    return int(matches[0]) if len(matches) else int(len(classes) - 1)


def run_rf_inference(
    raw_df: pd.DataFrame,
    bundle_path: str | Path,
    *,
    min_measures_override: int | None = None,
    merge_window_days_override: float | None = None,
) -> pd.DataFrame:
    bundle = safe_load_rf_bundle(bundle_path)
    if not isinstance(bundle, Mapping):
        raise TypeError(
            "Le fichier RF doit être le bundle du notebook contenant au moins "
            "'model', 'normalization_stats' et 'transformer'."
        )

    classifier = bundle["model"]
    catch22 = bundle.get("transformer")
    if catch22 is None:
        catch22 = Catch22(
            features="all",
            outlier_norm=False,
            replace_nans=True,
        )

    stats = bundle["normalization_stats"]
    threshold = float(bundle.get("threshold", 0.5))
    saved_cfg = bundle.get("config", {})

    object_id_col = cfg_get(
        saved_cfg, "object_id_col", "lensComponentSourceId"
    )
    time_col = cfg_get(saved_cfg, "time_col", "time")
    flux_col = cfg_get(saved_cfg, "flux_col", "flux_obs")
    min_measures = int(cfg_get(saved_cfg, "min_measures", 20))
    window_days = float(cfg_get(saved_cfg, "merge_window_days", 2.0))

    if min_measures_override is not None:
        min_measures = int(min_measures_override)
    if merge_window_days_override is not None:
        window_days = float(merge_window_days_override)

    prepared = preprocess_inference_dataframe(
        df=model_dataframe(raw_df, object_id_col, time_col, flux_col),
        object_id_col=object_id_col,
        time_col=time_col,
        flux_col=flux_col,
        min_measures=min_measures,
        window_days=window_days,
    )
    if prepared.empty:
        return pd.DataFrame(columns=["componentID", "RFprob", "RFpred"])

    normalized = rf_apply_normalization(prepared, flux_col, stats)
    panel = rf_build_sktime_panel(
        normalized, object_id_col, time_col, flux_col
    )
    features = catch22.transform(panel[["flux"]].copy())

    feature_columns = bundle.get("feature_columns")
    if feature_columns is not None:
        feature_columns = list(feature_columns)
        missing = [name for name in feature_columns if name not in features.columns]
        if missing:
            raise ValueError(
                "Incompatibilité des features Catch22. "
                f"Colonnes manquantes : {missing}"
            )
        features = features.loc[:, feature_columns]

    probabilities = classifier.predict_proba(features)
    probabilities = probabilities[:, positive_class_index(classifier)]
    predictions = probabilities >= threshold

    return pd.DataFrame(
        {
            "componentID": panel[object_id_col].to_numpy(),
            "RFprob": probabilities.astype(float),
            "RFpred": predictions.astype(bool),
        }
    )


# ---------------------------------------------------------------------------
# EDSM-Lite
# ---------------------------------------------------------------------------


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = np.std(values)
    if scale < 1e-6:
        scale = 1.0
    return (values - median) / scale


def compute_sequence_stats(
    flux_z: np.ndarray,
    dt: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    del flux_z, dt, slope
    return np.zeros((1,), dtype=np.float32)


class EventEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        event_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, event_dim),
            nn.LayerNorm(event_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(event_dim, event_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecayCell(nn.Module):
    def __init__(self, event_dim: int = 32, hidden_dim: int = 48) -> None:
        super().__init__()
        self.decay = nn.Linear(1, hidden_dim)
        self.in_proj = nn.Linear(event_dim, hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cand = nn.Linear(event_dim + hidden_dim, hidden_dim)
        self.mix = nn.Linear(event_dim + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        event: torch.Tensor,
        log_dt: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        gamma = torch.sigmoid(self.decay(log_dt))
        decayed_hidden = gamma * previous_hidden

        update_gate = torch.sigmoid(
            self.in_proj(event) + self.h_proj(decayed_hidden)
        )
        candidate = F.silu(
            self.cand(torch.cat([event, decayed_hidden], dim=-1))
        )
        mix = torch.sigmoid(
            self.mix(torch.cat([event, decayed_hidden], dim=-1))
        )

        hidden = (
            update_gate * decayed_hidden
            + (1.0 - update_gate) * candidate
        )
        hidden = mix * hidden + (1.0 - mix) * decayed_hidden
        return self.norm(hidden)


class GLUHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation, gate = self.proj(x).chunk(2, dim=-1)
        x = F.silu(activation) * torch.sigmoid(gate)
        return self.out(self.dropout(x)).squeeze(-1)


class EDSMLiteClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        event_dim: int = 32,
        hidden_dim: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.event_encoder = EventEncoder(
            input_dim=input_dim,
            event_dim=event_dim,
            dropout=dropout,
        )
        self.decay_cell = DecayCell(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
        )
        self.event_gate = nn.Linear(event_dim + 1, 1)
        self.state_gate = nn.Linear(hidden_dim + event_dim + 1, 1)
        fusion_dim = hidden_dim * 2 + event_dim
        self.head = GLUHead(
            fusion_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

    def encode_features(
        self,
        x: torch.Tensor,
        stats: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del stats  # Le checkpoint entraîné n'utilise pas encore ces statistiques.
        mask = (
            torch.arange(x.size(1), device=x.device)[None, :]
            < lengths[:, None]
        ).float()
        log_dt = x[..., 1:2]

        encoded_events = self.event_encoder(x)
        batch_size, time_steps, _ = encoded_events.shape
        hidden_dim = self.decay_cell.h_proj.out_features

        hidden = torch.zeros(
            batch_size,
            hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        states: list[torch.Tensor] = []

        for time_index in range(time_steps):
            new_hidden = self.decay_cell(
                encoded_events[:, time_index, :],
                log_dt[:, time_index, :],
                hidden,
            )
            valid = mask[:, time_index : time_index + 1]
            hidden = valid * new_hidden + (1.0 - valid) * hidden
            states.append(hidden.unsqueeze(1))

        state_tensor = torch.cat(states, dim=1)

        event_weights = torch.sigmoid(
            self.event_gate(torch.cat([encoded_events, log_dt], dim=-1))
        ).squeeze(-1)
        event_weights = event_weights * mask
        event_weights = event_weights / event_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        event_summary = torch.sum(
            event_weights.unsqueeze(-1) * encoded_events,
            dim=1,
        )

        state_weights = torch.sigmoid(
            self.state_gate(
                torch.cat([state_tensor, encoded_events, log_dt], dim=-1)
            )
        ).squeeze(-1)
        state_weights = state_weights * mask
        state_weights = state_weights / state_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        state_summary = torch.sum(
            state_weights.unsqueeze(-1) * state_tensor,
            dim=1,
        )

        batch_indices = torch.arange(batch_size, device=x.device)
        last_indices = (lengths - 1).clamp_min(0)
        last_hidden = state_tensor[batch_indices, last_indices, :]
        return torch.cat(
            [last_hidden, state_summary, event_summary],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        stats: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode_features(x, stats, lengths))


class EDSMInferenceDataset(Dataset):
    def __init__(
        self,
        events: np.ndarray,
        stats: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.events = torch.tensor(events, dtype=torch.float32)
        self.stats = torch.tensor(stats, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return (
            self.events[index],
            self.stats[index],
            self.lengths[index],
            index,
        )


def build_edsm_sequences(
    df: pd.DataFrame,
    object_id_col: str,
    time_col: str,
    flux_col: str,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    event_list: list[np.ndarray] = []
    stats_list: list[np.ndarray] = []
    lengths: list[int] = []
    object_ids: list[Any] = []

    for object_id, group in df.groupby(object_id_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        times = group[time_col].to_numpy(dtype=np.float32)
        flux = group[flux_col].to_numpy(dtype=np.float32)

        if len(times) > max_len:
            times = times[-max_len:]
            flux = flux[-max_len:]

        flux_z = robust_standardize(flux).astype(np.float32)
        relative_time = times - times[0]
        if len(relative_time) > 1 and relative_time[-1] > 0:
            normalized_time = relative_time / relative_time[-1]
        else:
            normalized_time = np.zeros_like(relative_time, dtype=np.float32)

        dt = np.diff(times, prepend=times[0]).astype(np.float32)
        dt = np.maximum(dt, 0.0)
        standardized_log_dt = robust_standardize(
            np.log1p(dt).astype(np.float32)
        ).astype(np.float32)

        slope = np.zeros_like(flux_z, dtype=np.float32)
        denominator = np.maximum(dt[1:], 1e-3)
        slope[1:] = (flux_z[1:] - flux_z[:-1]) / denominator
        slope = np.clip(slope, -20.0, 20.0).astype(np.float32)

        events = np.stack(
            [
                flux_z,
                standardized_log_dt,
                normalized_time,
                slope,
                np.abs(slope),
            ],
            axis=-1,
        ).astype(np.float32)
        sequence_stats = compute_sequence_stats(flux_z, dt, slope)
        length = len(events)

        padding = max_len - length
        if padding > 0:
            events = np.pad(
                events,
                ((0, padding), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        event_list.append(events)
        stats_list.append(sequence_stats)
        lengths.append(length)
        object_ids.append(object_id)

    if not event_list:
        return None

    return (
        np.stack(event_list).astype(np.float32),
        np.stack(stats_list).astype(np.float32),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(object_ids, dtype=object),
    )


def load_torch_checkpoint(path: str | Path, device: torch.device) -> Any:
    """Charge aussi bien les checkpoints récents que les anciennes versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch antérieur à l'argument weights_only
        return torch.load(path, map_location=device)


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Le checkpoint EDSM doit contenir config et model_state_dict."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        tensor_values = all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        )
        if not tensor_values:
            raise KeyError(
                "Aucune clé 'model_state_dict' ou 'state_dict' dans le checkpoint."
            )
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return dict(state_dict)


def infer_edsm_dimensions(
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[int, int, int, float]:
    config = checkpoint.get("config", {})

    encoder_weight = state_dict.get("event_encoder.net.0.weight")
    hidden_weight = state_dict.get("decay_cell.h_proj.weight")

    input_dim = int(
        encoder_weight.shape[1]
        if encoder_weight is not None
        else checkpoint.get("input_dim", 5)
    )
    event_dim = int(
        encoder_weight.shape[0]
        if encoder_weight is not None
        else cfg_get(config, "event_dim", 32)
    )
    hidden_dim = int(
        hidden_weight.shape[0]
        if hidden_weight is not None
        else cfg_get(config, "hidden_dim", 48)
    )
    dropout = float(cfg_get(config, "dropout", 0.15))
    return input_dim, event_dim, hidden_dim, dropout


def run_edsm_inference(
    raw_df: pd.DataFrame,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    batch_size: int = 256,
    min_measures_override: int | None = None,
    merge_window_days_override: float | None = None,
) -> pd.DataFrame:
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    config = checkpoint.get("config", {})

    object_id_col = cfg_get(
        config, "object_id_col", "lensComponentSourceId"
    )
    time_col = cfg_get(config, "time_col", "time")
    flux_col = cfg_get(config, "flux_col", "flux_obs")
    min_measures = int(cfg_get(config, "min_measures", 20))
    window_days = float(cfg_get(config, "merge_window_days", 2.0))
    max_len = int(cfg_get(config, "max_len", 128))
    threshold = float(
        checkpoint.get(
            "best_threshold",
            checkpoint.get("threshold", 0.5),
        )
    )

    if min_measures_override is not None:
        min_measures = int(min_measures_override)
    if merge_window_days_override is not None:
        window_days = float(merge_window_days_override)

    prepared = preprocess_inference_dataframe(
        df=model_dataframe(raw_df, object_id_col, time_col, flux_col),
        object_id_col=object_id_col,
        time_col=time_col,
        flux_col=flux_col,
        min_measures=min_measures,
        window_days=window_days,
    )
    if prepared.empty:
        return pd.DataFrame(columns=["componentID", "EDSMprob", "EDSMpred"])

    sequences = build_edsm_sequences(
        prepared,
        object_id_col=object_id_col,
        time_col=time_col,
        flux_col=flux_col,
        max_len=max_len,
    )
    if sequences is None:
        return pd.DataFrame(columns=["componentID", "EDSMprob", "EDSMpred"])

    events, stats, lengths, object_ids = sequences
    dataset = EDSMInferenceDataset(events, stats, lengths)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
    )

    input_dim, event_dim, hidden_dim, dropout = infer_edsm_dimensions(
        checkpoint,
        state_dict,
    )
    model = EDSMLiteClassifier(
        input_dim=input_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    probabilities = np.empty(len(object_ids), dtype=np.float64)
    with torch.inference_mode():
        for event_batch, stats_batch, length_batch, indices in loader:
            logits = model(
                event_batch.to(device),
                stats_batch.to(device),
                length_batch.to(device),
            )
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities[indices.numpy()] = batch_probabilities

    return pd.DataFrame(
        {
            "componentID": object_ids,
            "EDSMprob": probabilities,
            "EDSMpred": (probabilities >= threshold).astype(bool),
        }
    )


# ---------------------------------------------------------------------------
# Orchestration des variantes réelle et synthétique
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé, mais aucun GPU CUDA n'est disponible.")
    return torch.device(requested)


def require_model(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} introuvable : {resolved}")
    return resolved


def paths_for_domain(
    domain: str,
    models_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    rf_override = getattr(args, f"rf_{domain}_model")
    edsm_override = getattr(args, f"edsm_{domain}_model")

    rf_path = (
        Path(rf_override)
        if rf_override
        else models_dir / MODEL_FILENAMES[domain]["rf"]
    )
    edsm_path = (
        Path(edsm_override)
        if edsm_override
        else models_dir / MODEL_FILENAMES[domain]["edsm"]
    )
    return (
        require_model(rf_path, f"Modèle RF {domain}"),
        require_model(edsm_path, f"Modèle EDSM {domain}"),
    )


def domain_predictions(
    raw_df: pd.DataFrame,
    domain: str,
    rf_path: Path,
    edsm_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    rf_min_measures: int | None,
    rf_window_days: float | None,
    edsm_min_measures: int | None,
    edsm_window_days: float | None,
    verbose: bool,
) -> pd.DataFrame:
    if verbose:
        print(f"\n[{domain}] RF : {rf_path}")
    rf_result = run_rf_inference(
        raw_df,
        rf_path,
        min_measures_override=rf_min_measures,
        merge_window_days_override=rf_window_days,
    )

    if verbose:
        print(f"[{domain}] EDSM : {edsm_path}")
    edsm_result = run_edsm_inference(
        raw_df,
        edsm_path,
        device=device,
        batch_size=batch_size,
        min_measures_override=edsm_min_measures,
        merge_window_days_override=edsm_window_days,
    )

    return rf_result.merge(edsm_result, on="componentID", how="outer")


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    raw_df = load_input_dataframe(args.input_csv)
    component_map = build_component_to_source_map(raw_df)

    root = project_root()
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else (root / "models").resolve()
    )
    domains = (
        ["real", "synthetic"]
        if args.domain == "both"
        else [args.domain]
    )

    if args.verbose:
        print(f"Lignes d'entrée : {len(raw_df):,}")
        print(
            "Composantes uniques : "
            f"{raw_df['lensComponentSourceId'].nunique():,}"
        )
        print(f"Device EDSM : {device}")

    result = component_map.copy()
    for domain in domains:
        rf_path, edsm_path = paths_for_domain(domain, models_dir, args)
        predictions = domain_predictions(
            raw_df=raw_df,
            domain=domain,
            rf_path=rf_path,
            edsm_path=edsm_path,
            device=device,
            batch_size=args.batch_size,
            rf_min_measures=args.rf_min_measures,
            rf_window_days=args.rf_merge_window_days,
            edsm_min_measures=args.edsm_min_measures,
            edsm_window_days=args.edsm_merge_window_days,
            verbose=args.verbose,
        )

        if args.domain == "both":
            predictions = predictions.rename(
                columns={
                    "RFprob": f"RFprob_{'synth' if domain == 'synthetic' else 'real'}",
                    "RFpred": f"RFpred_{'synth' if domain == 'synthetic' else 'real'}",
                    "EDSMprob": f"EDSMprob_{'synth' if domain == 'synthetic' else 'real'}",
                    "EDSMpred": f"EDSMpred_{'synth' if domain == 'synthetic' else 'real'}",
                }
            )
        result = result.merge(predictions, on="componentID", how="left")

    prediction_columns = [
        column for column in result.columns if "pred" in column.lower()
    ]
    probability_columns = [
        column for column in result.columns if "prob" in column.lower()
    ]

    for column in prediction_columns:
        result[column] = result[column].fillna(False).astype(bool)

    if args.add_final_pred:
        if args.domain == "both":
            result["final_pred_real"] = (
                result["RFpred_real"] | result["EDSMpred_real"]
            )
            result["final_pred_synth"] = (
                result["RFpred_synth"] | result["EDSMpred_synth"]
            )
            result["final_pred"] = (
                result["final_pred_real"] | result["final_pred_synth"]
            )
        else:
            result["final_pred"] = result["RFpred"] | result["EDSMpred"]

    if not args.include_probabilities:
        result = result.drop(columns=probability_columns)

    result = result.sort_values(
        ["SourceID", "componentID"],
        kind="stable",
    ).reset_index(drop=True)

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else root / "results" / f"inference_{args.domain}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    if args.verbose:
        print(f"\nPrédictions enregistrées dans : {output_path.resolve()}")
        for column in prediction_columns:
            print(f"{column} : {result[column].mean():.2%} de True")
        print(result.head())

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inférence par composante avec les modèles RF/Catch22 et "
            "EDSM-Lite réels, synthétiques, ou les quatre."
        )
    )
    parser.add_argument(
        "input_csv",
        help="CSV contenant les courbes de lumière.",
    )
    parser.add_argument(
        "--domain",
        choices=("real", "synthetic", "both"),
        default="both",
        help="Jeu de modèles à exécuter (défaut : both).",
    )
    parser.add_argument(
        "--output",
        help="Chemin du CSV de sortie.",
    )
    parser.add_argument(
        "--models-dir",
        help="Dossier des modèles (défaut : <racine_du_projet>/models).",
    )

    parser.add_argument("--rf-real-model")
    parser.add_argument("--edsm-real-model")
    parser.add_argument("--rf-synthetic-model")
    parser.add_argument("--edsm-synthetic-model")

    parser.add_argument("--rf-min-measures", type=int)
    parser.add_argument("--rf-merge-window-days", type=float)
    parser.add_argument("--edsm-min-measures", type=int)
    parser.add_argument("--edsm-merge-window-days", type=float)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille des lots EDSM (défaut : 256).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Ajoute RFprob et EDSMprob au CSV.",
    )
    parser.add_argument(
        "--add-final-pred",
        action="store_true",
        help="Ajoute la combinaison logique RFpred OU EDSMpred.",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Réduit les messages affichés.",
    )
    parser.set_defaults(verbose=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
