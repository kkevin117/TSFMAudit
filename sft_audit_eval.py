"""
SFT Audit Evaluation (FP-constrained thresholding)

输出终端表格字段：
  - candidate / reference（无参考写“无”）/ condition(含 epoch)
  - AUROC / AUPRC / Macro-F1 / MCC / Balanced Accuracy

协议要点：
  - 评估单位是 (model, dataset)
  - 数据集级重复划分：校准集优先覆盖更多 Clean
  - 阈值 τ* 由校准集允许 FP 个数 k 决定（score > τ* 判为 Leaked）
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from path_config import RESULTS_ROOT


@dataclass(frozen=True)
class ModelConfig:
    key: str
    base_dir: str
    model_name: str


CANDIDATES: Dict[str, ModelConfig] = {
    "chronos": ModelConfig("chronos", "chronos_sft", "Chronos"),
    "kairos": ModelConfig("kairos", "kairos_sft", "Kairos"),
    "moirai1": ModelConfig("moirai1", "moirai1_sft", "Moirai1"),
    "moirai2": ModelConfig("moirai2", "moirai2_sft", "Moirai2"),
    "timesfm": ModelConfig("timesfm", "timesfm2p0_sft", "TimesFM2.0"),
    "tirex": ModelConfig("tirex", "tirex_sft", "TiRex"),
    # Controlled contamination sanity-check: ScratchCNN probe runs with exposed θ0.
    "controlled": ModelConfig("controlled", "controlled_sft", "ScratchCNN"),
}

REFERENCES: Dict[str, ModelConfig] = {
    "scratch_cnn": ModelConfig("scratch_cnn", "scratch_cnn_sft", "ScratchCNN"),
    "scratch_transformer": ModelConfig("scratch_transformer", "scratch_transformer_sft", "ScratchTransformer"),
    "stat": ModelConfig("stat", "stat_sft", "Stat"),
    "visionts": ModelConfig("visionts", "visionts_sft", "VisionTS"),
}

DEFAULT_NOREF_SCORES: List[str] = [
    "drop",
    "aulc",
    "loss",
    "eff",
    "grad_loss_ratio",
    "grad",
    "ws",
    "logreg_subset",
    "logreg_all",
    "thr2_loss_drop",
    "thr2_loss_aulc",
    "thr2_drop_aulc",
]
DEFAULT_WITHREF_SCORES: List[str] = [
    "d_drop",
    "d_loss",
    "d_aulc",
    "d_eff",
    "inter_ld",
    "r_loss",
    "r_drop",
    "r_grad",
    "r_ws",
    "d_loss_norm",
    "d_grad_norm_n",
    "d_grad",
    "d_ws",
    "logreg_delta",
    "logreg_delta_all_epochs",
    "logreg_allrefs_plus_det",
    "logreg_allrefs_only",
    "logreg_allrefs_plus_det_all_epochs",
    "logreg_allrefs_only_all_epochs",
    "thr2_d_loss_d_drop",
    "thr2_d_loss_d_aulc",
    "thr2_d_loss_d_grad",
    "thr2_d_drop_d_aulc",
    "thr2_d_drop_d_grad",
]


LOGREG_DETECTORS: Dict[str, List[str]] = {
    "logreg_subset": ["loss", "drop", "aulc", "eff", "grad_loss_ratio"],
    "logreg_all": ["loss", "drop", "aulc", "grad", "ws", "eff", "grad_loss_ratio"],
    "logreg_delta": [
        "d_loss",
        "d_drop",
        "d_aulc",
        "d_grad",
        "d_ws",
        "r_loss",
        "r_grad",
        "d_eff",
        "inter_ld",
        "r_drop",
        "r_ws",
        "d_loss_norm",
        "d_grad_norm_n",
    ],
    # Legacy alias kept for backward compatibility
    "logreg_deltas": [
        "d_loss",
        "d_drop",
        "d_aulc",
        "d_grad",
        "d_ws",
        "r_loss",
        "r_grad",
        "d_eff",
        "inter_ld",
        "r_drop",
        "r_ws",
        "d_loss_norm",
        "d_grad_norm_n",
    ],
    # Dynamic feature matrix provided at runtime via custom_logreg_mats.
    "logreg_allrefs_plus_det": [],
    "logreg_allrefs_only": [],
    "logreg_delta_all_epochs": [],
    "logreg_allrefs_plus_det_all_epochs": [],
    "logreg_allrefs_only_all_epochs": [],
}


LOGREG_ALL_EPOCH_DETECTORS = {
    "logreg_delta_all_epochs",
    "logreg_allrefs_plus_det_all_epochs",
    "logreg_allrefs_only_all_epochs",
}


THR2_DETECTORS: Dict[str, Tuple[str, str]] = {
    "thr2_loss_drop": ("loss", "drop"),
    "thr2_loss_aulc": ("loss", "aulc"),
    "thr2_drop_aulc": ("drop", "aulc"),
    "thr2_d_loss_d_drop": ("d_loss", "d_drop"),
    "thr2_d_loss_d_aulc": ("d_loss", "d_aulc"),
    "thr2_d_loss_d_grad": ("d_loss", "d_grad"),
    "thr2_d_drop_d_aulc": ("d_drop", "d_aulc"),
    "thr2_d_drop_d_grad": ("d_drop", "d_grad"),
}


def _normalize_candidate_key(raw: str) -> str:
    s = raw.strip().lower()
    if not s:
        return s
    if s.endswith(".py"):
        s = s[:-3]
    if s.startswith("sft_"):
        s = s[4:]
    if s.endswith("_sft"):
        s = s[:-4]
    if s in {"timesfm2p0", "timesfm2", "timesfm2.0", "timesfm_2p0", "timesfm_2"}:
        s = "timesfm"
    return s


def _parse_candidates(values: Optional[List[str]]) -> List[str]:
    if not values:
        return list(CANDIDATES.keys())
    keys: List[str] = []
    for v in values:
        for part in v.split(","):
            k = _normalize_candidate_key(part)
            if k:
                keys.append(k)
    uniq: List[str] = []
    seen = set()
    for k in keys:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    unknown = [k for k in uniq if k not in CANDIDATES]
    if unknown:
        avail = ", ".join(sorted(CANDIDATES.keys()))
        raise SystemExit(f"Unknown candidates: {unknown}. Available: {avail}")
    return uniq


def _parse_score_list(raw: Optional[str], default: List[str]) -> List[str]:
    if raw is None:
        return list(default)
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or list(default)


def _parse_fixed_condition(raw: Optional[str]) -> Optional[Tuple[str, int, str]]:
    """
    Parse fixed condition string "<score_name>@e<epoch>".
    Returns (score_name, epoch, kind) where kind is 'thr2', 'logreg', or 'score'.
    Returns None if raw is None.
    """
    if raw is None:
        return None
    # Parse format: score_name@e<epoch>
    if "@e" not in raw:
        raise ValueError(f"Invalid fixed_condition format: {raw}. Expected '<score_name>@e<epoch>'")
    parts = raw.rsplit("@e", 1)
    score_name = parts[0].strip()
    try:
        epoch = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid epoch in fixed_condition: {raw}")

    # Infer kind
    if score_name in THR2_DETECTORS:
        kind = "thr2"
    elif score_name in LOGREG_DETECTORS:
        kind = "logreg"
    else:
        kind = "score"

    return (score_name, epoch, kind)


def _dataset_family(name: str) -> str:
    parts = name.split("/")
    if len(parts) <= 1:
        return name
    # Special-case "pretrain/<dataset>" to avoid collapsing all pretrain datasets into one giant group.
    if parts[0] == "pretrain" and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def _load_group_map_csv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    if "dataset" not in df.columns or "group" not in df.columns:
        raise ValueError(f"group map csv must have columns: dataset, group. Got: {list(df.columns)}")
    return dict(zip(df["dataset"].astype(str), df["group"].astype(str)))


def _safe_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return (mean, std, int(len(arr)))


def _fmt_mean_std(values: Sequence[float], digits: int = 3) -> str:
    mean, std, n = _safe_mean_std(values)
    if n == 0 or not np.isfinite(mean):
        return "N/A"
    if n == 1 or (not np.isfinite(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, scores))
    except Exception:
        return float("nan")


def _safe_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(f1_score(y, pred, average="macro", zero_division=0))
    except Exception:
        return float("nan")


def _safe_mcc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(matthews_corrcoef(y, pred))
    except Exception:
        return float("nan")


def _safe_balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(balanced_accuracy_score(y, pred))
    except Exception:
        return float("nan")


def _selection_value(metric: str, y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> float:
    if metric == "tpr":
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        return tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if metric == "mcc":
        return _safe_mcc(y_true, pred)
    if metric in {"balanced", "balanced_acc"}:
        return _safe_balanced_acc(y_true, pred)
    if metric in {"macro_f1", "f1_macro"}:
        return _safe_macro_f1(y_true, pred)
    if metric == "auroc":
        return _safe_auroc(y_true, scores)
    raise ValueError(f"unknown select_metric: {metric}")


def _select_tau_by_fp_k(clean_scores: np.ndarray, fp_k: int) -> float:
    clean_scores = clean_scores[np.isfinite(clean_scores)]
    if len(clean_scores) == 0:
        return float("inf")
    sorted_desc = np.sort(clean_scores)[::-1]
    if fp_k >= len(sorted_desc):
        return float(sorted_desc[-1]) - 1e-12
    return float(sorted_desc[fp_k])


def _select_tau_unconstrained(y: np.ndarray, scores: np.ndarray, select_metric: str) -> float:
    """
    Select tau on calibration data without FP-k constraint.
    Tau is chosen to maximize select_metric on calibration predictions.
    """
    ok = np.isfinite(scores)
    if int(np.sum(ok)) == 0:
        return float("inf")
    y_c = y[ok].astype(int)
    s_c = scores[ok].astype(float)

    uniq = np.unique(s_c)
    if len(uniq) == 0:
        return float("inf")

    # AUROC does not depend on threshold; fall back to MCC to get a usable tau.
    metric_for_tau = "mcc" if select_metric == "auroc" else select_metric

    tau_candidates: List[float] = [float(uniq[0]) - 1e-12]
    if len(uniq) >= 2:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        tau_candidates.extend(float(v) for v in mids.tolist())
    tau_candidates.append(float(uniq[-1]) + 1e-12)

    best_rank = None
    best_tau = float("inf")
    auroc = _safe_auroc(y_c, s_c)
    for tau in tau_candidates:
        pred = (s_c > tau).astype(int)
        sel = _selection_value(metric_for_tau, y_c, pred, s_c)
        if not np.isfinite(sel):
            continue
        tpr = _selection_value("tpr", y_c, pred, s_c)
        fp = int(np.sum((y_c == 0) & (pred == 1)))
        rank = (
            float(sel),
            float(tpr) if np.isfinite(tpr) else -1.0,
            float(auroc) if np.isfinite(auroc) else -1.0,
            -float(fp),
            -float(tau),
        )
        if (best_rank is None) or (rank > best_rank):
            best_rank = rank
            best_tau = float(tau)

    return float(best_tau)


def _select_tau_calibration(
    *,
    y_calib: np.ndarray,
    scores_calib: np.ndarray,
    fp_k: Optional[int],
    select_metric: str,
) -> float:
    if fp_k is None:
        return _select_tau_unconstrained(y_calib, scores_calib, select_metric)
    clean_scores = scores_calib[y_calib == 0]
    if len(clean_scores) == 0:
        return float("inf")
    return _select_tau_by_fp_k(clean_scores, int(fp_k))


def _fit_logreg_proba_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    """
    Fit logistic regression on (X_train, y_train) and return p(y=1|X_eval).
    Caller must ensure X is finite and y has both classes.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=int(random_state),
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train.astype(int))
    proba = pipe.predict_proba(X_eval)
    idx1 = int(np.where(pipe.named_steps["clf"].classes_ == 1)[0][0])
    return proba[:, idx1].astype(float)


def _search_thr2_rule_fp_k(
    *,
    y: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    name1: str,
    name2: str,
    fp_k: Optional[int],
    select_metric: str,
) -> Optional[dict]:
    """
    Two-variable rule search on calibration data.
    If fp_k is not None, enforce FP(clean)<=k; otherwise unconstrained.
    Returns dict with rule params + pred + score (0/1) for calibration.
    """
    if len(y) < 4 or len(np.unique(y)) < 2:
        return None

    # Use a coarse grid (percentiles) like the legacy script.
    t1_candidates = np.percentile(v1, [10, 25, 50, 75, 90])
    t2_candidates = np.percentile(v2, [10, 25, 50, 75, 90])

    best_rank = None
    best = None

    for t1 in t1_candidates:
        for t2 in t2_candidates:
            for dir1 in ("gt", "lt"):
                for dir2 in ("gt", "lt"):
                    c1 = (v1 > t1) if dir1 == "gt" else (v1 < t1)
                    c2 = (v2 > t2) if dir2 == "gt" else (v2 < t2)
                    for op in ("and", "or"):
                        pred = (c1 & c2) if op == "and" else (c1 | c2)
                        pred = pred.astype(int)

                        fp = int(np.sum((y == 0) & (pred == 1)))
                        if fp_k is not None and fp > int(fp_k):
                            continue

                        scores = pred.astype(float)
                        sel = _selection_value(select_metric, y, pred, scores)
                        if not np.isfinite(sel):
                            continue

                        tpr = _selection_value("tpr", y, pred, scores)
                        auroc = _safe_auroc(y, scores)
                        rank = (
                            float(sel),
                            float(tpr) if np.isfinite(tpr) else -1.0,
                            float(auroc) if np.isfinite(auroc) else -1.0,
                            -float(fp),
                        )
                        if best_rank is None or rank > best_rank:
                            best_rank = rank
                            best = {
                                "t1": float(t1),
                                "t2": float(t2),
                                "dir1": dir1,
                                "dir2": dir2,
                                "op": op,
                                "var1": name1,
                                "var2": name2,
                                "pred": pred,
                                "scores": scores,
                                "rule": f"({name1} {'>=' if dir1=='gt' else '<'} {t1:.4g}) {op.upper()} ({name2} {'>=' if dir2=='gt' else '<'} {t2:.4g})",
                            }

    return best


@dataclass
class EpochData:
    datasets: List[str]
    epochs: List[int]
    loss: np.ndarray
    grad: np.ndarray
    ws: np.ndarray
    drop: np.ndarray
    aulc: np.ndarray
    eff: np.ndarray
    grad_loss_ratio: np.ndarray


def _cumulative_aulc(loss_vec: np.ndarray) -> np.ndarray:
    E = loss_vec.shape[0]
    out = np.full(E, np.nan, dtype=float)
    if E == 0:
        return out
    if not np.isfinite(loss_vec[0]):
        return out
    out[0] = float(loss_vec[0])
    traps = 0.0
    for e in range(1, E):
        if not (np.isfinite(loss_vec[e - 1]) and np.isfinite(loss_vec[e])):
            out[e] = np.nan
            continue
        traps += 0.5 * (loss_vec[e - 1] + loss_vec[e])
        out[e] = traps / e
    return out


def _load_labels(results_root: Path, cfg: ModelConfig, min_samples: int) -> Tuple[List[str], np.ndarray]:
    path = results_root / cfg.base_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"] == cfg.model_name]
    if "num_samples" in df.columns:
        df = df[df["num_samples"] >= min_samples]
    if "dataset" not in df.columns or "is_leaked" not in df.columns:
        raise ValueError(f"summary.csv must contain dataset,is_leaked columns: {path}")
    df = df[["dataset", "is_leaked"]].drop_duplicates()
    datasets = df["dataset"].astype(str).tolist()
    y = df["is_leaked"].astype(int).to_numpy()
    return datasets, y


def _load_datasets(results_root: Path, cfg: ModelConfig, min_samples: int) -> List[str]:
    path = results_root / cfg.base_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"] == cfg.model_name]
    if "num_samples" in df.columns:
        df = df[df["num_samples"] >= min_samples]
    if "dataset" not in df.columns:
        raise ValueError(f"summary.csv must contain dataset column: {path}")
    return df["dataset"].astype(str).drop_duplicates().tolist()


def _load_epoch_data(
    results_root: Path,
    cfg: ModelConfig,
    datasets: List[str],
    *,
    epochs: Optional[List[int]] = None,
    max_epoch: Optional[int] = None,
    eps: float = 1e-6,
) -> EpochData:
    path = results_root / cfg.base_dir / "epoch_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"] == cfg.model_name]
    df = df[df["dataset"].isin(datasets)]
    if max_epoch is not None and "epoch" in df.columns:
        df = df[df["epoch"] <= max_epoch]

    if epochs is None:
        if "epoch" not in df.columns:
            raise ValueError(f"epoch_metrics.csv missing epoch column: {path}")
        epochs = sorted(df["epoch"].astype(int).unique().tolist())
    else:
        epochs = [int(e) for e in epochs]
        if max_epoch is not None:
            epochs = [e for e in epochs if e <= max_epoch]

    D = len(datasets)
    E = len(epochs)
    if E == 0:
        raise ValueError(f"no epochs after filtering in: {path}")

    ds2i = {d: i for i, d in enumerate(datasets)}
    ep2j = {e: j for j, e in enumerate(epochs)}

    loss = np.full((D, E), np.nan, dtype=float)
    grad = np.full((D, E), np.nan, dtype=float)
    ws = np.full((D, E), np.nan, dtype=float)

    # columns: dataset, model_name, is_leaked, epoch, loss, grad_norm, weight_shift
    for r in df.itertuples(index=False):
        d = str(getattr(r, "dataset"))
        e = int(getattr(r, "epoch"))
        i = ds2i.get(d)
        j = ep2j.get(e)
        if i is None or j is None:
            continue
        loss[i, j] = float(getattr(r, "loss"))
        grad[i, j] = float(getattr(r, "grad_norm"))
        ws[i, j] = float(getattr(r, "weight_shift"))

    drop = np.full_like(loss, np.nan)
    aulc = np.full_like(loss, np.nan)
    eff = np.full_like(loss, np.nan)
    grad_loss_ratio = np.full_like(loss, np.nan)

    for i in range(D):
        loss_i = loss[i, :]
        aulc[i, :] = _cumulative_aulc(loss_i)

        base_loss = loss_i[0]
        if np.isfinite(base_loss) and base_loss > 0:
            drop[i, :] = (base_loss - loss_i) / base_loss

        eff[i, :] = drop[i, :] / (ws[i, :] + eps)
        grad_loss_ratio[i, :] = grad[i, :] / (loss[i, :] + eps)

    return EpochData(
        datasets=datasets,
        epochs=epochs,
        loss=loss,
        grad=grad,
        ws=ws,
        drop=drop,
        aulc=aulc,
        eff=eff,
        grad_loss_ratio=grad_loss_ratio,
    )


def _build_scores_noref(cand: EpochData) -> Dict[str, np.ndarray]:
    return {
        "loss": cand.loss,
        "drop": cand.drop,
        "aulc": cand.aulc,
        "eff": cand.eff,
        "grad_loss_ratio": cand.grad_loss_ratio,
        "grad": cand.grad,
        "ws": cand.ws,
    }


def _build_scores_withref(cand: EpochData, ref: EpochData, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    d_loss = cand.loss - ref.loss
    d_drop = cand.drop - ref.drop
    d_aulc = cand.aulc - ref.aulc
    d_grad = cand.grad - ref.grad
    d_ws = cand.ws - ref.ws
    d_eff = cand.eff - ref.eff
    r_loss = cand.loss / (ref.loss + eps)
    r_grad = cand.grad / (ref.grad + eps)
    r_drop = cand.drop / (ref.drop + eps)
    r_ws = cand.ws / (ref.ws + eps)
    inter_ld = d_loss * d_drop
    d_loss_norm = (cand.loss - ref.loss) / (cand.loss + ref.loss + eps)
    d_grad_norm_n = (cand.grad - ref.grad) / (cand.grad + ref.grad + eps)

    return {
        "d_loss": d_loss,
        "d_drop": d_drop,
        "d_aulc": d_aulc,
        "d_grad": d_grad,
        "d_ws": d_ws,
        "d_eff": d_eff,
        "r_loss": r_loss,
        "r_grad": r_grad,
        "r_drop": r_drop,
        "r_ws": r_ws,
        "inter_ld": inter_ld,
        "d_loss_norm": d_loss_norm,
        "d_grad_norm_n": d_grad_norm_n,
    }


def _build_multiref_logreg_mats(
    *,
    score_map_noref: Dict[str, np.ndarray],
    score_maps_withref: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, List[np.ndarray]]:
    """
    Build feature matrices for multi-reference logreg detectors.

    - logreg_allrefs_only: concat logreg_delta features from all references.
    - logreg_allrefs_plus_det: logreg_all (detector self features) + all references' delta features.
    """
    mats_allrefs_only: List[np.ndarray] = []
    delta_feats = LOGREG_DETECTORS["logreg_delta"]
    for ref_key in sorted(score_maps_withref.keys()):
        sm = score_maps_withref[ref_key]
        for feat in delta_feats:
            if feat in sm:
                mats_allrefs_only.append(sm[feat])

    mats_allrefs_plus_det: List[np.ndarray] = []
    for feat in LOGREG_DETECTORS["logreg_all"]:
        if feat in score_map_noref:
            mats_allrefs_plus_det.append(score_map_noref[feat])
    mats_allrefs_plus_det.extend(mats_allrefs_only)

    return {
        "logreg_allrefs_only": mats_allrefs_only,
        "logreg_allrefs_plus_det": mats_allrefs_plus_det,
    }


def _flatten_feature_mats_all_epochs(mats: List[np.ndarray]) -> Optional[np.ndarray]:
    if not mats:
        return None
    if any(m.ndim != 2 for m in mats):
        return None
    n_rows = int(mats[0].shape[0])
    if any(int(m.shape[0]) != n_rows for m in mats):
        return None
    parts = [m.astype(float).reshape(n_rows, -1) for m in mats]
    return np.concatenate(parts, axis=1)


def _build_single_ref_all_epoch_logreg_X(score_map_ref: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    mats: List[np.ndarray] = []
    for feat in LOGREG_DETECTORS["logreg_delta"]:
        if feat in score_map_ref:
            mats.append(score_map_ref[feat])
    return _flatten_feature_mats_all_epochs(mats)


def _build_multiref_logreg_X_all_epochs(
    *,
    score_map_noref: Dict[str, np.ndarray],
    score_maps_withref: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    mats_allrefs_only: List[np.ndarray] = []
    for ref_key in sorted(score_maps_withref.keys()):
        sm = score_maps_withref[ref_key]
        for feat in LOGREG_DETECTORS["logreg_delta"]:
            if feat in sm:
                mats_allrefs_only.append(sm[feat])

    mats_allrefs_plus_det: List[np.ndarray] = []
    for feat in LOGREG_DETECTORS["logreg_all"]:
        if feat in score_map_noref:
            mats_allrefs_plus_det.append(score_map_noref[feat])
    mats_allrefs_plus_det.extend(mats_allrefs_only)

    out: Dict[str, np.ndarray] = {}
    x_only = _flatten_feature_mats_all_epochs(mats_allrefs_only)
    x_plus = _flatten_feature_mats_all_epochs(mats_allrefs_plus_det)
    if x_only is not None:
        out["logreg_allrefs_only_all_epochs"] = x_only
    if x_plus is not None:
        out["logreg_allrefs_plus_det_all_epochs"] = x_plus
    return out


def _build_units(
    datasets: List[str],
    y: np.ndarray,
    group_split: str,
    group_map: Optional[Dict[str, str]],
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Build disjoint units for splitting. Each unit is a list of dataset indices.
    If a unit contains mixed labels, it is automatically split back to per-dataset units.
    """
    if group_split == "none":
        units = [[i] for i in range(len(datasets))]
        labels = y.astype(int)
        return units, labels

    gid2idxs: Dict[str, List[int]] = {}
    for i, ds in enumerate(datasets):
        if group_split == "family":
            gid = _dataset_family(ds)
        elif group_split == "map":
            gid = (group_map or {}).get(ds, ds)
        else:
            raise ValueError(f"unknown group_split: {group_split}")
        gid2idxs.setdefault(gid, []).append(i)

    units: List[List[int]] = []
    labels: List[int] = []
    for gid, idxs in gid2idxs.items():
        uniq = set(int(v) for v in y[idxs])
        if len(uniq) == 1:
            units.append(list(idxs))
            labels.append(next(iter(uniq)))
        else:
            # mixed labels in one group: fall back to per-dataset units
            for i in idxs:
                units.append([i])
                labels.append(int(y[i]))

    return units, np.asarray(labels, dtype=int)


def _sample_split_units(
    rng: np.random.Generator,
    units: List[List[int]],
    unit_labels: np.ndarray,
    clean_calib_frac: float,
    leaked_calib_ratio: float,
    *,
    require_both_in_calib: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    U = len(units)
    if U < 2:
        return None

    clean_units = [u for u in range(U) if int(unit_labels[u]) == 0]
    leaked_units = [u for u in range(U) if int(unit_labels[u]) == 1]

    total_clean = sum(len(units[u]) for u in clean_units)
    total_leaked = sum(len(units[u]) for u in leaked_units)
    if total_clean == 0:
        return None

    target_clean = int(round(clean_calib_frac * total_clean))
    target_clean = max(1, min(target_clean, total_clean))

    rng.shuffle(clean_units)
    calib_units: List[int] = []
    calib_clean = 0
    for u in clean_units:
        remaining_after = total_clean - (calib_clean + len(units[u]))
        # Prefer leaving some Clean for test, but don't block split creation when only one clean unit exists.
        if remaining_after == 0 and total_clean > 1 and calib_clean > 0:
            continue
        calib_units.append(u)
        calib_clean += len(units[u])
        if calib_clean >= target_clean:
            break
    if calib_clean == 0:
        return None

    target_leaked = int(round(leaked_calib_ratio * calib_clean))
    target_leaked = min(target_leaked, total_leaked)
    if total_leaked > 0 and target_leaked == 0:
        target_leaked = 1

    rng.shuffle(leaked_units)
    calib_leaked = 0
    for u in leaked_units:
        remaining_after = total_leaked - (calib_leaked + len(units[u]))
        # Prefer leaving some Leaked for test, but allow consuming all if it's the only viable option.
        if remaining_after == 0 and total_leaked > 1 and calib_leaked > 0:
            continue
        calib_units.append(u)
        calib_leaked += len(units[u])
        if calib_leaked >= target_leaked:
            break

    calib_set = set(calib_units)
    calib_idx = sorted(i for u in calib_units for i in units[u])
    test_idx = sorted(i for u in range(U) if u not in calib_set for i in units[u])
    if len(test_idx) == 0:
        return None

    if require_both_in_calib and total_leaked > 0:
        y_calib = np.asarray([unit_labels[u] for u in calib_units], dtype=int)
        if len(np.unique(y_calib)) < 2:
            return None

    return np.asarray(calib_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _make_repeated_splits(
    y: np.ndarray,
    datasets: List[str],
    n_repeats: int,
    random_state: int,
    clean_calib_frac: float,
    leaked_calib_ratio: float,
    group_split: str,
    group_map: Optional[Dict[str, str]],
    max_attempts: int = 200,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    units, unit_labels = _build_units(datasets, y, group_split, group_map)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        rng = np.random.default_rng(int(random_state) + 10007 * r)
        split = None
        for _ in range(max_attempts):
            split = _sample_split_units(
                rng,
                units,
                unit_labels,
                clean_calib_frac,
                leaked_calib_ratio,
                require_both_in_calib=True,
            )
            if split is not None:
                break
        if split is not None:
            splits.append(split)
    return splits


def _evaluate_one_setting(
    *,
    y: np.ndarray,
    epochs: List[int],
    score_map: Dict[str, np.ndarray],  # name -> (D,E)
    score_names: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    fp_k: Optional[int],
    select_metric: str,
    random_state: int,
    fixed_condition: Optional[Tuple[str, int, str]] = None,  # (score_name, epoch, kind)
    custom_logreg_mats: Optional[Dict[str, List[np.ndarray]]] = None,
    custom_logreg_X: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[dict, dict]:
    """
    Per-split:
      - On calibration: choose best (score_name, epoch) by select_metric, with τ* determined by FP<=k on calib clean.
      - On test: compute metrics using continuous scores + τ* predictions.

    Returns:
      summary: {"metrics": {metric: [values...]}, "cond": {...}, "n_splits_used": int}
      pooled:  {"y":..., "scores":..., "pred":..., "fp_test": int, "clean_test": int}
    """
    split_metrics: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "macro_f1": [],
        "mcc": [],
        "balanced_acc": [],
    }
    chosen: List[dict] = []

    pooled_y: List[int] = []
    pooled_scores: List[float] = []
    pooled_pred: List[int] = []
    pooled_clean: int = 0
    pooled_fp: int = 0

    ep2j = {e: j for j, e in enumerate(epochs)}

    def selection_value(metric: str, y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> float:
        return _selection_value(metric, y_true, pred, scores)

    def build_logreg_X(det_name: str, epoch_index: int) -> Optional[np.ndarray]:
        if custom_logreg_X is not None and det_name in custom_logreg_X:
            X = custom_logreg_X.get(det_name)
            if X is None:
                return None
            X = np.asarray(X, dtype=float)
            if X.ndim != 2 or X.shape[0] != len(y):
                return None
            return X

        if custom_logreg_mats is not None and det_name in custom_logreg_mats:
            mats = custom_logreg_mats.get(det_name) or []
            if not mats:
                return None
            return np.vstack([m[:, epoch_index].astype(float) for m in mats]).T

        feats = LOGREG_DETECTORS.get(det_name) or []
        # Some detectors rely on runtime-provided custom matrices.
        # If no static feature list is defined, skip in this context.
        if not feats:
            return None
        if any(f not in score_map for f in feats):
            return None
        return np.vstack([score_map[f][:, epoch_index].astype(float) for f in feats]).T

    single_score_names = [n for n in score_names if n not in LOGREG_DETECTORS and n not in THR2_DETECTORS]
    logreg_names = [n for n in score_names if n in LOGREG_DETECTORS]
    thr2_names = [n for n in score_names if n in THR2_DETECTORS]

    for calib_idx, test_idx in splits:
        best = None
        best_rank = None

        # ==================================================================
        # FIXED CONDITION MODE: skip search, evaluate only the fixed config
        # ==================================================================
        if fixed_condition is not None:
            fixed_score, fixed_epoch, fixed_kind = fixed_condition
            if fixed_epoch not in ep2j:
                # epoch not available, skip this split
                continue
            j = ep2j[fixed_epoch]

            if fixed_kind == "score":
                mat = score_map.get(fixed_score)
                if mat is None:
                    continue
                scores_all = mat[:, j].astype(float)
                y_calib_all = y[calib_idx].astype(int)
                s_calib_all = scores_all[calib_idx]

                ok_calib = np.isfinite(s_calib_all)
                if ok_calib.sum() < 4:
                    continue
                y_c = y_calib_all[ok_calib]
                s_c = s_calib_all[ok_calib]
                if len(np.unique(y_c)) < 2:
                    continue

                # orient score using calibration AUROC: make higher => leaked
                sign = 1.0
                auroc_raw = _safe_auroc(y_c, s_c)
                if np.isfinite(auroc_raw) and auroc_raw < 0.5:
                    sign = -1.0
                    s_c = -s_c

                tau = _select_tau_calibration(
                    y_calib=y_c,
                    scores_calib=s_c,
                    fp_k=fp_k,
                    select_metric=select_metric,
                )
                if not np.isfinite(tau):
                    continue
                best = {"kind": "score", "score": fixed_score, "epoch": int(fixed_epoch), "tau": float(tau), "sign": float(sign)}

            elif fixed_kind == "logreg":
                X_all = build_logreg_X(fixed_score, j)
                if X_all is None:
                    continue
                X_c_all = X_all[calib_idx]
                y_c_all = y[calib_idx].astype(int)

                ok_calib = np.isfinite(X_c_all).all(axis=1)
                if ok_calib.sum() < 8:
                    continue
                X_c = X_c_all[ok_calib]
                y_c = y_c_all[ok_calib]
                if len(np.unique(y_c)) < 2:
                    continue

                best = {"kind": "logreg", "score": fixed_score, "epoch": int(fixed_epoch)}

            elif fixed_kind == "thr2":
                var1, var2 = THR2_DETECTORS.get(fixed_score, (None, None))
                if var1 is None or var2 is None:
                    continue
                mat1 = score_map.get(var1)
                mat2 = score_map.get(var2)
                if mat1 is None or mat2 is None:
                    continue

                x1_all = mat1[:, j].astype(float)
                x2_all = mat2[:, j].astype(float)
                y_calib_all = y[calib_idx].astype(int)
                v1_all = x1_all[calib_idx]
                v2_all = x2_all[calib_idx]

                ok_calib = np.isfinite(v1_all) & np.isfinite(v2_all)
                if ok_calib.sum() < 4:
                    continue
                y_c = y_calib_all[ok_calib]
                v1 = v1_all[ok_calib]
                v2 = v2_all[ok_calib]
                if len(np.unique(y_c)) < 2:
                    continue

                res = _search_thr2_rule_fp_k(
                    y=y_c,
                    v1=v1,
                    v2=v2,
                    name1=var1,
                    name2=var2,
                    fp_k=fp_k,
                    select_metric=select_metric,
                )
                if res is None:
                    continue

                best = {"kind": "thr2", "score": fixed_score, "epoch": int(fixed_epoch), "thr2": res}

            else:
                continue

        # ==================================================================
        # SEARCH MODE: original nested search over (score_names x epochs)
        # ==================================================================
        else:
            # ------------------------------------------------------------------
            # 1) Single-score detectors
            # ------------------------------------------------------------------
            for score_name in single_score_names:
                mat = score_map.get(score_name)
                if mat is None:
                    continue
                for e in epochs:
                    j = ep2j[e]
                    scores_all = mat[:, j].astype(float)
                    y_calib_all = y[calib_idx].astype(int)
                    s_calib_all = scores_all[calib_idx]

                    ok_calib = np.isfinite(s_calib_all)
                    if ok_calib.sum() < 4:
                        continue
                    y_c = y_calib_all[ok_calib]
                    s_c = s_calib_all[ok_calib]
                    if len(np.unique(y_c)) < 2:
                        continue

                    # orient score using calibration AUROC: make higher => leaked
                    sign = 1.0
                    auroc_raw = _safe_auroc(y_c, s_c)
                    if np.isfinite(auroc_raw) and auroc_raw < 0.5:
                        sign = -1.0
                        s_c = -s_c
                    auroc_c = _safe_auroc(y_c, s_c)

                    tau = _select_tau_calibration(
                        y_calib=y_c,
                        scores_calib=s_c,
                        fp_k=fp_k,
                        select_metric=select_metric,
                    )
                    if not np.isfinite(tau):
                        continue
                    pred_c = (s_c > tau).astype(int)

                    sel = selection_value(select_metric, y_c, pred_c, s_c)
                    if not np.isfinite(sel):
                        continue

                    tpr_c = selection_value("tpr", y_c, pred_c, s_c)
                    rank = (
                        float(sel),
                        float(tpr_c) if np.isfinite(tpr_c) else -1.0,
                        float(auroc_c) if np.isfinite(auroc_c) else -1.0,
                        -float(tau),
                    )
                    if (best_rank is None) or (rank > best_rank):
                        best_rank = rank
                        best = {"kind": "score", "score": score_name, "epoch": int(e), "tau": float(tau), "sign": float(sign)}

            # ------------------------------------------------------------------
            # 2) Multi-feature logistic regression detectors
            # ------------------------------------------------------------------
            for det_name in logreg_names:
                eval_epochs = [-1] if det_name in LOGREG_ALL_EPOCH_DETECTORS else epochs
                for e in eval_epochs:
                    j = ep2j[e] if e in ep2j else -1
                    X_all = build_logreg_X(det_name, j)
                    if X_all is None:
                        continue
                    X_c_all = X_all[calib_idx]
                    y_c_all = y[calib_idx].astype(int)

                    ok_calib = np.isfinite(X_c_all).all(axis=1)
                    if ok_calib.sum() < 8:
                        continue
                    X_c = X_c_all[ok_calib]
                    y_c = y_c_all[ok_calib]
                    if len(np.unique(y_c)) < 2:
                        continue

                    e_seed = max(0, int(e))
                    s_c = _fit_logreg_proba_scores(X_c, y_c, X_c, random_state=int(random_state) + 1009 * e_seed)
                    tau = _select_tau_calibration(
                        y_calib=y_c,
                        scores_calib=s_c,
                        fp_k=fp_k,
                        select_metric=select_metric,
                    )
                    if not np.isfinite(tau):
                        continue
                    pred_c = (s_c > tau).astype(int)

                    sel = selection_value(select_metric, y_c, pred_c, s_c)
                    if not np.isfinite(sel):
                        continue

                    tpr_c = selection_value("tpr", y_c, pred_c, s_c)
                    auroc_c = _safe_auroc(y_c, s_c)
                    rank = (
                        float(sel),
                        float(tpr_c) if np.isfinite(tpr_c) else -1.0,
                        float(auroc_c) if np.isfinite(auroc_c) else -1.0,
                        -float(tau),
                    )
                    if (best_rank is None) or (rank > best_rank):
                        best_rank = rank
                        best = {"kind": "logreg", "score": det_name, "epoch": int(e)}

            # ------------------------------------------------------------------
            # 3) Two-variable rule detectors (thr2_*)
            # ------------------------------------------------------------------
            for det_name in thr2_names:
                var1, var2 = THR2_DETECTORS[det_name]
                mat1 = score_map.get(var1)
                mat2 = score_map.get(var2)
                if mat1 is None or mat2 is None:
                    continue
                for e in epochs:
                    j = ep2j[e]
                    x1_all = mat1[:, j].astype(float)
                    x2_all = mat2[:, j].astype(float)

                    y_calib_all = y[calib_idx].astype(int)
                    v1_all = x1_all[calib_idx]
                    v2_all = x2_all[calib_idx]

                    ok_calib = np.isfinite(v1_all) & np.isfinite(v2_all)
                    if ok_calib.sum() < 4:
                        continue
                    y_c = y_calib_all[ok_calib]
                    v1 = v1_all[ok_calib]
                    v2 = v2_all[ok_calib]
                    if len(np.unique(y_c)) < 2:
                        continue

                    res = _search_thr2_rule_fp_k(
                        y=y_c,
                        v1=v1,
                        v2=v2,
                        name1=var1,
                        name2=var2,
                        fp_k=fp_k,
                        select_metric=select_metric,
                    )
                    if res is None:
                        continue

                    pred_c = res["pred"]
                    s_c = res["scores"]
                    sel = selection_value(select_metric, y_c, pred_c, s_c)
                    if not np.isfinite(sel):
                        continue

                    tpr_c = selection_value("tpr", y_c, pred_c, s_c)
                    auroc_c = _safe_auroc(y_c, s_c)
                    fp_c = int(np.sum((y_c == 0) & (pred_c == 1)))
                    rank = (
                        float(sel),
                        float(tpr_c) if np.isfinite(tpr_c) else -1.0,
                        float(auroc_c) if np.isfinite(auroc_c) else -1.0,
                        -float(fp_c),
                    )
                    if (best_rank is None) or (rank > best_rank):
                        best_rank = rank
                        best = {"kind": "thr2", "score": det_name, "epoch": int(e), "thr2": res}

        # END OF SEARCH MODE (else block)

        if best is None:
            continue

        kind = str(best["kind"])
        score_name = str(best["score"])
        e = int(best["epoch"])
        j = ep2j[e] if e in ep2j else -1

        rule_str: Optional[str] = None

        if kind == "score":
            tau = float(best["tau"])
            sign = float(best["sign"])

            scores_test_all = score_map[score_name][:, j].astype(float) * sign
            y_test_all = y[test_idx].astype(int)
            s_test_all = scores_test_all[test_idx]

            ok_test = np.isfinite(s_test_all)
            if ok_test.sum() < 2:
                continue
            y_t = y_test_all[ok_test]
            s_t = s_test_all[ok_test]
            pred_t = (s_t > tau).astype(int)

        elif kind == "logreg":
            X_all = build_logreg_X(score_name, j)
            if X_all is None:
                continue
            X_c_all = X_all[calib_idx]
            y_c_all = y[calib_idx].astype(int)
            ok_calib = np.isfinite(X_c_all).all(axis=1)
            if ok_calib.sum() < 8:
                continue
            X_c = X_c_all[ok_calib]
            y_c = y_c_all[ok_calib]
            if len(np.unique(y_c)) < 2:
                continue

            # Fit on calibration, then calibrate threshold (FP-k constrained or unconstrained).
            e_seed = max(0, int(e))
            s_c = _fit_logreg_proba_scores(X_c, y_c, X_c, random_state=int(random_state) + 1009 * e_seed)
            tau = _select_tau_calibration(
                y_calib=y_c,
                scores_calib=s_c,
                fp_k=fp_k,
                select_metric=select_metric,
            )
            if not np.isfinite(tau):
                continue

            X_t_all = X_all[test_idx]
            y_test_all = y[test_idx].astype(int)
            ok_test = np.isfinite(X_t_all).all(axis=1)
            if ok_test.sum() < 2:
                continue
            X_t = X_t_all[ok_test]
            y_t = y_test_all[ok_test]
            s_t = _fit_logreg_proba_scores(X_c, y_c, X_t, random_state=int(random_state) + 1009 * e_seed)
            pred_t = (s_t > tau).astype(int)

        elif kind == "thr2":
            res = best.get("thr2") or {}
            var1, var2 = THR2_DETECTORS[score_name]
            mat1 = score_map.get(var1)
            mat2 = score_map.get(var2)
            if mat1 is None or mat2 is None:
                continue

            x1_test_all = mat1[:, j].astype(float)[test_idx]
            x2_test_all = mat2[:, j].astype(float)[test_idx]
            y_test_all = y[test_idx].astype(int)

            ok_test = np.isfinite(x1_test_all) & np.isfinite(x2_test_all)
            if ok_test.sum() < 2:
                continue

            y_t = y_test_all[ok_test]
            v1_t = x1_test_all[ok_test]
            v2_t = x2_test_all[ok_test]

            t1 = float(res.get("t1", float("nan")))
            t2 = float(res.get("t2", float("nan")))
            dir1 = str(res.get("dir1", "gt"))
            dir2 = str(res.get("dir2", "gt"))
            op = str(res.get("op", "or"))
            c1 = (v1_t > t1) if dir1 == "gt" else (v1_t < t1)
            c2 = (v2_t > t2) if dir2 == "gt" else (v2_t < t2)
            pred_t = (c1 & c2) if op == "and" else (c1 | c2)
            pred_t = pred_t.astype(int)
            s_t = pred_t.astype(float)
            tau = 0.5
            rule_str = str(res.get("rule")) if res.get("rule") is not None else None

        else:
            continue

        split_metrics["auroc"].append(_safe_auroc(y_t, s_t))
        split_metrics["auprc"].append(_safe_auprc(y_t, s_t))
        split_metrics["macro_f1"].append(_safe_macro_f1(y_t, pred_t))
        split_metrics["mcc"].append(_safe_mcc(y_t, pred_t))
        split_metrics["balanced_acc"].append(_safe_balanced_acc(y_t, pred_t))

        chosen.append({"score": score_name, "epoch": int(e), "tau": float(tau), "rule": rule_str})

        pooled_y.extend(int(v) for v in y_t.tolist())
        pooled_scores.extend(float(v) for v in s_t.tolist())
        pooled_pred.extend(int(v) for v in pred_t.tolist())
        pooled_clean += int(np.sum(y_t == 0))
        pooled_fp += int(np.sum((y_t == 0) & (pred_t == 1)))

    cond = {"score": None, "epoch": None, "tau_median": None, "rule": None, "fp_test": pooled_fp, "clean_test": pooled_clean}
    if chosen:
        cnt = Counter((str(c.get("score")), int(c.get("epoch"))) for c in chosen)
        (mode_score, mode_epoch), _ = cnt.most_common(1)[0]

        taus: List[float] = []
        rules: List[str] = []
        for c in chosen:
            if str(c.get("score")) != mode_score or int(c.get("epoch")) != int(mode_epoch):
                continue
            t = c.get("tau")
            if t is not None and np.isfinite(float(t)):
                taus.append(float(t))
            r = c.get("rule")
            if r:
                rules.append(str(r))

        tau_med = float(np.nanmedian(np.asarray(taus, dtype=float))) if taus else float("nan")
        rule_mode = Counter(rules).most_common(1)[0][0] if rules else None
        cond.update({"score": mode_score, "epoch": int(mode_epoch), "tau_median": tau_med, "rule": rule_mode})

    summary = {"metrics": split_metrics, "cond": cond, "n_splits_used": int(len(chosen))}
    pooled = {
        "y": np.asarray(pooled_y, dtype=int),
        "scores": np.asarray(pooled_scores, dtype=float),
        "pred": np.asarray(pooled_pred, dtype=int),
        "fp_test": int(pooled_fp),
        "clean_test": int(pooled_clean),
    }
    return summary, pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--max_epoch", type=int, default=None)
    ap.add_argument("--n_repeats", type=int, default=30)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--clean_calib_frac", type=float, default=0.8)
    ap.add_argument("--leaked_calib_ratio", type=float, default=1.0)
    ap.add_argument("--rule", type=str, default="S0", choices=["S0", "S1"])
    ap.add_argument(
        "--fp_mode",
        type=str,
        default="fp_k",
        choices=["fp_k", "unconstrained"],
        help="Threshold calibration mode: fp_k=enforce FP(clean)<=k; unconstrained=ignore FP-k limit.",
    )
    ap.add_argument("--fp_k", type=int, default=None)
    ap.add_argument("--select_metric", type=str, default="mcc", choices=["mcc", "tpr", "balanced_acc", "macro_f1", "auroc"])
    ap.add_argument("--scores_noref", type=str, default=None)
    ap.add_argument("--scores_withref", type=str, default=None)
    ap.add_argument("--group_split", type=str, default="family", choices=["none", "family", "map"])
    ap.add_argument("--group_map_csv", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel workers for per-candidate reference evaluation. 0=auto(use all logical cores).",
    )
    ap.add_argument("--fixed_condition", type=str, default=None,
                    help="Fixed condition: '<score_name>@e<epoch>', e.g. 'd_drop@e8' or 'logreg_delta@e10'. "
                         "Disables per-repeat condition selection to eliminate overfit.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    cand_keys = _parse_candidates(args.candidates)

    fp_k: Optional[int] = None
    if args.fp_mode == "fp_k":
        fp_k = args.fp_k
        if fp_k is None:
            fp_k = 0 if args.rule == "S0" else 1
        fp_k = int(fp_k)
    k_label = "none" if fp_k is None else str(int(fp_k))
    if fp_k is None:
        print("[THRESHOLD MODE] unconstrained (no FP-k limit)")
    else:
        print(f"[THRESHOLD MODE] fp_k (k={k_label})")

    if int(args.num_workers) <= 0:
        num_workers = max(1, int(os.cpu_count() or 1))
    else:
        num_workers = max(1, int(args.num_workers))
    print(f"[PARALLEL] reference workers={num_workers}")

    score_noref = _parse_score_list(args.scores_noref, DEFAULT_NOREF_SCORES)
    score_withref = _parse_score_list(args.scores_withref, DEFAULT_WITHREF_SCORES)

    # Parse fixed condition if provided
    fixed_cond = _parse_fixed_condition(args.fixed_condition)
    if fixed_cond is not None:
        fixed_score, fixed_epoch, fixed_kind = fixed_cond
        print(f"[FIXED CONDITION MODE] score={fixed_score}, epoch={fixed_epoch}, kind={fixed_kind}")

    group_map = None
    if args.group_split == "map":
        if not args.group_map_csv:
            raise SystemExit("--group_split=map requires --group_map_csv")
        group_map = _load_group_map_csv(Path(args.group_map_csv))

    def _epoch_label(v: object) -> str:
        if v is None:
            return "?"
        try:
            iv = int(v)
            return "ALL" if iv < 0 else str(iv)
        except Exception:
            return str(v)

    def _format_condition(cond: dict, *, fixed_mode: bool) -> str:
        if cond.get("score") is None:
            return "N/A"
        tau_med = cond.get("tau_median")
        tau_str = "nan" if tau_med is None or not np.isfinite(float(tau_med)) else f"{float(tau_med):.4g}"
        epoch_tag = _epoch_label(cond.get("epoch"))
        prefix = "FIXED:" if fixed_mode else ""
        if cond.get("rule"):
            return (
                f"{prefix}{cond['score']}@e{epoch_tag} "
                f"(k={k_label}, rule={cond['rule']}, FPtest={cond['fp_test']}/{cond['clean_test']})"
            )
        return (
            f"{prefix}{cond['score']}@e{epoch_tag} "
            f"(k={k_label}, tau~{tau_str}, FPtest={cond['fp_test']}/{cond['clean_test']})"
        )

    rows: List[dict] = []
    pooled_by_ref: Dict[str, List[dict]] = {}

    for cand_key in cand_keys:
        cand_cfg = CANDIDATES[cand_key]
        try:
            datasets, y = _load_labels(results_root, cand_cfg, args.min_samples)
        except FileNotFoundError as e:
            print(f"[SKIP] {cand_cfg.model_name}: {e}")
            continue

        if len(datasets) < 4 or len(np.unique(y)) < 2:
            n_clean = int(np.sum(y == 0))
            n_leaked = int(np.sum(y == 1))
            print(
                f"[SKIP] {cand_cfg.model_name}: not enough labeled datasets (n={len(datasets)}, clean={n_clean}, leaked={n_leaked})"
            )
            continue

        n_datasets = int(len(datasets))
        n_clean = int(np.sum(y == 0))
        n_leaked = int(np.sum(y == 1))
        units, unit_labels = _build_units(datasets, y, args.group_split, group_map)
        n_units = int(len(units))
        n_clean_units = int(np.sum(unit_labels == 0))
        n_leaked_units = int(np.sum(unit_labels == 1))

        try:
            cand_epoch = _load_epoch_data(
                results_root,
                cand_cfg,
                datasets,
                epochs=None,
                max_epoch=args.max_epoch,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {cand_cfg.model_name}: {e}")
            continue

        epochs = cand_epoch.epochs
        splits = _make_repeated_splits(
            y=y,
            datasets=datasets,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            clean_calib_frac=args.clean_calib_frac,
            leaked_calib_ratio=args.leaked_calib_ratio,
            group_split=args.group_split,
            group_map=group_map,
        )
        if not splits:
            print(
                f"[SKIP] {cand_cfg.model_name}: could not create any valid splits "
                f"(datasets={n_datasets}, clean={n_clean}, leaked={n_leaked}, "
                f"units={n_units}, clean_units={n_clean_units}, leaked_units={n_leaked_units})"
            )
            continue

        # ---------------- NO Reference ----------------
        score_map_noref = _build_scores_noref(cand_epoch)
        summary, pooled = _evaluate_one_setting(
            y=y,
            epochs=epochs,
            score_map=score_map_noref,
            score_names=score_noref,
            splits=splits,
            fp_k=fp_k,
            select_metric=args.select_metric,
            random_state=args.random_state,
            fixed_condition=fixed_cond,
        )
        cond = summary["cond"]
        tau_med = cond.get("tau_median")
        condition = _format_condition(cond, fixed_mode=(fixed_cond is not None))
        auroc_mean, auroc_std, auroc_n = _safe_mean_std(summary["metrics"]["auroc"])
        auprc_mean, auprc_std, auprc_n = _safe_mean_std(summary["metrics"]["auprc"])
        mf1_mean, mf1_std, mf1_n = _safe_mean_std(summary["metrics"]["macro_f1"])
        mcc_mean, mcc_std, mcc_n = _safe_mean_std(summary["metrics"]["mcc"])
        bacc_mean, bacc_std, bacc_n = _safe_mean_std(summary["metrics"]["balanced_acc"])
        rows.append(
            {
                "candidate_key": cand_key,
                "candidate": cand_cfg.model_name,
                "reference_key": "none",
                "reference": "none",
                "n_datasets": n_datasets,
                "n_clean": n_clean,
                "n_leaked": n_leaked,
                "n_units": n_units,
                "n_clean_units": n_clean_units,
                "n_leaked_units": n_leaked_units,
                "condition": condition,
                "cond_score": cond.get("score"),
                "cond_epoch": cond.get("epoch"),
                "fp_mode": args.fp_mode,
                "k": float(fp_k) if fp_k is not None else float("nan"),
                "tau_median": float(tau_med) if tau_med is not None else float("nan"),
                "fp_test": int(cond.get("fp_test", 0)),
                "clean_test": int(cond.get("clean_test", 0)),
                "n_splits_used": int(summary.get("n_splits_used", 0)),
                "AUROC": _fmt_mean_std(summary["metrics"]["auroc"]),
                "AUROC_mean": auroc_mean,
                "AUROC_std": auroc_std,
                "AUROC_n": auroc_n,
                "AUPRC": _fmt_mean_std(summary["metrics"]["auprc"]),
                "AUPRC_mean": auprc_mean,
                "AUPRC_std": auprc_std,
                "AUPRC_n": auprc_n,
                "Macro-F1": _fmt_mean_std(summary["metrics"]["macro_f1"]),
                "Macro-F1_mean": mf1_mean,
                "Macro-F1_std": mf1_std,
                "Macro-F1_n": mf1_n,
                "MCC": _fmt_mean_std(summary["metrics"]["mcc"]),
                "MCC_mean": mcc_mean,
                "MCC_std": mcc_std,
                "MCC_n": mcc_n,
                "BalancedAcc": _fmt_mean_std(summary["metrics"]["balanced_acc"]),
                "BalancedAcc_mean": bacc_mean,
                "BalancedAcc_std": bacc_std,
                "BalancedAcc_n": bacc_n,
            }
        )
        pooled_by_ref.setdefault("none", []).append(pooled)

        # ---------------- WITH Reference ----------------
        ref_epochs_by_key: Dict[str, EpochData] = {}
        for ref_key, ref_cfg in REFERENCES.items():
            try:
                ref_epoch = _load_epoch_data(
                    results_root,
                    ref_cfg,
                    datasets,
                    epochs=epochs,
                    max_epoch=args.max_epoch,
                )
            except FileNotFoundError as e:
                print(f"[REF SKIP] {cand_cfg.model_name} vs {ref_cfg.model_name}: {e}")
                continue
            ref_epochs_by_key[ref_key] = ref_epoch

        score_maps_withref_by_key: Dict[str, Dict[str, np.ndarray]] = {
            ref_key: _build_scores_withref(cand_epoch, ref_epoch)
            for ref_key, ref_epoch in ref_epochs_by_key.items()
        }

        ref_eval_keys = [rk for rk in REFERENCES.keys() if rk in score_maps_withref_by_key]

        def _evaluate_one_ref_bundle(ref_key: str) -> Tuple[List[dict], List[Tuple[str, dict]]]:
            ref_cfg = REFERENCES[ref_key]
            score_map_ref = score_maps_withref_by_key.get(ref_key)
            if score_map_ref is None:
                return [], []

            local_rows: List[dict] = []
            local_pooled: List[Tuple[str, dict]] = []

            custom_logreg_X_ref: Dict[str, np.ndarray] = {}
            if "logreg_delta_all_epochs" in score_withref:
                X_all_epochs = _build_single_ref_all_epoch_logreg_X(score_map_ref)
                if X_all_epochs is not None:
                    custom_logreg_X_ref["logreg_delta_all_epochs"] = X_all_epochs

            summary, pooled = _evaluate_one_setting(
                y=y,
                epochs=epochs,
                score_map=score_map_ref,
                score_names=score_withref,
                splits=splits,
                fp_k=fp_k,
                select_metric=args.select_metric,
                random_state=args.random_state,
                fixed_condition=fixed_cond,
                custom_logreg_X=(custom_logreg_X_ref or None),
            )
            cond = summary["cond"]
            tau_med = cond.get("tau_median")
            condition = _format_condition(cond, fixed_mode=(fixed_cond is not None))
            auroc_mean, auroc_std, auroc_n = _safe_mean_std(summary["metrics"]["auroc"])
            auprc_mean, auprc_std, auprc_n = _safe_mean_std(summary["metrics"]["auprc"])
            mf1_mean, mf1_std, mf1_n = _safe_mean_std(summary["metrics"]["macro_f1"])
            mcc_mean, mcc_std, mcc_n = _safe_mean_std(summary["metrics"]["mcc"])
            bacc_mean, bacc_std, bacc_n = _safe_mean_std(summary["metrics"]["balanced_acc"])
            local_rows.append(
                {
                    "candidate_key": cand_key,
                    "candidate": cand_cfg.model_name,
                    "reference_key": ref_key,
                    "reference": ref_cfg.model_name,
                    "n_datasets": n_datasets,
                    "n_clean": n_clean,
                    "n_leaked": n_leaked,
                    "n_units": n_units,
                    "n_clean_units": n_clean_units,
                    "n_leaked_units": n_leaked_units,
                    "condition": condition,
                    "cond_score": cond.get("score"),
                    "cond_epoch": cond.get("epoch"),
                    "fp_mode": args.fp_mode,
                    "k": float(fp_k) if fp_k is not None else float("nan"),
                    "tau_median": float(tau_med) if tau_med is not None else float("nan"),
                    "fp_test": int(cond.get("fp_test", 0)),
                    "clean_test": int(cond.get("clean_test", 0)),
                    "n_splits_used": int(summary.get("n_splits_used", 0)),
                    "AUROC": _fmt_mean_std(summary["metrics"]["auroc"]),
                    "AUROC_mean": auroc_mean,
                    "AUROC_std": auroc_std,
                    "AUROC_n": auroc_n,
                    "AUPRC": _fmt_mean_std(summary["metrics"]["auprc"]),
                    "AUPRC_mean": auprc_mean,
                    "AUPRC_std": auprc_std,
                    "AUPRC_n": auprc_n,
                    "Macro-F1": _fmt_mean_std(summary["metrics"]["macro_f1"]),
                    "Macro-F1_mean": mf1_mean,
                    "Macro-F1_std": mf1_std,
                    "Macro-F1_n": mf1_n,
                    "MCC": _fmt_mean_std(summary["metrics"]["mcc"]),
                    "MCC_mean": mcc_mean,
                    "MCC_std": mcc_std,
                    "MCC_n": mcc_n,
                    "BalancedAcc": _fmt_mean_std(summary["metrics"]["balanced_acc"]),
                    "BalancedAcc_mean": bacc_mean,
                    "BalancedAcc_std": bacc_std,
                    "BalancedAcc_n": bacc_n,
                }
            )
            local_pooled.append((ref_cfg.model_name, pooled))

            # Single-ref all-epoch control row.
            if "logreg_delta_all_epochs" in score_withref:
                X_all_epochs = _build_single_ref_all_epoch_logreg_X(score_map_ref)
                if X_all_epochs is not None:
                    summary_eall, pooled_eall = _evaluate_one_setting(
                        y=y,
                        epochs=epochs,
                        score_map=score_map_ref,
                        score_names=["logreg_delta_all_epochs"],
                        splits=splits,
                        fp_k=fp_k,
                        select_metric=args.select_metric,
                        random_state=args.random_state,
                        fixed_condition=fixed_cond,
                        custom_logreg_X={"logreg_delta_all_epochs": X_all_epochs},
                    )
                    cond = summary_eall["cond"]
                    tau_med = cond.get("tau_median")
                    condition = _format_condition(cond, fixed_mode=(fixed_cond is not None))
                    auroc_mean, auroc_std, auroc_n = _safe_mean_std(summary_eall["metrics"]["auroc"])
                    auprc_mean, auprc_std, auprc_n = _safe_mean_std(summary_eall["metrics"]["auprc"])
                    mf1_mean, mf1_std, mf1_n = _safe_mean_std(summary_eall["metrics"]["macro_f1"])
                    mcc_mean, mcc_std, mcc_n = _safe_mean_std(summary_eall["metrics"]["mcc"])
                    bacc_mean, bacc_std, bacc_n = _safe_mean_std(summary_eall["metrics"]["balanced_acc"])
                    ref_name_eall = f"{ref_cfg.model_name} [EALL-LR]"
                    local_rows.append(
                        {
                            "candidate_key": cand_key,
                            "candidate": cand_cfg.model_name,
                            "reference_key": f"{ref_key}_all_epochs_lr",
                            "reference": ref_name_eall,
                            "n_datasets": n_datasets,
                            "n_clean": n_clean,
                            "n_leaked": n_leaked,
                            "n_units": n_units,
                            "n_clean_units": n_clean_units,
                            "n_leaked_units": n_leaked_units,
                            "condition": condition,
                            "cond_score": cond.get("score"),
                            "cond_epoch": cond.get("epoch"),
                            "fp_mode": args.fp_mode,
                            "k": float(fp_k) if fp_k is not None else float("nan"),
                            "tau_median": float(tau_med) if tau_med is not None else float("nan"),
                            "fp_test": int(cond.get("fp_test", 0)),
                            "clean_test": int(cond.get("clean_test", 0)),
                            "n_splits_used": int(summary_eall.get("n_splits_used", 0)),
                            "AUROC": _fmt_mean_std(summary_eall["metrics"]["auroc"]),
                            "AUROC_mean": auroc_mean,
                            "AUROC_std": auroc_std,
                            "AUROC_n": auroc_n,
                            "AUPRC": _fmt_mean_std(summary_eall["metrics"]["auprc"]),
                            "AUPRC_mean": auprc_mean,
                            "AUPRC_std": auprc_std,
                            "AUPRC_n": auprc_n,
                            "Macro-F1": _fmt_mean_std(summary_eall["metrics"]["macro_f1"]),
                            "Macro-F1_mean": mf1_mean,
                            "Macro-F1_std": mf1_std,
                            "Macro-F1_n": mf1_n,
                            "MCC": _fmt_mean_std(summary_eall["metrics"]["mcc"]),
                            "MCC_mean": mcc_mean,
                            "MCC_std": mcc_std,
                            "MCC_n": mcc_n,
                            "BalancedAcc": _fmt_mean_std(summary_eall["metrics"]["balanced_acc"]),
                            "BalancedAcc_mean": bacc_mean,
                            "BalancedAcc_std": bacc_std,
                            "BalancedAcc_n": bacc_n,
                        }
                    )
                    local_pooled.append((ref_name_eall, pooled_eall))

            return local_rows, local_pooled

        ref_bundle_results: List[Tuple[List[dict], List[Tuple[str, dict]]]] = []
        if ref_eval_keys:
            if num_workers > 1 and len(ref_eval_keys) > 1:
                worker_n = min(num_workers, len(ref_eval_keys))
                with ThreadPoolExecutor(max_workers=worker_n) as ex:
                    ref_bundle_results = list(ex.map(_evaluate_one_ref_bundle, ref_eval_keys))
            else:
                ref_bundle_results = [_evaluate_one_ref_bundle(k) for k in ref_eval_keys]

        for bundle_rows, bundle_pooled in ref_bundle_results:
            rows.extend(bundle_rows)
            for ref_name, pooled in bundle_pooled:
                pooled_by_ref.setdefault(ref_name, []).append(pooled)

        # ---------------- MULTI-REF LOGREG ----------------
        have_all_refs = all(k in score_maps_withref_by_key for k in REFERENCES.keys())
        selected_multiref_detectors = [
            n
            for n in score_withref
            if n
            in {
                "logreg_allrefs_plus_det",
                "logreg_allrefs_only",
                "logreg_allrefs_plus_det_all_epochs",
                "logreg_allrefs_only_all_epochs",
            }
        ]
        if selected_multiref_detectors and not have_all_refs:
            missing = [k for k in REFERENCES.keys() if k not in score_maps_withref_by_key]
            print(
                f"[MULTIREF SKIP] {cand_cfg.model_name}: missing reference(s) for all-ref logreg: {missing}"
            )

        if selected_multiref_detectors and have_all_refs:
            custom_logreg_mats = _build_multiref_logreg_mats(
                score_map_noref=score_map_noref,
                score_maps_withref=score_maps_withref_by_key,
            )
            custom_logreg_X = _build_multiref_logreg_X_all_epochs(
                score_map_noref=score_map_noref,
                score_maps_withref=score_maps_withref_by_key,
            )

            multiref_meta = [
                ("logreg_allrefs_plus_det", "all_refs_plus_det", "ALL_REFS+DET"),
                ("logreg_allrefs_only", "all_refs_only", "ALL_REFS"),
                ("logreg_allrefs_plus_det_all_epochs", "all_refs_plus_det_all_epochs", "ALL_REFS+DET [EALL-LR]"),
                ("logreg_allrefs_only_all_epochs", "all_refs_only_all_epochs", "ALL_REFS [EALL-LR]"),
            ]
            for det_name, det_ref_key, det_ref_name in multiref_meta:
                if det_name not in selected_multiref_detectors:
                    continue

                summary, pooled = _evaluate_one_setting(
                    y=y,
                    epochs=epochs,
                    score_map=score_map_noref,
                    score_names=[det_name],
                    splits=splits,
                    fp_k=fp_k,
                    select_metric=args.select_metric,
                    random_state=args.random_state,
                    fixed_condition=fixed_cond,
                    custom_logreg_mats=custom_logreg_mats,
                    custom_logreg_X=custom_logreg_X,
                )
                cond = summary["cond"]
                tau_med = cond.get("tau_median")
                condition = _format_condition(cond, fixed_mode=(fixed_cond is not None))
                auroc_mean, auroc_std, auroc_n = _safe_mean_std(summary["metrics"]["auroc"])
                auprc_mean, auprc_std, auprc_n = _safe_mean_std(summary["metrics"]["auprc"])
                mf1_mean, mf1_std, mf1_n = _safe_mean_std(summary["metrics"]["macro_f1"])
                mcc_mean, mcc_std, mcc_n = _safe_mean_std(summary["metrics"]["mcc"])
                bacc_mean, bacc_std, bacc_n = _safe_mean_std(summary["metrics"]["balanced_acc"])
                rows.append(
                    {
                        "candidate_key": cand_key,
                        "candidate": cand_cfg.model_name,
                        "reference_key": det_ref_key,
                        "reference": det_ref_name,
                        "n_datasets": n_datasets,
                        "n_clean": n_clean,
                        "n_leaked": n_leaked,
                        "n_units": n_units,
                        "n_clean_units": n_clean_units,
                        "n_leaked_units": n_leaked_units,
                        "condition": condition,
                        "cond_score": cond.get("score"),
                        "cond_epoch": cond.get("epoch"),
                        "fp_mode": args.fp_mode,
                        "k": float(fp_k) if fp_k is not None else float("nan"),
                        "tau_median": float(tau_med) if tau_med is not None else float("nan"),
                        "fp_test": int(cond.get("fp_test", 0)),
                        "clean_test": int(cond.get("clean_test", 0)),
                        "n_splits_used": int(summary.get("n_splits_used", 0)),
                        "AUROC": _fmt_mean_std(summary["metrics"]["auroc"]),
                        "AUROC_mean": auroc_mean,
                        "AUROC_std": auroc_std,
                        "AUROC_n": auroc_n,
                        "AUPRC": _fmt_mean_std(summary["metrics"]["auprc"]),
                        "AUPRC_mean": auprc_mean,
                        "AUPRC_std": auprc_std,
                        "AUPRC_n": auprc_n,
                        "Macro-F1": _fmt_mean_std(summary["metrics"]["macro_f1"]),
                        "Macro-F1_mean": mf1_mean,
                        "Macro-F1_std": mf1_std,
                        "Macro-F1_n": mf1_n,
                        "MCC": _fmt_mean_std(summary["metrics"]["mcc"]),
                        "MCC_mean": mcc_mean,
                        "MCC_std": mcc_std,
                        "MCC_n": mcc_n,
                        "BalancedAcc": _fmt_mean_std(summary["metrics"]["balanced_acc"]),
                        "BalancedAcc_mean": bacc_mean,
                        "BalancedAcc_std": bacc_std,
                        "BalancedAcc_n": bacc_n,
                    }
                )
                pooled_by_ref.setdefault(det_ref_name, []).append(pooled)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["candidate", "reference"]).reset_index(drop=True)

    # MACRO aggregates per reference: compute per-model pooled metrics then average over models
    macro_rows: List[dict] = []
    for ref_name, pooled_list in pooled_by_ref.items():
        per_auroc: List[float] = []
        per_auprc: List[float] = []
        per_mf1: List[float] = []
        per_mcc: List[float] = []
        per_bacc: List[float] = []
        fp_total = int(sum(int(p.get("fp_test", 0)) for p in pooled_list))
        clean_total = int(sum(int(p.get("clean_test", 0)) for p in pooled_list))

        for p in pooled_list:
            y_arr = p.get("y", np.array([], dtype=int))
            s_arr = p.get("scores", np.array([], dtype=float))
            pred_arr = p.get("pred", np.array([], dtype=int))
            if len(y_arr) < 2:
                continue
            per_auroc.append(_safe_auroc(y_arr, s_arr))
            per_auprc.append(_safe_auprc(y_arr, s_arr))
            per_mf1.append(_safe_macro_f1(y_arr, pred_arr))
            per_mcc.append(_safe_mcc(y_arr, pred_arr))
            per_bacc.append(_safe_balanced_acc(y_arr, pred_arr))

        if not per_mcc:
            continue

        macro_rows.append(
            {
                "candidate": "MACRO",
                "reference": ref_name,
                "condition": f"avg over models (FPtest={fp_total}/{clean_total})",
                "AUROC": _fmt_mean_std(per_auroc),
                "AUPRC": _fmt_mean_std(per_auprc),
                "Macro-F1": _fmt_mean_std(per_mf1),
                "MCC": _fmt_mean_std(per_mcc),
                "BalancedAcc": _fmt_mean_std(per_bacc),
            }
        )

    if macro_rows:
        df = pd.concat([df, pd.DataFrame(macro_rows)], ignore_index=True)

    # MICRO aggregates per reference: pool all (model,dataset) test occurrences across repeats
    micro_rows: List[dict] = []
    for ref_name, pooled_list in pooled_by_ref.items():
        y_list = [p["y"] for p in pooled_list if len(p.get("y", [])) > 0]
        s_list = [p["scores"] for p in pooled_list if len(p.get("scores", [])) > 0]
        p_list = [p["pred"] for p in pooled_list if len(p.get("pred", [])) > 0]
        if not y_list or not s_list or not p_list:
            continue

        y_all = np.concatenate(y_list, axis=0)
        s_all = np.concatenate(s_list, axis=0)
        pred_all = np.concatenate(p_list, axis=0)
        fp_total = int(sum(int(p.get("fp_test", 0)) for p in pooled_list))
        clean_total = int(sum(int(p.get("clean_test", 0)) for p in pooled_list))

        auroc = _safe_auroc(y_all, s_all)
        auprc = _safe_auprc(y_all, s_all)
        mf1 = _safe_macro_f1(y_all, pred_all)
        mcc = _safe_mcc(y_all, pred_all)
        bacc = _safe_balanced_acc(y_all, pred_all)

        micro_rows.append(
            {
                "candidate": "MICRO",
                "reference": ref_name,
                "condition": f"pooled (FPtest={fp_total}/{clean_total})",
                "AUROC": f"{auroc:.3f}" if np.isfinite(auroc) else "nan",
                "AUPRC": f"{auprc:.3f}" if np.isfinite(auprc) else "nan",
                "Macro-F1": f"{mf1:.3f}" if np.isfinite(mf1) else "nan",
                "MCC": f"{mcc:.3f}" if np.isfinite(mcc) else "nan",
                "BalancedAcc": f"{bacc:.3f}" if np.isfinite(bacc) else "nan",
            }
        )

    if micro_rows:
        df = pd.concat([df, pd.DataFrame(micro_rows)], ignore_index=True)

    preferred_cols = [
        "candidate",
        "reference",
        "condition",
        "AUROC",
        "AUPRC",
        "Macro-F1",
        "MCC",
        "BalancedAcc",
        "n_splits_used",
        "candidate_key",
        "reference_key",
        "n_datasets",
        "n_clean",
        "n_leaked",
        "n_units",
        "n_clean_units",
        "n_leaked_units",
        "cond_score",
        "cond_epoch",
        "fp_mode",
        "k",
        "tau_median",
        "fp_test",
        "clean_test",
        "AUROC_mean",
        "AUROC_std",
        "AUROC_n",
        "AUPRC_mean",
        "AUPRC_std",
        "AUPRC_n",
        "Macro-F1_mean",
        "Macro-F1_std",
        "Macro-F1_n",
        "MCC_mean",
        "MCC_std",
        "MCC_n",
        "BalancedAcc_mean",
        "BalancedAcc_std",
        "BalancedAcc_n",
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    rest = [c for c in df.columns if c not in set(existing)]
    df = df[existing + rest]

    pd.set_option("display.max_colwidth", 120)
    print_cols = ["candidate", "reference", "condition", "AUROC", "AUPRC", "Macro-F1", "MCC", "BalancedAcc"]
    df_print = df[print_cols].copy()
    print(df_print.to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
    else:
        default_name = "sft_audit_eval_table.csv" if args.fp_mode == "fp_k" else "sft_audit_eval_table_unconstrained.csv"
        out = results_root / default_name
    out_full = out.with_name(out.stem + "_full" + out.suffix)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        df_print.to_csv(out, index=False, encoding="utf-8")
        df.to_csv(out_full, index=False, encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
