"""
TS-Min-K% Frequency Domain Detection for TiRex
================================================
Adapts the LLM Min-K% method to time series:
- Compute prediction residuals
- Apply FFT to residuals
- Extract energy in top-K% highest frequencies
- Lower high-freq energy = model memorized the data

Usage:
    python ts_mink_fft.py
"""

import sys
from pathlib import Path
import os

from path_config import GIFT_EVAL_ROOT, PROJECT_ROOT, RESULTS_ROOT, optional_repo_path

sys.path.insert(0, str(PROJECT_ROOT / "src"))
_tirex_src = optional_repo_path("TIREX_REPO", ["tirex"], "src")
if _tirex_src is not None:
    sys.path.insert(0, str(_tirex_src))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneOut
import pyarrow as pa
import pyarrow.ipc as ipc
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
TIREX_MODEL = os.environ.get("TIREX_MODEL", "NX-AI/TiRex")

CONTEXT_LENGTH = 128
PRED_LENGTH = 24
BATCH_SIZE = 8
NUM_SAMPLES = 50
MIN_SAMPLES_THRESHOLD = 20

# K values to compare
K_VALUES = [10, 20, 30]

# Datasets (excluding pretrain/)
ALL_DATASETS = [
    "bitbrains_fast_storage/H", "bitbrains_rnd/H", "LOOP_SEATTLE/H",
    "M_DENSE/H", "SZ_TAXI/H", "electricity/H", "electricity/W",
    "hierarchical_sales/D", "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H",
    "solar/D", "saugeenday/D", "us_births/D",
    "bizitobs_service", "covid_deaths", "m4_daily", "m4_hourly",
    "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant",
    "temperature_rain_with_missing",
]

LEAKED_DATASETS = {
    "electricity/H", "electricity/W", "kdd_cup_2018_with_missing/D",
    "kdd_cup_2018_with_missing/H", "m4_daily", "m4_hourly", "m4_monthly", "m4_weekly",
    "temperature_rain_with_missing",
}


# ============================================================
# Data Loading
# ============================================================
def load_dataset_data(dataset_name: str, max_samples: int = NUM_SAMPLES) -> List[np.ndarray]:
    """Load time series data from arrow files."""
    data_path = GIFT_EVAL_ROOT / dataset_name
    if not data_path.exists():
        return []
    
    arrow_files = list(data_path.glob("*.arrow"))
    if not arrow_files:
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                arrow_files.extend(subdir.glob("*.arrow"))
    
    if not arrow_files:
        return []
    
    series_list = []
    min_len = CONTEXT_LENGTH + PRED_LENGTH
    
    for arrow_path in arrow_files:
        try:
            with pa.memory_map(str(arrow_path), 'r') as source:
                reader = ipc.open_stream(source)
                table = reader.read_all()
            
            if "target" not in table.column_names:
                continue
            
            for i in range(table.num_rows):
                if max_samples and len(series_list) >= max_samples:
                    break
                
                try:
                    target_raw = table["target"][i].as_py()
                    if target_raw is None:
                        continue
                    
                    arr = np.asarray(target_raw, dtype=np.float32)
                    if arr.ndim != 1:
                        arr = arr.flatten()
                    
                    if len(arr) < min_len:
                        continue
                    
                    if not np.isfinite(arr).all():
                        mask = np.isnan(arr) | np.isinf(arr)
                        if mask.sum() < len(arr) * 0.5 and (~mask).sum() > 0:
                            arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
                        else:
                            continue
                    
                    if np.std(arr) > 1e-6:
                        series_list.append(arr)
                except Exception:
                    continue
            
            if max_samples and len(series_list) >= max_samples:
                break
        except Exception:
            continue
    
    return series_list


class ForecastDataset(Dataset):
    """Forecast dataset with Z-score normalization."""
    
    def __init__(self, series_list: List[np.ndarray], context_len: int, pred_len: int):
        self.samples = []
        for idx, s in enumerate(series_list):
            if len(s) >= context_len + pred_len:
                context = s[:context_len].astype(np.float32)
                target = s[context_len:context_len + pred_len].astype(np.float32)
                
                mean = np.mean(context)
                std = np.std(context)
                if std < 1e-6:
                    std = 1.0
                
                context_norm = (context - mean) / std
                target_norm = (target - mean) / std
                
                self.samples.append((context_norm, target_norm, mean, std, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target, mean, std, original_idx = self.samples[idx]
        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'mean': torch.tensor(mean, dtype=torch.float32),
            'std': torch.tensor(std, dtype=torch.float32),
            'original_idx': torch.tensor(original_idx, dtype=torch.long)
        }


# ============================================================
# TiRex Model Wrapper
# ============================================================
class TiRexWrapper(nn.Module):
    """TiRex model wrapper for prediction."""
    
    def __init__(self, tirex_model, pred_len: int, hidden_dim: int = None):
        super().__init__()
        self.tirex = tirex_model
        self.pred_len = pred_len
        inferred_dim = None
        if getattr(self.tirex, "blocks", None):
            inferred_dim = getattr(getattr(self.tirex.blocks[0], "config", None), "embedding_dim", None)
        self.hidden_dim = hidden_dim or inferred_dim or 512
        self.pred_head = nn.Linear(self.hidden_dim, pred_len)
        self.patch_size = getattr(getattr(self.tirex, "config", None), "input_patch_size", 32)
        self.nan_mask_value = getattr(getattr(self.tirex, "config", None), "nan_mask_value", 0.0)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B, L = context.shape
        
        try:
            pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
            if pad_len > 0:
                context = F.pad(context, (pad_len, 0), mode='replicate')
            
            tokens, _ = self.tirex.tokenizer.input_transform(context)
            token_mask = (~torch.isnan(tokens)).to(tokens.dtype)
            tokens = torch.nan_to_num(tokens, nan=self.nan_mask_value)
            token_features = torch.cat((tokens, token_mask), dim=2)
            
            hidden = self.tirex.input_patch_embedding(token_features)
            for blk in self.tirex.blocks:
                hidden = blk(hidden)
            hidden = self.tirex.out_norm(hidden)
            
            last_hidden = hidden[:, -1, :]
            return self.pred_head(last_hidden)
            
        except Exception as e:
            pooled = F.adaptive_avg_pool1d(context.unsqueeze(1), self.hidden_dim).squeeze(1)
            return self.pred_head(pooled)


# ============================================================
# FFT Min-K% Score Computation
# ============================================================
def compute_mink_fft_scores(residuals: np.ndarray, k_percent: int) -> float:
    """
    Compute Min-K% FFT score from residuals.
    
    Args:
        residuals: Array of shape [num_samples, pred_len]
        k_percent: Percentage of highest frequencies to consider
    
    Returns:
        Score: Mean energy in top-K% frequencies (lower = more memorized)
    """
    scores = []
    
    for res in residuals:
        # Apply FFT
        fft_result = np.fft.fft(res)
        fft_magnitude = np.abs(fft_result)
        
        # Get frequency indices (sorted by frequency, high to low)
        n = len(fft_magnitude)
        # In FFT, high frequencies are in the middle, we take from n//2
        half_n = n // 2
        high_freq_start = int(half_n * (1 - k_percent / 100))
        
        # Extract high-frequency energy
        high_freq_energy = np.mean(fft_magnitude[high_freq_start:half_n] ** 2)
        scores.append(high_freq_energy)
    
    return np.mean(scores)


def run_inference_and_compute_scores(model, dataloader, device, k_values: List[int]) -> Dict[int, float]:
    """Run inference and compute Min-K% scores for multiple K values."""
    model.eval()
    all_residuals = []
    
    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            
            try:
                pred = model(context)
                if torch.isnan(pred).any():
                    continue
                
                residual = (target - pred).cpu().numpy()
                all_residuals.append(residual)
            except Exception:
                continue
    
    if not all_residuals:
        return {k: np.nan for k in k_values}
    
    residuals = np.vstack(all_residuals)
    
    scores = {}
    for k in k_values:
        scores[k] = compute_mink_fft_scores(residuals, k)
    
    return scores


# ============================================================
# Evaluation
# ============================================================
def fit_predict_loo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-One-Out CV with LogReg."""
    loo = LeaveOneOut()
    oof_probs = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        if len(np.unique(y_train)) < 2:
            oof_probs[test_idx] = 0.5
            continue
        
        valid_mask = np.isfinite(X_train).all(axis=1)
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]
        
        if len(y_train_clean) < 2 or len(np.unique(y_train_clean)) < 2:
            oof_probs[test_idx] = 0.5
            continue
        
        if not np.isfinite(X_test).all():
            oof_probs[test_idx] = 0.5
            continue
        
        try:
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', solver='liblinear'))
            model.fit(X_train_clean, y_train_clean)
            oof_probs[test_idx] = model.predict_proba(X_test)[:, 1]
        except Exception:
            oof_probs[test_idx] = 0.5
    
    return oof_probs


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Compute metrics with optimal threshold."""
    best = {"f1": 0.0, "thr": 0.5, "rec": 0.0, "prec": 0.0, "spec": 0.0, "auc": 0.0}
    
    try:
        best["auc"] = roc_auc_score(y_true, y_prob)
    except:
        best["auc"] = 0.5
    
    thresholds = np.unique(y_prob)
    thresholds = np.concatenate([[0.5], thresholds])
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 >= best["f1"]:
            best.update({
                "f1": f1,
                "thr": thr,
                "rec": tpr,
                "prec": precision_score(y_true, y_pred, zero_division=0),
                "spec": tnr,
                "conf_matrix": [int(tn), int(fp), int(fn), int(tp)]
            })
    
    return best


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 100)
    print("TS-Min-K% Frequency Domain Detection (TiRex)")
    print(f"K values: {K_VALUES}")
    print("=" * 100)
    
    # Select GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load TiRex
    print("\nLoading TiRex model...")
    from tirex import load_model
    tirex_base = load_model(TIREX_MODEL, device=str(device))
    model = TiRexWrapper(tirex_base, PRED_LENGTH).to(device)
    print("Model loaded.\n")
    
    # Process datasets
    all_scores = {k: [] for k in K_VALUES}
    all_labels = []
    all_raw_losses = []
    all_datasets = []
    
    print(f"Processing {len(ALL_DATASETS)} datasets...")
    print("-" * 80)
    
    for ds_name in ALL_DATASETS:
        series_list = load_dataset_data(ds_name)
        
        if len(series_list) < MIN_SAMPLES_THRESHOLD:
            print(f"  [{ds_name}] SKIP - insufficient samples ({len(series_list)})")
            continue
        
        is_leaked = ds_name in LEAKED_DATASETS
        
        dataset = ForecastDataset(series_list, CONTEXT_LENGTH, PRED_LENGTH)
        if len(dataset) < MIN_SAMPLES_THRESHOLD:
            print(f"  [{ds_name}] SKIP - insufficient valid samples")
            continue
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Compute scores
        scores = run_inference_and_compute_scores(model, dataloader, device, K_VALUES)
        
        # Also compute raw loss for comparison
        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                context = batch['context'].to(device)
                target = batch['target'].to(device)
                try:
                    pred = model(context)
                    if not torch.isnan(pred).any():
                        loss = F.mse_loss(pred, target).item()
                        total_loss += loss
                        count += 1
                except:
                    pass
        raw_loss = total_loss / count if count > 0 else np.nan
        
        # Store results
        for k in K_VALUES:
            all_scores[k].append(scores[k])
        all_labels.append(1 if is_leaked else 0)
        all_raw_losses.append(raw_loss)
        all_datasets.append(ds_name)
        
        leaked_str = "LEAKED" if is_leaked else "CLEAN"
        print(f"  [{ds_name}] {leaked_str} | Loss={raw_loss:.4f} | K10={scores[10]:.4f} | K20={scores[20]:.4f} | K30={scores[30]:.4f}")
    
    print("-" * 80)
    print(f"\nTotal datasets: {len(all_labels)}")
    
    # Convert to arrays
    y = np.array(all_labels)
    n_leaked = int(np.sum(y))
    n_clean = len(y) - n_leaked
    print(f"Leaked: {n_leaked}, Clean: {n_clean}\n")
    
    # Evaluate each method
    results = []
    
    print("=" * 120)
    print(f"{'Method':<20} {'AUC':<8} {'F1':<8} {'Rec(TPR)':<10} {'Prec':<8} {'Spec(TNR)':<10} {'ConfMat[TN,FP,FN,TP]'}")
    print("-" * 120)
    
    # Raw Loss baseline
    X_raw = np.array(all_raw_losses).reshape(-1, 1) * -1  # Invert: lower = more leaked
    probs = fit_predict_loo(X_raw, y)
    m = evaluate_metrics(y, probs)
    tn, fp, fn, tp = m["conf_matrix"]
    print(f"{'Raw Loss':<20} {m['auc']:<8.3f} {m['f1']:<8.3f} {m['rec']:<10.3f} {m['prec']:<8.3f} {m['spec']:<10.3f} [{tn}, {fp}, {fn}, {tp}]")
    results.append({"method": "Raw Loss", **m})
    
    # Min-K% FFT for each K
    for k in K_VALUES:
        X = np.array(all_scores[k]).reshape(-1, 1) * -1  # Invert: lower high-freq energy = more memorized
        probs = fit_predict_loo(X, y)
        m = evaluate_metrics(y, probs)
        tn, fp, fn, tp = m["conf_matrix"]
        print(f"{'Min-K% FFT (K=' + str(k) + ')':<20} {m['auc']:<8.3f} {m['f1']:<8.3f} {m['rec']:<10.3f} {m['prec']:<8.3f} {m['spec']:<10.3f} [{tn}, {fp}, {fn}, {tp}]")
        results.append({"method": f"Min-K% FFT (K={k})", **m})
    
    print("=" * 120)
    
    # Save results
    out_dir = RESULTS_ROOT / "ts_mink_fft"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed scores
    detail_df = pd.DataFrame({
        "dataset": all_datasets,
        "is_leaked": all_labels,
        "raw_loss": all_raw_losses,
        **{f"mink_k{k}": all_scores[k] for k in K_VALUES}
    })
    detail_df.to_csv(out_dir / "detailed_scores.csv", index=False)
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
