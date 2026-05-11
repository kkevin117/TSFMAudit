"""
TiRex 单模型微调实验：统一损失函数训练动态追踪
使用统一的 MSE 预测损失函数追踪模型的训练动态。
"""

import sys
from pathlib import Path
import os

from path_config import GIFT_EVAL_ROOT, GIFTEVAL_PRETRAIN_ROOT, METADATA_ROOT, PROJECT_ROOT, optional_repo_path, output_dir as resolve_output_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))
_tirex_src = optional_repo_path("TIREX_REPO", ["tirex"], "src")
if _tirex_src is not None:
    sys.path.insert(0, str(_tirex_src))

import copy
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

def get_gpu_memory_usage() -> List[Tuple[int, float]]:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id, mem_used, mem_total = int(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())
                gpu_info.append((gpu_id, mem_used / mem_total * 100))
        return gpu_info
    except Exception:
        return [(0, 0.0)]

def select_best_gpu() -> torch.device:
    if not torch.cuda.is_available(): return torch.device("cpu")
    gpu_info = get_gpu_memory_usage()
    if not gpu_info: return torch.device("cuda:0")
    best_gpu = min(gpu_info, key=lambda x: x[1])
    print(f"选择 GPU {best_gpu[0]} (显存占用: {best_gpu[1]:.1f}%)")
    return torch.device(f"cuda:{best_gpu[0]}")

def set_random_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

TIREX_MODEL = os.environ.get("TIREX_MODEL", "NX-AI/TiRex")
PRETRAIN_ROOT = GIFTEVAL_PRETRAIN_ROOT
LEAKED_NAMES_FILE = Path(os.environ.get("TIREX_LEAKED_NAMES_FILE", METADATA_ROOT / "chronos_datasets_names.txt"))

NUM_EPOCHS, NUM_SAMPLES, NUM_WINDOWS, BATCH_SIZE, LEARNING_RATE = 10, 50, 1, 4, 1e-3
CONTEXT_LENGTH, PRED_LENGTH, PATCH_SIZE, MIN_SAMPLES_THRESHOLD = 128, 24, 32, 20

ALL_DATASETS = ["bitbrains_fast_storage/H", "bitbrains_rnd/H", "LOOP_SEATTLE/H", "M_DENSE/H", "SZ_TAXI/H", "electricity/H", "electricity/W", "hierarchical_sales/D", "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H", "solar/D", "saugeenday/D", "us_births/D", "bizitobs_service", "covid_deaths", "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant", "temperature_rain_with_missing"]

def load_leaked_names() -> set:
    if not LEAKED_NAMES_FILE.exists(): return set()
    names = set()
    with open(LEAKED_NAMES_FILE, 'r') as f:
        for line in f:
            name = line.strip().lower()
            if name: names.add(name)
    return names

LEAKED_NAMES = load_leaked_names()

def get_pretrain_datasets() -> List[str]:
    if not PRETRAIN_ROOT.exists(): return []
    datasets = []
    for d in PRETRAIN_ROOT.iterdir():
        if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('zzz'):
            if not d.name.endswith('.py') and not d.name.endswith('.txt') and not d.name.endswith('.md'):
                datasets.append(f"pretrain/{d.name}")
    return sorted(datasets)

def check_is_leaked(dataset_name: str) -> bool:
    if dataset_name.startswith("pretrain/"): base = dataset_name.split("/")[1].lower()
    else: base = dataset_name.split("/")[0].lower()
    if base in LEAKED_NAMES: return True
    for leaked_name in LEAKED_NAMES:
        if leaked_name in base or base in leaked_name: return True
    return False

@dataclass
class EpochMetrics:
    dataset: str; model_name: str; is_leaked: bool; epoch: int; loss: float; grad_norm: float; weight_shift: float

@dataclass
class ModelSummary:
    dataset: str; model_name: str; is_leaked: bool; num_samples: int; initial_loss: float; final_loss: float; loss_decrease_rate: float; aulc: float; first_batch_grad_norm: float; total_weight_shift: float

def load_dataset_data(dataset_name: str, max_samples: int = NUM_SAMPLES) -> List[np.ndarray]:
    import pyarrow as pa; import pyarrow.ipc as ipc
    if dataset_name.startswith("pretrain/"):
        actual_name = dataset_name.split("/", 1)[1]
        data_path = PRETRAIN_ROOT / actual_name
    else:
        data_path = GIFT_EVAL_ROOT / dataset_name
    if not data_path.exists(): return []
    arrow_files = list(data_path.glob("*.arrow"))
    if not arrow_files:
        for subdir in data_path.iterdir():
            if subdir.is_dir(): arrow_files.extend(subdir.glob("*.arrow"))
    if not arrow_files: return []
    series_list, min_len = [], CONTEXT_LENGTH + PRED_LENGTH
    for arrow_path in arrow_files:
        try:
            with pa.memory_map(str(arrow_path), 'r') as source:
                table = ipc.open_stream(source).read_all()
            if "target" not in table.column_names: continue
            for i in range(table.num_rows):
                if max_samples and len(series_list) >= max_samples: break
                try:
                    target_raw = table["target"][i].as_py()
                    if target_raw is None: continue
                    arr = np.asarray(target_raw, dtype=np.float32)
                    if arr.ndim != 1: arr = arr.flatten()
                    if len(arr) < min_len: continue
                    if not np.isfinite(arr).all():
                        mask = np.isnan(arr) | np.isinf(arr)
                        if mask.sum() < len(arr) * 0.5 and (~mask).sum() > 0: arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
                        else: continue
                    if np.std(arr) > 1e-6: series_list.append(arr)
                except Exception: continue
            if max_samples and len(series_list) >= max_samples: break
        except Exception: continue
    return series_list

class ForecastDataset(Dataset):
    """预测数据集（带 Z-score 标准化）"""
    def __init__(self, series_list: List[np.ndarray], context_len: int, pred_len: int):
        self.samples = []
        for idx, s in enumerate(series_list):
            if len(s) >= context_len + pred_len:
                context = s[:context_len].astype(np.float32)
                target = s[context_len:context_len + pred_len].astype(np.float32)
                mean, std = np.mean(context), np.std(context)
                if std < 1e-6: std = 1.0
                self.samples.append(((context - mean) / std, (target - mean) / std, mean, std, idx))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        context, target, mean, std, original_idx = self.samples[idx]
        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'mean': torch.tensor(mean, dtype=torch.float32),
            'std': torch.tensor(std, dtype=torch.float32),
            'original_idx': torch.tensor(original_idx, dtype=torch.long)
        }

class TiRexWrapper(nn.Module):
    def __init__(self, tirex_model, pred_len: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.tirex, self.pred_len = tirex_model, pred_len
        inferred_dim = getattr(getattr(self.tirex.blocks[0], "config", None), "embedding_dim", None) if getattr(self.tirex, "blocks", None) else None
        self.hidden_dim = hidden_dim or inferred_dim or 512
        self.pred_head = nn.Linear(self.hidden_dim, pred_len)
        self.patch_size = getattr(getattr(self.tirex, "config", None), "input_patch_size", 32)
        self.nan_mask_value = getattr(getattr(self.tirex, "config", None), "nan_mask_value", 0.0)
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B, L = context.shape
        try:
            pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
            if pad_len > 0: context = F.pad(context, (pad_len, 0), mode='replicate')
            tokens, _ = self.tirex.tokenizer.input_transform(context)
            token_mask = (~torch.isnan(tokens)).to(tokens.dtype)
            tokens = torch.nan_to_num(tokens, nan=self.nan_mask_value)
            hidden = self.tirex.input_patch_embedding(torch.cat((tokens, token_mask), dim=2))
            for blk in self.tirex.blocks: hidden = blk(hidden)
            return self.pred_head(self.tirex.out_norm(hidden)[:, -1, :])
        except Exception as e:
            print(f"TiRex forward error: {e}")
            return self.pred_head(F.adaptive_avg_pool1d(context.unsqueeze(1), self.hidden_dim).squeeze(1))

def compute_gradient_norm(model) -> float:
    return sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

def compute_weight_shift(initial_state: dict, current_state: dict) -> float:
    return sum(((current_state[k].cpu().float() - initial_state[k].cpu().float()) ** 2).sum().item() for k in initial_state if k in current_state) ** 0.5

def compute_aulc(loss_history: List[float]) -> float:
    if len(loss_history) < 2: return loss_history[0] if loss_history else 0.0
    return sum((loss_history[i-1] + loss_history[i]) / 2 for i in range(1, len(loss_history))) / (len(loss_history) - 1)

def evaluate_per_sample_loss(model, data_loader, device):
    """计算每个样本的 Loss"""
    model.eval()
    losses = {}
    with torch.no_grad():
        for batch in data_loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            indices = batch['original_idx'].cpu().numpy()
            try:
                pred = model(context)
                if torch.isnan(pred).any(): continue
                batch_losses = F.mse_loss(pred, target, reduction='none').mean(dim=1).cpu().numpy()
                for idx, loss in zip(indices, batch_losses):
                    if not (np.isnan(loss) or np.isinf(loss)): losses[idx] = float(loss)
            except Exception: continue
    return losses

def train_model(model, train_loader, device, initial_state, dataset_name, model_name, is_leaked, num_samples):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
    loss_history, epoch_metrics, first_batch_grad_norm, is_first_batch = [], [], 0.0, True

    # 1. 初始评估
    initial_losses_dict = evaluate_per_sample_loss(model, train_loader, device)
    init_loss = float(np.mean(list(initial_losses_dict.values()))) if initial_losses_dict else float('nan')
    loss_history.append(init_loss)
    epoch_metrics.append(EpochMetrics(dataset_name, model_name, is_leaked, 0, init_loss, 0.0, 0.0))

    # 2. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train(); epoch_grad_norms = []
        for batch in train_loader:
            context, target = batch['context'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            try:
                pred = model(context)
                if torch.isnan(pred).any(): continue
                loss = F.mse_loss(pred, target)
                if torch.isnan(loss) or torch.isinf(loss): continue
                loss.backward()
                if is_first_batch: first_batch_grad_norm, is_first_batch = compute_gradient_norm(model), False
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm): optimizer.zero_grad(); continue
                epoch_grad_norms.append(float(grad_norm)); optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e): torch.cuda.empty_cache()
            except Exception: continue

        # Epoch 评估
        current_losses_dict = evaluate_per_sample_loss(model, train_loader, device)
        epoch_loss = float(np.mean(list(current_losses_dict.values()))) if current_losses_dict else float('nan')
        loss_history.append(epoch_loss)
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        epoch_metrics.append(EpochMetrics(dataset_name, model_name, is_leaked, epoch + 1, epoch_loss, np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0, compute_weight_shift(initial_state, current_state)))

    final_loss = float(np.mean(list(evaluate_per_sample_loss(model, train_loader, device).values())))
    current_state = {k: v.clone() for k, v in model.state_dict().items()}
    return ModelSummary(dataset_name, model_name, is_leaked, num_samples, init_loss, final_loss, (init_loss - final_loss) / init_loss if init_loss > 0 else 0.0, compute_aulc(loss_history), first_batch_grad_norm, compute_weight_shift(initial_state, current_state)), epoch_metrics

def main():
    set_random_seed(42); logging.getLogger().setLevel(logging.WARNING)
    print("=" * 100 + f"\nTiRex 单模型微调实验\n配置: epochs={NUM_EPOCHS}, samples={NUM_SAMPLES}, windows={NUM_WINDOWS}\n配置: context={CONTEXT_LENGTH}, pred={PRED_LENGTH}\n筛选: 样本数 >= {MIN_SAMPLES_THRESHOLD}\n" + "=" * 100)
    device = select_best_gpu(); print(f"设备: {device}")
    from tirex import load_model
    print("加载 TiRex...")
    tirex_base = load_model(TIREX_MODEL, device=str(device))
    tirex_model = TiRexWrapper(tirex_base, PRED_LENGTH).to(device)
    tirex_initial_state = copy.deepcopy(tirex_model.state_dict())
    print("模型加载完成\n" + "=" * 100 + f"\n{'数据集':<30} {'泄露':<4} {'Init Loss':<12} {'Final Loss':<12} {'Decrease Rate':<12}\n" + "=" * 100)
    summaries, all_epochs = [], []
    # 合并 GiftEval 和 Pretrain 数据集
    pretrain_datasets = get_pretrain_datasets()
    all_datasets_combined = ALL_DATASETS + pretrain_datasets
    print(f"数据集总数: {len(all_datasets_combined)} (GiftEval: {len(ALL_DATASETS)}, Pretrain: {len(pretrain_datasets)})")
    for dataset_name in all_datasets_combined:
        series_list = load_dataset_data(dataset_name)
        if len(series_list) < MIN_SAMPLES_THRESHOLD: print(f"{dataset_name:<30} {'SKIP':<4} 序列不足 ({len(series_list)} < {MIN_SAMPLES_THRESHOLD})"); continue
        is_leaked = check_is_leaked(dataset_name)
        train_dataset = ForecastDataset(series_list, CONTEXT_LENGTH, PRED_LENGTH)
        if len(train_dataset) < MIN_SAMPLES_THRESHOLD: print(f"{dataset_name:<30} {'SKIP':<4} 有效样本不足"); continue
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        set_random_seed(42)
        tirex_model.load_state_dict(copy.deepcopy(tirex_initial_state))
        summary, epoch_metrics = train_model(tirex_model, train_loader, device, tirex_initial_state, dataset_name, "TiRex", is_leaked, len(train_dataset))
        summaries.append(summary); all_epochs.extend(epoch_metrics)
        print(f"{dataset_name:<30} {'Y' if is_leaked else 'N':<4} {summary.initial_loss:<12.4f} {summary.final_loss:<12.4f} {summary.loss_decrease_rate:<12.4f}")
    output_dir = resolve_output_dir("tirex_sft"); output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(s) for s in summaries]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame([vars(e) for e in all_epochs]).to_csv(output_dir / "epoch_metrics.csv", index=False)
    print(f"\n结果已保存到: {output_dir}\n" + "=" * 100 + "\n实验完成!")

if __name__ == "__main__":
    main()
