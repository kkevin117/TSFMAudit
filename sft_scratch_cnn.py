"""
Scratch CNN Reference 单模型训练 - 对齐单模型 SFT 脚本架构

使用简单 CNN 作为参考基线, 与其他单模型脚本输出格式一致。
"""

import argparse
import os
import sys
import random
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import copy

from path_config import CHECKPOINT_ROOT, GIFT_EVAL_ROOT, GIFTEVAL_PRETRAIN_ROOT, PROJECT_ROOT, output_dir as resolve_output_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRETRAIN_ROOT = GIFTEVAL_PRETRAIN_ROOT

# ============================================================
# 训练配置 (对齐单模型 SFT 脚本)
# ============================================================
CONTEXT_LENGTH = 128
PRED_LENGTH = 24
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_SAMPLES = 50
NUM_WINDOWS = 1  # 单窗口模式以确保候选/参考模型数据一致性
MIN_SAMPLES_THRESHOLD = 20
MODEL_DIM = 128

DEFAULT_CONTROLLED_EXPOSED_N = 8
DEFAULT_CONTROLLED_EXPOSURE_EPOCHS = 1
DEFAULT_CONTROLLED_PROBE_EPOCHS = NUM_EPOCHS
CONTROLLED_MODEL_NAME = "ScratchCNN"
CONTROLLED_EXPOSURE_DIR = CHECKPOINT_ROOT / "controlled_exposure"
CONTROLLED_CKPT_PATH = CONTROLLED_EXPOSURE_DIR / "ckpt.pt"
CONTROLLED_PROBE_DIR = resolve_output_dir("controlled_sft")

ALL_DATASETS = [
    "bitbrains_fast_storage/H", "bitbrains_rnd/H", "LOOP_SEATTLE/H",
    "M_DENSE/H", "SZ_TAXI/H", "electricity/H", "electricity/W",
    "hierarchical_sales/D", "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H",
    "solar/D", "saugeenday/D", "us_births/D",
    "bizitobs_service", "covid_deaths", "m4_daily", "m4_hourly",
    "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant",
    "temperature_rain_with_missing",
]

# Scratch 模型无泄露概念，但为了统一接口，标记所有为 False
def check_is_leaked(dataset_name: str) -> bool:
    return False  # 参考模型不区分泄露


# Pretrain 数据集（从可配置根目录动态加载）
def get_pretrain_datasets() -> List[str]:
    """获取 pretrain 数据集列表"""
    if not PRETRAIN_ROOT.exists():
        return []
    datasets = []
    for d in PRETRAIN_ROOT.iterdir():
        if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('zzz'):
            if not d.name.endswith('.py') and not d.name.endswith('.txt') and not d.name.endswith('.md'):
                datasets.append(f"pretrain/{d.name}")
    return sorted(datasets)


# ============================================================
# GPU 选择
# ============================================================
def get_gpu_memory_usage() -> List[Tuple[int, float]]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                gpu_info.append((gpu_id, mem_used / mem_total * 100))
        return gpu_info
    except Exception:
        return [(0, 0.0)]


def select_best_gpu() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_info = get_gpu_memory_usage()
    if not gpu_info:
        return torch.device("cuda:0")
    best_gpu = min(gpu_info, key=lambda x: x[1])
    gpu_id, usage = best_gpu
    print(f"选择 GPU {gpu_id} (显存占用: {usage:.1f}%)")
    return torch.device(f"cuda:{gpu_id}")


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================
# 数据加载 (对齐 SFT 脚本)
# ============================================================
def load_dataset_data(dataset_name: str, max_samples: int = NUM_SAMPLES) -> List[np.ndarray]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    
    if dataset_name.startswith("pretrain/"):
        actual_name = dataset_name.split("/", 1)[1]
        data_path = PRETRAIN_ROOT / actual_name
    else:
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
    """预测数据集（带 Z-score 标准化）"""
    
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
# CNN 模型
# ============================================================
class SimpleCNNForecaster(nn.Module):
    def __init__(self, context_length: int, pred_length: int, d_model: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.proj = nn.Linear(d_model * 2, pred_length)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        return self.proj(combined)


# ============================================================
# 数据类
# ============================================================
@dataclass
class EpochMetrics:
    dataset: str
    model_name: str
    is_leaked: bool
    epoch: int
    loss: float
    grad_norm: float
    weight_shift: float


@dataclass
class ModelSummary:
    dataset: str
    model_name: str
    is_leaked: bool
    num_samples: int
    initial_loss: float
    final_loss: float
    loss_decrease_rate: float
    aulc: float
    first_batch_grad_norm: float
    total_weight_shift: float


# ============================================================
# 训练函数
# ============================================================
def compute_gradient_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_weight_shift(initial_state: dict, current_state: dict) -> float:
    total_shift = 0.0
    for key in initial_state:
        if key in current_state:
            diff = current_state[key].cpu().float() - initial_state[key].cpu().float()
            total_shift += (diff ** 2).sum().item()
    return total_shift ** 0.5


def compute_aulc(loss_history: List[float]) -> float:
    if len(loss_history) < 2:
        return loss_history[0] if loss_history else 0.0
    n = len(loss_history)
    area = sum((loss_history[i-1] + loss_history[i]) / 2 for i in range(1, n)) / (n - 1)
    return area


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
                
                if torch.isnan(pred).any():
                    continue
                
                batch_losses = F.mse_loss(pred, target, reduction='none').mean(dim=1).cpu().numpy()
                
                for idx, loss in zip(indices, batch_losses):
                    if not (np.isnan(loss) or np.isinf(loss)):
                        losses[idx] = float(loss)
            except Exception:
                continue
    return losses


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    initial_state: dict,
    dataset_name: str,
    model_name: str,
    is_leaked: bool,
    num_samples: int,
    num_epochs: int = NUM_EPOCHS,
) -> Tuple[ModelSummary, List[EpochMetrics]]:
    """训练函数，追踪 Mean 指标"""
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    
    loss_history = []
    epoch_metrics = []
    first_batch_grad_norm = 0.0
    is_first_batch = True
    
    # 1. 初始评估（使用 evaluate_per_sample_loss）
    initial_losses_dict = evaluate_per_sample_loss(model, train_loader, device)
    
    init_loss = float(np.mean(list(initial_losses_dict.values()))) if initial_losses_dict else float('nan')
    loss_history.append(init_loss)
    
    epoch_metrics.append(EpochMetrics(
        dataset=dataset_name, model_name=model_name, is_leaked=is_leaked,
        epoch=0, loss=init_loss, grad_norm=0.0, weight_shift=0.0
    ))
    
    # 2. 训练循环
    for epoch in range(int(num_epochs)):
        model.train()
        epoch_grad_norms = []
        
        for batch in train_loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            optimizer.zero_grad()
            
            try:
                pred = model(context)
                if torch.isnan(pred).any():
                    continue
                
                loss = F.mse_loss(pred, target)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                if is_first_batch:
                    first_batch_grad_norm = compute_gradient_norm(model)
                    is_first_batch = False
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad()
                    continue
                
                epoch_grad_norms.append(float(grad_norm))
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue
            except Exception:
                continue
        
        # Epoch 结束后的评估
        current_losses_dict = evaluate_per_sample_loss(model, train_loader, device)
        
        epoch_loss = float(np.mean(list(current_losses_dict.values()))) if current_losses_dict else float('nan')
        loss_history.append(epoch_loss)
        
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        weight_shift = compute_weight_shift(initial_state, current_state)
        grad_norm_avg = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        epoch_metrics.append(EpochMetrics(
            dataset=dataset_name, model_name=model_name, is_leaked=is_leaked,
            epoch=epoch + 1, loss=epoch_loss,
            grad_norm=grad_norm_avg, weight_shift=weight_shift
        ))
    
    # 3. 最终评估
    final_losses_dict = evaluate_per_sample_loss(model, train_loader, device)
    
    final_loss = float(np.mean(list(final_losses_dict.values()))) if final_losses_dict else float('nan')
    current_state = {k: v.clone() for k, v in model.state_dict().items()}
    total_weight_shift = compute_weight_shift(initial_state, current_state)
    loss_decrease_rate = (init_loss - final_loss) / init_loss if init_loss > 0 else 0.0
    
    summary = ModelSummary(
        dataset=dataset_name, model_name=model_name, is_leaked=is_leaked,
        num_samples=num_samples, initial_loss=init_loss, final_loss=final_loss,
        loss_decrease_rate=loss_decrease_rate, aulc=compute_aulc(loss_history),
        first_batch_grad_norm=first_batch_grad_norm, total_weight_shift=total_weight_shift
    )
        
    return summary, epoch_metrics


# ============================================================
# 主函数
# ============================================================
def _build_train_loader(dataset_name: str, *, strict: bool) -> Tuple[DataLoader, int]:
    series_list = load_dataset_data(dataset_name)
    if len(series_list) < MIN_SAMPLES_THRESHOLD:
        msg = f"序列不足 ({len(series_list)} < {MIN_SAMPLES_THRESHOLD})"
        if strict:
            raise SystemExit(f"[ERROR] {dataset_name}: {msg}")
        raise ValueError(msg)

    train_dataset = ForecastDataset(series_list, CONTEXT_LENGTH, PRED_LENGTH)
    if len(train_dataset) < MIN_SAMPLES_THRESHOLD:
        msg = f"有效样本不足 ({len(train_dataset)} < {MIN_SAMPLES_THRESHOLD})"
        if strict:
            raise SystemExit(f"[ERROR] {dataset_name}: {msg}")
        raise ValueError(msg)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader, int(len(train_dataset))


def _save_controlled_ckpt(
    model: nn.Module,
    *,
    ckpt_path: Path,
    seed: int,
    exposed_datasets: List[str],
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": CONTROLLED_MODEL_NAME,
        "arch": "scratch_cnn",
        "seed": int(seed),
        "exposed_datasets": list(exposed_datasets),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, ckpt_path)


def _load_controlled_ckpt(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        raise SystemExit(f"[ERROR] missing checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"[ERROR] invalid checkpoint format: {ckpt_path}")
    if "exposed_datasets" not in payload or not isinstance(payload["exposed_datasets"], list):
        raise SystemExit(f"[ERROR] checkpoint missing exposed_datasets list: {ckpt_path}")
    return payload


def run_standard_sft(*, device: torch.device, seed: int):
    set_random_seed(seed)

    print("=" * 100)
    print("Scratch CNN Reference 单模型训练")
    print(f"配置: epochs={NUM_EPOCHS}, samples={NUM_SAMPLES}, windows={NUM_WINDOWS}")
    print(f"配置: context={CONTEXT_LENGTH}, pred={PRED_LENGTH}")
    print(f"筛选: 样本数 >= {MIN_SAMPLES_THRESHOLD}")
    print("=" * 100)

    print(f"设备: {device}")
    print("模型加载完成\n")

    summaries = []
    all_epochs = []

    print("=" * 100)
    print(f"{'数据集':<30} {'泄露':<4} {'Init Loss':<12} {'Final Loss':<12} {'Decrease Rate':<12}")
    print("=" * 100)

    pretrain_datasets = get_pretrain_datasets()
    all_datasets_combined = ALL_DATASETS + pretrain_datasets
    print(f"数据集总数: {len(all_datasets_combined)} (GiftEval: {len(ALL_DATASETS)}, Pretrain: {len(pretrain_datasets)})")

    model = SimpleCNNForecaster(CONTEXT_LENGTH, PRED_LENGTH, d_model=MODEL_DIM).to(device)
    initial_state = copy.deepcopy(model.state_dict())

    for dataset_name in all_datasets_combined:
        try:
            train_loader, num_samples = _build_train_loader(dataset_name, strict=False)
        except ValueError as e:
            print(f"{dataset_name:<30} {'SKIP':<4} {e}")
            continue

        is_leaked = check_is_leaked(dataset_name)

        set_random_seed(seed)
        model.load_state_dict(copy.deepcopy(initial_state))

        summary, epoch_metrics = train_model(
            model,
            train_loader,
            device,
            initial_state,
            dataset_name,
            CONTROLLED_MODEL_NAME,
            is_leaked,
            num_samples,
            num_epochs=NUM_EPOCHS,
        )

        summaries.append(summary)
        all_epochs.extend(epoch_metrics)

        leaked_str = "Y" if is_leaked else "N"
        print(
            f"{dataset_name:<30} {leaked_str:<4} {summary.initial_loss:<12.4f} {summary.final_loss:<12.4f} {summary.loss_decrease_rate:<12.4f}"
        )

    output_dir = resolve_output_dir("scratch_cnn_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([vars(s) for s in summaries])
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    epochs_df = pd.DataFrame([vars(e) for e in all_epochs])
    epochs_df.to_csv(output_dir / "epoch_metrics.csv", index=False)

    print(f"\n结果已保存到: {output_dir}")
    print("=" * 100)
    print("实验完成!")


def run_controlled_exposure(
    *,
    device: torch.device,
    seed: int,
    exposed_n: int,
    exposure_epochs: int,
    ckpt_path: Path,
):
    set_random_seed(seed)

    exposed_datasets = ALL_DATASETS[: int(exposed_n)]
    if len(exposed_datasets) < int(exposed_n):
        raise SystemExit(f"[ERROR] exposed_n too large: {exposed_n} > {len(ALL_DATASETS)}")

    print("=" * 100)
    print("Controlled Contamination - Stage A (Exposure)")
    print(f"模型: {CONTROLLED_MODEL_NAME} | seed={seed} | exposure_epochs={exposure_epochs} | exposed_n={exposed_n}")
    print(f"exposed_datasets = {exposed_datasets}")
    print("=" * 100)

    model = SimpleCNNForecaster(CONTEXT_LENGTH, PRED_LENGTH, d_model=MODEL_DIM).to(device)
    base_initial_state = copy.deepcopy(model.state_dict())

    for dataset_name in exposed_datasets:
        train_loader, num_samples = _build_train_loader(dataset_name, strict=True)
        # 不重置权重：连续训练形成 “exposed base checkpoint”
        _summary, _epoch_metrics = train_model(
            model,
            train_loader,
            device,
            base_initial_state,
            dataset_name,
            CONTROLLED_MODEL_NAME,
            True,
            num_samples,
            num_epochs=int(exposure_epochs),
        )

    _save_controlled_ckpt(model, ckpt_path=ckpt_path, seed=seed, exposed_datasets=exposed_datasets)
    print(f"[OK] saved exposed checkpoint: {ckpt_path}")


def run_controlled_probe(
    *,
    device: torch.device,
    seed: int,
    ckpt_path: Path,
    probe_epochs: int,
):
    payload = _load_controlled_ckpt(ckpt_path)
    exposed_datasets = payload["exposed_datasets"]
    exposed_set = set(exposed_datasets)

    print("=" * 100)
    print("Controlled Contamination - Stage B (Audit Probe)")
    print(f"模型: {CONTROLLED_MODEL_NAME} | seed={seed} | probe_epochs={probe_epochs}")
    print(f"ckpt: {ckpt_path}")
    print(f"exposed_datasets = {exposed_datasets}")
    print("=" * 100)

    model = SimpleCNNForecaster(CONTEXT_LENGTH, PRED_LENGTH, d_model=MODEL_DIM).to(device)
    model.load_state_dict(payload["state_dict"])
    initial_state = copy.deepcopy(model.state_dict())

    summaries: List[ModelSummary] = []
    all_epochs: List[EpochMetrics] = []

    print("=" * 100)
    print(f"{'数据集':<30} {'exposed':<7} {'Init Loss':<12} {'Final Loss':<12} {'Decrease Rate':<12}")
    print("=" * 100)

    for dataset_name in ALL_DATASETS:
        try:
            train_loader, num_samples = _build_train_loader(dataset_name, strict=False)
        except ValueError as e:
            print(f"{dataset_name:<30} {'SKIP':<7} {e}")
            continue

        is_exposed = dataset_name in exposed_set

        # 每个 dataset 都从 “exposed θ0” 重置开始，形成标准 probe 记录（10 epochs 对齐现有 SFT）
        set_random_seed(seed)
        model.load_state_dict(copy.deepcopy(initial_state))

        summary, epoch_metrics = train_model(
            model,
            train_loader,
            device,
            initial_state,
            dataset_name,
            CONTROLLED_MODEL_NAME,
            bool(is_exposed),
            num_samples,
            num_epochs=int(probe_epochs),
        )

        summaries.append(summary)
        all_epochs.extend(epoch_metrics)

        exposed_str = "Y" if is_exposed else "N"
        print(
            f"{dataset_name:<30} {exposed_str:<7} {summary.initial_loss:<12.4f} {summary.final_loss:<12.4f} {summary.loss_decrease_rate:<12.4f}"
        )

    CONTROLLED_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(s) for s in summaries]).to_csv(CONTROLLED_PROBE_DIR / "summary.csv", index=False)
    pd.DataFrame([vars(e) for e in all_epochs]).to_csv(CONTROLLED_PROBE_DIR / "epoch_metrics.csv", index=False)
    print(f"\n[OK] probe results saved to: {CONTROLLED_PROBE_DIR}")


def main():
    ap = argparse.ArgumentParser(description="ScratchCNN SFT + controlled contamination (exposure/probe)")
    ap.add_argument("--mode", type=str, default="standard", choices=["standard", "exposure", "probe"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exposed_n", type=int, default=DEFAULT_CONTROLLED_EXPOSED_N)
    ap.add_argument("--exposure_epochs", type=int, default=DEFAULT_CONTROLLED_EXPOSURE_EPOCHS)
    ap.add_argument("--probe_epochs", type=int, default=DEFAULT_CONTROLLED_PROBE_EPOCHS)
    ap.add_argument("--ckpt_path", type=str, default=str(CONTROLLED_CKPT_PATH))
    args = ap.parse_args()

    device = select_best_gpu()

    if args.mode == "standard":
        run_standard_sft(device=device, seed=int(args.seed))
        return

    ckpt_path = Path(args.ckpt_path)
    if args.mode == "exposure":
        run_controlled_exposure(
            device=device,
            seed=int(args.seed),
            exposed_n=int(args.exposed_n),
            exposure_epochs=int(args.exposure_epochs),
            ckpt_path=ckpt_path,
        )
        return

    if args.mode == "probe":
        run_controlled_probe(
            device=device,
            seed=int(args.seed),
            ckpt_path=ckpt_path,
            probe_epochs=int(args.probe_epochs),
        )
        return

    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
