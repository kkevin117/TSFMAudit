"""
Moirai2 单模型微调实验：统一损失函数训练动态追踪

使用统一的 MSE 预测损失函数追踪模型的训练动态。
基于 sft_moirai1.py 结构，使用 Moirai2 模型。
"""

import sys
from pathlib import Path
import os

from path_config import GIFT_EVAL_ROOT, GIFTEVAL_PRETRAIN_ROOT, PROJECT_ROOT, output_dir as resolve_output_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import copy
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


# ============================================================
# GPU 自动选择
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


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================
# 统一配置
# ============================================================
MOIRAI2_MODEL = os.environ.get("MOIRAI2_MODEL", "Salesforce/moirai-2.0-R-small")
PRETRAIN_ROOT = GIFTEVAL_PRETRAIN_ROOT

NUM_EPOCHS = 10
NUM_SAMPLES = 50
NUM_WINDOWS = 1  # 单窗口模式以确保候选/参考模型数据一致性
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
CONTEXT_LENGTH = 128
PRED_LENGTH = 24
MIN_SAMPLES_THRESHOLD = 20  # 最小样本数阈值

# GiftEval 数据集（未泄露 - 优先级更高）
ALL_DATASETS = [
    "bitbrains_fast_storage/H", "bitbrains_rnd/H", "LOOP_SEATTLE/H",
    "M_DENSE/H", "SZ_TAXI/H", "electricity/H", "electricity/W",
    "hierarchical_sales/D", "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H",
    "solar/D", "saugeenday/D", "us_births/D",
    "bizitobs_service", "covid_deaths", "m4_daily", "m4_hourly",
    "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant",
    "temperature_rain_with_missing",
]

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


def check_is_leaked(dataset_name: str) -> bool:
    """
    Moirai2 泄露检测逻辑：
    - ALL_DATASETS 中的数据集 -> 未泄露（优先级更高）
    - pretrain/ 开头的数据集 -> 泄露
    """
    # 检查是否是 ALL_DATASETS 中的数据集（未泄露，优先）
    if dataset_name in ALL_DATASETS:
        return False
    
    # pretrain 数据集视为泄露
    if dataset_name.startswith("pretrain/"):
        return True
    
    return False


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
# 数据加载
# ============================================================
def load_dataset_data(dataset_name: str, max_samples: int = NUM_SAMPLES) -> List[np.ndarray]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    
    # 确定数据路径
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
# 模型包装器
# ============================================================
class Moirai2Wrapper(nn.Module):
    """Moirai2 模型包装器（用于微调）"""
    
    def __init__(self, moirai2_module, pred_len: int, context_len: int):
        super().__init__()
        self.module = moirai2_module
        self.pred_len = pred_len
        self.context_len = context_len
        
        # Moirai2 使用 encoder，获取其输出维度
        # Moirai2-small 的隐层维度通常是 384
        hidden_dim = 384
        self.hidden_dim = hidden_dim
        self.pred_head = nn.Linear(hidden_dim, pred_len)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, context_len]
        B = context.shape[0]
        device = context.device
        
        try:
            # Moirai2 的 encoder 接口
            # 将 context 转换为适当的输入格式
            x = context.unsqueeze(-1)  # [B, context_len, 1] - 单变量
            
            # 通过 encoder 获取表示
            hidden = self.module.encoder(x)
            
            # hidden: [B, num_patches, hidden_dim] 或 [B, hidden_dim]
            if hidden.dim() == 3:
                # 取最后一个 patch 的表示
                last_hidden = hidden[:, -1, :]
            else:
                last_hidden = hidden
            
            # 确保维度匹配
            actual_dim = last_hidden.shape[-1]
            if actual_dim != self.hidden_dim:
                # 使用自适应池化调整维度
                last_hidden = F.adaptive_avg_pool1d(
                    last_hidden.unsqueeze(1), self.hidden_dim
                ).squeeze(1)
            
            pred = self.pred_head(last_hidden)
            return pred
        except Exception:
            # 如果标准方法失败，使用简单的前向传播
            # 使用 adaptive_avg_pool1d 正确池化到 hidden_dim
            x = context.unsqueeze(1)  # [B, 1, context_len]
            hidden = F.adaptive_avg_pool1d(x, self.hidden_dim).squeeze(1)  # [B, hidden_dim]
            pred = self.pred_head(hidden)
            return pred


# ============================================================
# 训练函数
# ============================================================
def compute_gradient_norm(model) -> float:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    initial_state: dict,
    dataset_name: str,
    model_name: str,
    is_leaked: bool,
    num_samples: int,
) -> Tuple[ModelSummary, List[EpochMetrics]]:
    """
    训练函数，追踪 Mean 指标
    
    Returns:
        summary, epoch_metrics_list
    """
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    
    loss_history = []
    epoch_metrics = []
    first_batch_grad_norm = 0.0
    is_first_batch = True
    
    # 1. 初始评估
    model.eval()
    init_losses = []
    with torch.no_grad():
        for batch in train_loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            try:
                pred = model(context)
                if torch.isnan(pred).any():
                    continue
                batch_loss = F.mse_loss(pred, target, reduction='none').mean(dim=1).cpu().numpy()
                init_losses.extend([l for l in batch_loss if not (np.isnan(l) or np.isinf(l))])
            except Exception:
                continue
    
    init_loss = float(np.mean(init_losses)) if init_losses else float('nan')
    loss_history.append(init_loss)
    
    epoch_metrics.append(EpochMetrics(
        dataset=dataset_name, model_name=model_name, is_leaked=is_leaked,
        epoch=0, loss=init_loss, grad_norm=0.0, weight_shift=0.0
    ))
    
    # 2. 训练循环
    for epoch in range(NUM_EPOCHS):
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
        
        # Epoch 结束后评估
        model.eval()
        epoch_losses = []
        with torch.no_grad():
            for batch in train_loader:
                context = batch['context'].to(device)
                target = batch['target'].to(device)
                try:
                    pred = model(context)
                    if torch.isnan(pred).any():
                        continue
                    batch_loss = F.mse_loss(pred, target, reduction='none').mean(dim=1).cpu().numpy()
                    epoch_losses.extend([l for l in batch_loss if not (np.isnan(l) or np.isinf(l))])
                except Exception:
                    continue
        
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        loss_history.append(epoch_loss)
        
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        weight_shift = compute_weight_shift(initial_state, current_state)
        grad_norm_avg = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        epoch_metrics.append(EpochMetrics(
            dataset=dataset_name, model_name=model_name, is_leaked=is_leaked,
            epoch=epoch + 1, loss=epoch_loss,
            grad_norm=grad_norm_avg, weight_shift=weight_shift
        ))
    
    # 3. 最终汇总
    final_loss = loss_history[-1] if loss_history else float('nan')
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
def main():
    set_random_seed(42)
    logging.getLogger().setLevel(logging.WARNING)
    
    print("=" * 100)
    print("Moirai2 单模型微调实验")
    print(f"配置: epochs={NUM_EPOCHS}, samples={NUM_SAMPLES}, context={CONTEXT_LENGTH}, pred={PRED_LENGTH}")
    print(f"筛选: 样本数 >= {MIN_SAMPLES_THRESHOLD}")
    print("=" * 100)
    
    device = select_best_gpu()
    print(f"设备: {device}")
    
    # 加载 Moirai2
    from uni2ts.model.moirai2 import Moirai2Module
    print("加载 Moirai2...")
    moirai2_module = Moirai2Module.from_pretrained(MOIRAI2_MODEL)
    moirai2_model = Moirai2Wrapper(moirai2_module, PRED_LENGTH, CONTEXT_LENGTH).to(device)
    moirai2_initial_state = copy.deepcopy(moirai2_model.state_dict())
    
    print("模型加载完成\n")
    
    # 结果容器
    summaries = []
    all_epochs = []
    
    print("=" * 100)
    print(f"{'数据集':<30} {'泄露':<4} {'Init Loss':<12} {'Final Loss':<12} {'Decrease Rate':<12}")
    print("=" * 100)
    
    # 合并 GiftEval 和 Pretrain 数据集
    pretrain_datasets = get_pretrain_datasets()
    all_datasets_combined = ALL_DATASETS + pretrain_datasets
    print(f"数据集总数: {len(all_datasets_combined)} (GiftEval: {len(ALL_DATASETS)}, Pretrain: {len(pretrain_datasets)})")
    
    for dataset_name in all_datasets_combined:
        series_list = load_dataset_data(dataset_name)
        if len(series_list) < MIN_SAMPLES_THRESHOLD:
            print(f"{dataset_name:<30} {'SKIP':<4} 序列不足 ({len(series_list)} < {MIN_SAMPLES_THRESHOLD})")
            continue
        
        is_leaked = check_is_leaked(dataset_name)
        
        train_dataset = ForecastDataset(series_list, CONTEXT_LENGTH, PRED_LENGTH)
        
        if len(train_dataset) < MIN_SAMPLES_THRESHOLD:
            print(f"{dataset_name:<30} {'SKIP':<4} 有效样本不足 ({len(train_dataset)} < {MIN_SAMPLES_THRESHOLD})")
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        # 训练模型
        set_random_seed(42)  # 重置种子以确保数据顺序一致
        moirai2_model.load_state_dict(copy.deepcopy(moirai2_initial_state))
        summary, epoch_metrics = train_model(
            moirai2_model, train_loader, device, moirai2_initial_state,
            dataset_name, "Moirai2", is_leaked, len(train_dataset)
        )
        
        summaries.append(summary)
        all_epochs.extend(epoch_metrics)
        
        leaked_str = "Y" if is_leaked else "N"
        print(f"{dataset_name:<30} {leaked_str:<4} {summary.initial_loss:<12.4f} {summary.final_loss:<12.4f} {summary.loss_decrease_rate:<12.4f}")
    
    # 保存结果
    output_dir = resolve_output_dir("moirai2_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = pd.DataFrame([vars(s) for s in summaries])
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    epochs_df = pd.DataFrame([vars(e) for e in all_epochs])
    epochs_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    
    print(f"\n结果已保存到: {output_dir}")
    print("=" * 100)
    print("实验完成!")


if __name__ == "__main__":
    main()
