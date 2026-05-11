"""
Statistical Reference 单模型训练 - 对齐单模型 SFT 脚本架构

使用 ETS (Exponential Smoothing) 作为统计基线。
移除 min20/max20 分组，只保留 Mean 指标。

特点:
- 单次拟合 (无 Epoch 概念，只输出 epoch 0-10 相同值)
- grad_norm / weight_shift = 0 (无学习动态)
"""

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
import warnings

warnings.filterwarnings('ignore')

from path_config import GIFT_EVAL_ROOT, GIFTEVAL_PRETRAIN_ROOT, PROJECT_ROOT, output_dir as resolve_output_dir

# ============================================================
# 路径配置
# ============================================================
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRETRAIN_ROOT = GIFTEVAL_PRETRAIN_ROOT

# ============================================================
# 训练配置 (对齐 SFT 脚本)
# ============================================================
CONTEXT_LENGTH = 128
PRED_LENGTH = 24
NUM_SAMPLES = 50
NUM_WINDOWS = 1  # 单窗口模式以确保候选/参考模型数据一致性
NUM_EPOCHS = 10
MIN_SAMPLES_THRESHOLD = 20

ALL_DATASETS = [
    "bitbrains_fast_storage/H", "bitbrains_rnd/H", "LOOP_SEATTLE/H",
    "M_DENSE/H", "SZ_TAXI/H", "electricity/H", "electricity/W",
    "hierarchical_sales/D", "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H",
    "solar/D", "saugeenday/D", "us_births/D",
    "bizitobs_service", "covid_deaths", "m4_daily", "m4_hourly",
    "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant",
    "temperature_rain_with_missing",
]


def get_pretrain_datasets() -> List[str]:
    """Get pretrain datasets under the configured root as pretrain/<name>."""
    if not PRETRAIN_ROOT.exists():
        return []
    datasets: List[str] = []
    for d in PRETRAIN_ROOT.iterdir():
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("zzz"):
            if not d.name.endswith(".py") and not d.name.endswith(".txt") and not d.name.endswith(".md"):
                datasets.append(f"pretrain/{d.name}")
    return sorted(datasets)


def check_is_leaked(dataset_name: str) -> bool:
    return False  # 参考模型不区分泄露


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


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


# ============================================================
# ETS 模型预测
# ============================================================
def ets_predict(context: np.ndarray, pred_len: int) -> np.ndarray:
    """使用 Exponential Smoothing 进行预测"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(
            context,
            trend='add',
            seasonal=None,
            damped_trend=True,
            initialization_method='estimated'
        )
        fit = model.fit(optimized=True, use_brute=False)
        forecast = fit.forecast(pred_len)
        
        if np.isnan(forecast).any():
            raise ValueError("NaN in forecast")
        
        return forecast
    except Exception:
        # Fallback: Naive method (repeat last value)
        return np.full(pred_len, context[-1])


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
# 评估函数
# ============================================================
def evaluate_dataset(series_list: List[np.ndarray], dataset_name: str, is_leaked: bool):
    """对一个数据集的所有序列进行 ETS 预测并计算 Loss"""
    losses = []
    
    for s in series_list:
        if len(s) < CONTEXT_LENGTH + PRED_LENGTH:
            continue
        
        context = s[:CONTEXT_LENGTH].astype(np.float32)
        target = s[CONTEXT_LENGTH:CONTEXT_LENGTH + PRED_LENGTH].astype(np.float32)
        
        # Z-score 标准化
        mean = np.mean(context)
        std = np.std(context)
        if std < 1e-6:
            std = 1.0
        
        context_norm = (context - mean) / std
        target_norm = (target - mean) / std
        
        # ETS 预测
        pred = ets_predict(context_norm, PRED_LENGTH)
        
        # 计算 MSE Loss
        loss = float(np.mean((pred - target_norm) ** 2))
        if np.isfinite(loss):
            losses.append(loss)
    
    if not losses:
        return None, []
    
    mean_loss = np.mean(losses)
    
    # 生成 epoch metrics (统计模型无训练，所有 epoch 值相同)
    epoch_metrics = []
    for epoch in range(NUM_EPOCHS + 1):
        epoch_metrics.append(EpochMetrics(
            dataset=dataset_name,
            model_name="Stat",
            is_leaked=is_leaked,
            epoch=epoch,
            loss=mean_loss,
            grad_norm=0.0,
            weight_shift=0.0
        ))
    
    # 生成 summary
    summary = ModelSummary(
        dataset=dataset_name,
        model_name="Stat",
        is_leaked=is_leaked,
        num_samples=len(losses),
        initial_loss=mean_loss,
        final_loss=mean_loss,
        loss_decrease_rate=0.0,
        aulc=mean_loss,
        first_batch_grad_norm=0.0,
        total_weight_shift=0.0
    )
    
    return summary, epoch_metrics


# ============================================================
# 主函数
# ============================================================
def main():
    set_random_seed(42)
    
    print("=" * 100)
    print("Statistical Reference 单模型训练 (ETS)")
    print(f"配置: samples={NUM_SAMPLES}, windows={NUM_WINDOWS}")
    print(f"配置: context={CONTEXT_LENGTH}, pred={PRED_LENGTH}")
    print(f"筛选: 样本数 >= {MIN_SAMPLES_THRESHOLD}")
    print("=" * 100)
    
    # 结果容器
    summaries = []
    all_epochs = []
    
    print("=" * 100)
    print(f"{'数据集':<30} {'泄露':<4} {'Init Loss':<12} {'Final Loss':<12} {'Decrease Rate':<12}")
    print("=" * 100)
    
    pretrain_datasets = get_pretrain_datasets()
    all_datasets_combined = ALL_DATASETS + pretrain_datasets
    print(f"Datasets: {len(all_datasets_combined)} (GiftEval: {len(ALL_DATASETS)}, Pretrain: {len(pretrain_datasets)})")
    for dataset_name in all_datasets_combined:
        series_list = load_dataset_data(dataset_name)
        if len(series_list) < MIN_SAMPLES_THRESHOLD:
            print(f"{dataset_name:<30} {'SKIP':<4} 序列不足 ({len(series_list)} < {MIN_SAMPLES_THRESHOLD})")
            continue
        
        is_leaked = check_is_leaked(dataset_name)
        
        summary, epoch_metrics = evaluate_dataset(series_list, dataset_name, is_leaked)
        
        if summary is None:
            print(f"{dataset_name:<30} {'SKIP':<4} 评估失败")
            continue
        
        summaries.append(summary)
        all_epochs.extend(epoch_metrics)
        
        leaked_str = "Y" if is_leaked else "N"
        print(f"{dataset_name:<30} {leaked_str:<4} {summary.initial_loss:<12.4f} {summary.final_loss:<12.4f} {summary.loss_decrease_rate:<12.4f}")
    
    # 保存结果
    output_dir = resolve_output_dir("stat_sft")
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
