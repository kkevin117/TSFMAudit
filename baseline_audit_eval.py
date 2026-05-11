#!/usr/bin/env python3
"""
Run all baseline detectors under the same FP-k audit protocol (one-click).

Baselines (from baseline_detectors.py + ts_mink_fft.py):
  - raw_loss:        candidate initial_loss
  - loss_drop_rate:  candidate loss_decrease_rate
  - lira_ratio:      initial_loss_cand / initial_loss_ref (per reference model)
  - ts_mink_fft_*:   optional, from results/ts_mink_fft/<candidate>/detailed_scores.csv

The protocol is aligned with sft_audit_eval.py:
  - dataset-level repeated splits (optionally group-split)
  - threshold selected on calibration Clean with FP(clean) <= k (S0/S1)
  - metrics reported on held-out test sets
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sft_audit_eval as audit


def _parse_csv_list(raw: Optional[str], default: List[str]) -> List[str]:
    if raw is None:
        return list(default)
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or list(default)


def _load_candidate_baselines(
    results_root: Path, cfg: audit.ModelConfig, min_samples: int
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    path = results_root / cfg.base_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"] == cfg.model_name]
    if "num_samples" in df.columns:
        df = df[df["num_samples"] >= min_samples]

    needed = {"dataset", "is_leaked", "initial_loss", "loss_decrease_rate"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing columns {missing}: {path}")

    df = df[["dataset", "is_leaked", "initial_loss", "loss_decrease_rate"]].drop_duplicates(subset=["dataset"])
    datasets = df["dataset"].astype(str).tolist()
    y = df["is_leaked"].astype(int).to_numpy()
    feats = {
        "raw_loss": df["initial_loss"].astype(float).to_numpy(),
        "loss_drop_rate": df["loss_decrease_rate"].astype(float).to_numpy(),
    }
    return datasets, y, feats


def _load_ref_initial_loss(results_root: Path, cfg: audit.ModelConfig, min_samples: int) -> Dict[str, float]:
    path = results_root / cfg.base_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"] == cfg.model_name]
    if "num_samples" in df.columns:
        df = df[df["num_samples"] >= min_samples]

    if "dataset" not in df.columns or "initial_loss" not in df.columns:
        raise ValueError(f"summary.csv must contain dataset,initial_loss columns: {path}")
    df = df[["dataset", "initial_loss"]].drop_duplicates(subset=["dataset"])
    return dict(zip(df["dataset"].astype(str), df["initial_loss"].astype(float)))


def _load_ts_mink_fft_file(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for col in ("mink_k10", "mink_k20", "mink_k30"):
        if col in df.columns:
            out[f"ts_mink_fft_{col}"] = dict(zip(df["dataset"].astype(str), df[col].astype(float)))
    return out


def _load_ts_mink_fft(results_root: Path, cfg: audit.ModelConfig) -> Dict[str, Dict[str, float]]:
    root = results_root / "ts_mink_fft"
    candidates = [
        root / cfg.key / "detailed_scores.csv",
        root / cfg.base_dir / "detailed_scores.csv",
        root / cfg.model_name / "detailed_scores.csv",
        results_root / f"ts_mink_fft_{cfg.key}" / "detailed_scores.csv",
    ]
    for path in candidates:
        out = _load_ts_mink_fft_file(path)
        if out:
            return out

    path = root / "detailed_scores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        return {}

    for col, values in (
        ("candidate_key", {cfg.key}),
        ("candidate", {cfg.model_name, cfg.key}),
        ("model_key", {cfg.key}),
        ("model_name", {cfg.model_name, cfg.key}),
        ("model", {cfg.model_name, cfg.key}),
    ):
        if col in df.columns:
            sub = df[df[col].astype(str).isin(values)]
            if sub.empty:
                return {}
            out: Dict[str, Dict[str, float]] = {}
            for score_col in ("mink_k10", "mink_k20", "mink_k30"):
                if score_col in sub.columns:
                    out[f"ts_mink_fft_{score_col}"] = dict(zip(sub["dataset"].astype(str), sub[score_col].astype(float)))
            return out

    if cfg.key == "tirex":
        return _load_ts_mink_fft_file(path)
    return {}


def _evaluate_one_score(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    fp_k: Optional[int],
    select_metric: str,
) -> Tuple[dict, dict]:
    split_metrics: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "macro_f1": [],
        "mcc": [],
        "balanced_acc": [],
    }
    taus: List[float] = []

    pooled_y: List[int] = []
    pooled_scores: List[float] = []
    pooled_pred: List[int] = []
    pooled_clean: int = 0
    pooled_fp: int = 0

    for calib_idx, test_idx in splits:
        y_calib_all = y[calib_idx].astype(int)
        s_calib_all = scores[calib_idx].astype(float)
        ok_calib = np.isfinite(s_calib_all)
        if ok_calib.sum() < 4:
            continue

        y_c = y_calib_all[ok_calib]
        s_c = s_calib_all[ok_calib]
        if len(np.unique(y_c)) < 2:
            continue

        # Orient score so "higher => leaked" based on calibration AUROC.
        sign = 1.0
        auroc_raw = audit._safe_auroc(y_c, s_c)
        if np.isfinite(auroc_raw) and auroc_raw < 0.5:
            sign = -1.0
            s_c = -s_c

        tau = audit._select_tau_calibration(
            y_calib=y_c,
            scores_calib=s_c,
            fp_k=fp_k,
            select_metric=select_metric,
        )
        if not np.isfinite(tau):
            continue

        y_test_all = y[test_idx].astype(int)
        s_test_all = (scores[test_idx].astype(float) * sign).astype(float)
        ok_test = np.isfinite(s_test_all)
        if ok_test.sum() < 2:
            continue
        y_t = y_test_all[ok_test]
        s_t = s_test_all[ok_test]
        pred_t = (s_t > tau).astype(int)

        split_metrics["auroc"].append(audit._safe_auroc(y_t, s_t))
        split_metrics["auprc"].append(audit._safe_auprc(y_t, s_t))
        split_metrics["macro_f1"].append(audit._safe_macro_f1(y_t, pred_t))
        split_metrics["mcc"].append(audit._safe_mcc(y_t, pred_t))
        split_metrics["balanced_acc"].append(audit._safe_balanced_acc(y_t, pred_t))

        taus.append(float(tau))

        pooled_y.extend(int(v) for v in y_t.tolist())
        pooled_scores.extend(float(v) for v in s_t.tolist())
        pooled_pred.extend(int(v) for v in pred_t.tolist())
        pooled_clean += int(np.sum(y_t == 0))
        pooled_fp += int(np.sum((y_t == 0) & (pred_t == 1)))

    tau_med = float(np.nanmedian(np.asarray(taus, dtype=float))) if taus else float("nan")
    summary = {
        "metrics": split_metrics,
        "n_splits_used": int(len(taus)),
        "tau_median": tau_med,
        "fp_test": int(pooled_fp),
        "clean_test": int(pooled_clean),
    }
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
    ap.add_argument("--results_root", type=str, default=str(audit.RESULTS_ROOT))
    ap.add_argument("--min_samples", type=int, default=30)
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
    ap.add_argument("--group_split", type=str, default="family", choices=["none", "family", "map"])
    ap.add_argument("--group_map_csv", type=str, default=None)

    ap.add_argument("--baselines", type=str, default=None, help="raw_loss,loss_drop_rate,lira_ratio,ts_mink_fft")
    ap.add_argument("--lira_refs", type=str, default=None, help="reference keys for lira_ratio (default: all refs)")
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    cand_keys = audit._parse_candidates(args.candidates)

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

    group_map = None
    if args.group_split == "map":
        if not args.group_map_csv:
            raise SystemExit("--group_split=map requires --group_map_csv")
        group_map = audit._load_group_map_csv(Path(args.group_map_csv))

    baselines = _parse_csv_list(args.baselines, ["raw_loss", "loss_drop_rate", "lira_ratio", "ts_mink_fft"])
    lira_refs = _parse_csv_list(args.lira_refs, list(audit.REFERENCES.keys()))
    for rk in lira_refs:
        if rk not in audit.REFERENCES:
            avail = ", ".join(sorted(audit.REFERENCES.keys()))
            raise SystemExit(f"Unknown lira ref: {rk}. Available: {avail}")

    ref_maps: Dict[str, Dict[str, float]] = {}
    if "lira_ratio" in baselines:
        for rk in lira_refs:
            try:
                ref_maps[rk] = _load_ref_initial_loss(results_root, audit.REFERENCES[rk], args.min_samples)
            except FileNotFoundError as e:
                print(f"[REF SKIP] {audit.REFERENCES[rk].model_name}: {e}")

    rows: List[dict] = []
    pooled_by_key: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for cand_key in cand_keys:
        cand_cfg = audit.CANDIDATES[cand_key]
        try:
            datasets, y, feats = _load_candidate_baselines(results_root, cand_cfg, args.min_samples)
        except FileNotFoundError as e:
            print(f"[SKIP] {cand_cfg.model_name}: {e}")
            continue
        except Exception as e:
            print(f"[SKIP] {cand_cfg.model_name}: {e}")
            continue

        if len(datasets) < 4 or len(np.unique(y)) < 2:
            n_clean = int(np.sum(y == 0))
            n_leaked = int(np.sum(y == 1))
            print(f"[SKIP] {cand_cfg.model_name}: not enough labeled datasets (n={len(datasets)}, clean={n_clean}, leaked={n_leaked})")
            continue

        units, unit_labels = audit._build_units(datasets, y, args.group_split, group_map)
        n_units = int(len(units))
        n_clean_units = int(np.sum(unit_labels == 0))
        n_leaked_units = int(np.sum(unit_labels == 1))

        splits = audit._make_repeated_splits(
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
                f"(datasets={len(datasets)}, clean={int(np.sum(y==0))}, leaked={int(np.sum(y==1))}, "
                f"units={n_units}, clean_units={n_clean_units}, leaked_units={n_leaked_units})"
            )
            continue

        # ---------------- Baselines ----------------
        def add_row(baseline: str, reference_key: str, reference: str, scores: np.ndarray) -> None:
            summary, pooled = _evaluate_one_score(
                y=y,
                scores=scores,
                splits=splits,
                fp_k=fp_k,
                select_metric=args.select_metric,
            )
            tau = float(summary["tau_median"])
            tau_str = "nan" if not np.isfinite(tau) else f"{tau:.4g}"
            condition = (
                f"{baseline} (k={k_label}, tau~{tau_str}, FPtest={summary['fp_test']}/{summary['clean_test']})"
                if summary["n_splits_used"] > 0
                else "N/A"
            )
            rows.append(
                {
                    "candidate_key": cand_key,
                    "candidate": cand_cfg.model_name,
                    "baseline": baseline,
                    "reference_key": reference_key,
                    "reference": reference,
                    "condition": condition,
                    "fp_mode": args.fp_mode,
                    "k": float(fp_k) if fp_k is not None else float("nan"),
                    "tau_median": tau,
                    "fp_test": int(summary["fp_test"]),
                    "clean_test": int(summary["clean_test"]),
                    "n_splits_used": int(summary["n_splits_used"]),
                    "n_datasets": int(len(datasets)),
                    "n_clean": int(np.sum(y == 0)),
                    "n_leaked": int(np.sum(y == 1)),
                    "n_units": n_units,
                    "n_clean_units": n_clean_units,
                    "n_leaked_units": n_leaked_units,
                    "AUROC": audit._fmt_mean_std(summary["metrics"]["auroc"]),
                    "AUPRC": audit._fmt_mean_std(summary["metrics"]["auprc"]),
                    "Macro-F1": audit._fmt_mean_std(summary["metrics"]["macro_f1"]),
                    "MCC": audit._fmt_mean_std(summary["metrics"]["mcc"]),
                    "BalancedAcc": audit._fmt_mean_std(summary["metrics"]["balanced_acc"]),
                }
            )
            pooled_by_key[(baseline, reference)].append(pooled)

        if "raw_loss" in baselines:
            add_row("raw_loss", "none", "none", feats["raw_loss"])
        if "loss_drop_rate" in baselines:
            add_row("loss_drop_rate", "none", "none", feats["loss_drop_rate"])
        if "lira_ratio" in baselines:
            eps = 1e-8
            raw = feats["raw_loss"].astype(float)
            for rk in lira_refs:
                rm = ref_maps.get(rk)
                if not rm:
                    continue
                ref_loss = np.asarray([rm.get(ds, np.nan) for ds in datasets], dtype=float)
                add_row("lira_ratio", rk, audit.REFERENCES[rk].model_name, raw / (ref_loss + eps))
        if "ts_mink_fft" in baselines:
            mink_maps = _load_ts_mink_fft(results_root, cand_cfg)
            for score_name, ds2score in mink_maps.items():
                scores = np.asarray([ds2score.get(ds, np.nan) for ds in datasets], dtype=float)
                add_row(score_name, "none", "none", scores)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["baseline", "reference", "candidate"]).reset_index(drop=True)

    # Add MACRO/MICRO summaries per (baseline, reference)
    macro_rows: List[dict] = []
    micro_rows: List[dict] = []
    for (baseline, ref_name), pooled_list in pooled_by_key.items():
        fp_total = int(sum(int(p.get("fp_test", 0)) for p in pooled_list))
        clean_total = int(sum(int(p.get("clean_test", 0)) for p in pooled_list))

        per_auroc: List[float] = []
        per_auprc: List[float] = []
        per_mf1: List[float] = []
        per_mcc: List[float] = []
        per_bacc: List[float] = []

        y_all_list = []
        s_all_list = []
        p_all_list = []

        for p in pooled_list:
            y_arr = p.get("y", np.array([], dtype=int))
            s_arr = p.get("scores", np.array([], dtype=float))
            pred_arr = p.get("pred", np.array([], dtype=int))
            if len(y_arr) < 2:
                continue
            per_auroc.append(audit._safe_auroc(y_arr, s_arr))
            per_auprc.append(audit._safe_auprc(y_arr, s_arr))
            per_mf1.append(audit._safe_macro_f1(y_arr, pred_arr))
            per_mcc.append(audit._safe_mcc(y_arr, pred_arr))
            per_bacc.append(audit._safe_balanced_acc(y_arr, pred_arr))
            y_all_list.append(y_arr)
            s_all_list.append(s_arr)
            p_all_list.append(pred_arr)

        if per_mcc:
            macro_rows.append(
                {
                    "candidate": "MACRO",
                    "baseline": baseline,
                    "reference": ref_name,
                    "condition": f"avg over models (FPtest={fp_total}/{clean_total})",
                    "AUROC": audit._fmt_mean_std(per_auroc),
                    "AUPRC": audit._fmt_mean_std(per_auprc),
                    "Macro-F1": audit._fmt_mean_std(per_mf1),
                    "MCC": audit._fmt_mean_std(per_mcc),
                    "BalancedAcc": audit._fmt_mean_std(per_bacc),
                }
            )

        if y_all_list:
            y_all = np.concatenate(y_all_list, axis=0)
            s_all = np.concatenate(s_all_list, axis=0)
            pred_all = np.concatenate(p_all_list, axis=0)
            micro_rows.append(
                {
                    "candidate": "MICRO",
                    "baseline": baseline,
                    "reference": ref_name,
                    "condition": f"pooled (FPtest={fp_total}/{clean_total})",
                    "AUROC": f"{audit._safe_auroc(y_all, s_all):.3f}",
                    "AUPRC": f"{audit._safe_auprc(y_all, s_all):.3f}",
                    "Macro-F1": f"{audit._safe_macro_f1(y_all, pred_all):.3f}",
                    "MCC": f"{audit._safe_mcc(y_all, pred_all):.3f}",
                    "BalancedAcc": f"{audit._safe_balanced_acc(y_all, pred_all):.3f}",
                }
            )

    if macro_rows:
        df = pd.concat([df, pd.DataFrame(macro_rows)], ignore_index=True)
    if micro_rows:
        df = pd.concat([df, pd.DataFrame(micro_rows)], ignore_index=True)

    pd.set_option("display.max_colwidth", 120)
    print_cols = ["candidate", "baseline", "reference", "condition", "AUROC", "AUPRC", "Macro-F1", "MCC", "BalancedAcc"]
    print(df[print_cols].to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
    else:
        default_name = "baseline_audit_eval_table.csv" if args.fp_mode == "fp_k" else "baseline_audit_eval_table_unconstrained.csv"
        out = results_root / default_name
    out_full = out.with_name(out.stem + "_full" + out.suffix)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        df[print_cols].to_csv(out, index=False, encoding="utf-8")
        df.to_csv(out_full, index=False, encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
