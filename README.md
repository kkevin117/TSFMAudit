# TSFMAudit

Official code release for TSFMAudit: Data Contamination Auditing in Forecasting Time Series Foundation Models.
This repository implements the TSFMAudit auditing framework for detecting data contamination in forecasting time-series foundation models. The audit protocol uses short fine-tuning traces, and reference models, and false-positive-constrained threshold selection to evaluate whether target datasets may have been exposed during model pretraining.
Datasets, pretrained checkpoints, logs, and generated tables are not included.

## 1. Environment

Create a base environment for the audit scripts and the lightweight
references (`sft_scratch_cnn.py`, `sft_scratch_transformer.py`,
`sft_stat.py`):

```bash
conda create -n anon-tsfm python=3.10 -y
conda activate anon-tsfm
pip install -r requirements.txt
```

Each of the seven TSFMs (six audit candidates: Chronos, Kairos, Moirai-1,
Moirai-2, TimesFM 2.0, TiRex; one reference: VisionTS) needs its own
model-specific environment that mirrors the upstream model's recipe. The
exact `pip install` lines are collected in
[`docs/per_model_envs.md`](docs/per_model_envs.md).

Hardware: scripts auto-select an idle CUDA device via `nvidia-smi`; CPU
fallback works but is slow. TiRex additionally requires CUDA compute
capability >= 8.0.

## 2. Data preparation

Download the two public datasets outside this repository:

```text
/path/to/data/
├── GIFT-Eval/
└── GIFT-Eval-Pretrain/
```

Point the scripts at them with environment variables (or with the matching
`--data_root`, `--gifteval_root`, `--gifteval_pretrain_root` CLI flags):

```bash
export DATA_ROOT=/path/to/data
export GIFTEVAL_ROOT=$DATA_ROOT/GIFT-Eval
export GIFTEVAL_PRETRAIN_ROOT=$DATA_ROOT/GIFT-Eval-Pretrain
export OUTPUT_DIR=outputs
export CHECKPOINT_ROOT=/path/to/checkpoints
```

Leakage-name metadata used by Chronos / TimesFM / TiRex matching is shipped
in `metadata/`:

```text
metadata/chronos_datasets_names.txt   # one short-name per line, used by Chronos and TiRex
metadata/timesfm_data.txt             # one short-name per line, used by TimesFM
```

A dataset is marked `is_leaked=True` if its short-name overlaps any line in
the corresponding model's file (case-insensitive substring match). Override
the file location with `CHRONOS_LEAKED_NAMES_FILE`,
`TIREX_LEAKED_NAMES_FILE`, or `TIMESFM_LEAKED_NAMES_FILE`. Kairos, Moirai-1
and Moirai-2 derive their leakage labels from their own pretrain manifests
rather than from a `metadata/` file (Moirai-1 reads `LOTSA_DATA_ROOT`).

## 3. Reproduce main results

After preparing the per-model environments (see
[`docs/per_model_envs.md`](docs/per_model_envs.md)) and the data, run the
six audit candidates, each in its own conda env:

```bash
bash scripts/run_chronos.sh
bash scripts/run_kairos.sh
bash scripts/run_moirai1.sh
bash scripts/run_moirai2.sh
bash scripts/run_timesfm.sh
bash scripts/run_tirex.sh
```

Then the four reference / debias models (the last three run in the base
env; VisionTS needs its own env):

```bash
bash scripts/run_visionts.sh
python sft_scratch_cnn.py
python sft_scratch_transformer.py
python sft_stat.py
```

Optional frequency-domain baseline (TiRex only):

```bash
bash scripts/run_ts_mink_fft.sh
```

Aggregate the two main tables:

```bash
bash scripts/run_main.sh            # -> outputs/sft_audit_eval_table.csv
bash scripts/run_all_baselines.sh   # -> outputs/baseline_audit_eval_table.csv
bash scripts/summarize_results.sh
```

Each per-model script writes `summary.csv` and `epoch_metrics.csv` to
`$OUTPUT_DIR/<model>_sft/`; the audit scripts read those CSVs.

## 4. Optional: controlled contamination sanity check

The `controlled` candidate row in `sft_audit_eval.py` is produced by a
controlled exposure / probe pipeline on top of the ScratchCNN reference:

```bash
python sft_scratch_cnn.py --mode exposure
python sft_scratch_cnn.py --mode probe
```

The probe step writes to `$OUTPUT_DIR/controlled_sft/`, which is then read
as the `controlled` candidate in `sft_audit_eval.py`. Skip this if you only
want to reproduce the main candidate tables.

## 5. Expected outputs

```text
outputs/
├── chronos_sft/
├── kairos_sft/
├── moirai1_sft/
├── moirai2_sft/
├── timesfm2p0_sft/
├── tirex_sft/
├── visionts_sft/
├── scratch_cnn_sft/
├── scratch_transformer_sft/
├── stat_sft/
├── controlled_sft/                   # only if §4 was run
├── ts_mink_fft/                      # only if scripts/run_ts_mink_fft.sh was run
├── sft_audit_eval_table.csv
└── baseline_audit_eval_table.csv
```

`results/`, `outputs/`, `logs/`, `runs/`, `data/`, and `checkpoints/` are
gitignored except for placeholder README files.

## 6. Repository layout

```text
.
├── path_config.py            # path / env / CLI plumbing for all scripts
├── sft_<model>.py            # per-model fine-tuning trace
├── sft_audit_eval.py         # main FP-constrained audit table
├── baseline_audit_eval.py    # baseline detectors under the same protocol
├── ts_mink_fft.py            # TS-Min-K% FFT baseline (TiRex)
├── scripts/                  # bash wrappers around the above
├── configs/*.yaml            # documentation-only templates (NOT auto-loaded)
├── docs/per_model_envs.md    # per-model conda / pip install instructions
├── metadata/                 # leakage-name lists (text files)
└── data/, checkpoints/,      # placeholder dirs; contents are gitignored
    results/, experiments/
```

## 7. Notes for reviewers

- All paths are controlled by environment variables or CLI flags; no
  hard-coded absolute paths exist.
- No external services, accounts, or API keys are required (no wandb, no
  HuggingFace token).
- `configs/*.yaml` are documentation templates only; the Python entry
  points read settings exclusively from environment variables and CLI
  flags.
- Random seeds are set to `42` in every per-model script.
- Citation will be added after the anonymous review period.
