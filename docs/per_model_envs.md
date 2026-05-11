# Per-model environments

Each of the seven TSFMs evaluated in this repository (Chronos, Kairos,
Moirai-1, Moirai-2, TimesFM 2.0, TiRex, VisionTS) ships with its own
upstream installation recipe. The commands below mirror those recipes;
they are *not* author contributions. Run each block in a fresh conda
environment so dependencies do not collide.

Common setup, used by all environments below:

```bash
export DATA_ROOT=/path/to/data
export GIFTEVAL_ROOT=$DATA_ROOT/GIFT-Eval
export GIFTEVAL_PRETRAIN_ROOT=$DATA_ROOT/GIFT-Eval-Pretrain
export GIFT_EVAL=$GIFTEVAL_ROOT
export OUTPUT_DIR=outputs
export CHECKPOINT_ROOT=/path/to/checkpoints
```

If you want the GIFT-Eval package itself inside an environment, clone it
*outside* the repository and install in editable mode:

```bash
mkdir -p external
git clone --depth 1 https://github.com/SalesforceAIResearch/gift-eval.git external/gift-eval
pip install -e "external/gift-eval[baseline]"
```

## Chronos / Chronos-Bolt

Defaults: `CHRONOS_MODEL=amazon/chronos-bolt-base`, wrapper imports
`from chronos import BaseChronosPipeline`.

```bash
conda create -n tsfm-chronos python=3.10 -y
conda activate tsfm-chronos

pip install -r requirements.txt
pip install chronos-forecasting

export CHRONOS_MODEL=amazon/chronos-bolt-base
bash scripts/run_chronos.sh
```

## Kairos

Defaults: `KAIROS_MODEL=mldi-lab/Kairos_50m`, optional `KAIROS_REPO`
points at a local Kairos source checkout.

```bash
conda create -n tsfm-kairos python=3.10 -y
conda activate tsfm-kairos

mkdir -p external
git clone --depth 1 https://github.com/foundation-model-research/Kairos.git external/Kairos
pip install -r external/Kairos/requirements.txt

pip install -r requirements.txt
pip install gluonts==0.14.4 python-dotenv datasets==3.5.1

export KAIROS_REPO=$PWD/external/Kairos
export KAIROS_MODEL=mldi-lab/Kairos_50m
bash scripts/run_kairos.sh
```

## Moirai-1

Defaults: `MOIRAI_MODEL=Salesforce/moirai-1.0-R-small`.

```bash
conda create -n tsfm-moirai1 python=3.10 -y
conda activate tsfm-moirai1

pip install -r requirements.txt
pip install uni2ts
pip install gluonts==0.15.1

export MOIRAI_MODEL=Salesforce/moirai-1.0-R-small
bash scripts/run_moirai1.sh
```

## Moirai-2

Defaults: `MOIRAI2_MODEL=Salesforce/moirai-2.0-R-small`.

```bash
conda create -n tsfm-moirai2 python=3.10 -y
conda activate tsfm-moirai2

pip install -r requirements.txt
pip install uni2ts
pip install gluonts==0.15.1

export MOIRAI2_MODEL=Salesforce/moirai-2.0-R-small
bash scripts/run_moirai2.sh
```

## TimesFM 2.0

Defaults: `TIMESFM_MODEL=google/timesfm-2.0-500m-pytorch`, wrapper uses
`timesfm.TimesFm`. This repository targets the PyTorch checkpoint.

```bash
conda create -n tsfm-timesfm python=3.11 -y
conda activate tsfm-timesfm

pip install -r requirements.txt
pip install "timesfm[torch]"

export TIMESFM_MODEL=google/timesfm-2.0-500m-pytorch
bash scripts/run_timesfm.sh
```

To use the Python 3.10 / PAX path instead, replace the install command
with `pip install "timesfm[pax]"` in a Python 3.10 env.

## TiRex

Defaults: `TIREX_MODEL=NX-AI/TiRex`, optional `TIREX_REPO` for a local
source checkout. TiRex requires an NVIDIA GPU with CUDA compute
capability >= 8.0.

```bash
conda create -c conda-forge \
  python=3.11 pip cuda=12.4 cuda-nvcc=12.4 \
  gxx_linux-64=11.4.0 compilers cmake ninja \
  cuda-toolkit=12.4 cuda-cccl=12.4 \
  --name tsfm-tirex -y
conda activate tsfm-tirex

pip install -r requirements.txt
pip install "tirex-ts[all]==1.3.0"

export TIREX_MODEL=NX-AI/TiRex
bash scripts/run_tirex.sh
```

If you use a local TiRex checkout instead of the PyPI package, set
`TIREX_REPO=/path/to/external/tirex`.

## VisionTS

Defaults: `VISIONTS_ARCH=mae_base`, checkpoint directory variable
`VISIONTS_CKPT_DIR`. VisionTS follows the VisionTS package's own
instructions rather than a GIFT-Eval notebook.

```bash
conda create -n tsfm-visionts python=3.8.18 -y
conda activate tsfm-visionts

pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install visionts
pip install numpy pandas scikit-learn pyarrow gluonts==0.15.1

export VISIONTS_ARCH=mae_base
export VISIONTS_CKPT_DIR=$CHECKPOINT_ROOT/visionts
bash scripts/run_visionts.sh
```

If your CUDA driver does not support the old PyTorch wheel, install a
locally supported PyTorch build first and then install `visionts`; minor
numerical differences may occur relative to the original VisionTS
evaluation environment.
