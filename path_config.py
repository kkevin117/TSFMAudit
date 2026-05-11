from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def _consume_cli_path_args(option_to_env: Dict[str, str]) -> None:
    argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        matched = False
        for option, env_name in option_to_env.items():
            if arg == option:
                if i + 1 < len(sys.argv):
                    os.environ[env_name] = sys.argv[i + 1]
                    i += 2
                    matched = True
                break
            prefix = option + "="
            if arg.startswith(prefix):
                os.environ[env_name] = arg[len(prefix):]
                i += 1
                matched = True
                break
        if not matched:
            argv.append(arg)
            i += 1
    sys.argv[:] = argv


_consume_cli_path_args(
    {
        "--data_root": "DATA_ROOT",
        "--gifteval_root": "GIFTEVAL_ROOT",
        "--gifteval_pretrain_root": "GIFTEVAL_PRETRAIN_ROOT",
        "--output_dir": "OUTPUT_DIR",
        "--checkpoint_root": "CHECKPOINT_ROOT",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
GIFT_EVAL_ROOT = Path(os.environ.get("GIFTEVAL_ROOT", DATA_ROOT / "GIFT-Eval")).expanduser()
GIFTEVAL_PRETRAIN_ROOT = Path(
    os.environ.get("GIFTEVAL_PRETRAIN_ROOT", DATA_ROOT / "GIFT-Eval-Pretrain")
).expanduser()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser()
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", OUTPUT_ROOT)).expanduser()
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")).expanduser()
METADATA_ROOT = Path(os.environ.get("METADATA_ROOT", PROJECT_ROOT / "metadata")).expanduser()
EXTERNAL_ROOT = Path(os.environ.get("EXTERNAL_ROOT", PROJECT_ROOT / "external")).expanduser()


def output_dir(name: str) -> Path:
    return RESULTS_ROOT / name


def optional_repo_path(env_name: str, default_parts: Iterable[str], subdir: Optional[str] = None) -> Optional[Path]:
    raw = os.environ.get(env_name)
    root = Path(raw).expanduser() if raw else EXTERNAL_ROOT.joinpath(*default_parts)
    path = root / subdir if subdir else root
    return path if path.exists() else None
