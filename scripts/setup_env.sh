#!/usr/bin/env bash
set -e

ENV_NAME=${ENV_NAME:-anonymous-tsfm}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

if command -v conda >/dev/null 2>&1; then
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
  echo "Activate with: conda activate $ENV_NAME"
else
  echo "conda was not found. Create a Python $PYTHON_VERSION environment manually."
fi

echo "Install base dependencies with: pip install -r requirements.txt"
echo "Install model-specific dependencies according to the GIFT-Eval or VisionTS environment references."
