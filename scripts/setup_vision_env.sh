#!/usr/bin/env bash
# Build the N0-VTLA conda environment on the VISION cluster, entirely inside
# /scratch/project/prj-02-phai-lab/yqq (never /home, never /tmp -- see @TAMU/CLAUDE.md).
#
# Run from the repo root (yqq/N0-VTLA) AFTER `source /scratch/project/prj-02-phai-lab/yqq/env.sh`
# (sets HF/GH/WANDB tokens under the qqyang identity and points caches at yqq):
#   cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA && bash scripts/setup_vision_env.sh
set -euo pipefail

YQQ=/scratch/project/prj-02-phai-lab/yqq
ENV_PREFIX="$YQQ/envs/n0vtla"

module load Miniforge3 CUDA/12.8.0

if [[ ! -d "$ENV_PREFIX" ]]; then
  conda create -p "$ENV_PREFIX" python=3.11 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

pip install -r requirements.txt
pip install -e . --no-deps

TRANSFORMERS_DIR="$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')"
cp -r n0vtla/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
echo "VISION_ENV_SETUP_COMPLETE: $ENV_PREFIX"
