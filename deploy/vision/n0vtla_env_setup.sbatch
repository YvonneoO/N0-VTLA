#!/bin/bash
#SBATCH --job-name=n0vtla_env_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_env_setup_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source env.sh

ENV_DIR=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
echo "creating conda env at $ENV_DIR"
conda create -p "$ENV_DIR" python=3.11 -y

source activate "$ENV_DIR"
python -m pip install --upgrade pip

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA
echo "installing torch/torchvision/torchcodec (cu128, matching lab's environment)"
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec==0.5

echo "installing the rest of requirements.txt"
pip install -r requirements.txt

echo "installing n0vtla package itself"
pip install -e .

echo "=== verifying ==="
python -c "
import torch
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
import n0vtla
print('n0vtla import OK')
from n0vtla.training import config as _config
print('n0vtla.training.config import OK')
cfg = _config.get_config('vtla_tactile_posttrain')
print('vtla_tactile_posttrain config resolved OK, name=', cfg.name)
"
echo "=== ENV SETUP DONE ==="
