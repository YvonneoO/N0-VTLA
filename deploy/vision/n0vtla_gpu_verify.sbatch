#!/bin/bash
#SBATCH --job-name=n0vtla_gpu_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_gpu_verify_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

python -c "
import torch
print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)
print('cuda.is_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device='cuda')
y = torch.randn(4096, 4096, device='cuda')
z = (x @ y).sum().item()
print('matmul on GPU OK, sum=', z)

import n0vtla.models_pytorch.pi0_pytorch as _pp
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy
print('n0vtla model imports OK on GPU node')
"
echo "=== GPU VERIFY DONE ==="
