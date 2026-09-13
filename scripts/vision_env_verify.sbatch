#!/bin/bash
#SBATCH --job-name=n0vtla-env-verify
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/vision_env_verify_%j.log
#
# The env's torch build (2.7.1+cu126, installed by an earlier login-node run) does not
# need to change: VISION's GPUs are H200 (Hopper, sm_90), which cu126 wheels already
# support -- there is no lab-style Blackwell (sm_120) "no kernel image" risk here. This
# job only VERIFIES that on an actual allocated GPU (a login node has none, so the earlier
# in-place check was meaningless) and opens group rw on the env + repo trees per the
# phai-lab request.
#
# Submit with: sbatch scripts/vision_env_verify.sbatch  (run from the repo root, i.e.
# after `cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA`)
set -euo pipefail

module load Miniforge3 CUDA/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'arch_list', torch.cuda.get_arch_list(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

chmod -R g+rwX /scratch/project/prj-02-phai-lab/yqq/N0-VTLA
chmod -R g+rwX /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

echo VISION_ENV_VERIFY_COMPLETE
