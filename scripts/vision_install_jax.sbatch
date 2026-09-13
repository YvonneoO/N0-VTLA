#!/bin/bash
#SBATCH --job-name=n0vtla-install-jax
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/install_jax_%j.log
#
# Job 534590 (online Stage-1 training smoke test) failed immediately with
# ModuleNotFoundError: No module named 'jax' -- jax IS listed in requirements.txt
# (jax==0.5.3, jax-cuda12-pjrt/plugin==0.5.3, jaxlib==0.5.3) but isn't actually present
# in the n0vtla env, meaning the original `pip install -r requirements.txt` during env
# setup silently skipped or failed on it (never re-verified after that first setup).
# n0vtla.training.data_loader/config.py import jax at module level even for the
# PyTorch training path (used for tree_map-style batch collation), so this blocks
# ANY training entrypoint (offline or online), not just the new online one.
#
# Requested a GPU here (not CPU-only) specifically to verify the CUDA jax backend
# initializes on this node's actual H200, not just that the import succeeds on CPU.
#
# Submit with: sbatch scripts/vision_install_jax.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

echo "before: $(pip show jax 2>&1 | head -1)"
pip install -r requirements.txt
echo "after: $(pip show jax 2>&1 | head -1)"

python -c "
import jax
print('jax version:', jax.__version__)
print('jax devices:', jax.devices())
"
echo VISION_JAX_INSTALL_COMPLETE
