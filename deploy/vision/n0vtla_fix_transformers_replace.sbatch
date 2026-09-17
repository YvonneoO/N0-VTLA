#!/bin/bash
#SBATCH --job-name=n0vtla_fix_tfreplace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_fix_tfreplace_%j.log

set -euo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

TRANSFORMERS_DIR="$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')"
echo "transformers installed at: $TRANSFORMERS_DIR"
cp -r n0vtla/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"

python -c "
import transformers
print('transformers', transformers.__version__)
from n0vtla.models_pytorch import pi0_pytorch
print('pi0_pytorch import OK -- transformers_replace check passes')
"
echo "=== TRANSFORMERS_REPLACE FIX DONE ==="
