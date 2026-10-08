#!/bin/bash
#SBATCH --job-name=n0vtla-predtac-off
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_predtac_off_%x_%j.log
#
# Offline sensitivity check, no training: the REAL-tactile post-train checkpoint (Stage-1 init, 10000 steps, the touchscale-project run) is scored
# (held-out flow-matching action loss, same settings as scripts/compute_posttrain_held_out_loss.py in the scaling study) on the SAME validation
# episodes with three tactile inputs:
#   real  = canonical_wetlab_task1_val / canonical_wetlab_smoketestv2_dev   (glove)
#   pred  = canonical_wetlab_<task>_predtacq_val                            (predicted: threshold + per-task decoder + quantile map)
#   zero  = canonical_wetlab_<task>_zerotac0_val                             (all-zero tactile = idle glove)
# Expected if predicted tactile is informative: loss(real) < loss(pred) < loss(zero).
#   sbatch --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_predtac_offline_check_siyuan.sbatch      (TASK task1 | sv2)
set -uo pipefail
: "${TASK:?TASK=task1|sv2}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
source activate $YQQ/envs/n0vtla
PY=$YQQ/envs/n0vtla/bin/python
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME=$YQQ/N0-VTLA/models/huggingface
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VTLA_ATTN_IMPL=eager VTLA_PREFIX_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=$YQQ/data/n0vtla
REF=$DATA/eval_ckpts
OUT_DIR="${OUT_DIR:-$YQQ/data/predtac_offline_check}"
mkdir -p "$OUT_DIR"
case "$TASK" in
  task1) CKPT="$REF/task1_10k/n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/ours/10000"; ASSET=task1_tuberack_train
         REAL=$DATA/canonical_wetlab_task1_val; PRED=$DATA/canonical_wetlab_task1_predtacq_val; ZERO=$DATA/canonical_wetlab_task1_zerotac0_val;;
  sv2)   CKPT="$REF/sv2_100pct/n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000"; ASSET=wetlab_smoketestv2_train
         REAL=$DATA/canonical_wetlab_smoketestv2_dev; PRED=$DATA/canonical_wetlab_smoketestv2_predtacq_val; ZERO=$DATA/canonical_wetlab_smoketestv2_zerotac0_val;;
  *) echo "unknown TASK $TASK"; exit 2;;
esac
[ -f "$CKPT/model.safetensors" ] || { echo "FATAL: checkpoint missing $CKPT"; exit 2; }
$PY - "$REAL" "$PRED" "$ZERO" <<'PYEOF'
import json, sys
L = [[json.loads(l)["length"] for l in open(f"{d}/meta/episodes.jsonl")] for d in sys.argv[1:]]
print("episode lengths equal across real/pred/zero:", L[0] == L[1] == L[2], [len(x) for x in L])
if not (L[0] == L[1] == L[2]): sys.exit(3)
PYEOF
[ $? -eq 0 ] || { echo "FATAL: datasets do not hold the same episodes"; exit 3; }
for v in real pred zero; do
  case "$v" in real) ds=$REAL;; pred) ds=$PRED;; zero) ds=$ZERO;; esac
  label="${TASK}_realckpt_on_${v}"
  out="$OUT_DIR/${label}.json"
  [[ -f "$out" ]] && { echo "SKIP $label (exists)"; continue; }
  echo "=== $label ($CKPT on $ds) ==="
  $PY scripts/compute_posttrain_held_out_loss.py --checkpoint "$CKPT" --checkpoint-label "$label" --dataset-root "$ds" --asset-id "$ASSET" --output "$out" \
    || echo "FAILED $label"
done
$PY - "$OUT_DIR" "$TASK" <<'PYEOF'
import json, sys
from pathlib import Path
d, task = Path(sys.argv[1]), sys.argv[2]
for v in ("real", "pred", "zero"):
    p = d / f"{task}_realckpt_on_{v}.json"
    if p.exists():
        r = json.loads(p.read_text())
        g = {k: round(x["mean"], 4) for k, x in r["by_group"].items()}
        print(f"{task} tactile={v:<5} loss(macro)={r['mean_action_loss_macro']:.4f} se={r['episode_se']:.4f} rows={r['n_scored_rows']} groups={g}")
PYEOF
echo PREDTAC_OFFLINE_CHECK_DONE
