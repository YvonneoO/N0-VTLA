#!/bin/bash
#SBATCH --job-name=n0vtla-t4-srvcli
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/t4_srvcli_%j.log
#
# Integration test of the real serving path for a Task4 dual-arm checkpoint: starts scripts/serve_policy.py
# (websocket server, exactly the documented command, NO VTLA_ASSET_ID set so the checkpoint-asset fallback is
# exercised) and replays one released training package through scripts/dual_arm_client_example.py.
# Required: CKPT. Optional: UUID (default: first episode dir), QUERIES(=6).
#   sbatch --export=ALL,CKPT=<ckpt dir> deploy/vision/n0vtla_check_dual_server_client_siyuan.sbatch
set -uo pipefail
: "${CKPT:?CKPT is required}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"
PY=$YQQ/envs/n0vtla/bin/python
cd $YQQ/N0-VTLA
export PYTHONPATH=$PWD:$PWD/scripts
RAW=$YQQ/data/n0vtla/wetlab_raw/task4_dual_arm_tube_pour_20261005
UUID="${UUID:-$(ls $RAW | grep -E '^[0-9a-f-]{36}$' | head -1)}"
PORT=8765

$PY scripts/serve_policy.py --policy.config=vtla_tactile_posttrain --policy.dir="$CKPT" --low-cpu-mem-usage --port $PORT > $YQQ/logs/t4_srvcli_server_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
for i in $(seq 1 90); do
  $PY -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -n 30 $YQQ/logs/t4_srvcli_server_$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
echo "server up after ~$((i*5))s; package $UUID"
$PY scripts/dual_arm_client_example.py --package $RAW/$UUID --norm scripts/wetlab_tactile_norm_task4_dual.json \
  --port $PORT --queries "${QUERIES:-6}"
STATUS=$?
kill $SERVER 2>/dev/null
grep -aE "asset|norm|fall" $YQQ/logs/t4_srvcli_server_$SLURM_JOB_ID.log | cut -c1-200 | tail -5
[ $STATUS -eq 0 ] && echo T4_SRVCLI_PASSED || echo "T4_SRVCLI_FAILED ($STATUS)"
exit $STATUS
