# code/ImageDifussionFake/scripts/eval_2.sh


set -euo pipefail

cd /scratch/sahil/projects/img_deepfake/code/ImageDifussionFake

EXP_DIR="${EXP_DIR:-experiments/FFPP_10}"
CKPT_DIR="${CKPT_DIR:-$EXP_DIR/ckpt}"
CONFIG="${CONFIG:-configs/eval_all_datasets.yaml}"
SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

BEST_CKPT="$(ls -1 "$CKPT_DIR"/best-eer-epoch=*\.ckpt 2>/dev/null | \
  sed -E 's/.*best-eer-epoch=([0-9]+)\.ckpt/\1 &/' | \
  sort -n | tail -1 | awk '{print $2}' || true)"

if [[ -n "${BEST_CKPT:-}" ]]; then
  CKPT="$BEST_CKPT"
elif [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  CKPT="$CKPT_DIR/last.ckpt"
else
  echo "[ERR] No checkpoint found in: $CKPT_DIR"
  exit 1
fi

OUT_DIR="$EXP_DIR/eval"
OUT_CSV="${OUT_CSV:-$OUT_DIR/generalization_all_${SPLIT}.csv}"
OUT_METRICS="${OUT_METRICS:-$OUT_DIR/generalization_all_${SPLIT}_metrics.json}"
PER_DS_DIR="${PER_DS_DIR:-$OUT_DIR/per_dataset_${SPLIT}}"

mkdir -p "$OUT_DIR"

echo "[EVAL-ALL] CKPT=$CKPT"
echo "[EVAL-ALL] CONFIG=$CONFIG SPLIT=$SPLIT"
echo "[EVAL-ALL] OUT_CSV=$OUT_CSV"
echo "[EVAL-ALL] OUT_METRICS=$OUT_METRICS"
echo "[EVAL-ALL] PER_DS_DIR=$PER_DS_DIR"

python -u eval_generalization.py \
  -c "$CONFIG" \
  --ckpt "$CKPT" \
  --split "$SPLIT" \
  --out_csv "$OUT_CSV" \
  --out_metrics "$OUT_METRICS" \
  --per_dataset_dir "$PER_DS_DIR"


# chmod +x scripts/eval_2.sh
# bash scripts/eval_2.sh