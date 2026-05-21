#!/usr/bin/env bash
set -euo pipefail

# Run test-time adaptation with visual prompts, text prompts, and IDM settings.
# Override any variable below from the command line, for example:
# CORRUPTION_TYPE=motion_blur SHOT_CAPACITY=0 bash run_tta_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py}"
CHECKPOINT="${CHECKPOINT:-download/groundingdino_swint_ogc_mmdet-822d7e9d.pth}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/JPEGImages-C}"
ANN_FILE="${ANN_FILE:-/root/autodl-fs/TTAOD-F/pascal_test2007.json}"
CORRUPTION_TYPE="${CORRUPTION_TYPE:-gaussian_noise}"
PROMPT_FILE="${PROMPT_FILE:-prompt_init/voc-c/prompt_voc_${CORRUPTION_TYPE}.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/ttaod}"
PROMPT_TYPE="${PROMPT_TYPE:-prepend}"
NUM_TOKENS="${NUM_TOKENS:-10}"
PROMPT_DEEP="${PROMPT_DEEP:-True}"
TEXT_PROMPT="${TEXT_PROMPT:-True}"
BACKBONE_WITH_CP="${BACKBONE_WITH_CP:-True}"
ENCODER_NUM_CP="${ENCODER_NUM_CP:-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
LR="${LR:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP_MAX_NORM="${GRAD_CLIP_MAX_NORM:-0.1}"
TUNABLE_LINEAR_LR_MULT="${TUNABLE_LINEAR_LR_MULT:-0.1}"
PSEUDO_LABEL_INITIAL_SCORE_THR="${PSEUDO_LABEL_INITIAL_SCORE_THR:-0.3}"
CLS_PSEUDO_THR="${CLS_PSEUDO_THR:-0.3}"
UNSUP_WEIGHT="${UNSUP_WEIGHT:-1.0}"
SHOT_CAPACITY="${SHOT_CAPACITY:-20}"
ALPHA="${ALPHA:-5.0}"
BETA="${BETA:-5.0}"
THRE_ME="${THRE_ME:-0.3}"
MEMORY_HALLUCINATION="${MEMORY_HALLUCINATION:-True}"
HALLUCINATION_MAX_INSTANCES="${HALLUCINATION_MAX_INSTANCES:-3}"
HALLUCINATION_BETA="${HALLUCINATION_BETA:-1.0}"
HALLUCINATION_IOU_THR="${HALLUCINATION_IOU_THR:-0.2}"
HALLUCINATION_MAX_TRIALS="${HALLUCINATION_MAX_TRIALS:-10}"
HALLUCINATION_SCALE_RANGE="${HALLUCINATION_SCALE_RANGE:-(0.5,1.5)}"
HALLUCINATION_VIS_DIR="${HALLUCINATION_VIS_DIR:-}"
HALLUCINATION_VIS_MAX="${HALLUCINATION_VIS_MAX:-8}"
DINOV2_REPO="${DINOV2_REPO:-download/dinov2}"
DINOV2_MODEL="${DINOV2_MODEL:-dinov2_vitl14}"
DINOV2_SOURCE="${DINOV2_SOURCE:-local}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-False}"
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-download/dinov2_vitl14_pretrain.pth}"

mkdir -p "${WORK_DIR}" logs

echo "Running test-time adaptation"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Annotation: ${ANN_FILE}"
echo "Corruption: ${CORRUPTION_TYPE}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Work dir: ${WORK_DIR}"
echo "Prompt: type=${PROMPT_TYPE}, tokens=${NUM_TOKENS}, deep=${PROMPT_DEEP}"
echo "Pseudo thresholds: initial=${PSEUDO_LABEL_INITIAL_SCORE_THR}, cls=${CLS_PSEUDO_THR}"
echo "IDM: shot_capacity=${SHOT_CAPACITY}, alpha=${ALPHA}, beta=${BETA}, thre_me=${THRE_ME}"
echo "MH: enabled=${MEMORY_HALLUCINATION}, max_instances=${HALLUCINATION_MAX_INSTANCES}, beta=${HALLUCINATION_BETA}, iou=${HALLUCINATION_IOU_THR}, trials=${HALLUCINATION_MAX_TRIALS}, scale=${HALLUCINATION_SCALE_RANGE}"
echo "DINOv2: repo=${DINOV2_REPO}, model=${DINOV2_MODEL}, source=${DINOV2_SOURCE}, pretrained=${DINOV2_PRETRAINED}, checkpoint=${DINOV2_CHECKPOINT}"

cfg_options=(
  "data_root='${DATA_ROOT}'"
  "ann_file='${ANN_FILE}'"
  "corruption_type='${CORRUPTION_TYPE}'"
  "load_from='${CHECKPOINT}'"
  "train_dataloader.batch_size=${TRAIN_BATCH_SIZE}"
  "train_dataloader.num_workers=${TRAIN_NUM_WORKERS}"
  "val_dataloader.num_workers=${VAL_NUM_WORKERS}"
  "test_dataloader.num_workers=${VAL_NUM_WORKERS}"
  "train_dataloader.dataset.data_root='${DATA_ROOT}'"
  "train_dataloader.dataset.ann_file='${ANN_FILE}'"
  "train_dataloader.dataset.data_prefix.img='${CORRUPTION_TYPE}'"
  "val_dataloader.dataset.data_root='${DATA_ROOT}'"
  "val_dataloader.dataset.ann_file='${ANN_FILE}'"
  "val_dataloader.dataset.data_prefix.img='${CORRUPTION_TYPE}'"
  "test_dataloader.dataset.data_root='${DATA_ROOT}'"
  "test_dataloader.dataset.ann_file='${ANN_FILE}'"
  "test_dataloader.dataset.data_prefix.img='${CORRUPTION_TYPE}'"
  "val_evaluator.ann_file='${ANN_FILE}'"
  "test_evaluator.ann_file='${ANN_FILE}'"
  "model.detector.text_prompt=${TEXT_PROMPT}"
  "model.detector.encoder.num_cp=${ENCODER_NUM_CP}"
  "model.detector.backbone.with_cp=${BACKBONE_WITH_CP}"
  "model.detector.backbone.prompt_type='${PROMPT_TYPE}'"
  "model.detector.backbone.num_tokens=${NUM_TOKENS}"
  "model.detector.backbone.prompt_deep=${PROMPT_DEEP}"
  "model.detector.backbone.TTWS_file='${PROMPT_FILE}'"
  "model.semi_train_cfg.pseudo_label_initial_score_thr=${PSEUDO_LABEL_INITIAL_SCORE_THR}"
  "model.semi_train_cfg.cls_pseudo_thr=${CLS_PSEUDO_THR}"
  "model.semi_train_cfg.unsup_weight=${UNSUP_WEIGHT}"
  "optim_wrapper.optimizer.lr=${LR}"
  "optim_wrapper.optimizer.weight_decay=${WEIGHT_DECAY}"
  "optim_wrapper.clip_grad.max_norm=${GRAD_CLIP_MAX_NORM}"
  "optim_wrapper.paramwise_cfg.custom_keys.tunable_linear.lr_mult=${TUNABLE_LINEAR_LR_MULT}"
  "train_cfg.shot_capacity=${SHOT_CAPACITY}"
  "train_cfg.alpha=${ALPHA}"
  "train_cfg.beta=${BETA}"
  "train_cfg.thre_me=${THRE_ME}"
  "train_cfg.memory_hallucination=${MEMORY_HALLUCINATION}"
  "train_cfg.hallucination_max_instances=${HALLUCINATION_MAX_INSTANCES}"
  "train_cfg.hallucination_beta=${HALLUCINATION_BETA}"
  "train_cfg.hallucination_iou_thr=${HALLUCINATION_IOU_THR}"
  "train_cfg.hallucination_max_trials=${HALLUCINATION_MAX_TRIALS}"
  "train_cfg.hallucination_scale_range=${HALLUCINATION_SCALE_RANGE}"
  "train_cfg.hallucination_vis_max=${HALLUCINATION_VIS_MAX}"
  "train_cfg.dinov2_repo='${DINOV2_REPO}'"
  "train_cfg.dinov2_model='${DINOV2_MODEL}'"
  "train_cfg.dinov2_source='${DINOV2_SOURCE}'"
  "train_cfg.dinov2_pretrained=${DINOV2_PRETRAINED}"
  "train_cfg.dinov2_checkpoint='${DINOV2_CHECKPOINT}'"
)

if [[ -n "${HALLUCINATION_VIS_DIR}" ]]; then
  cfg_options+=("train_cfg.hallucination_vis_dir='${HALLUCINATION_VIS_DIR}'")
else
  cfg_options+=("train_cfg.hallucination_vis_dir=None")
fi

"${PYTHON_BIN}" train.py "${CONFIG}" \
  --cfg-options "${cfg_options[@]}" \
  --work-dir "${WORK_DIR}" 2>&1 | tee ./logs/run_tta_train_${CORRUPTION_TYPE}.log
