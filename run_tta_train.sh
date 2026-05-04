#!/usr/bin/env bash
set -euo pipefail

# Run test-time adaptation with visual prompts, text prompts, and IDM settings.
# Override any variable below from the command line, for example:
# CORRUPTION_TYPE=motion_blur SHOT_CAPACITY=0 bash run_tta_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py}"
CORRUPTION_TYPE="${CORRUPTION_TYPE:-gaussian_noise}"
PROMPT_FILE="${PROMPT_FILE:-prompt_init/voc-c/prompt_voc_${CORRUPTION_TYPE}.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/ttaod}"
PROMPT_TYPE="${PROMPT_TYPE:-prepend}"
NUM_TOKENS="${NUM_TOKENS:-10}"
PROMPT_DEEP="${PROMPT_DEEP:-True}"
SHOT_CAPACITY="${SHOT_CAPACITY:-15}"
ALPHA="${ALPHA:-5.0}"
BETA="${BETA:-5.0}"
THRE_ME="${THRE_ME:-0.3}"

mkdir -p "${WORK_DIR}"

echo "Running test-time adaptation"
echo "Config: ${CONFIG}"
echo "Corruption: ${CORRUPTION_TYPE}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Work dir: ${WORK_DIR}"
echo "Prompt: type=${PROMPT_TYPE}, tokens=${NUM_TOKENS}, deep=${PROMPT_DEEP}"
echo "IDM: shot_capacity=${SHOT_CAPACITY}, alpha=${ALPHA}, beta=${BETA}, thre_me=${THRE_ME}"

"${PYTHON_BIN}" train.py "${CONFIG}" \
  --cfg-options \
  corruption_type="'${CORRUPTION_TYPE}'" \
  model.detector.backbone.prompt_type="'${PROMPT_TYPE}'" \
  model.detector.backbone.num_tokens="${NUM_TOKENS}" \
  model.detector.backbone.prompt_deep="${PROMPT_DEEP}" \
  model.detector.backbone.TTWS_file="'${PROMPT_FILE}'" \
  train_cfg.shot_capacity="${SHOT_CAPACITY}" \
  train_cfg.alpha="${ALPHA}" \
  train_cfg.beta="${BETA}" \
  train_cfg.thre_me="${THRE_ME}" \
  --work-dir "${WORK_DIR}"
