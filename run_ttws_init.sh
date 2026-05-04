#!/usr/bin/env bash
set -euo pipefail

# Initialize TTWS visual prompts for one corruption type.
# Override any variable below from the command line, for example:
# CORRUPTION_TYPE=motion_blur PROMPT_FILE=prompt_init/voc-c/prompt_voc_motion_blur.pth bash run_ttws_init.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py}"
CHECKPOINT="${CHECKPOINT:-download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth}"
CORRUPTION_TYPE="${CORRUPTION_TYPE:-gaussian_noise}"
PROMPT_FILE="${PROMPT_FILE:-prompt_init/voc-c/prompt_voc_${CORRUPTION_TYPE}.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/prompt_init}"

mkdir -p "$(dirname "${PROMPT_FILE}")" "${WORK_DIR}"

echo "Running TTWS initialization"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Corruption: ${CORRUPTION_TYPE}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Work dir: ${WORK_DIR}"

"${PYTHON_BIN}" test.py "${CONFIG}" "${CHECKPOINT}" \
  --cfg-options \
  corruption_type="'${CORRUPTION_TYPE}'" \
  model.detector.backbone.TTWS_init=True \
  model.detector.backbone.TTWS_file="'${PROMPT_FILE}'" \
  --work-dir "${WORK_DIR}"
