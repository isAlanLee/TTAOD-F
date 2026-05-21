#!/usr/bin/env bash
set -euo pipefail

# Initialize TTWS visual prompts for one corruption type.
# Override any variable below from the command line, for example:
# CORRUPTION_TYPE=motion_blur PROMPT_FILE=prompt_init/voc-c/prompt_voc_motion_blur.pth bash run_ttws_init.sh

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
WORK_DIR="${WORK_DIR:-work_dirs/prompt_init}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"

mkdir -p "$(dirname "${PROMPT_FILE}")" "${WORK_DIR}" logs

echo "Running TTWS initialization"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Annotation: ${ANN_FILE}"
echo "Corruption: ${CORRUPTION_TYPE}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Work dir: ${WORK_DIR}"

cfg_options=(
  "data_root='${DATA_ROOT}'"
  "ann_file='${ANN_FILE}'"
  "corruption_type='${CORRUPTION_TYPE}'"
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
  "model.detector.backbone.prompt_type=None"
  "model.detector.backbone.num_tokens=0"
  "model.detector.backbone.prompt_deep=False"
  "model.detector.backbone.TTWS_init=True"
  "model.detector.backbone.TTWS_file='${PROMPT_FILE}'"
)

"${PYTHON_BIN}" test.py "${CONFIG}" "${CHECKPOINT}" \
  --cfg-options "${cfg_options[@]}" \
  --work-dir "${WORK_DIR}" 2>&1 | tee ./logs/run_ttws_init_${CORRUPTION_TYPE}.log

PROMPT_FILE="${PROMPT_FILE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

prompt_file = os.environ["PROMPT_FILE"]
expected_shapes = {
    "prompt_embeddings": (1, 1, 96),
    "deep_prompt_embeddings_0": (1, 1, 96),
    "deep_prompt_embeddings_1": (2, 1, 192),
    "deep_prompt_embeddings_2": (6, 1, 384),
    "deep_prompt_embeddings_3": (2, 1, 768),
}

if not os.path.isfile(prompt_file):
    raise SystemExit(f"TTWS prompt file was not created: {prompt_file}")

prompt = torch.load(prompt_file, map_location="cpu")
missing = [key for key in expected_shapes if key not in prompt]
unexpected = [key for key in prompt if key not in expected_shapes]
if missing or unexpected:
    raise SystemExit(
        f"Invalid TTWS prompt keys: missing={missing}, unexpected={unexpected}")

for key, shape in expected_shapes.items():
    tensor = prompt[key]
    if not torch.is_tensor(tensor):
        raise SystemExit(f"{key} is not a tensor")
    if tuple(tensor.shape) != shape:
        raise SystemExit(f"{key} shape {tuple(tensor.shape)} != {shape}")
    if not torch.isfinite(tensor).all().item():
        raise SystemExit(f"{key} contains NaN or Inf")
    if tensor.float().abs().max().item() <= 0:
        raise SystemExit(f"{key} is all zeros")

print(f"TTWS prompt validation passed: {prompt_file}")
PY
