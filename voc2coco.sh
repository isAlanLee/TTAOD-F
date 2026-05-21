#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VOC_ROOT="${VOC_ROOT:-/root/autodl-tmp/VOC2007}"
COCO_OUTPUT="${COCO_OUTPUT:-pascal_test2007.json}"

echo "Converting VOC annotations to COCO"
echo "VOC root: ${VOC_ROOT}"
echo "COCO output: ${COCO_OUTPUT}"

"${PYTHON_BIN}" voc2coco.py \
  --voc-root "${VOC_ROOT}" \
  --output "${COCO_OUTPUT}"
