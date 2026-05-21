#!/bin/bash

# 图像腐化处理脚本
# 使用 imagecorruptions 库对图像进行多种类型的腐化处理

# 参数配置
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-/root/autodl-tmp/VOC2007}"        # 输入图像目录
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/PASCAL-C}"      # 输出目录
OUTPUT_TYPE="${OUTPUT_TYPE:-subdirs}"                       # 输出类型: subdirs 或 filename
SUBSET="${SUBSET:-all}"                                      # 腐化类型子集: common, validation, all, noise, blur, weather, digital
SEVERITY_LEVELS="${SEVERITY_LEVELS:-5}"                      # 严重程度级别
NUM_CORES="${NUM_CORES:-20}"                                 # 并行处理核心数

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 检查输入目录是否存在
if [ ! -d "${INPUT_DIR}" ]; then
    echo "错误: 输入目录 ${INPUT_DIR} 不存在!"
    exit 1
fi

# 检查Python脚本是否存在
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_SCRIPT="${SCRIPT_DIR}/corrupt_images.py"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误: Python脚本 ${PYTHON_SCRIPT} 不存在!"
    exit 1
fi

# 执行图像腐化处理
echo "开始图像腐化处理..."
echo "输入目录: ${INPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "输出类型: ${OUTPUT_TYPE}"
echo "腐化子集: ${SUBSET}"
echo "严重程度: ${SEVERITY_LEVELS}"
echo "并行核心数: ${NUM_CORES}"
echo "----------------------------------------"

"${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    "${OUTPUT_TYPE}" \
    --subset "${SUBSET}" \
    --severity ${SEVERITY_LEVELS} \
    -j "${NUM_CORES}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "图像腐化处理完成!"
else
    echo "----------------------------------------"
    echo "图像腐化处理失败!"
    exit 1
fi
