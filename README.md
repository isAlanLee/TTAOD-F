# ttaod_foundation

This is the implementation of the paper "Test-Time Adaptive Object Detection with Foundation Model" (Neurips 2025, [arxiv](https://arxiv.org/pdf/2510.25175))

## Installation

Please refer to mmdetection [Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions. Note:

* pytorch==1.11.0+cu113
* mmcv==2.0.0rc4, mmengine==0.10.7

## Prepare Data

We construct cross-corruption benchmark by transforming PASCAL into **PASCAL-C** and COCO into **COCO-C**. Both datasets are generated using the [imagecorruptions](https://github.com/bethgelab/imagecorruptions) package, which applies 15 types of corruption to each image at a severity level of 5.


## Training
All paths and reproduction parameters are passed through shell scripts. Override
the defaults with environment variables instead of editing Python config files.

1. Convert VOC XML annotations to COCO JSON if needed:
```
VOC_ROOT=/root/autodl-tmp/VOC2007 \
COCO_OUTPUT=/root/autodl-fs/TTAOD-F/pascal_test2007.json \
bash voc2coco.sh
```

2. Generate PASCAL-C/COCO-C corruptions:
```
INPUT_DIR=/root/autodl-tmp/VOC2007 \
OUTPUT_DIR=/root/autodl-tmp/JPEGImages-C \
SEVERITY_LEVELS=5 \
bash corrupt_images.sh
```

3. Run TTWS for one corruption type:
```
DATA_ROOT=/root/autodl-tmp/JPEGImages-C \
ANN_FILE=/root/autodl-fs/TTAOD-F/pascal_test2007.json \
CORRUPTION_TYPE=gaussian_noise \
bash run_ttws_init.sh
```

4. Run test-time adaptation:
```
DATA_ROOT=/root/autodl-tmp/JPEGImages-C \
ANN_FILE=/root/autodl-fs/TTAOD-F/pascal_test2007.json \
CORRUPTION_TYPE=gaussian_noise \
SHOT_CAPACITY=20 \
ALPHA=5.0 \
BETA=5.0 \
THRE_ME=0.3 \
MEMORY_HALLUCINATION=True \
HALLUCINATION_IOU_THR=0.2 \
bash run_tta_train.sh
```

Useful overrides include `CHECKPOINT`, `PROMPT_FILE`, `WORK_DIR`,
`PSEUDO_LABEL_INITIAL_SCORE_THR`, `CLS_PSEUDO_THR`, `DINOV2_REPO`,
`DINOV2_CHECKPOINT`, `HALLUCINATION_MAX_INSTANCES`,
`HALLUCINATION_BETA`, `HALLUCINATION_MAX_TRIALS`, and
`HALLUCINATION_SCALE_RANGE`.

## Citing

You can cite our paper with such bibtex:
```
@article{gao2025test,
  title={Test-Time Adaptive Object Detection with Foundation Model},
  author={Gao, Yingjie and Zhang, Yanan and Cai, Zhi and Huang, Di},
  journal={arXiv preprint arXiv:2510.25175},
  year={2025}
}
```
