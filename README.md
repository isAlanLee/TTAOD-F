# ttaod_foundation

This is the implementation of the paper "Test-Time Adaptive Object Detection with Foundation Model" (Neurips 2025, [arxiv](https://arxiv.org/pdf/2510.25175))

## Installation

Please refer to mmdetection [Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions. Note:

* pytorch==1.11.0+cu113
* mmcv==2.0.0rc4, mmengine==0.10.7

## Prepare Data

We construct cross-corruption benchmark by transforming PASCAL into **PASCAL-C** and COCO into **COCO-C**. Both datasets are generated using the [imagecorruptions](https://github.com/bethgelab/imagecorruptions) package, which applies 15 types of corruption to each image at a severity level of 5.


## Training
1. Run TTWS:
```
python test.py configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py
  download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
  --cfg-options corruption_type='gaussian_noise' 
  detector.backbone.TTWS_init=True  
  detector.backbone.TTWS_file='prompt_init/voc-c/prompt_voc_gaussian_noise.pth'
  --work-dir work_dirs/prompt_init
```

2. Test Time Adaptation:
```
python train.py configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py
  --cfg-options corruption_type='gaussian_noise' 
  detector.backbone.prompt_type='prepend'
  detector.backbone.num_tokens=10
  detector.backbone.prompt_deep=True
  detector.backbone.TTWS_file='prompt_init/voc-c/prompt_voc_gaussian_noise.pth'
  train_cfg.shot_capacity=15
  train_cfg.alpha=5.0 train_cfg.beta=5.0
  train_cfg.thre_me=0.3
  --work-dir work_dirs/ttaod
```



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