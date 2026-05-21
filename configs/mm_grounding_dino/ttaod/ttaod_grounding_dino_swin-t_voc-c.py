_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

detector = _base_.model

# Match the OGC checkpoint config
# `grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py`.
detector.language_model.add_pooling_layer = True
detector.bbox_head.num_classes = 80
detector.bbox_head.contrastive_cfg = dict(max_text_len=256)
detector.text_prompt=True

# Keep activation checkpointing enabled for TTA training memory usage. This
# does not affect checkpoint key compatibility with the OGC weights.
detector.encoder.num_cp = 6

detector.backbone = dict(
    detector.backbone,
    init_cfg=None,
    with_cp=True,
    convert_weights=False,
    prompt_type='prepend',
    num_tokens=10,
    prompt_deep=True,
    TTWS_init=False,
    TTWS_file='prompt_init/voc-c/prompt_voc_gaussian_noise.pth',
)

model = dict(
    _delete_=True,
    type='TTAODSoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0, 
        pseudo_label_initial_score_thr=0.3, 
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.3,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

dataset_type = 'CocoDataset'
data_root = 'data/JPEGImages-C'
ann_file = 'data/pascal_test2007.json'
corruption_type = 'gaussian_noise'
class_name = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
                'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor',)
palette = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
           (0, 80, 100), (0, 0, 230), (119, 11, 32)]

metainfo = dict(classes=class_name, palette=palette)

backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

scale = [(1333, 400), (1333, 1200)]

branch_field = ['unsup_teacher', 'unsup_student', 'test']
# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'homography_matrix'))
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'homography_matrix'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'homography_matrix'))
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
        test=test_pipeline,
    )
]

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file=ann_file,
    data_prefix=dict(img=corruption_type),
    return_classes=True,
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=unlabeled_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file,
        data_prefix=dict(img=corruption_type),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_file,
    metric='bbox',
    # classwise=True,
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.2, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'tunable_linear': dict(lr_mult=0.1),
        }))

# learning policy
max_epochs = 1
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[1, 1],
        gamma=0.1)
]

train_cfg = dict(type='TTAODLoop', 
    max_epochs=max_epochs, 
    val_interval=1,
    shot_capacity=20,
    alpha=5.0,
    beta=5.0,
    thre_me=0.3,
    memory_hallucination=True,
    hallucination_max_instances=3,
    hallucination_beta=1.0,
    hallucination_iou_thr=0.2,
    hallucination_max_trials=10,
    hallucination_scale_range=(0.5, 1.5),
    hallucination_vis_dir=None,
    hallucination_vis_max=8,
    dinov2_repo=None,
    dinov2_model=None,
    dinov2_source='local',
    dinov2_pretrained=False,
    dinov2_checkpoint=None)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))
custom_hooks = [dict(type='MeanTeacherHook')]

# GroundingDINO-SwinT OGC checkpoint trained on Objects365, GoldG, and Cap4M.
load_from = 'download/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
