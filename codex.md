# Codex Progress

## 2026-05-04

### 当前任务
- 根据 GitHub issue #2 修正 TTAOD-F 中 text prompt 与 TTWS 配置/实现不一致的问题。
- 结合项目 README 整理整体训练流程。

### 已确认
- issue #2 指出 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 默认关闭 `detector.text_prompt`，与论文声称使用 text prompts 不一致。
- issue #2 指出 `mmdet/engine/runner/ttaod_loop.py` 只解冻名称包含 `prompt_embeddings` 的参数，未解冻 text prompt 的 `tunable_linear`。
- issue #2 指出 README 中 TTWS warm start 参数层级可能需要写成 `model.detector.backbone.*`，以更新 `TTAODSoftTeacher` 外层模型内的 detector。

### 下一步
- 阅读配置、runner loop、模型封装和 README 的相关实现。
- 修改 text prompt 默认启用、解冻逻辑和 TTWS 命令示例。
- 做基础静态校验，并更新训练流程说明。

### 阶段进展
- 已将 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 中的 `detector.text_prompt` 默认值改为 `True`。
- 已将 `mmdet/engine/runner/ttaod_loop.py` 的训练参数解冻逻辑扩展到 `tunable_linear`，使 text prompt 参数可训练。
- 已将 README 中 TTWS 和 TTA 命令的 backbone 覆盖路径改为 `model.detector.backbone.*`，匹配 `TTAODSoftTeacher` 的外层 `model.detector` 配置结构。
- 已执行 `python -m py_compile` 对修改过的 Python 文件做语法检查，检查通过；生成的临时 `__pycache__` 已清理。

### 待交付说明
- 最终回复中需要给出修改摘要、验证结果，以及基于项目主页 README 的整体训练流程。

### 追加阶段进展
- 已新增 `run_ttws_init.sh`，用于 Linux 环境下运行 TTWS visual prompt 初始化。
- 已新增 `run_tta_train.sh`，用于 Linux 环境下运行测试时自适应训练。
- 两个脚本均支持通过环境变量覆盖配置、腐化类型、prompt 文件、work dir 等常用参数。

### 日志检查
- 用户提供的 `run_ttws_init_gaussian_noise.log` 未发现 Traceback 或 fatal error。
- TTWS 初始化已完成，并打印 `Saved visual prompt to prompt_init/voc-c/prompt_voc_gaussian_noise.pth`。
- 日志开头有 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，说明运行环境中的 `OMP_NUM_THREADS` 值非法；建议在脚本中显式设置为正整数。
- checkpoint 加载时的 `missing keys ... tunable_linear.weight` 与启用 text prompt 后新增参数有关，属于预期现象；`unexpected ... position_ids` 通常是 BERT position ids buffer 差异，不影响本次保存 visual prompt。

## 2026-05-07

### 当前任务
- 用户指出项目尚未实现论文中的 Memory Hallucination，需要查询论文具体方案并在现有代码上补齐。

### 已确认
- 论文《Test-Time Adaptive Object Detection with Foundation Model》Sec. 3.4 说明：IDM 需要为每类维护高质量伪标签实例三元组 `(img, feat, s)`，其中 `img` 是实例 crop，`feat` 是 DINOv2 特征，`s` 是分类分数。
- 现有 `mmdet/engine/runner/ttaod_loop.py` 已实现 IDM 与 Memory Enhancement，但 `IDM_cache` 条目只保存 `feature + score`，没有保存实例图像 crop，因此无法执行 Memory Hallucination。
- 论文中的 Memory Hallucination 针对没有可用伪标签的负样本：从 IDM 随机采样高质量实例图像，随机缩放后以 Beta 分布采样的混合系数贴到负图像上；每张负图最多混入 3 个实例；为避免实例互相过度重叠，超过 IoU 阈值时重选位置并最多重试 10 次。

### 下一步
- 扩展 IDM cache 条目，使其同时保存实例 crop。
- 在 TTA 训练循环中检测 `unsup_student` 分支伪标签为空的样本，并基于 IDM 合成 hallucinated positive image 与对应伪标签。
- 为 Memory Hallucination 增加可配置开关和参数，并做基础语法校验。

### 阶段进展
- 已将 `mmdet/engine/runner/ttaod_loop.py` 中的 `IDM_cache` 条目扩展为 `feature / score / image`，使 IDM 符合论文中的 `(img, feat, s)` 三元组。
- 已新增 Memory Hallucination 逻辑：从 IDM 随机采样实例 crop，随机缩放后按 Beta 分布混合系数贴入无可用伪标签的 `unsup_student` 图像，并生成对应 `bboxes / labels / scores` 伪标签。
- 已实现每张负样本最多混入 3 个实例、IoU 阈值检查、最多 10 次位置重试，并处理 detector 原始输入 BGR/RGB 通道约定。
- 已在 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 中增加 Memory Hallucination 配置项，默认启用。
- 已执行 `python -m py_compile mmdet/engine/runner/ttaod_loop.py`，语法检查通过；本地当前 Python 环境缺少 `torch`，未做完整训练 dry run。

### Text Prompt 对齐检查
- `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 已设置 `detector.text_prompt=True`，文本提示已启用。
- `mmdet/engine/runner/ttaod_loop.py` 已解冻 `tunable_linear`，文本提示参数可训练。
- 已在 `optim_wrapper.paramwise_cfg.custom_keys` 下补充 `'tunable_linear': dict(lr_mult=0.1)`，使 text prompt 学习率为基础学习率的 0.1 倍。注意 MMEngine/MMDet 使用的键名是 `lr_mult`，不是 `lr_multi`。

### OGC Checkpoint 对齐
- 用户要求将当前使用的 `download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth` 切换为论文一致的 `download/groundingdino_swint_ogc_mmdet-822d7e9d.pth`。
- 已确认主 TTAOD 配置 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 的 `load_from` 指向 `download/groundingdino_swint_ogc_mmdet-822d7e9d.pth`，并补充注释说明该权重为 Objects365、GoldG、Cap4M 上训练的 OGC checkpoint。
- 已将 README 中 TTWS 初始化示例的 checkpoint 参数从旧的 Objects365 + GoldG + GRIT9M + V3Det 权重改为 `download/groundingdino_swint_ogc_mmdet-822d7e9d.pth`。
- 已用 `rg` 检查 README、configs、脚本入口、train.py、test.py，旧 checkpoint 字符串已不再出现。
- 已执行 `python -m py_compile configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`，配置文件语法检查通过；本轮生成的 `configs/mm_grounding_dino/ttaod/__pycache__` 已清理。

### OGC 权重指标为 0 的日志诊断
- 用户提供 `C:/Users/33168/Downloads/run_tta_train_gaussian_noise (2).log`，日志只记录到 `Epoch(train) [1][50/1238]`，未包含最终评估表。
- 日志显示加载 `download/groundingdino_swint_ogc_mmdet-822d7e9d.pth` 后，`student/teacher.bbox_head.cls_branches.*.bias` 缺失；当前项目配置 `contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True)` 会创建分类头 bias，而 OGC 官方配置 `grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py` 使用 `contrastive_cfg=dict(max_text_len=256)`，不带 bias/log_scale。
- 当前配置与 OGC 官方配置还存在 `language_model.add_pooling_layer=False` vs 官方 `True`、`backbone.with_cp=True/convert_weights=True` vs 官方 `False/False` 等差异。
- 日志第 50 iter 显示 `loss=0.0000`、`grad_norm=0.0000`、所有 `unsup_*` loss 为 0，说明 teacher 伪标签经 `pseudo_label_initial_score_thr=0.3` / `cls_pseudo_thr=0.3` 过滤后基本为空，student 没有实际训练信号。
- 初步结论：旧 GRIT9M+V3Det 权重与当前 base 配置兼容；OGC 权重需要配套 OGC Cap4M 配置，尤其要去掉分类头 bias/log_scale，否则分数会被额外负偏置压低，导致伪标签为空并最终 mAP 为 0。

### OGC 配置修正
- 已在 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 中显式覆盖 OGC 关键配置：`detector.language_model.add_pooling_layer=True`、`detector.bbox_head.contrastive_cfg=dict(max_text_len=256)`、`detector.backbone.with_cp=False`、`detector.backbone.convert_weights=False`。
- 该修正保留 TTAOD 需要的 `detector.text_prompt=True` 和 visual prompt 配置项，但去掉与 OGC checkpoint 不匹配的分类头 bias/log_scale。
- 已执行 `python -m py_compile configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`，语法检查通过；本机 Windows 环境缺少 `mmengine`，未能解析完整 MMEngine 配置。
- 本轮语法检查生成的 `configs/mm_grounding_dino/ttaod/__pycache__` 已清理。
- 后续在服务器上需要重新运行 `run_ttws_init.sh` 生成新的 `prompt_init/voc-c/prompt_voc_gaussian_noise.pth`，再运行 `run_tta_train.sh`；不建议复用修正前生成的 prompt 文件。

### OGC 配置解析错误修复
- 用户运行 `test.py` 时报错 `TypeError: type object got multiple values for keyword argument 'with_cp'`。
- 原因是 `dict(**detector.backbone, with_cp=False, convert_weights=False, ...)` 在 `detector.backbone` 已包含同名键时会形成重复关键字。
- 已将写法改为 `dict(detector.backbone, with_cp=False, convert_weights=False, ...)`，允许后续关键字覆盖原有键。
- 已重新执行 `python -m py_compile configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`，语法检查通过。

### Swin Backbone 初始化错误修复
- 用户运行 `train.py` 时报错 `IndexError: list index out of range`，位置在 `mmdet/models/backbones/swin.py::init_weights`。
- 原因是 base 配置仍保留 Swin backbone 的 ImageNet 预训练 `init_cfg`，模型初始化阶段会先尝试加载该 backbone checkpoint；当前 Swin 加载逻辑只保留 `backbone.` 前缀键，若 checkpoint 键名不匹配会得到空 `state_dict` 并触发索引错误。
- OGC 官方配置不设置 backbone `init_cfg`，应由整模型 `load_from=download/groundingdino_swint_ogc_mmdet-822d7e9d.pth` 加载权重。
- 已在 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 的 `detector.backbone` 覆盖中加入 `init_cfg=None`。
- 已重新执行 `python -m py_compile configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`，语法检查通过。

### 论文对齐审查
- 当前任务：审查仓库实现是否与论文《Test-Time Adaptive Object Detection with Foundation Model》对齐，重点核对 MPMT、TTWS、IDM、Memory Enhancement、Memory Hallucination 以及关键超参数。

### 已确认对齐
- 论文方法层面的核心组件已经落地：`GroundingDINO` 中实现了 text prompt (`tunable_linear`)；`SwinTransformer` 中实现了 visual prompt、deep prompt 与 TTWS 初始化逻辑；`TTAODLoop` 中实现了 IDM、Memory Enhancement、Memory Hallucination。
- `MeanTeacherHook` 默认 `momentum=0.001`，对应论文公式中的 `gamma=0.999`（即 `teacher = 0.999 * teacher + 0.001 * student`），语义对齐。
- `run_tta_train.sh` 默认覆盖 `prompt_type=prepend`、`num_tokens=10`、`prompt_deep=True`、`shot_capacity=20`、`alpha=5.0`、`beta=5.0`、`thre_me=0.3`，与论文 Pascal-C 主设置基本一致。

### 发现的偏差
- `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 当前默认 `train_cfg.shot_capacity=15`，而论文在 Pascal-C 主实验中使用的 IDM 容量是 20；README 的训练示例也仍写成 15。虽然 `run_tta_train.sh` 已改为 20，但裸跑配置或按 README 手动执行会偏离论文设置。
- 同一配置文件当前默认 `train_cfg.hallucination_iou_thr=0.5`，而论文 Memory Hallucination 的实现细节使用的是 `thIoU=0.2`。该参数当前也没有在 `run_tta_train.sh` 中覆盖，因此脚本默认行为仍与论文不一致。

### 下一步
- 若继续追求严格复现论文，需要先修正 `shot_capacity` 与 `hallucination_iou_thr` 的默认值，并同步更新 README 示例，避免配置、脚本、文档三者继续分叉。

### 论文对齐修正
- 已按论文 Sec. 4.2 将 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 中的 `train_cfg.hallucination_iou_thr` 从 `0.5` 改为 `0.2`，使 Memory Hallucination 的 IoU 重叠阈值与 cross-corruption benchmark 设置一致。

### 种子设置检查
- 已检查 `train.py`、`test.py`、`configs/_base_/default_runtime.py`、当前 TTAOD 配置以及启动脚本。
- 当前仓库没有在训练/测试入口或 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 中显式设置全局随机种子，也没有定义 MMEngine 的 `randomness = dict(seed=..., deterministic=...)`。
- 目前唯一与 seed 直接相关的默认配置是 `configs/_base_/default_runtime.py` 中的 `sampler_seed=dict(type='DistSamplerSeedHook')`，它用于分布式 sampler 的 epoch 级 seed 同步，不等同于固定整个实验的随机性。

### 进一步论文对齐审查
- 结合论文 Sec. 4.2 与 Table 3 继续复核后，确认当前仓库还存在一个容易被忽略但影响很大的偏差：`configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 的默认 backbone 配置仍是 `prompt_type=None`、`num_tokens=0`、`prompt_deep=False`，这意味着如果直接用该配置启动 `train.py`，实际上不会启用论文主方法中的 Visual Prompt Tuning，只会保留 Text Prompt Tuning。
- 论文 Table 3 显示：仅启用 TPT 的平均 AP50 为 45.4，而启用 `TPT + VPT + TTWS + ME + MH` 的完整方法为 56.2，差距较大。因此当前仓库“基础配置默认值”与“论文主方法默认值”仍不一致，只是 `run_tta_train.sh` 通过命令行覆盖把它改回了论文设定。
- 当前仍未对齐的另一处确定项是 `train_cfg.shot_capacity`：基础配置和 README 示例仍是 15，而论文 Sec. 4.2 与 Fig. 5 分析后给出的最终设置是 20；`run_tta_train.sh` 已覆盖为 20，但配置与文档尚未收齐。

### 独立论文/官方实现对齐复查
- 本轮按用户要求忽略既有结论，重新读取 arXiv 论文、官方仓库 `gaoyingjay/TTAOD-F` 的 `main` 分支和本地关键代码进行核对。
- 官方仓库 HEAD 为 `16635fd26505d38cab79ef6dccc4efb2d24fca02`；与本地 HEAD 相比，核心差异集中在 `README.md`、`configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`、`mmdet/engine/runner/ttaod_loop.py`，`swin.py`、`grounding_dino.py`、`ttaod_soft_teacher.py`、`mean_teacher_hook.py` 与官方仓库当前版本无差异。
- 已确认本地已比官方仓库更接近论文的部分：启用 `detector.text_prompt=True`，解冻 `tunable_linear`，以 `lr_mult=0.1` 使 text prompt 学习率为 0.02；使用 OGC/Cap4M 权重相关配置；IDM cache 已保存 `image/feature/score`；已补充 Memory Hallucination，并使用最多 3 个实例、Beta 混合、随机缩放、最多 10 次重选、配置中 `thIoU=0.2`。
- 仍未严格对齐论文主实验的确定项：基础配置默认 `prompt_type=None`、`num_tokens=0`、`prompt_deep=False`，直接运行配置不会启用 VPT/TTWS；基础配置与 README 仍写 `shot_capacity=15`，论文 Sec. 4.2/Fig. 5 设置为 20；`TTAODLoop.__init__` 的 `hallucination_iou_thr` 默认值仍是 0.5，虽然当前配置覆盖为 0.2。
- 风险项：本地只覆盖了 OGC 配置中最关键的 `add_pooling_layer=True`、`contrastive_cfg=dict(max_text_len=256)`、`with_cp=False`、`convert_weights=False`、`init_cfg=None`，没有完整继承官方 `grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py`；其中 `bbox_head.num_classes`、`encoder.num_cp` 等差异目前看不直接影响 GroundingDINOHead 的文本 token 分类路径，但仍建议后续做完整 MMEngine Config dump 和权重加载日志比对。
- 本轮执行了 `python -m py_compile` 检查 `ttaod_loop.py`、TTAOD 配置、`grounding_dino.py`、`swin.py`，语法检查通过；编译产生的 Python 3.13 `__pycache__` 已清理/恢复。

### 论文对齐与 pipeline 修正
- 已将 TTAOD 基础配置默认改为论文主方法设置：`prompt_type='prepend'`、`num_tokens=10`、`prompt_deep=True`、默认读取 `prompt_init/voc-c/prompt_voc_gaussian_noise.pth`，避免直接运行配置时只启用 TPT 而不启用 VPT/TTWS。
- 已将 `train_cfg.shot_capacity` 从 15 改为 20，并同步 README；已将 `TTAODLoop.__init__` 中 Memory Hallucination 的默认 `hallucination_iou_thr` 从 0.5 改为 0.2。
- 已进一步对齐 OGC/Cap4M 配置：`detector.encoder.num_cp=-1`、`detector.bbox_head.num_classes=80`，并保留 TTA 训练必须使用的 detector `train_cfg` assigner。
- 已修正 TTWS 初始化和默认 VPT 的冲突：`SwinTransformer` 在 `TTWS_init=True` 时不再加载尚未生成的 `TTWS_file`，forward 时也临时禁用 prompt token，确保 TTWS 使用原始视觉 token 均值生成 warm-start prompt。
- 已将 TTAOD 的 `LoadImageFromFile` 后端改为 `imdecode_backend='pillow'`，与 OGC 测试 pipeline 对齐，同时保留 `homography_matrix` meta 以支持 teacher 到 student 的伪框投影。
- pipeline 静态审查结论：`LoadEmptyAnnotations` 仅用于 unsupervised teacher/student 分支以避免 GT 泄漏；test 分支仍通过 `LoadAnnotations` 读取评估标注；`return_classes=True` 与 `text/custom_entities` meta 可供 GroundingDINO 生成 text prompt；hallucinated pseudo labels 只需写入 `bboxes/labels/scores`，`GroundingDINO.loss` 会在训练时根据 labels 和 text 重新生成 `positive_maps`。
- 已修复两个脚本的运行问题：创建 `logs` 目录，并在 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 缺失或非法时设为 1；`run_ttws_init.sh` 显式覆盖为无 prompt 初始化模式。
- 已执行无 pycache 写入的 Python `compile()` 语法检查，覆盖 TTAOD 配置、`ttaod_loop.py`、`swin.py`，检查通过；本机缺少 `mmengine`，无法在 Windows 环境解析完整配置或做训练 dry run；本机也没有 `bash`，无法执行 `bash -n`。

### TTWS prompt 初始化校验
- 已在 `mmdet/models/backbones/swin.py` 中新增 TTWS prompt 字典校验：保存前和训练加载时都会检查 5 个必需键、精确 shape、tensor 类型、无 NaN/Inf、非全零；保存时会将 prompt 转为 CPU tensor，避免后续 `torch.load(..., map_location='cpu')` 兼容性问题。
- 已在 `run_ttws_init.sh` 中追加初始化后独立校验：`test.py` 生成 prompt 文件成功后，脚本会重新加载 `PROMPT_FILE`，检查 Swin-T 期望 shape：`(1,1,96)`、`(1,1,96)`、`(2,1,192)`、`(6,1,384)`、`(2,1,768)`；不通过会直接退出并报错。
- 已执行 Python `compile()` 语法检查，`swin.py` 通过；`git diff --check` 通过。本机缺少 `torch`，因此未能本地实际加载 `.pth` 或运行 prompt 初始化。

### OOM 日志诊断与显存修复
- 用户提供 `C:/Users/33168/Downloads/run_tta_train_gaussian_noise (3).log`，OOM 发生在 student 训练阶段的 GroundingDINO encoder FFN：`grounding_dino_layers.py:246 -> detr_layers.py:235 -> FFN Linear`，报错时已分配约 19.76GiB、reserved 约 20.95GiB。
- 日志中的实际配置为 `model.detector.backbone.with_cp=False`、`model.detector.encoder.num_cp=-1`，这是最近为贴近 OGC 配置而关闭的 activation checkpointing；该设置会显著提高训练激活显存，且不影响 OGC 权重 key 兼容性。
- 已将 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 改回训练省显存设置：`detector.encoder.num_cp=6`、`detector.backbone.with_cp=True`，保留 `convert_weights=False`、`init_cfg=None`、OGC contrastive/head 配置。
- 已在 `mmdet/engine/runner/ttaod_loop.py` 的 test phase `val_step` 外增加 `torch.no_grad()`，避免训练循环中每 iter 的评估预测构建和保留不必要计算图。
- 已执行 Python `compile()` 语法检查，覆盖 TTAOD 配置、`ttaod_loop.py`、`swin.py`，通过；`git diff --check` 通过。本机缺少服务器依赖，未能本地复跑训练显存。

## 2026-05-09

### Memory Hallucination 实现审查
- 当前任务：详细审查本地 Memory Hallucination 实现是否符合论文与 GitHub issue #3 的要求。
- 已读取 issue #3：用户指出公开版缺少 Memory Hallucination、`thIoU=0.2` 和 crop paste pipeline；作者 `gaoyingjay` 于 2026-05-06 回复确认公开版当时不包含 MH，并建议保存 cropped images 后与 test images mixing 实现。
- 已核对论文要点：IDM 应保存每类高质量伪标签实例，包含实例图像、DINOv2 特征与分数；MH 针对无可用伪标签的 negative images，从 IDM 随机采样高质量实例并粘贴生成 positive hallucinated images；cross-corruption 主设置包含 `thpl=0.3`、`thme=0.3`、`thIoU=0.2`、`|Qc|max=20`、Pascal-C 的 `alpha=5.0`、`beta=5.0`。
- 本地 `mmdet/engine/runner/ttaod_loop.py` 已实现 issue #3 列出的主要缺口：`IDM_cache` 条目保存 `feature/score/image`；从 IDM 随机采样 crop；对无可用伪标签的 `unsup_student` 图像进行粘贴；使用 `hallucination_iou_thr` 拒绝重叠位置；生成 `bboxes/labels/scores` 后进入 adaptation。
- 本地配置 `configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py` 已设置 `shot_capacity=20`、`thre_me=0.3`、`memory_hallucination=True`、`hallucination_iou_thr=0.2`、最多 3 个实例、最多 10 次重试，并已将 text prompt 学习率通过 `tunable_linear: lr_mult=0.1` 调到全局 0.2 的 0.1 倍。
- 结论：当前本地实现已经不是官方 issue 中描述的缺失状态，功能路径与论文 Memory Hallucination 的文字描述基本对齐。
- 仍需注意的偏差/风险：
  - 论文没有公开足够细的实现代码；当前的 `hallucination_beta=1.0`、`hallucination_scale_range=(0.5, 1.5)`、矩形 crop paste、按类别先均匀采样再按实例采样，属于合理实现选择，但无法证明与作者私有实现逐行一致。
  - IDM 在当前 batch teacher 伪标签生成后立即更新，然后再对同一 batch 的 negative `unsup_student` 样本做 MH；这可能允许同一 batch 内其他图像的 crop 被用于 hallucination，而论文措辞更偏向 previous test samples。
  - IoU 检查目前只检查新粘贴实例之间的重叠；negative image 本身没有可靠伪标签，因此无法避免贴到真实但未检出的物体上。这与论文“negative images”场景可接受，但不是更强的无遮挡保证。
  - crop 来自原图 RGB/PIL，粘贴到强增强后的 student tensor；代码根据 `bgr_to_rgb` 尝试做通道对齐。由于 pipeline 当前使用 `imdecode_backend='pillow'`，后续最好在服务器上抽样可视化 hallucinated tensor，确认颜色没有通道反转。
- 已执行 `python -B -m py_compile mmdet/engine/runner/ttaod_loop.py configs/mm_grounding_dino/ttaod/ttaod_grounding_dino_swin-t_voc-c.py`，语法检查通过且未生成 pycache。

### Memory Hallucination 可视化调试开关
- 用户要求提供服务器上可视化 hallucinated student tensor 的命令；由于此前代码没有保存开关，已在 `mmdet/engine/runner/ttaod_loop.py` 中新增环境变量调试能力。
- 新增 `VIS_HALLUCINATION_DIR`：设置后，只有发生 Memory Hallucination 的样本会保存调试图；默认不设置时不影响训练。
- 新增 `VIS_HALLUCINATION_MAX`：限制最多保存多少个 hallucinated 样本，默认 8，避免长训练写出过多图片。
- 每个样本保存 3 张图：`*_before_raw.png`、`*_after_raw.png`、`*_after_model_rgb.png`。其中 `after_*` 带红色 hallucinated bbox 与类别/分数；`after_model_rgb` 会按当前 data preprocessor 的 `bgr_to_rgb` 设置做通道交换，便于检查 RGB/BGR 是否反转。
- 推荐服务器命令：
  - `VIS_HALLUCINATION_DIR=debug_hallucination VIS_HALLUCINATION_MAX=12 bash run_tta_train.sh`
- 已执行 `python -m py_compile mmdet/engine/runner/ttaod_loop.py` 和 `git diff --check`，检查通过；生成的临时 `__pycache__` 已清理。
