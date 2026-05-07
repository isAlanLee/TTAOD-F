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
