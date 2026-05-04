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
