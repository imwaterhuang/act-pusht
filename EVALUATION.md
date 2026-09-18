# ACT 评测与验收

## 发布规则（2026-09-18）

ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）的发布成功定义为：最多 300 步内任意一步 `coverage > 0.87`，首次达到即停止。比较严格使用 `>`。

当前表格由原始逐步轨迹重算，原始评测停止阈值为 0.95。因降低阈值只会提前停止，已有动作前缀足以确定是否达到 0.87；重算不依赖新的模型推理。原轨迹结束覆盖率另列，避免将其当成新阈值下的最终覆盖率。

## 原选点规则

全部 206 条示范用于训练；50 个冻结开发场景按 >95% 成功数选点，同分比较平均最终覆盖率，再同分保留较早检查点。此规则仍用于历史训练配置与恢复。原选中检查点为 30,000 步。

50 个复核场景的种子与状态和开发池分离；它们已用于三检查点比较与发布阈值重算，因此后续实验应称为已使用的复核集，不能再称为未触碰的最终测试集。未对历史项目封存测试做评测。

## 证据

- [训练完成](reports/act/training/completed.json)、[原选点](reports/act/training/selection.json)、[完整训练日志](reports/act/training/train_metrics.csv)。
- [原始 50 场景评测](reports/act/fresh50_20260918/REPORT.md)、[场景清单](reports/act/fresh50_20260918/scenes.json)。
- [发布 87% 统计](reports/act/release87/summary.json)、[视频逐步校验](reports/act/release87/videos.json)。
- [发布验证记录](reports/act/RELEASE_VERIFICATION.md)。

只有一个训练种子。30k/35k/40k 的发布成功率为 64%/62%/62%，不据此宣称检查点间具有可靠优势。旧项目数据范围与场景不同，不作受控架构提升结论。
