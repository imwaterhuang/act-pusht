# 复现指南

所有命令从仓库根目录执行。依赖清单为 `requirements-demo.txt`；推理权重见 [v0.1.0 Release](https://github.com/imwaterhuang/act-pusht/releases/tag/v0.1.0)。

## 无需训练：评测模型

```bash
python scripts/verify_release.py --checkpoint artifacts/studio/models/act_step_30000.pt
python scripts/evaluate_act.py --checkpoint artifacts/studio/models/act_step_30000.pt --scenes reports/act/fresh50_20260918/scenes.json --split review --success-coverage 0.87 --output-dir outputs/review87 --device cpu
```

添加 `--save-videos` 可保存全部回合视频。原训练与评测阈值为 0.95；复查原规则时将 `--success-coverage` 设为 `0.95` 并使用不同输出目录。新设备推理可能与原 L4 的动作轨迹有数值差异。

## 无需模型：重建已发布证据

```bash
python -m pip install -r requirements-evidence.txt
python scripts/build_release_evidence.py
python scripts/verify_release.py
```

脚本使用原始逐步动作和冻结场景，重算全部 150 个回合的 87% 指标；渲染 5 个展示回合，逐步核对覆盖率；从完整训练日志重建曲线。失败时立即报错。不会重新推理或重写原始评测记录。

视频重建依赖相同模拟器行为；视频编码字节可能随编码器版本变化。重新生成媒体后若要发布新版本，应重新生成发布校验清单，不能冒用旧校验值。

## 数据与训练

```bash
python scripts/download_dataset.py
python scripts/train_act.py --config configs/act_smoke.yaml --run-dir runs/act/smoke --device cpu
python scripts/train_act.py --config configs/act_seed0.yaml --run-dir runs/act/seed0 --device cuda
```

全部 206 回合参与训练，固定的 50 个开发场景负责检查点选择。正式配置保留历史 95% 选点规则，以复现实验；发布评测和交互界面使用 87%。训练核心默认保留旧配置的 95% 选点语义。

恢复示例：

```bash
python scripts/train_act.py --config configs/act_seed0.yaml --resume runs/act/seed0/checkpoints/last.pt --run-dir runs/act/seed0 --device cuda
```

也可使用 [Colab 恢复笔记本](../notebooks/act_colab_resume.ipynb)。发布权重只含模型、配置与必要元数据，不能用来恢复优化器；完整恢复需要训练产生的 `last.pt`。

## 本地交互

```bash
python scripts/demo_act.py --checkpoint artifacts/studio/models/act_step_30000.pt --port 7860
```

浏览器访问 `http://127.0.0.1:7860/live`。可暂停、重置和拖动物体；该模式不作为评测，不会达到阈值就终止。

## 测试

```bash
python -m pytest -q
```

没有数据时跳过数据相关测试；没有权重时跳过真实模型适配测试。发布验证报告区分这些条件，不把跳过当作通过。
