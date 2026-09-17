# ACT 模型实现记录

日期：2026-09-16。用户本轮明确要求实现模型剩余部分。
ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）模型及损失位于 `src/mini_wam/models/act.py`。

## 配置与数据流

默认配置：隐藏维度 256，8 个注意力头，4 层策略编码器、4 层潜变量编码器、1 层已有策略解码器，潜变量维度 32，动作块长度 16。
默认模型实测参数量为 **18,744,834**。这些是本项目配置，不代表原论文默认配置或逐项复现。

- 视觉：ResNet-18（Residual Network，残差网络）保留空间特征，`[B,3,96,96] → [B,512,3,3] → [B,9,256]`；添加固定二维正弦/余弦位置编码。
- 状态：二维位置 `[B,2] → [B,1,256]`，修正原代码三维输入。
- CVAE（Conditional Variational Autoencoder，条件变分自编码器）：汇总 token、位置和动作序列，加固定时间编码；4 层编码后输出 `mu/logvar`，重参数化采样 `[B,32]`。
- 填充动作在投影前清零，并在每层注意力中作为键被屏蔽；`True` 表示有效动作。每个样本必须至少有一个有效动作。
- 视觉、状态、潜变量分别添加可学习类型向量，拼接为 `[B,11,256]`；策略编码器接已有可学习动作查询解码器，最后线性投影到 `[B,16,2]`。
- 策略编码器已补齐真正多头注意力、输出投影、残差和层归一化。解码器保持已有单层实现及可学习动作查询。
- 评估模式不调用潜变量编码器、不采样，固定使用 `z=0`；传入专家动作会报错，防止误用训练路径。

## 使用接口

```python
import torch
from mini_wam.models.act import ActionPolicy, act_loss

model = ActionPolicy(pretrained=True)
model.train()
output = model(image, position, actions, valid_mask)
losses = act_loss(
    output["actions"], actions, valid_mask,
    output["mu"], output["logvar"], beta=10.0,
)
losses["loss"].backward()

model.eval()
with torch.no_grad():
    predicted_actions = model(image, position)
```

上例由调用方提供批次张量。
图像由调用方做 ImageNet 均值/标准差预处理；位置和动作使用全量示范的新统计标准化。
预测保持标准化坐标，执行前必须反归一化并裁剪环境范围。
`pretrained=True` 使用 torchvision 的 ImageNet 权重，未缓存时需要下载；本次验证使用本地已有权重。

损失返回 `loss`、`action_l1`、`kl`，均保留计算图。
L1 按有效动作坐标数量平均；KL（Kullback–Leibler）散度按潜变量维度求和、批次平均。
默认 `beta=10.0` 是可调整接口默认值，正式训练前仍需配置并登记。
推理输出张量；训练输出包含动作和后验参数的字典。

## 本地验证

环境：项目 `.venv`，PyTorch 2.11.0，torchvision 0.26.0。
运行 `.venv/bin/python -m pytest -q tests/test_act.py`：**12 passed**。

覆盖：编码器和解码器与 PyTorch 参考层的输出/梯度对比；空间网格；填充动作的非有限值隔离和零梯度；有效动作顺序影响后验；损失手算对照；整网有限梯度和优化器更新；推理零潜变量、不消耗随机数、保存重载逐元素一致。

另外用默认完整配置、预训练视觉骨干、批大小 1 运行合成输入：训练和推理输出均为 `[1,16,2]`，所有可训练参数获得有限梯度。
这些检查验证模型实现，不是示范数据过拟合、正式训练、设备吞吐或闭环控制效果的证据。
全量数据适配、独立训练/恢复入口和随机场景闭环评测仍待实现。

全仓 `.venv/bin/python -m pytest -q`：**67 passed, 4 skipped, 2 failed**。
失败为 `test_action_only_mode_skips_future_without_changing_used_fields` 和 `test_cached_windows_exact_and_samples_cannot_mutate_cache`。
任务开始前已有的 `dataset.py` 工作区修改删除了 `include_future_observations` 参数/分支及图像缓存读取路径；两项失败涉及这些旧数据层接口。本轮未修改该文件。
