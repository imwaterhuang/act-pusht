# ACT 接口与实验规格

ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）适配 Push-T。正式训练配置见 [act_seed0.yaml](configs/act_seed0.yaml)，训练完成证据见 [PROGRESS.md](PROGRESS.md)。

## 输入与输出

- 当前图像 `[B,3,96,96]`；当前位置 `[B,2]`。
- 预测动作 `[B,16,2]`，二维绝对目标坐标，推理后反归一化并裁剪到环境坐标范围。
- 动作有效掩码 `[B,16]`；尾部填充不参加损失与后验注意力。
- `B` 为批大小。执行前 4 步后获取新观测，不跨回合构造训练窗口。

## 模型与损失

ResNet-18（Residual Network，残差网络）空间视觉特征、位置编码、固定时间位置编码与 Transformer 编解码器输出动作块。训练使用 CVAE（Conditional Variational Autoencoder，条件变分自编码器）后验编码有效专家动作并采样潜变量；推理固定 `z=0`。

损失为有效动作 L1 重建损失与 `beta=10` 的 KL（Kullback–Leibler）散度之和。模型核心见 [act.py](src/mini_wam/models/act.py)。

## 数据

`lerobot/pusht_image`，全部 206 回合、25,650 帧、25,444 个有效动作窗口参与训练。均值和标准差按全量有效原始记录统计；不按重叠窗口重复计数。图像使用预训练视觉骨干对应的归一化。

检查点附带本次归一化元数据，训练与推理一致。无示范验证划分；检查点用随机模拟开发场景选取。

## 训练与恢复

种子 0，批大小 64，40,000 步，学习率 0.0001，预热 500 步，梯度范数裁剪 1.0。目标运行设备为 NVIDIA L4。每 1,000 步保存、每 5,000 步评测。

训练恢复保存模型、优化器、调度器、随机状态及已消费采样位置；模型发布权重不含优化器状态，不作为恢复权重。精确恢复只在已验证的软件与设备环境内作承诺。

## 阈值与选点

发布评测默认 `coverage > 0.87`；每次运行把实际阈值写入结果。正式历史训练的检查点选择保留 `>0.95`，详见 [EVALUATION.md](EVALUATION.md)。交互演示按 87% 显示达到目标，但为观察后续行为而持续运行。

## 参考

- [ACT 项目与论文入口](https://tonyzhaozh.github.io/aloha/)
- [ACT 作者实现](https://github.com/tonyzhaozh/act)
- [Push-T 数据](https://huggingface.co/datasets/lerobot/pusht_image)
