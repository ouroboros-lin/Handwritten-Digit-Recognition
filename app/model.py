# -*- coding: utf-8 -*-
"""手写数字识别 CNN 网络定义。

本模块被「训练脚本 train.py」和「后端推理 app/main.py」**同时引用**，
保证训练与推理使用的是同一份网络结构，避免两边各写一份导致权重加载失败。

网络结构（Conv-Pool-Conv-Pool-FC-FC，输入 1×28×28，输出 10 类）
------------------------------------------------------------------
    输入                                            [N, 1, 28, 28]
    Conv2d(1 -> 32, 3x3, padding=1) + ReLU          [N, 32, 28, 28]
    MaxPool2d(2)                                    [N, 32, 14, 14]
    Conv2d(32 -> 64, 3x3, padding=1) + ReLU         [N, 64, 14, 14]
    MaxPool2d(2)                                    [N, 64,  7,  7]
    Flatten                                         [N, 3136]
    Linear(3136 -> 128) + ReLU + Dropout(0.25)      [N, 128]
    Linear(128 -> 10)                               [N, 10]  ← logits

padding=1 的作用：3×3 卷积不改变长宽，让每层特征图尺寸只由池化决定，尺寸推导直观。

关于 dropout 的取舍
-------------------
- 保留（默认 0.25）：MNIST 上小 CNN 极易过拟合（不加 dropout 时 5 个 epoch 就能把
  训练集准确率刷到 99.7% 以上），dropout 能轻微改善泛化，也让面试有可讲的设计点。
- 代价：收敛略慢，且训练集准确率会**明显低于**不加 dropout 的情形——这是正常现象，
  不代表模型变差。实测 5 个 epoch 仍可稳定达到 99% 以上的测试准确率。
- 若不需要，把构造参数 dropout 传 0.0 即可。

关于 forward 的返回值
---------------------
forward 返回的是 **logits（未归一化的分数）**，不是概率：
- 训练时配合 ``nn.CrossEntropyLoss``（内部自带 log_softmax），无需手动 softmax；
- 推理时由调用方自行 softmax 得到概率（见 app/main.py）。

注意：不要在 forward 里直接加 softmax——那会让 CrossEntropyLoss 变成"双重 softmax"，
训练会明显变慢甚至不收敛。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 全局约定：这些常量被训练脚本与后端推理共同依赖，改动需两边同步
# ---------------------------------------------------------------------------
NUM_CLASSES = 10            # 输出类别数：数字 0-9
INPUT_SHAPE = (1, 28, 28)   # 输入张量形状（通道, 高, 宽），与 MNIST 一致
DROPOUT = 0.25              # 默认 dropout 概率
MODEL_PATH = "models/mnist_cnn.pt"  # 默认权重保存路径（相对项目根目录）


class MnistCNN(nn.Module):
    """用于 MNIST 手写数字识别的卷积神经网络。

    结构刻意保持简洁可读（两个卷积块 + 两个全连接层），
    参数量约 0.42M（实测 421,642），CPU 上单张图片推理耗时在毫秒级。
    """

    def __init__(self, dropout: float = DROPOUT) -> None:
        """初始化网络层。

        Args:
            dropout: 全连接层之间的 dropout 概率，0 表示不使用 dropout。
        """
        super().__init__()

        # ---- 特征提取部分：卷积 + 池化 ----
        self.features = nn.Sequential(
            # 第 1 个卷积块：1 -> 32 通道，尺寸 28x28 -> 14x14
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 第 2 个卷积块：32 -> 64 通道，尺寸 14x14 -> 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ---- 分类部分：展平 + 全连接 ----
        # 展平后维度 = 64 通道 * 7 * 7 = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=NUM_CLASSES),
        )

        # 权重初始化：卷积层用 Kaiming，全连接层用截断正态。
        # 相比 PyTorch 默认初始化，这里只是让训练初期更稳定，非必需但无害。
        self._init_weights()

    def _init_weights(self) -> None:
        """对卷积层与全连接层做初始化。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``[N, 1, 28, 28]``，取值范围 ``[0, 1]``
               （训练与推理都只做 /255，不做 mean/std 标准化，两端必须一致）。

        Returns:
            logits，形状 ``[N, 10]``，**未经过 softmax**。
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 权重读写工具：训练脚本与后端共用，避免两边各写一份 torch.save/load
# ---------------------------------------------------------------------------
def build_model(dropout: float = DROPOUT) -> MnistCNN:
    """构建一个未加载权重的模型实例。"""
    return MnistCNN(dropout=dropout)


def save_model(model: nn.Module, path: str = MODEL_PATH) -> None:
    """保存模型权重（只保存 state_dict，不保存整个模型对象）。

    只存 state_dict 的好处：文件小、不绑定具体代码路径，加载时必须先实例化
    ``MnistCNN``，能强制保证结构在训练端和推理端一致。

    Args:
        model: 待保存的模型。
        path: 保存路径，父目录不存在会自动创建。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str = MODEL_PATH, device: torch.device | str = "cpu") -> MnistCNN:
    """加载模型权重并返回处于 eval 模式的模型。

    Args:
        path: 权重文件路径。
        device: 加载到的设备。

    Returns:
        已加载权重、``eval()`` 状态、位于指定 device 上的 ``MnistCNN``。

    Raises:
        FileNotFoundError: 权重文件不存在时抛出，并给出明确的处理提示。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到模型文件：{os.path.abspath(path)}\n"
            f"请先在项目根目录运行：python train.py"
        )

    model = MnistCNN()
    # weights_only=True 是 PyTorch 2.6+ 的默认值，这里显式写出来表明意图：
    # 只反序列化张量，不允许 pickle 任意对象，加载外部权重时更安全。
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # 便于在 PyCharm 里直接右键运行本文件，快速自检网络结构是否正确。
    # 预期输出：
    #   output shape : torch.Size([2, 10])
    #   params       : 421,642
    #   prob sum     : [1.0, 1.0]
    _model = MnistCNN()
    _x = torch.randn(2, *INPUT_SHAPE)
    _logits = _model(_x)
    print("output shape :", _logits.shape)
    print("params       :", f"{sum(p.numel() for p in _model.parameters()):,}")
    print("prob sum     :", torch.softmax(_logits, dim=1).sum(dim=1).tolist())
