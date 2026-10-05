# -*- coding: utf-8 -*-
"""MNIST 手写数字识别 —— CNN 训练脚本。

功能
----
1. 自动下载 MNIST 数据集到**项目内** ``./data/`` 目录（首次运行需联网，4 个文件共约 12 MB）；
2. 训练一个结构简洁的 CNN，网络定义见 ``app/model.py``（训练与推理共用同一份结构）；
3. 每个 epoch 记录训练集 / 测试集的 loss 与 accuracy；
4. 保存「测试集准确率最高的那一轮」权重到 ``models/mnist_cnn.pt``（只存最佳，不存最后一轮）；
5. 在 ``reports/`` 下产出四件套：

   ==========================  ==========================================
   training_curve.png          双子图：左 loss 曲线，右 accuracy 曲线，train/test 同图对比
   confusion_matrix.png        带数值标注的 10x10 混淆矩阵
   classification_report.txt   sklearn 的 precision / recall / f1-score 分类报告
   test_predictions.png        4x5=20 张测试样本，标题标注「预测 / 真实」
   ==========================  ==========================================

运行
----
    python train.py

设计说明（面试可讲）
--------------------
- **本机无 CUDA**，脚本自动回落到 CPU；实测 24 逻辑核（torch 默认 12 线程）机器上
  5 个 epoch 全量 MNIST 约 1.5 分钟，核心数少的机器会明显更慢。
- **归一化约定**：图像只做 ``ToTensor()``（像素缩放到 ``[0, 1]``），
  **不额外做 ``(x - 0.1307) / 0.3081`` 的 MNIST 标准化**。
  这样后端预处理只需 ``/255`` 就能与训练分布完全一致，前后端约定单一、好讲解。
  若改成带标准化的版本，后端 ``app/preprocess.py`` 必须同步改动，否则精度会掉。
- **测试集兼作验证集**：按需求「保存测试集准确率最高的那一轮」实现。
  严格工程做法应另切一份验证集，这里为保持代码简单、结果可复现而省略，
  面试时可主动说明这一取舍。
- 刻意**不加**数据增强、early stopping、学习率调度，保持代码简单可读。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# matplotlib 必须在 pyplot 之前指定无界面后端 Agg，
# 否则在服务器 / 无显示环境（或 PyCharm 的非交互运行）下可能报错或卡住。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (必须在 matplotlib.use 之后导入)
from matplotlib import font_manager  # noqa: E402

from app.model import NUM_CLASSES, MnistCNN, load_model, save_model  # noqa: E402

# ===========================================================================
# 训练超参数（集中在此处，便于调参；改动后重跑 python train.py 即可）
# ===========================================================================
EPOCHS = 5                     # 训练轮数。CPU 上 5 轮已足够到 99% 左右
BATCH_SIZE = 64                # 批大小
LR = 1e-3                      # 学习率，配合 Adam 使用
SEED = 42                      # 随机种子，固定后结果可复现
DATA_DIR = "data"              # MNIST 下载目录（项目内，便于 Docker 打包）
MODEL_PATH = "models/mnist_cnn.pt"   # 最佳权重保存路径
REPORTS_DIR = "reports"        # 训练报告输出目录

# DataLoader 工作进程数。Windows 下多进程会重复导入主模块、容易出问题，
# 且本任务数据量小，用 0（主进程加载）更稳。
NUM_WORKERS = 0

# 测试集准确率的展示目标，低于此值会在结尾给出提示
TARGET_ACCURACY = 0.98


# ===========================================================================
# 基础工具
# ===========================================================================
def set_seed(seed: int = SEED) -> None:
    """固定所有随机源，保证多次运行结果一致（可复现）。

    覆盖 random / numpy / torch 三处随机源；DataLoader 的 shuffle 在没有传
    generator 时会使用 torch 的全局随机数，因此也被这个种子统一控制。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 本机无 CUDA，调用无副作用

    # cuDNN 是 CUDA 上的加速库，CPU 环境下这两行不生效，
    # 写上是为了脚本拿到 GPU 机器上跑时行为依然确定。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """自动选择设备：有 CUDA 用 CUDA，否则用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_chinese_font() -> str:
    """为 matplotlib 挑选一个可用的中文字体，避免图表标题/图例出现方块乱码。

    做法：按候选列表依次检查系统已安装字体，命中第一个就设为默认无衬线字体。
    Windows 上通常命中 "Microsoft YaHei"（微软雅黑）。

    Returns:
        实际选用的字体名；若一个都没找到则返回 "DejaVu Sans"（不含中文字形，
        此时图表中的中文会显示为方块，但不会报错——见文末常见问题）。
    """
    candidates = [
        "Microsoft YaHei",   # Windows 微软雅黑
        "SimHei",            # Windows 黑体
        "SimSun",            # Windows 宋体
        "Noto Sans CJK SC",  # Linux 常见
        "Source Han Sans SC",
        "PingFang SC",       # macOS
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False   # 负号不能显示成方块
            return name

    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def build_dataloaders() -> tuple[DataLoader, DataLoader, datasets.MNIST]:
    """构建 MNIST 训练 / 测试 DataLoader。

    归一化说明见模块 docstring：只做 ToTensor()，把 uint8 的 [0,255] 变成
    float32 的 [0,1]，并在同时把形状从 HWC 变成 CHW（符合 PyTorch 卷积输入要求）。

    Returns:
        (train_loader, test_loader, test_dataset)
        额外返回 test_dataset 是为了后面画 20 张预测样例时能拿到原始图像和标签。
    """
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,               # 训练集打乱，避免按类别顺序喂数据
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,              # 测试集不需要打乱
        num_workers=NUM_WORKERS,
    )
    return train_loader, test_loader, test_set


# ===========================================================================
# 训练与评估
# ===========================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch。

    Returns:
        (平均 loss, 准确率)，均为按样本数加权的整体值。
    """
    model.train()   # 切到训练模式：启用 dropout、BatchNorm 用当前批统计量
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1) 清空上一轮的梯度。set_to_none=True 比置零更省内存，是 PyTorch 2.x 推荐写法
        optimizer.zero_grad(set_to_none=True)

        # 2) 前向：拿到 logits（未过 softmax）
        logits = model(images)

        # 3) 计算损失：CrossEntropyLoss 内部自带 log_softmax，输入 logits 即可
        loss = criterion(logits, labels)

        # 4) 反向传播 + 更新参数
        loss.backward()
        optimizer.step()

        # ---- 统计 ----
        total_loss += loss.item() * images.size(0)     # 乘回批大小，最后再除以总样本数
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()   # 评估不需要梯度，关掉可省内存并加速
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """在给定数据集上评估模型。

    Returns:
        (平均 loss, 准确率, 预测标签数组, 真实标签数组)
        后两项用于生成混淆矩阵和分类报告。
    """
    model.eval()    # 切到评估模式：关闭 dropout，用确定的推理行为
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # 收集到 CPU 上，避免长时间占用显存
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, y_pred, y_true


# ===========================================================================
# 报告产出
# ===========================================================================
def save_training_curve(history: dict[str, list[float]], save_path: str) -> None:
    """绘制并保存训练曲线（左：loss，右：accuracy，train/test 同图对比）。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- 左图：Loss 曲线 ----
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=1.8, label="训练集")
    ax.plot(epochs, history["test_loss"], marker="s", linewidth=1.8, label="测试集")
    ax.set_title("损失曲线 (Loss)")
    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()

    # ---- 右图：Accuracy 曲线（换算成百分比，更直观）----
    ax = axes[1]
    train_acc = [acc * 100 for acc in history["train_acc"]]
    test_acc = [acc * 100 for acc in history["test_acc"]]
    ax.plot(epochs, train_acc, marker="o", linewidth=1.8, label="训练集")
    ax.plot(epochs, test_acc, marker="s", linewidth=1.8, label="测试集")
    ax.set_title("准确率曲线 (Accuracy)")
    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("准确率 (%)")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()

    # 在右图末尾标注最终测试准确率，方便截图展示
    if test_acc:
        ax.annotate(
            f"{test_acc[-1]:.2f}%",
            xy=(epochs[-1], test_acc[-1]),
            xytext=(-42, 12),
            textcoords="offset points",
            fontsize=10,
            color="tab:orange",
        )

    fig.suptitle("MNIST CNN 训练过程", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str
) -> np.ndarray:
    """绘制并保存带数值标注的混淆矩阵。

    Args:
        y_true: 真实标签数组。
        y_pred: 预测标签数组。
        save_path: 输出图片路径。

    Returns:
        计算出的混淆矩阵（行=真实标签，列=预测标签）。
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(cm, cmap="Blues")

    # 在每个格子里写上数量；底色深的格子用白字，否则黑字，保证可读
    threshold = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > threshold else "black",
            )

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("预测标签 (Predicted)")
    ax.set_ylabel("真实标签 (True)")
    ax.set_title("混淆矩阵 (Confusion Matrix)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="样本数")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return cm


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    extra_header: str = "",
) -> str:
    """生成并保存 sklearn 分类报告（precision / recall / f1-score）。

    Returns:
        报告文本，便于同时打印到控制台。
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[str(i) for i in range(NUM_CLASSES)],
        digits=4,
        zero_division=0,
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("MNIST CNN 分类报告 (classification_report)\n")
        f.write("=" * 62 + "\n")
        if extra_header:
            f.write(extra_header)
            f.write("=" * 62 + "\n")
        f.write(report)
        f.write("\n")

    return report


def save_test_predictions(
    model: nn.Module,
    dataset: datasets.MNIST,
    device: torch.device,
    save_path: str,
    num_samples: int = 20,
    cols: int = 5,
) -> None:
    """抽样若干测试样本，画成 4x5 网格，标题标注「预测 / 真实」与置信度。

    绿色标题表示预测正确，红色表示预测错误——一眼就能看出模型的错误案例。
    """
    rng = np.random.default_rng(SEED)   # 独立随机源，保证每次抽样同一批图
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    # 逐张取出后堆成一个 batch，一次性送进模型（比 for 循环逐张推理快得多）
    images = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
    labels = np.array([int(dataset[int(i)][1]) for i in indices])

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()

    for k, ax in enumerate(axes):
        if k >= num_samples:
            ax.axis("off")
            continue

        ax.imshow(images[k].cpu().squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        is_correct = preds[k] == labels[k]
        ax.set_title(
            f"预测:{preds[k]} / 真实:{labels[k]}\n{probs[k].max() * 100:.1f}%",
            fontsize=10,
            color="green" if is_correct else "red",
        )
        ax.axis("off")

    fig.suptitle("测试集预测样例（绿=正确，红=错误）", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> None:
    """训练主流程：准备 -> 训练 -> 选最佳 -> 评估 -> 出报告。"""
    font_name = setup_chinese_font()
    set_seed(SEED)

    # ---- 目录准备（幂等，已存在不会报错）----
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    device = get_device()

    print("=" * 66)
    print("MNIST CNN training")
    print("=" * 66)
    print(f"  device        : {device}")
    print(f"  epochs        : {EPOCHS}")
    print(f"  batch_size    : {BATCH_SIZE}")
    print(f"  lr            : {LR} (Adam)")
    print(f"  seed          : {SEED}")
    print(f"  data_dir      : {os.path.abspath(DATA_DIR)}")
    print(f"  model_path    : {MODEL_PATH}")
    print(f"  reports_dir   : {REPORTS_DIR}")
    print(f"  chinese_font  : {font_name}")
    print("=" * 66)

    # ---- 数据 ----
    print("Loading MNIST (首次运行会自动下载到 ./data/) ...")
    train_loader, test_loader, test_dataset = build_dataloaders()
    print(f"  train samples : {len(train_loader.dataset):,}")
    print(f"  test  samples : {len(test_loader.dataset):,}")

    # ---- 模型 / 损失 / 优化器 ----
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  model params  : {num_params:,}")
    print("-" * 66)

    # ---- 训练循环 ----
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
    }
    best_acc = 0.0      # 历史最佳测试准确率
    best_epoch = 0      # 取得最佳准确率的轮次

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc * 100:6.2f}%  |  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc * 100:6.2f}%"
        )

        # 只保存「目前为止最好」的那一轮，避免最后一轮过拟合反而变差
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(model, MODEL_PATH)
            print(f"  -> new best, checkpoint saved to {MODEL_PATH}")

    print("-" * 66)
    print(f"Training finished. best epoch = {best_epoch}, "
          f"best test acc = {best_acc * 100:.2f}%")

    # ---- 重新载入最佳权重，用它来出报告（而不是用最后一轮的模型）----
    print(f"Reloading best checkpoint from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH, device)

    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, device
    )

    # ---- 产出 reports/ 四件套 ----
    curve_path = os.path.join(REPORTS_DIR, "training_curve.png")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    samples_path = os.path.join(REPORTS_DIR, "test_predictions.png")

    print("Generating reports ...")
    save_training_curve(history, curve_path)
    save_confusion_matrix(y_true, y_pred, cm_path)

    header = (
        f"best_epoch      : {best_epoch}\n"
        f"epochs          : {EPOCHS}\n"
        f"batch_size      : {BATCH_SIZE}\n"
        f"optimizer       : Adam (lr={LR})\n"
        f"seed            : {SEED}\n"
        f"device          : {device}\n"
        f"test_loss       : {test_loss:.4f}\n"
        f"test_accuracy   : {test_acc * 100:.2f}%\n"
    )
    report_text = save_classification_report(y_true, y_pred, report_path, header)

    save_test_predictions(model, test_dataset, device, samples_path)

    # ---- 最终输出 ----
    print("=" * 66)
    print("FINAL RESULT")
    print("=" * 66)
    print(f"  best epoch    : {best_epoch}")
    print(f"  test loss     : {test_loss:.4f}")
    print(f"  test accuracy : {test_acc * 100:.2f}%")
    print("-" * 66)
    print("Classification report:")
    print(report_text)
    print("-" * 66)
    print("Artifacts:")
    print(f"  model         : {MODEL_PATH}")
    print(f"  curve         : {curve_path}")
    print(f"  confusion mat : {cm_path}")
    print(f"  report        : {report_path}")
    print(f"  samples       : {samples_path}")

    if test_acc < TARGET_ACCURACY:
        print("-" * 66)
        print(f"  [WARN] test accuracy {test_acc * 100:.2f}% is below "
              f"the target {TARGET_ACCURACY * 100:.0f}%.")
        print("         Try increasing EPOCHS at the top of train.py.")
    else:
        print("-" * 66)
        print(f"  [OK] target accuracy {TARGET_ACCURACY * 100:.0f}% reached.")
    print("=" * 66)


if __name__ == "__main__":
    main()
