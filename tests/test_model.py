# -*- coding: utf-8 -*-
"""模型与训练产物测试：``app/model.py``。

运行：
    pytest tests/test_model.py -v

这里测的是**推理侧真正依赖的那几条性质**，而不是重复验证网络结构本身：
    - 权重文件能加载，且路径不存在时报的是明确的 FileNotFoundError
    - 加载后处于 eval 模式，**Dropout 确实被关掉**（同一个输入两次前向结果一致）
    - 前向输出形状与数值域正确
    - softmax 能给出合法的概率分布
    - 推理是确定的（``torch.no_grad`` + eval 的组合不该有任何随机性）

其中"Dropout 是否真的关掉"这条最容易出事也最难肉眼发现：
如果忘了 ``model.eval()``，同一个手写数字每次识别的结果都会不一样。
下面用**功能验证**（两次前向是否一致）来锁住它，而不是只看 ``model.training`` 标志位。

模型文件不存在时相关用例会 ``skip`` 并提示先运行 ``python train.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.model import (
    INPUT_SHAPE,
    MODEL_PATH,
    NUM_CLASSES,
    MnistCNN,
    build_model,
    load_model,
    save_model,
)

BASE_DIR = Path(__file__).resolve().parent.parent


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture(scope="module")
def trained_model():
    """加载训练好的模型；文件不存在时跳过。"""
    path = Path(MODEL_PATH)
    if not path.is_file():
        pytest.skip(f"模型文件不存在（{MODEL_PATH}），请先运行 python train.py")
    return load_model(MODEL_PATH, torch.device("cpu"))


@pytest.fixture
def sample_input() -> torch.Tensor:
    """一批形状合法的随机输入（取值 [0,1]，与训练端 ToTensor() 的约定一致）。"""
    return torch.rand(2, *INPUT_SHAPE)


# ===========================================================================
# 网络结构
# ===========================================================================
def test_build_model_returns_mnist_cnn():
    """build_model 应返回 MnistCNN 实例。"""
    assert isinstance(build_model(), MnistCNN)


def test_forward_output_shape(sample_input):
    """前向输出必须是 [N, 10]——10 个数字的 logits。"""
    model = MnistCNN()
    output = model(sample_input)

    assert output.shape == (2, NUM_CLASSES)


def test_forward_accepts_single_sample(sample_input):
    """单张图（batch=1）也应能正常前向——接口推理就是这个形状。"""
    model = MnistCNN()
    output = model(sample_input[:1])

    assert output.shape == (1, NUM_CLASSES)


def test_forward_returns_logits_not_probabilities(sample_input):
    """forward 输出的是 logits，**不能**是概率。

    这条很重要：训练用 CrossEntropyLoss（内部自带 log_softmax）。
    若 forward 里也加了 softmax，就成了"双重 softmax"，训练会明显变慢甚至不收敛。
    判据：logits 可以有负值、且绝不满足"每行和为 1"。
    """
    model = MnistCNN()
    with torch.no_grad():
        output = model(sample_input)

    assert output.min() < 0, "输出全是非负数，疑似在 forward 里过了 softmax"
    assert not torch.allclose(output.sum(dim=1), torch.ones(2))


def test_softmax_produces_valid_distribution(sample_input):
    """对 logits 做 softmax 后应得到合法的概率分布（非负、每行和为 1）。"""
    model = MnistCNN()
    with torch.no_grad():
        probabilities = torch.softmax(model(sample_input), dim=1)

    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)


def test_parameter_count_matches_readme():
    """参数量应为 421,642——README 与前端页脚都引用了这个数字，改结构必须同步。"""
    assert sum(p.numel() for p in MnistCNN().parameters()) == 421_642


def test_dropout_zero_makes_training_deterministic(sample_input):
    """dropout=0.0 时，即使在 train 模式下两次前向也应完全一致。

    这里断言的是**行为**而不是"Dropout 模块不存在"，因为行为才要紧：
    实现上 ``dropout=0.0`` 并不会把 Dropout 层整个删掉，而是构造一个 ``p=0`` 的
    Dropout——``nn.Dropout(0.0)`` 在前向时是恒等变换，所以效果上等价于"没有 dropout"。
    （这一点是写这条测试时才确认的，容易想当然地以为层会被移除。）
    """
    model = MnistCNN(dropout=0.0)
    model.train()

    with torch.no_grad():
        first = model(sample_input)
        second = model(sample_input)

    assert torch.equal(first, second), "dropout=0.0 时 train 模式的两次前向仍不一致"


# ===========================================================================
# 权重的保存与加载
# ===========================================================================
def test_save_and_load_roundtrip(tmp_path):
    """保存再加载后，权重应逐位一致——这是"训练与推理共用同一份结构"的底线。"""
    original = MnistCNN()
    path = tmp_path / "roundtrip.pt"

    save_model(original, str(path))
    assert path.is_file()

    restored = load_model(str(path), torch.device("cpu"))

    for (name_a, param_a), (name_b, param_b) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"参数 {name_a} 在保存/加载后发生了变化"


def test_save_model_creates_missing_parent_directory(tmp_path):
    """保存路径的父目录不存在时应自动创建，而不是报错。"""
    nested = tmp_path / "models" / "nested" / "model.pt"
    save_model(MnistCNN(), str(nested))

    assert nested.is_file()


def test_load_model_raises_filenotfound_with_helpful_message():
    """权重文件不存在时，应抛出带**处理建议**的 FileNotFoundError。

    启动服务时模型缺失是高频问题，报错信息必须能直接告诉用户下一步做什么。
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model("models/definitely_not_here.pt", torch.device("cpu"))

    message = str(excinfo.value)
    assert "train.py" in message, "报错信息里应当提示运行 python train.py"


def test_load_model_returns_model_in_eval_mode(trained_model):
    """加载后必须已经是 eval 模式，并位于目标设备上。"""
    assert trained_model.training is False
    assert not any(m.training for m in trained_model.modules())


# ===========================================================================
# eval 模式：Dropout 是否真的关掉（最有价值的一条）
# ===========================================================================
def test_trained_model_dropout_is_disabled(trained_model, sample_input):
    """功能验证：eval 模式下同一个输入两次前向必须得到**完全相同**的输出。

    若 dropout 还在生效，两次结果会不同——这正是"忘了 model.eval() 会导致
    同一张手写图每次识别结果都不一样"的根因。
    """
    with torch.no_grad():
        first = trained_model(sample_input)
        second = trained_model(sample_input)

    assert torch.equal(first, second), "两次前向结果不一致，Dropout 可能没有关闭"


def test_train_mode_does_activate_dropout(sample_input):
    """反向验证：train 模式下 dropout 应当生效，两次前向结果不同。

    这条证明上面那条测试是有意义的——如果网络里根本没有 dropout，
    两条测试都会通过，eval() 的重要性也就无从体现。
    """
    model = MnistCNN()          # 默认 dropout=0.25
    model.train()

    with torch.no_grad():
        first = model(sample_input)
        second = model(sample_input)

    assert not torch.equal(first, second), (
        "train 模式下两次前向结果相同，说明网络里没有生效的 Dropout，"
        "请检查 MnistCNN 的 DROPOUT 默认值"
    )


def test_inference_is_deterministic_across_repeated_calls(trained_model, sample_input):
    """推理应当完全确定：连续 5 次前向的预测结果和概率都一致。"""
    predictions = []
    probabilities = []

    with torch.no_grad():
        for _ in range(5):
            output = trained_model(sample_input)
            predictions.append(output.argmax(dim=1))
            probabilities.append(torch.softmax(output, dim=1))

    for predicted in predictions[1:]:
        assert torch.equal(predicted, predictions[0])
    for probs in probabilities[1:]:
        assert torch.allclose(probs, probabilities[0], atol=0)


# ===========================================================================
# 训练产物
# ===========================================================================
def test_trained_model_gives_high_confidence_on_mnist(trained_model):
    """在真实 MNIST 图像上，模型应当给出高置信度且预测正确。

    这条把"模型文件确实是训练好的、不是随机权重"钉死：
    随机初始化的网络在 10 类上的平均置信度约 10%，不可能达到 0.9 以上。
    """
    dataset_ready = (BASE_DIR / "data" / "MNIST" / "raw" / "t10k-images-idx3-ubyte").is_file()
    if not dataset_ready:
        pytest.skip("MNIST 数据不存在，请先运行 python train.py 完成下载")

    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root=str(BASE_DIR / "data"), train=False, download=False,
        transform=transforms.ToTensor(),   # 与训练端完全相同的变换
    )

    correct = 0
    confidences = []
    with torch.no_grad():
        for index in range(20):
            image, label = dataset[index]
            probabilities = torch.softmax(trained_model(image.unsqueeze(0)), dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)
            confidences.append(float(confidence))
            correct += int(prediction.item() == label)

    assert correct >= 18, f"前 20 张只对了 {correct} 张，模型文件可能不是训练好的权重"
    assert sum(confidences) / len(confidences) > 0.9, "平均置信度过低，疑似随机权重"
