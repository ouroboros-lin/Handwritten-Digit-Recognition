# -*- coding: utf-8 -*-
"""接口测试：``/predict``、``/sample``、``/health``、静态页面。

运行：
    pytest tests/test_api.py -v

特点：
- 用 ``fastapi.testclient.TestClient``，**不需要真的启动 uvicorn**；
- 用 ``with TestClient(app)`` 进入上下文，才能触发 lifespan 完成模型加载；
- 模型文件或 MNIST 数据不存在时，相关用例 ``skip`` 并给出原因，而不是报一堆红。

关于「小字」回归用例的设计（阶段6 的重点）
------------------------------------------
阶段5 曾建议断言「字高 50px 时 confidence < 0.9」。**实测证明这个断言不可靠**：
50px 字高下 108 个错例里有 24 个（22%）置信度 ≥ 0.9，最高到 100%，
写成断言会有约 12% 的概率随机失败。

所以这里改成两条统计上稳健的断言：
1. 小字场景**仍然返回可用的预测值**（接口不该因为不确定就拒答）；
2. 小字场景的**平均置信度显著低于**正常字高——这是可复现的整体性质，不依赖单张图。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MODEL_PATH, SAMPLE_SIZE, app

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MNIST_RAW = DATA_DIR / "MNIST" / "raw"

# 一张 MNIST 图片经过完整链路后，模型应当预测正确。
# 这里允许少量容错，避免模型重训后偶发的单张波动导致测试变红。
SAMPLE_COUNT = 20
MIN_CORRECT = 18

# /sample -> /predict 的往返链路（含 280x280 最近邻放大）实测 40/40 全对，
# 这里仍然留一档余量，抗住模型重训带来的轻微波动。
ROUNDTRIP_COUNT = 5
MIN_ROUNDTRIP_CORRECT = 4

# ---- 小字回归用例的参数（依据阶段5/6 的实测曲线）----
# 字高 50px 时正确率约 46%，100px 时约 97.5%；
# 平均置信度 50px 约 80.3%、100px 约 96.4%，差距约 16 个百分点。
SMALL_SIZE = 50
NORMAL_SIZE = 100
SMALL_SAMPLES_PER_CLASS = 3              # 3 x 10 类 = 30 张，跑得快且足够稳定
MIN_CONFIDENCE_GAP = 0.05                # 只要求 5 个百分点的差距，远小于实测的 16


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture(scope="module")
def client():
    """带 lifespan 的测试客户端（模块级复用，模型只加载一次）。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def require_model(client: TestClient):
    """模型未加载时跳过用例，并说明怎么修。"""
    if client.app.state.model is None:
        pytest.skip("模型未加载，请先运行 python train.py 生成 models/mnist_cnn.pt")


@pytest.fixture(scope="module")
def mnist_testset():
    """加载 MNIST 测试集（不下载）。数据缺失时跳过相关用例。"""
    if not (MNIST_RAW / "t10k-images-idx3-ubyte").is_file():
        pytest.skip("MNIST 数据不存在，请先运行 python train.py 完成下载")

    from torchvision import datasets

    return datasets.MNIST(root=str(DATA_DIR), train=False, download=False)


@pytest.fixture(scope="module")
def synthesized_digits():
    """数字合成工具（复用边界探测脚本，保证画布参数只有一处定义）。

    延迟到 fixture 里导入：``tools.edge_cases`` 会拉起 matplotlib，
    放在模块顶层会拖慢每次 pytest 收集。

    Returns:
        函数 ``(size, per_class) -> list[(data_url, label)]``，
        按指定字高合成每类若干张图。
    """
    from tools.edge_cases import mnist_stroke_mask, paste_on_canvas, to_data_url

    def build(size: int, per_class: int):
        items = []
        for digit in range(10):
            index = _first_index_of_class(digit)
            mask, label = mnist_stroke_mask(index, size)
            canvas = paste_on_canvas(mask, (140, 140))
            items.append((to_data_url(canvas), label))
            if per_class > 1:
                # 同一类多取几张：用后续同类样本，形状更多样
                for extra in range(1, per_class):
                    idx = _nth_index_of_class(digit, extra)
                    mask, label = mnist_stroke_mask(idx, size)
                    items.append((to_data_url(paste_on_canvas(mask, (140, 140))), label))
        return items

    return build


# 每个数字类在 MNIST 测试集里的前若干个下标（延迟填充并缓存，只扫一遍数据集）
_CLASS_INDEX_CACHE: dict[int, list[int]] = {}


def _class_indexes() -> dict[int, list[int]]:
    """返回每个数字类的样本下标缓存，首次调用时扫描 MNIST 测试集填充。"""
    if _CLASS_INDEX_CACHE:
        return _CLASS_INDEX_CACHE

    from torchvision import datasets

    dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    for index in range(len(dataset)):
        label = int(dataset[index][1])
        bucket = _CLASS_INDEX_CACHE.setdefault(label, [])
        if len(bucket) < 5:
            bucket.append(index)
        if sum(len(v) for v in _CLASS_INDEX_CACHE.values()) >= 50:
            break
    return _CLASS_INDEX_CACHE


def _first_index_of_class(digit: int) -> int:
    return _class_indexes()[digit][0]


def _nth_index_of_class(digit: int, n: int) -> int:
    bucket = _class_indexes()[digit]
    return bucket[n % len(bucket)]


# ===========================================================================
# 工具函数
# ===========================================================================
def png_base64(image: Image.Image, with_prefix: bool = True) -> str:
    """把 PIL 图片编码成 base64 PNG，模拟前端 ``canvas.toDataURL()`` 的输出。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if with_prefix else encoded


def solid_image(color: int, size: int = 280) -> Image.Image:
    """生成一张纯色图片，用于模拟空白画布 / 纯白图。"""
    return Image.new("L", (size, size), color=color)


def decode_data_url(data_url: str) -> Image.Image:
    """把 ``data:image/png;base64,...`` 解回 PIL 图片，用于校验接口返回的图片。"""
    assert data_url.startswith("data:image/png;base64,"), "data URL 前缀不正确"
    payload = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload)))


# ===========================================================================
# /predict：空白与非法输入 -> 4xx
# ===========================================================================
def test_blank_canvas_returns_400(client: TestClient, require_model):
    """全黑（空）画布应返回 400，且 detail 明确提示要先写数字。"""
    response = client.post("/predict", json={"image": png_base64(solid_image(0))})

    assert response.status_code == 400, response.text
    assert "未检测到" in response.json()["detail"]


def test_all_white_image_returns_400(client: TestClient, require_model):
    """纯白图片（整幅都是"笔画"）同样应被判为没有数字，而不是硬猜一个结果。"""
    response = client.post("/predict", json={"image": png_base64(solid_image(255))})

    assert response.status_code == 400, response.text
    assert "未检测到" in response.json()["detail"]


@pytest.mark.parametrize(
    "bad_image",
    [
        "这不是base64!!!",                          # 含非法字符
        base64.b64encode(b"hello world").decode(),   # 合法 base64 但不是图片
        "",                                          # 空字符串
    ],
)
def test_invalid_image_returns_4xx(client: TestClient, require_model, bad_image):
    """乱码 / 非图片内容应返回 4xx，而不是 500。"""
    response = client.post("/predict", json={"image": bad_image})

    assert 400 <= response.status_code < 500, response.text
    assert "detail" in response.json()


def test_missing_image_field_returns_422(client: TestClient, require_model):
    """请求体缺少 image 字段时，由 pydantic 直接拦下（422）。"""
    response = client.post("/predict", json={})

    assert response.status_code == 422


# ===========================================================================
# /predict：正常路径
# ===========================================================================
def test_real_mnist_image_predicted_correctly(
    client: TestClient, require_model, mnist_testset
):
    """取 MNIST 测试集的第 0 张真实图片，预测结果应等于真实标签。

    这条用例把「预处理 -> 模型 -> 接口」整条链路一次打通，
    如果预处理与训练端不一致（例如方向搞反、没缩放、没居中），这里必然失败。
    """
    image, label = mnist_testset[0]
    response = client.post("/predict", json={"image": png_base64(image)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] == label, (
        f"预测 {body['prediction']} != 真实标签 {label}，"
        f"概率分布 {body['probabilities']}"
    )


def test_mnist_batch_accuracy(client: TestClient, require_model, mnist_testset):
    """对前 20 张真实图片跑一遍，正确率不低于 90%。

    单张用例容易受模型重训影响，这里再加一条整体性检查：
    真实数据上准确率正常，说明预处理链路整体没跑偏。
    """
    correct = 0
    for index in range(SAMPLE_COUNT):
        image, label = mnist_testset[index]
        response = client.post("/predict", json={"image": png_base64(image)})
        assert response.status_code == 200, response.text
        if response.json()["prediction"] == label:
            correct += 1

    assert correct >= MIN_CORRECT, f"前 {SAMPLE_COUNT} 张只对了 {correct} 张，预处理链路可疑"


def test_response_schema_is_valid(client: TestClient, require_model, mnist_testset):
    """响应结构符合约定：10 个概率、和为 1、confidence 与最大值一致。"""
    image, _ = mnist_testset[1]
    body = client.post("/predict", json={"image": png_base64(image)}).json()

    probabilities = body["probabilities"]
    assert len(probabilities) == 10
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    # 概率四舍五入到 6 位小数，总和允许 1e-4 的偏差
    assert abs(sum(probabilities) - 1.0) < 1e-4
    # confidence 应当就是 probabilities 里最大的那个
    assert body["confidence"] == max(probabilities)
    assert probabilities.index(max(probabilities)) == body["prediction"]


# ===========================================================================
# 小字回归用例（阶段6 新增，替代阶段5 那条不可靠的建议）
# ===========================================================================
def test_small_digit_still_returns_usable_prediction(
    client: TestClient, require_model, mnist_testset, synthesized_digits
):
    """字高 50px 时识别率会明显下降，但接口**仍然要返回合法的预测值**。

    这是产品决策：即使模型不确定，前端也要把预测值显示出来，
    而不是让接口拒答或返回空。断言落在"响应是否可用"上，
    不断言具体预测值——那个在小字场景下本来就不可靠。
    """
    samples = synthesized_digits(SMALL_SIZE, per_class=1)
    assert samples, "没有合成出任何样本"

    for data_url, _ in samples:
        response = client.post("/predict", json={"image": data_url})

        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body["prediction"], int)
        assert 0 <= body["prediction"] <= 9
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["probabilities"]) == 10


def test_small_digit_confidence_is_lower_on_average(
    client: TestClient, require_model, mnist_testset, synthesized_digits
):
    """小字场景的**平均**置信度应显著低于正常字高。

    这条断言的是整体性质而不是单张图：实测 50px 平均置信度约 80.3%、
    100px 约 96.4%，差距约 16 个百分点，这里只要求 5 个百分点，余量充足。

    为什么不断言单张图的置信度：50px 下 22% 的错例置信度仍然 ≥ 0.9，
    单张断言会随机失败。
    """
    small = synthesized_digits(SMALL_SIZE, per_class=SMALL_SAMPLES_PER_CLASS)
    normal = synthesized_digits(NORMAL_SIZE, per_class=SMALL_SAMPLES_PER_CLASS)

    def mean_confidence(samples):
        values = []
        for data_url, _ in samples:
            body = client.post("/predict", json={"image": data_url}).json()
            values.append(body["confidence"])
        return sum(values) / len(values)

    small_mean = mean_confidence(small)
    normal_mean = mean_confidence(normal)

    assert small_mean < normal_mean - MIN_CONFIDENCE_GAP, (
        f"小字平均置信度 {small_mean:.3f} 未明显低于正常字高 {normal_mean:.3f}，"
        f"置信度对输入退化不敏感了？"
    )


# ===========================================================================
# /sample
# ===========================================================================
def test_sample_returns_valid_image(client: TestClient, mnist_testset):
    """GET /sample 返回 200，image 前缀正确，label 在 0-9 之间。"""
    response = client.get("/sample")

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["image"].startswith("data:image/png;base64,")
    assert isinstance(body["label"], int)
    assert 0 <= body["label"] <= 9


def test_sample_image_is_decodable_and_enlarged(client: TestClient, mnist_testset):
    """返回的图片应能正常解码，且是最近邻放大到 280x280 的灰度图。

    放大到 280 是为了前端贴到画布上时边缘锐利；这里顺带确认尺寸没被改错。
    """
    body = client.get("/sample").json()
    image = decode_data_url(body["image"])

    assert image.size == (SAMPLE_SIZE, SAMPLE_SIZE)
    assert image.mode in ("L", "RGBA", "RGB")


def test_sample_returns_varied_images(client: TestClient, mnist_testset):
    """连续取 5 次应至少有 2 张不同，确认"随机"确实生效而不是固定返回某张。"""
    images = {client.get("/sample").json()["image"] for _ in range(5)}

    assert len(images) >= 2


def test_sample_to_predict_roundtrip(client: TestClient, require_model, mnist_testset):
    """端到端往返：/sample 取图 -> 送 /predict -> 结果与真实标签一致。

    这是前端「随机样例」按钮走的完整链路（含 280x280 放大这一步），
    实测 40/40 全对，这里留一档余量。
    """
    correct = 0
    for _ in range(ROUNDTRIP_COUNT):
        sample = client.get("/sample").json()
        response = client.post("/predict", json={"image": sample["image"]})
        assert response.status_code == 200, response.text
        if response.json()["prediction"] == sample["label"]:
            correct += 1

    assert correct >= MIN_ROUNDTRIP_CORRECT, (
        f"往返链路 {ROUNDTRIP_COUNT} 次只对了 {correct} 次，/sample 与 /predict 之间可能不一致"
    )


# ===========================================================================
# 健康检查、静态资源与产物
# ===========================================================================
def test_health_reports_model_loaded(client: TestClient, require_model):
    """健康检查应当报告模型已加载。"""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_path"] == "models/mnist_cnn.pt"


def test_index_page_served(client: TestClient):
    """GET / 应返回前端页面（static/index.html）。"""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_static_assets_served(client: TestClient):
    """三个静态资源都应能通过 /static 访问到。"""
    for name in ("app.js", "style.css"):
        response = client.get(f"/static/{name}")
        assert response.status_code == 200, f"{name} 未正确托管"


def test_openapi_schema_documents_all_endpoints(client: TestClient):
    """OpenAPI 文档应包含全部对外接口——面试时 /docs 是要给人看的。"""
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())

    assert {"/predict", "/sample", "/health"} <= paths


def test_model_file_exists():
    """训练产物应当存在——接口完全依赖它。"""
    if not MODEL_PATH.is_file():
        pytest.skip("模型文件不存在，请先运行 python train.py")
    assert MODEL_PATH.stat().st_size > 0
