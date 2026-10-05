# -*- coding: utf-8 -*-
"""FastAPI 应用入口：加载模型、托管静态页面、提供 ``POST /predict`` 与 ``GET /sample``。

启动
----
    uvicorn app.main:app --reload

访问
----
    页面      http://127.0.0.1:8000/
    接口文档  http://127.0.0.1:8000/docs
    健康检查  http://127.0.0.1:8000/health
    随机样例  http://127.0.0.1:8000/sample

设计要点
--------
1. **模型在启动时加载一次，常驻内存**（``lifespan``），避免每次请求都读盘。
2. 加载后**显式调用 model.eval()**：``MnistCNN`` 带 Dropout，若忘了切到评估模式，
   dropout 在推理时仍会随机丢弃神经元，同一个输入每次结果都不一样。
3. 推理包在 ``torch.no_grad()`` 里，省内存、提速。
4. ``/predict`` 与 ``/sample`` 用**同步 def** 而不是 async def：两者都是 CPU / 磁盘
   密集的阻塞操作，写成同步函数后 FastAPI 会自动丢到线程池执行，不会卡住事件循环。
5. 模型缺失时**服务照常启动**，只是 ``/predict`` 返回 503 并说明原因——
   这样页面和 /docs 依然能打开，排查问题更方便。
"""

from __future__ import annotations

import base64
import io
import random
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.model import load_model
from app.preprocess import EmptyCanvasError, InvalidImageError, preprocess_to_tensor
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SampleResponse,
)

# ===========================================================================
# 路径配置：基于本文件位置推导，保证从任意工作目录启动 uvicorn 都能找到资源
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent      # 项目根目录
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.pt"
MODEL_PATH_DISPLAY = "models/mnist_cnn.pt"             # 报错信息里显示的相对路径

# MNIST 测试集解压后的文件名，用于判断本地数据是否可用（避免触发 torchvision 的下载）
MNIST_RAW_DIR = DATA_DIR / "MNIST" / "raw"
MNIST_TEST_FILES = ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

# 统一提示语
MODEL_MISSING_MESSAGE = (
    f"模型文件 {MODEL_PATH_DISPLAY} 不存在，请先在项目根目录运行：python train.py"
)
MNIST_MISSING_MESSAGE = "本地未找到 MNIST 数据，请先运行 python train.py"

# /sample 返回的图片边长：MNIST 原图 28x28，用最近邻放大 10 倍到 280x280，
# 前端贴到画布上边缘锐利（最近邻不会产生模糊的过渡像素）。放大倍数即 SAMPLE_SCALE。
SAMPLE_SIZE = 280
SAMPLE_SCALE = SAMPLE_SIZE // 28

# 前端还没做时的占位页面（正常情况不会走到，见 GET / 的说明）
PLACEHOLDER_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>手写数字识别 - 前端缺失</title>
  <style>
    body { font-family: "Microsoft YaHei", system-ui, sans-serif; max-width: 680px;
           margin: 80px auto; padding: 0 24px; line-height: 1.8; color: #24292f; }
    code { background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }
    .ok { color: #1a7f37; font-weight: 600; }
    .warn { color: #9a6700; }
    li { margin: 4px 0; }
  </style>
</head>
<body>
  <h1>后端已启动 <span class="ok">✓</span></h1>
  <p class="warn">未找到 <code>static/index.html</code>，前端页面缺失。</p>
  <p>可直接用这些地址确认后端工作正常：</p>
  <ul>
    <li><a href="/docs">/docs</a> — 交互式接口文档，可试 <code>POST /predict</code></li>
    <li><a href="/health">/health</a> — 模型是否加载成功、推理设备</li>
    <li><a href="/sample">/sample</a> — 取一张带标签的 MNIST 测试图</li>
  </ul>
</body>
</html>
"""


# ===========================================================================
# 应用生命周期：启动时加载模型，关闭时释放
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用的启动与关闭。

    启动阶段把模型放进 ``app.state.model``，请求处理函数从这里取。
    模型文件不存在时**不抛异常中断启动**，而是记录错误信息，让 /predict 返回 503。
    """
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = None
    app.state.model_error = None
    app.state.mnist_testset = None      # /sample 用的 MNIST 测试集，首次调用时懒加载

    print("=" * 66)
    print("Starting handwritten digit recognition service")
    print(f"  device     : {app.state.device}")
    print(f"  model path : {MODEL_PATH}")

    try:
        model = load_model(str(MODEL_PATH), app.state.device)
        # load_model 内部已经调过 eval()，这里再显式调一次：
        # 这是最容易被忽略、又最能毁掉结果的一步（Dropout 在推理时仍生效会让结果抖动），
        # 所以不依赖上游的实现细节，就地再确认一次。
        model.eval()
        app.state.model = model
        print("  model      : loaded OK (eval mode)")
    except FileNotFoundError as exc:
        app.state.model_error = str(exc)
        print("  model      : NOT FOUND")
        print(f"  [WARN] {MODEL_MISSING_MESSAGE}")
        print("  service will still start; POST /predict returns 503 until trained.")

    mnist_ready = all((MNIST_RAW_DIR / name).is_file() for name in MNIST_TEST_FILES)
    print(f"  mnist data : {'ready' if mnist_ready else 'NOT FOUND'}")
    if not mnist_ready:
        print(f"  [WARN] {MNIST_MISSING_MESSAGE}")

    print("=" * 66)
    yield

    # ---- 关闭 ----
    app.state.model = None
    app.state.mnist_testset = None
    print("Service stopped.")


app = FastAPI(
    title="手写数字识别 API",
    description=(
        "接收网页 Canvas 导出的 PNG base64，用本地训练的 PyTorch CNN（MNIST）"
        "识别 0-9 的单个手写数字。全流程本地推理，不调用任何外部云 API。"
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# ===========================================================================
# CORS：允许所有来源，方便本地前端调试（生产环境应改为白名单）
# ===========================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # 注意：allow_origins=["*"] 时不能开启 credentials
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 静态资源与首页
# ===========================================================================
STATIC_DIR.mkdir(parents=True, exist_ok=True)      # 目录不存在时 StaticFiles 会报错
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# response_model=None：本项目返回的是 Response 对象（FileResponse / HTMLResponse），
# 不加这个参数的话 FastAPI 会拿返回类型注解去构建 pydantic 响应模型并直接报错。
@app.get("/", include_in_schema=False, response_model=None)
def index() -> FileResponse | HTMLResponse:
    """返回前端页面。

    ``static/index.html`` 存在时直接返回；否则返回一个说明性的占位页，
    避免访问首页看到 404 一脸茫然。
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return HTMLResponse(content=PLACEHOLDER_HTML, status_code=200)


# ===========================================================================
# 健康检查
# ===========================================================================
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """返回服务与模型状态，用于快速判断环境是否就绪。"""
    model_loaded = app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=MODEL_PATH_DISPLAY,
        device=str(app.state.device),
        detail=None if model_loaded else MODEL_MISSING_MESSAGE,
    )


# ===========================================================================
# 随机样例：给前端「随机样例」按钮提供带标签的真实 MNIST 图片
# ===========================================================================
def _get_mnist_testset():
    """懒加载并缓存 MNIST 测试集；数据缺失时抛出 503。

    ``download=False`` 是刻意的：这里只做本地读取，绝不在请求里触发下载
    （否则用户点一次按钮可能卡住几分钟）。数据由 train.py 负责下载。

    Returns:
        ``torchvision.datasets.MNIST`` 实例（transform 为 None，返回 PIL Image）。
    """
    if app.state.mnist_testset is not None:
        return app.state.mnist_testset

    if not all((MNIST_RAW_DIR / name).is_file() for name in MNIST_TEST_FILES):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MNIST_MISSING_MESSAGE,
        )

    # 延迟导入：torchvision 导入较慢，放在函数里可以让应用启动更快
    from torchvision import datasets

    app.state.mnist_testset = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=False
    )
    return app.state.mnist_testset


@app.get(
    "/sample",
    response_model=SampleResponse,
    tags=["inference"],
    summary="随机取一张带标签的 MNIST 测试图",
    responses={
        503: {"model": ErrorResponse, "description": "本地没有 MNIST 数据"},
    },
)
def sample() -> SampleResponse:
    """随机返回一张 MNIST 测试图片（base64 PNG）及其真实标签。

    前端拿它贴到画布上并自动调用 /predict，相当于给演示一个"标准答案"：
    如果预测结果和 label 不一致，说明链路上有问题。
    """
    dataset = _get_mnist_testset()

    index = random.randrange(len(dataset))
    image, label = dataset[index]

    # 最近邻放大 10 倍：不引入模糊的过渡像素，前端贴到画布上边缘锐利
    enlarged = image.resize((SAMPLE_SIZE, SAMPLE_SIZE), Image.NEAREST)

    buffer = io.BytesIO()
    enlarged.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    return SampleResponse(image=f"data:image/png;base64,{payload}", label=int(label))


# ===========================================================================
# 核心接口：预测
# ===========================================================================
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="识别手写数字",
    responses={
        400: {"model": ErrorResponse, "description": "图片无法解析，或画布上没有数字"},
        503: {"model": ErrorResponse, "description": "模型未加载（未训练）"},
    },
)
def predict(payload: PredictRequest) -> PredictResponse:
    """识别 base64 图片中的单个手写数字。

    刻意用同步函数：推理是阻塞的 CPU 操作，FastAPI 会把同步路由自动放进
    线程池执行，从而不阻塞事件循环、可以并发处理请求。
    """
    # ---- 前置检查：模型是否就绪 ----
    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_MISSING_MESSAGE,
        )

    # ---- 预处理：与训练端逐位对齐（见 app/preprocess.py 顶部说明）----
    try:
        tensor = preprocess_to_tensor(payload.image)
    except EmptyCanvasError as exc:
        # 空白画布是用户的常见误操作，用 400 而不是 500
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未检测到手写数字，请先写一个数字。",
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"图片解析失败：{exc}",
        ) from exc

    # ---- 推理 ----
    tensor = tensor.to(app.state.device)
    with torch.no_grad():
        logits = model(tensor)                       # [1, 10]，未过 softmax
        probabilities = torch.softmax(logits, dim=1)[0]   # [10]，归一化为概率

    confidence, prediction = torch.max(probabilities, dim=0)

    # 显式转成 Python 原生类型：numpy / torch 标量 pydantic 虽然大多能自动处理，
    # 但明确转换可以避免某些版本下 JSON 序列化报错。
    return PredictResponse(
        prediction=int(prediction.item()),
        confidence=round(float(confidence.item()), 6),
        # 保留 6 位小数便于阅读；四舍五入后总和可能偏差不超过 1e-5，属预期行为
        probabilities=[round(float(p), 6) for p in probabilities.tolist()],
    )


if __name__ == "__main__":
    # 便于在 PyCharm 中直接右键运行本文件调试（等价于 uvicorn 命令）
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
