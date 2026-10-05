# ==========================================================================
# 手写数字识别服务 · 生产镜像
# --------------------------------------------------------------------------
# 构建：docker build -t mnist-web .
# 运行：docker run -p 8000:8000 mnist-web
# 访问：http://127.0.0.1:8000/
#
# 设计要点见每一层的注释，核心是三条：
#   1. 只装 CPU 版 PyTorch —— 镜像从 3GB+ 降到约 300MB
#   2. 依赖层与源码层分开 COPY —— 改业务代码不必重装依赖
#   3. 模型权重直接打进镜像 —— 部署时不用额外挂载卷
# ==========================================================================

FROM python:3.13-slim

# --------------------------------------------------------------------------
# 环境变量
#   PYTHONDONTWRITEBYTECODE : 不在容器里生成 .pyc，避免无谓的文件写入
#   PYTHONUNBUFFERED        : 日志实时输出，否则 docker logs 要等缓冲满才看得到
#   PIP_NO_CACHE_DIR        : 不保留 pip 缓存，镜像更小
# --------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 是 OpenMP 运行时。torch / scipy / scikit-learn 都依赖它，
# 而 python:slim 基础镜像默认不带。约 150KB，装上可以免掉一类"导入即崩溃"的问题。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------
# 第 1 层：CPU 版 PyTorch
# --------------------------------------------------------------------------
# 为什么单独装：直接从 PyPI 执行 `pip install torch` 会连带拉取 CUDA 相关的
# nvidia-* 依赖包（好几个 GB），而本项目做的是 CPU 推理，这些包一点用都没有。
# 显式指定 PyTorch 官方的 CPU 索引，镜像体积能小一个数量级。
#
# 若所在网络访问不了 download.pytorch.org，把下面两行删掉，
# 并去掉第 2 层里 `-r requirements.txt` 前面的注释即可（镜像会大很多，但能装上）。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# --------------------------------------------------------------------------
# 第 2 层：其余依赖
# --------------------------------------------------------------------------
# 先只 COPY 依赖清单再安装，是为了利用 Docker 的层缓存：
# 只要 requirements.txt 没变，改业务代码重新构建时这一层直接命中缓存、秒过。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# 第 3 层：源码与模型
# --------------------------------------------------------------------------
# 哪些东西会被打包由 .dockerignore 决定：
#   - 不打 data/（MNIST 约 64MB）→ 云端 /sample 接口会返回 503，属预期行为
#   - 不打 reports/（训练曲线等纯展示材料）
#   - 打 models/mnist_cnn.pt（仅 1.69MB，打进镜像最省事，部署时无需挂载卷）
COPY . ./

# uvicorn 默认端口，与 README 里的命令保持一致
EXPOSE 8000

# --------------------------------------------------------------------------
# 健康检查
# --------------------------------------------------------------------------
# 不依赖 curl（slim 镜像里没有），用 Python 标准库发一个请求即可。
# start-period 给模型加载留出时间，避免容器刚启动就被判为不健康。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

# 必须监听 0.0.0.0：只听 127.0.0.1 的话，容器外部访问不到
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
