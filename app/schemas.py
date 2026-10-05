# -*- coding: utf-8 -*-
"""接口的请求 / 响应模型（pydantic）。

集中定义的好处：
1. FastAPI 自动生成 OpenAPI 文档与请求校验，非法请求在进入业务代码前就被拦下；
2. 响应字段有类型约束，避免不小心返回 numpy 标量（如 np.int64）导致序列化失败；
3. 面试时能体现接口契约是显式声明的，而不是"约定俗成"。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.model import NUM_CLASSES


class PredictRequest(BaseModel):
    """``POST /predict`` 的请求体。"""

    image: str = Field(
        ...,
        min_length=1,
        description=(
            "Canvas 导出的 PNG 图片 base64 字符串。"
            "允许带 'data:image/png;base64,' 前缀，也接受纯 base64。"
        ),
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
    )


class PredictResponse(BaseModel):
    """``POST /predict`` 的响应体。"""

    prediction: int = Field(
        ...,
        ge=0,
        lt=NUM_CLASSES,
        description="预测的数字（0-9）",
        examples=[7],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="该数字的置信度，即 softmax 概率，范围 0-1",
        examples=[0.998],
    )
    probabilities: list[float] = Field(
        ...,
        description=(
            f"长度为 {NUM_CLASSES} 的概率数组，下标即数字 0-9，元素为该类的 softmax 概率，"
            "总和为 1。前端用它画概率分布条形图。"
        ),
        examples=[[0.0001, 0.0, 0.0002, 0.0003, 0.0001, 0.0004, 0.0, 0.9980, 0.0005, 0.0004]],
    )

    @field_validator("probabilities")
    @classmethod
    def _check_length(cls, value: list[float]) -> list[float]:
        """确保概率数组长度与类别数一致，否则前端渲染会错位。"""
        if len(value) != NUM_CLASSES:
            raise ValueError(
                f"probabilities 长度必须为 {NUM_CLASSES}，实际为 {len(value)}"
            )
        return value


class SampleResponse(BaseModel):
    """``GET /sample`` 的响应体：一张带真实标签的 MNIST 测试图片。

    前端点「随机样例」时用它把图片贴到画布上并自动识别，
    相当于给演示提供了一个"标准答案"，也方便快速验证整条链路。
    """

    image: str = Field(
        ...,
        description=(
            "MNIST 测试图片的 base64 PNG，带 'data:image/png;base64,' 前缀。"
            "已用最近邻放大到 280x280，前端可直接贴到画布上而不糊。"
        ),
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
    )
    label: int = Field(
        ...,
        ge=0,
        lt=NUM_CLASSES,
        description="该图片的真实标签（0-9）",
        examples=[7],
    )


class HealthResponse(BaseModel):
    """``GET /health`` 的响应体：用于确认服务和模型是否就绪。"""

    status: str = Field(..., description="服务状态：ok 或 degraded", examples=["ok"])
    model_loaded: bool = Field(..., description="模型是否已成功加载", examples=[True])
    model_path: str = Field(..., description="模型权重路径", examples=["models/mnist_cnn.pt"])
    device: str = Field(..., description="推理设备", examples=["cpu"])
    detail: str | None = Field(
        default=None,
        description="模型未加载时的具体原因与处理建议",
    )


class ErrorResponse(BaseModel):
    """错误响应体。FastAPI 默认用 ``{"detail": "..."}``，这里显式声明便于文档展示。"""

    detail: str = Field(..., description="错误说明", examples=["未检测到手写数字，请先写一个数字。"])
