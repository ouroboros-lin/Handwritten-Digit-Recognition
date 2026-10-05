"""手写数字识别后端包。

模块划分（后续阶段逐步实现）：
- app.model      : CNN 网络结构定义（训练与推理共用，保证结构一致）
- app.preprocess : MNIST 风格图像预处理（灰度化、裁剪、缩放、居中、归一化）
- app.main       : FastAPI 应用入口，提供 /predict 接口与静态页面托管
"""