# -*- coding: utf-8 -*-
"""图像预处理：网页 Canvas 的 base64 图片 -> MNIST 风格 28x28 张量。

=========================== 归一化约定（最重要） ===========================
**归一化方式引自 train.py 第 147 行**：

    train.py:147    transform = transforms.ToTensor()

这是训练流程中**唯一**的图像变换，全文没有 ``Normalize``（已 grep 确认）。
因此训练时的输入是：

    - dtype  : float32
    - 数值域 : [0.0, 1.0]（uint8 的 [0,255] 直接除以 255）
    - 形状   : [C, H, W] = [1, 28, 28]
    - 极性   : 白笔画（接近 1.0）在黑背景（接近 0.0）上

本模块的输出必须与上述**逐位对齐**。特别强调两点：

1. **不做 (x - 0.1307) / 0.3081 标准化**。
   训练没用，推理若加了会让输入分布整体偏移，精度直接掉。
2. **不反色**。MNIST 是黑底白字；TASK0 第 49 条规定前端 Canvas 也是黑底白字，
   两者极性本来就一致（实证：reports/test_predictions.png 中 20 张样例均为黑底白字）。
   TASK3 第二节 B.3「反色：黑底白字 -> 白底黑字（与 MNIST 一致）」这一条描述有误，
   照做会得到与训练分布相反的白底黑字，模型输出会变成垃圾。
   是否反色由 ``INVERT_INPUT`` 常量控制，默认 False；阶段5 联调若发现真实 Canvas
   是白底黑笔画，把它改成 True 即可，无需改动其它代码。
==========================================================================

预处理流程（对应 TASK0 第 11 条）
---------------------------------
    base64 解码 -> 灰度化 -> (可选)反色 -> 二值化 -> 找数字外接框
    -> 裁剪 -> 等比缩放到 20x20 -> 按质心放入 28x28 -> /255 归一化

关于「按质心对齐」
-----------------
MNIST 官方的构造方式是：先做尺寸归一化（长边缩到 20 像素），再把数字放进 28x28
的框里，并使**笔画的质心落在画面中心**。这与「简单几何居中」的区别在于：
一个写在纸上偏左下的「7」，其外接框居中后笔画重心仍在左上，而质心对齐会把
重心拉回中心——这更接近 MNIST 里数字的实际分布，对真实手写体的准确率有明显帮助。

关于多数字扩展（TASK0 第六节）
------------------------------
``find_digit_boxes()`` 返回的是**外接框列表**并按 x 坐标从左到右排序。
单个数字时取列表第一项；未来升级多数字识别时，直接遍历列表逐框处理、拼接结果即可，
本模块的其余部分无需改动。
"""

from __future__ import annotations

import base64
import binascii
import io
import re

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

# ===========================================================================
# 常量（集中在此处，便于阶段5 联调调参）
# ===========================================================================
CANVAS_SIZE = 28            # 输出画布边长，与 MNIST 一致
DIGIT_BOX_SIZE = 20         # 数字等比缩放后的长边像素，MNIST 官方为 20
BINARY_THRESHOLD = 127      # 二值化阈值：大于该值视为笔画（阶段5 可调）
INVERT_INPUT = False        # 是否需要把输入反色，默认 False（理由见模块 docstring）
MIN_DIGIT_AREA = 20         # 小于该像素数的连通域视为噪点丢弃（阶段5 可调）
MAX_INK_RATIO = 0.85        # 笔画像素占比超过该值即判定"不是手写数字"（见下方说明）
MAX_INPUT_PIXELS = 4_000_000  # 输入图片像素上限，防止超大图导致内存问题

# 外接框类型：(left, top, right, bottom)，采用 PIL 的 crop 约定（右/下为开区间）
Box = tuple[int, int, int, int]


class PreprocessError(ValueError):
    """预处理相关错误的基类。"""


class InvalidImageError(PreprocessError):
    """输入不是可解析的图片（base64 损坏、格式不支持、内容为空等）。"""


class EmptyCanvasError(PreprocessError):
    """图片解析正常，但未检测到任何手写笔画（空白画布）。"""


# ===========================================================================
# 第 1 步：base64 -> PIL Image
# ===========================================================================
def strip_data_url_prefix(image_b64: str) -> str:
    """剥离 ``data:image/png;base64,`` 前缀，并去掉所有空白字符。

    前端 ``canvas.toDataURL()`` 得到的字符串带前缀，但直接传纯 base64 也应支持，
    所以这里做容错处理。

    Args:
        image_b64: 可能带 data URL 前缀的 base64 字符串。

    Returns:
        纯 base64 字符串（仅内容部分）。

    Raises:
        InvalidImageError: data URL 缺少逗号分隔符，无法定位内容起始位置。
    """
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise InvalidImageError("image 字段为空，请提交 Canvas 导出的 PNG base64 字符串。")

    text = image_b64.strip()
    if text.startswith("data:"):
        if "," not in text:
            raise InvalidImageError("data URL 格式不正确：缺少逗号分隔符。")
        text = text.split(",", 1)[1]

    # base64 中出现的换行/空格是合法的填充噪声，统一去掉
    return re.sub(r"\s+", "", text)


def decode_base64_image(image_b64: str) -> Image.Image:
    """把 base64 字符串解码成 PIL Image。

    Args:
        image_b64: 带或不带 data URL 前缀的 base64 字符串。

    Returns:
        解码得到的 PIL Image（RGB 或 L 模式，调用方按需转换）。

    Raises:
        InvalidImageError: base64 非法、内容不是图片、或图片过大。
    """
    payload = strip_data_url_prefix(image_b64)

    try:
        # validate=True 让非法字符直接报错，避免静默丢弃字符后解出错误的字节
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"base64 解码失败：{exc}") from exc

    if not raw:
        raise InvalidImageError("base64 解码结果为空。")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()      # 立刻触发解析，否则损坏的图片会在后面访问像素时才报错
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"无法识别的图片内容：{exc}") from exc

    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError(f"图片尺寸非法：{width}x{height}。")
    if width * height > MAX_INPUT_PIXELS:
        raise InvalidImageError(
            f"图片过大（{width}x{height}），请提交分辨率更小的画布截图。"
        )

    return image


# ===========================================================================
# 第 2、3、4 步：灰度化 -> (可选)反色 -> 二值化
# ===========================================================================
def binarize(
    image: Image.Image,
    threshold: int = BINARY_THRESHOLD,
    invert: bool = INVERT_INPUT,
) -> np.ndarray:
    """灰度化并二值化，返回布尔掩码（True 表示该像素属于笔画）。

    Args:
        image: 输入图片。
        threshold: 二值化阈值，像素值 > threshold 视为笔画。
        invert: 是否先反色。默认 False，理由见模块 docstring。

    Returns:
        ``bool`` 类型的二维数组，形状 ``[H, W]``。
    """
    # convert("L") 用 ITU-R 601-2 亮度公式做灰度化，对纯黑白画布等价于取原值
    gray = np.asarray(image.convert("L"), dtype=np.uint8)

    if invert:
        gray = 255 - gray

    # 直接阈值，不加高斯模糊（TASK3 第二节 B.5），保持链路简单可解释
    return gray > threshold


# ===========================================================================
# 第 5、6 步：找连通域 -> 数字外接框列表 -> 裁剪
# ===========================================================================
def find_digit_boxes(
    mask: np.ndarray,
    min_area: int = MIN_DIGIT_AREA,
) -> list[Box]:
    """找出掩码中所有笔画的连通域外接框，按 x 坐标从左到右排序。

    单个数字时调用方取第一个框即可；多数字时依次遍历（见模块 docstring）。

    Args:
        mask: ``binarize()`` 返回的布尔掩码。
        min_area: 面积小于该值的连通域视为噪点丢弃。

    Returns:
        外接框列表，元素为 ``(left, top, right, bottom)``（PIL crop 约定，
        右/下为开区间）。找不到任何有效笔画时返回空列表。
    """
    if mask.size == 0 or not mask.any():
        return []

    # 8 邻域标记：斜向相连的笔画算同一个数字（例如手写 "4" 的斜杠和竖线）
    structure = np.ones((3, 3), dtype=bool)
    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        return []

    boxes: list[Box] = []
    # find_objects 返回的列表下标 0 对应标签 1
    for label_id, slices in enumerate(ndimage.find_objects(labeled), start=1):
        if slices is None:
            continue
        area = int(np.count_nonzero(labeled[slices] == label_id))
        if area < min_area:
            continue     # 噪点（画布上的零星像素）
        row_slice, col_slice = slices
        boxes.append((col_slice.start, row_slice.start, col_slice.stop, row_slice.stop))

    boxes.sort(key=lambda box: box[0])   # 按 x 从左到右，为多数字识别预留顺序
    return boxes


# ===========================================================================
# 第 7、8 步：等比缩放到 20x20 -> 按质心放入 28x28
# ===========================================================================
def _shift_without_wrap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把图像平移 (dy, dx)，超出边界的部分直接裁掉。

    不用 ``np.roll`` 是因为它会做循环环绕，把移出边界的内容从另一侧卷回来，
    在没有额外留白的输入上会产生鬼影。
    """
    height, width = image.shape
    out = np.zeros_like(image)

    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return out      # 平移量超过整幅图，结果就是全零

    src_y0 = max(0, -dy)
    src_x0 = max(0, -dx)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def crop_and_center(
    mask: np.ndarray,
    box: Box,
    canvas_size: int = CANVAS_SIZE,
    digit_box_size: int = DIGIT_BOX_SIZE,
) -> np.ndarray:
    """裁剪出数字 -> 等比缩放到 20x20 -> 按质心放入 28x28。

    Args:
        mask: ``binarize()`` 返回的布尔掩码。
        box: ``find_digit_boxes()`` 给出的外接框。
        canvas_size: 输出画布边长，默认 28。
        digit_box_size: 数字等比缩放后的长边像素，默认 20。

    Returns:
        ``float32`` 数组，形状 ``[canvas_size, canvas_size]``，取值范围 ``[0, 1]``。
    """
    left, top, right, bottom = box
    digit = mask[top:bottom, left:right].astype(np.uint8) * 255

    # ---- 等比缩放：长边缩放到 digit_box_size，短边按比例，保证不变形 ----
    height, width = digit.shape
    scale = digit_box_size / float(max(height, width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    # LANCZOS 重采样：从二值掩码得到带抗锯齿边缘的灰度图，
    # 与 MNIST 数字"边缘有过渡灰度"的特征一致（MNIST 本身就是抗锯齿的）。
    resized = Image.fromarray(digit).resize((new_w, new_h), Image.LANCZOS)
    digit_float = np.asarray(resized, dtype=np.float32) / 255.0

    # ---- 先放在 canvas 中央，再按质心微调 ----
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset_y = (canvas_size - new_h) // 2
    offset_x = (canvas_size - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = digit_float

    total = float(canvas.sum())
    if total > 0:
        rows, cols = np.mgrid[0:canvas_size, 0:canvas_size]
        com_y = float((rows * canvas).sum()) / total      # 质心的 y 坐标
        com_x = float((cols * canvas).sum()) / total      # 质心的 x 坐标

        # 目标：质心落在画面正中心。canvas_size=28 时中心是 13.5（即像素 13 与 14 之间）
        center = (canvas_size - 1) / 2.0
        shift_y = int(round(center - com_y))
        shift_x = int(round(center - com_x))
        canvas = _shift_without_wrap(canvas, shift_y, shift_x)

    return canvas


# ===========================================================================
# 对外主入口
# ===========================================================================
def preprocess_to_array(
    image_b64: str,
    threshold: int = BINARY_THRESHOLD,
    invert: bool = INVERT_INPUT,
) -> np.ndarray:
    """完整预处理链路，返回 ``[1, 28, 28]`` 的 float32 数组（取值 [0,1]）。

    单数字识别时取 ``find_digit_boxes()`` 返回的第一个框；
    多数字扩展时改为遍历全部框即可。

    Args:
        image_b64: Canvas 导出的 PNG base64（可带 data URL 前缀）。
        threshold: 二值化阈值。
        invert: 是否反色。

    Returns:
        ``np.ndarray``，形状 ``[1, CANVAS_SIZE, CANVAS_SIZE]``，dtype ``float32``。

    Raises:
        InvalidImageError: 图片无法解码。
        EmptyCanvasError: 图中没有任何笔画。
    """
    image = decode_base64_image(image_b64)
    mask = binarize(image, threshold=threshold, invert=invert)

    # ---- 满幅笔画拦截 ----
    # 在"黑底白笔画"的极性下，一张纯白图片经阈值化后会变成整幅都是笔画，
    # 若直接往下走，会被当成一个占满画布的"数字"送进模型，输出必然荒谬。
    # 真实手写数字的笔画像素占比通常只有 0.05-0.25，远低于 0.85，
    # 因此用占比上限可以稳稳区分"手写"与"整幅填充"，且不会误伤写满画布的大数字。
    if float(mask.mean()) > MAX_INK_RATIO:
        raise EmptyCanvasError("未检测到手写数字，请先写一个数字。")

    boxes = find_digit_boxes(mask)
    if not boxes:
        raise EmptyCanvasError("未检测到手写数字，请先写一个数字。")

    # 当前只做单数字识别：取最左侧（也是唯一）的数字框
    digit = crop_and_center(mask, boxes[0])

    return digit[np.newaxis, :, :]     # 增加通道维 -> [1, 28, 28]


def preprocess_to_tensor(
    image_b64: str,
    threshold: int = BINARY_THRESHOLD,
    invert: bool = INVERT_INPUT,
) -> torch.Tensor:
    """同 :func:`preprocess_to_array`，但返回可直接送进模型的张量。

    Returns:
        ``torch.Tensor``，形状 ``[1, 1, 28, 28]``，dtype ``float32``，取值 ``[0,1]``，
        与 ``train.py:147`` 的 ``transforms.ToTensor()`` 输出逐位对齐。
    """
    array = preprocess_to_array(image_b64, threshold=threshold, invert=invert)
    # [1, 28, 28] -> [1, 1, 28, 28]：最前面是 batch 维，单张图 batch=1
    return torch.from_numpy(array).unsqueeze(0)
