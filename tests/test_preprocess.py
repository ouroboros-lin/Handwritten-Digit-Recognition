# -*- coding: utf-8 -*-
"""预处理单元测试：``app/preprocess.py``。

运行：
    pytest tests/test_preprocess.py -v

这里测的是**纯函数行为**，不依赖模型、不依赖 HTTP：
    - base64 解码与 data URL 前缀容错
    - 灰度化 / 二值化的极性（黑白是否搞反）
    - 笔画提取与连通域外接框（多数字扩展的基础）
    - 输出尺寸、取值范围是否与训练端一致
    - 质心对齐是否真的把重心放到了画面中心
    - 各种非法输入是否抛出**明确的**异常类型

依赖 MNIST 数据的用例在没有数据时会 ``skip`` 并说明原因。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from app.preprocess import (
    BINARY_THRESHOLD,
    CANVAS_SIZE,
    MAX_INK_RATIO,
    EmptyCanvasError,
    InvalidImageError,
    binarize,
    crop_and_center,
    decode_base64_image,
    find_digit_boxes,
    preprocess_to_array,
    preprocess_to_tensor,
    strip_data_url_prefix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MNIST_RAW = DATA_DIR / "MNIST" / "raw"

# 质心允许的偏差。实现在做质心对齐时把平移量取整到像素，
# 因此残差理论上不超过 0.5px，这里放宽到 1px 留出余量。
CENTROID_TOLERANCE = 1.0

# 质心对齐的目标位置：28 像素画面的中心在 13.5（即像素 13 与 14 之间）
CENTROID_TARGET = (CANVAS_SIZE - 1) / 2.0


# ===========================================================================
# Fixtures 与工具
# ===========================================================================
@pytest.fixture(scope="module")
def mnist_testset():
    """MNIST 测试集（不下载）；数据缺失时跳过依赖它的用例。"""
    if not (MNIST_RAW / "t10k-images-idx3-ubyte").is_file():
        pytest.skip("MNIST 数据不存在，请先运行 python train.py 完成下载")

    from torchvision import datasets

    return datasets.MNIST(root=str(DATA_DIR), train=False, download=False)


def to_data_url(image: Image.Image, with_prefix: bool = True) -> str:
    """把 PIL 图片编码成 base64 PNG（可带 / 不带 data URL 前缀）。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if with_prefix else encoded


def blank_canvas(size: int = 280) -> Image.Image:
    """纯黑画布——等价于前端"什么都没写"的状态。"""
    return Image.new("L", (size, size), color=0)


def solid_canvas(size: int = 280, color: int = 255) -> Image.Image:
    """纯色画布——纯白等价于"整幅都被涂满"。"""
    return Image.new("L", (size, size), color=color)


def canvas_with_rects(rects: list[tuple[int, int, int, int]],
                      size: int = 280) -> Image.Image:
    """在黑色画布上画若干白色矩形，用于构造"多个笔画块"的场景。"""
    image = Image.new("L", (size, size), color=0)
    drawer = ImageDraw.Draw(image)
    for box in rects:
        drawer.rectangle(box, fill=255)
    return image


def centroid(array: np.ndarray) -> tuple[float, float]:
    """计算二维数组的质心坐标 (y, x)。"""
    rows, cols = np.mgrid[0:array.shape[0], 0:array.shape[1]]
    total = float(array.sum())
    assert total > 0, "数组全零，质心无定义"
    return float((rows * array).sum() / total), float((cols * array).sum() / total)


# ===========================================================================
# base64 解码
# ===========================================================================
def test_strip_data_url_prefix_handles_both_forms():
    """带前缀与不带前缀应得到同样的纯 base64；空白字符应被清掉。"""
    raw = base64.b64encode(b"hello").decode()
    assert strip_data_url_prefix(f"data:image/png;base64,{raw}") == raw
    assert strip_data_url_prefix(raw) == raw
    assert strip_data_url_prefix(f"data:image/png;base64,{raw[:4]}\n{raw[4:]}") == raw


def test_strip_data_url_prefix_rejects_empty():
    """空字符串应被明确拒绝，而不是走到后面才报奇怪的错。"""
    with pytest.raises(InvalidImageError):
        strip_data_url_prefix("")


def test_strip_data_url_prefix_rejects_malformed_data_url():
    """缺少逗号的 data URL 应报错。"""
    with pytest.raises(InvalidImageError):
        strip_data_url_prefix("data:image/png;base64")


def test_decode_accepts_plain_base64_and_prefixed():
    """两种提交形式解出来的图片应完全一致。"""
    image = canvas_with_rects([(60, 60, 200, 200)])
    without = decode_base64_image(to_data_url(image, with_prefix=False))
    with_prefix = decode_base64_image(to_data_url(image, with_prefix=True))
    assert np.array_equal(np.asarray(without), np.asarray(with_prefix))


def test_decode_rejects_non_base64_characters():
    """含非法字符的字符串应抛 InvalidImageError，而不是静默解出错误字节。"""
    with pytest.raises(InvalidImageError):
        decode_base64_image("这不是base64!!!")


def test_decode_rejects_non_image_bytes():
    """合法 base64 但内容不是图片，应抛 InvalidImageError。"""
    with pytest.raises(InvalidImageError):
        decode_base64_image(base64.b64encode(b"hello world").decode())


def test_decode_rejects_oversized_image():
    """超大图片应被拒绝，避免一张图打爆内存。"""
    huge = Image.new("L", (3000, 3000), color=0)      # 900 万像素，超过 400 万上限
    with pytest.raises(InvalidImageError):
        decode_base64_image(to_data_url(huge))


# ===========================================================================
# 二值化：极性是最容易搞反的地方
# ===========================================================================
def test_binarize_marks_white_strokes_on_black_background():
    """黑底白笔画（MNIST 与前端 Canvas 的极性）：白色区域应被判定为笔画。"""
    mask = binarize(canvas_with_rects([(60, 60, 200, 200)]))

    assert mask.dtype == np.bool_
    assert bool(mask[130, 130])      # 矩形内是笔画
    assert not bool(mask[10, 10])    # 角落是背景


def test_binarize_threshold_boundary():
    """恰好等于阈值不算笔画，比阈值大 1 才算——与 > 127 的定义一致。"""
    at_threshold = np.full((8, 8), BINARY_THRESHOLD, dtype=np.uint8)
    above_threshold = np.full((8, 8), BINARY_THRESHOLD + 1, dtype=np.uint8)

    assert not binarize(Image.fromarray(at_threshold)).any()
    assert binarize(Image.fromarray(above_threshold)).all()


def test_binarize_invert_flips_polarity():
    """invert=True 时应把白底黑笔画转成黑底白笔画（保留给反向 Canvas 的开关）。"""
    white_background = solid_canvas(color=255)
    assert binarize(white_background, invert=False).all()
    assert not binarize(white_background, invert=True).any()


# ===========================================================================
# 连通域外接框：多数字扩展的基础
# ===========================================================================
def test_find_digit_boxes_returns_empty_for_blank():
    """空白画布应返回空列表，而不是抛异常或返回全图框。"""
    assert find_digit_boxes(binarize(blank_canvas())) == []


def test_find_digit_boxes_is_sorted_left_to_right():
    """两个分离的笔画块应得到两个框，并按 x 从左到右排序。

    这是未来多数字识别的排序依据（TASK0 第六节），必须现在就锁住行为。
    """
    # 故意先画右边那块，验证排序不是"画图顺序"
    image = canvas_with_rects([(180, 100, 230, 180), (40, 100, 90, 180)])
    boxes = find_digit_boxes(binarize(image))

    assert len(boxes) == 2
    lefts = [box[0] for box in boxes]
    assert lefts == sorted(lefts)
    assert boxes[0][0] < boxes[1][0]


def test_find_digit_boxes_merges_diagonally_touching_strokes():
    """斜向相接的笔画应算作同一个连通域（手写"4"的斜笔与竖笔就是这种情况）。

    两个矩形刻意只做**对角相邻**：第一个的右下角像素 (139,139) 与第二个的左上角
    像素 (140,140) 是斜邻居，只有 8 邻域标记才会把它们连成一块，
    4 邻域会判成两块——这正是 find_digit_boxes 使用 3x3 结构元素的原因。
    """
    image = canvas_with_rects([(60, 60, 140, 140), (140, 140, 200, 200)])
    assert len(find_digit_boxes(binarize(image))) == 1


def test_find_digit_boxes_filters_tiny_noise():
    """面积过小的连通域应被当作噪点丢掉。"""
    image = canvas_with_rects([(100, 100, 180, 180), (250, 250, 251, 251)])
    boxes = find_digit_boxes(binarize(image))
    assert len(boxes) == 1


def test_find_digit_boxes_respects_min_area_parameter():
    """min_area 参数应当生效——阶段5 已确认它是可调开关。"""
    image = canvas_with_rects([(100, 100, 120, 120)])      # 20x20 = 400 像素
    assert len(find_digit_boxes(binarize(image), min_area=100)) == 1
    assert len(find_digit_boxes(binarize(image), min_area=1000)) == 0


# ===========================================================================
# 裁剪 + 缩放 + 质心对齐
# ===========================================================================
def test_crop_and_center_outputs_expected_shape_and_range():
    """输出应为 28x28 的 float32，取值范围 [0, 1]。"""
    image = canvas_with_rects([(60, 40, 180, 240)])       # 120x200 的长条
    boxes = find_digit_boxes(binarize(image))
    result = crop_and_center(binarize(image), boxes[0])

    assert result.shape == (CANVAS_SIZE, CANVAS_SIZE)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0


def test_crop_and_center_preserves_aspect_ratio():
    """等比缩放：细长矩形的长边应占满 20 像素，短边按比例，不能变形。"""
    image = canvas_with_rects([(100, 40, 140, 240)])      # 40 宽 x 200 高
    boxes = find_digit_boxes(binarize(image))
    result = crop_and_center(binarize(image), boxes[0])

    rows_with_ink = np.where(result.sum(axis=1) > 0)[0]
    cols_with_ink = np.where(result.sum(axis=0) > 0)[0]

    assert rows_with_ink.max() - rows_with_ink.min() + 1 == 20   # 长边缩到 20
    assert cols_with_ink.max() - cols_with_ink.min() + 1 == 4    # 短边 200:40 -> 20:4


def test_crop_and_center_puts_centroid_at_canvas_center():
    """质心对齐的核心断言：重心必须落在画面中心 13.5 附近。

    用非对称图形（偏心的"L"形）测试才有意义——对称图形居中本来就居中了。
    """
    image = canvas_with_rects([
        (60, 60, 90, 240),        # 竖笔
        (60, 210, 220, 240),      # 底横（整体重心偏左下）
    ])
    mask = binarize(image)
    boxes = find_digit_boxes(mask)
    result = crop_and_center(mask, boxes[0])

    com_y, com_x = centroid(result)
    assert abs(com_y - CENTROID_TARGET) <= CENTROID_TOLERANCE, f"质心 y={com_y}"
    assert abs(com_x - CENTROID_TARGET) <= CENTROID_TOLERANCE, f"质心 x={com_x}"


def test_crop_and_center_is_not_merely_bounding_box_centered():
    """区分"质心对齐"与"外接框居中"——这正是 MNIST 官方做法与图省事的差别。

    一个重心明显偏上的图形：外接框居中后重心仍在偏上位置；质心对齐后重心回到中心。
    """
    image = canvas_with_rects([
        (100, 40, 180, 90),        # 上方一个粗块（重心偏上）
        (130, 90, 150, 240),       # 一根向下的细杆
    ])
    mask = binarize(image)
    boxes = find_digit_boxes(mask)
    result = crop_and_center(mask, boxes[0])

    com_y, _ = centroid(result)
    assert abs(com_y - CENTROID_TARGET) <= CENTROID_TOLERANCE


# ===========================================================================
# 端到端：完整预处理链路
# ===========================================================================
def test_blank_canvas_raises_empty_canvas_error():
    """空白画布：应抛 EmptyCanvasError，接口层据此返回 400。"""
    with pytest.raises(EmptyCanvasError):
        preprocess_to_array(to_data_url(blank_canvas()))


def test_all_ink_canvas_raises_empty_canvas_error():
    """满幅笔画（纯白图）：应被 MAX_INK_RATIO 拦住，抛 EmptyCanvasError。

    如果不拦，整幅白图会被当成一个占满画布的"数字"送进模型，输出必然荒谬。
    """
    with pytest.raises(EmptyCanvasError):
        preprocess_to_array(to_data_url(solid_canvas(color=255)))


def test_max_ink_ratio_threshold_is_sane():
    """确认阈值本身合理：真实手写笔画的占比远低于它。"""
    assert 0.0 < MAX_INK_RATIO < 1.0
    # 一个正常的数字，笔画占比应在 0.05-0.25 之间，离 0.85 有很大余量
    image = canvas_with_rects([(100, 40, 140, 240)])
    ink_ratio = float(binarize(image).mean())
    assert ink_ratio < MAX_INK_RATIO / 2


def test_preprocess_to_array_shape_and_dtype(mnist_testset):
    """输出形状必须是 [1, 28, 28]、dtype float32——训练端 ToTensor() 的约定。"""
    image, _ = mnist_testset[0]
    result = preprocess_to_array(to_data_url(image))

    assert result.shape == (1, CANVAS_SIZE, CANVAS_SIZE)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0


def test_preprocess_to_tensor_matches_training_convention(mnist_testset):
    """张量形状 [1, 1, 28, 28]、取值 [0,1]、float32。

    这三条正是 ``train.py:147`` 的 ``transforms.ToTensor()`` 的输出约定，
    对不上就会和训练分布错位。
    """
    image, _ = mnist_testset[0]
    tensor = preprocess_to_tensor(to_data_url(image))

    assert tuple(tensor.shape) == (1, 1, CANVAS_SIZE, CANVAS_SIZE)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_centroid_alignment_on_real_digits(mnist_testset, index):
    """在真实 MNIST 数字上验证质心对齐：5 个样本的质心都应落在中心附近。"""
    image, _ = mnist_testset[index]
    result = preprocess_to_array(to_data_url(image))[0]

    com_y, com_x = centroid(result)
    assert abs(com_y - CENTROID_TARGET) <= CENTROID_TOLERANCE, f"样本 {index} 质心 y={com_y}"
    assert abs(com_x - CENTROID_TARGET) <= CENTROID_TOLERANCE, f"样本 {index} 质心 x={com_x}"


def test_preprocessing_is_deterministic(mnist_testset):
    """同样的输入必须得到逐位相同的结果——预处理里不能有随机性。"""
    image, _ = mnist_testset[0]
    payload = to_data_url(image)

    assert np.array_equal(preprocess_to_array(payload), preprocess_to_array(payload))
