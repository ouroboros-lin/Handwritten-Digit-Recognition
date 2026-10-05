# -*- coding: utf-8 -*-
"""边界情况探测脚本：在**真实推理链路**上跑一遍手写数字的各种极端写法。

这个脚本解决什么问题
--------------------
模型在 MNIST 测试集上 99% 准确率，不代表真实使用体验——用户会写歪、写小、写满、
只写一条竖线当"1"、把"4"写得不闭合。这些情况 MNIST 测试集覆盖不到，
只能在"模拟真实画布"的输入上验证。

本脚本严格按 ``static/app.js`` 的画布参数生成输入：

    - 画布    : 280 x 280，纯黑底
    - 笔刷    : 白色，线宽 20px，圆头（round cap / round join）
    - 提交方式: 与前端一致，PNG -> base64 data URL -> POST /predict

输出：

    - 控制台：每个 case 的预测值、置信度、是否被画布裁切、是否与预期一致
    - 图片  : reports/edge_cases.png（4x4 对照图，标题写「case 名 / 预测 / 真实」）

运行：
    python tools/edge_cases.py

关于"被画布裁切"这一列
----------------------
数字必须完整落在画布内才谈得上"识别"。如果用例把数字摆到了画布外，Canvas 会把它
裁掉（真人在画布边缘写字也是同样效果），此时输入本身就是残缺的，识别错误属于
garbage in garbage out，不是模型缺陷。脚本把这一列显式列出来，避免误判。

注意：脚本里的"手写"是**程序合成**的近似（把 MNIST 笔画放大到指定尺寸后按需加粗，
或用折线直接画），用于复现几何上的极端情况，不等同于真人笔迹。
真人在浏览器里的实测结论以 README「边界测试」小节为准。
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))      # 让 `from app...` / `from train...` 能正常导入

# 让中文在 Windows 终端 / PyCharm 里都能正常打印（默认 cp936 会把中文打成乱码）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app                   # noqa: E402
from train import setup_chinese_font        # noqa: E402  复用训练脚本里的中文字体探测

import matplotlib                          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

# ---------------------------------------------------------------------------
# 与 static/app.js 保持一致的画布参数
# ---------------------------------------------------------------------------
CANVAS_SIZE = 280        # app.js: CANVAS_SIZE
BRUSH_WIDTH = 20         # app.js: BRUSH_WIDTH
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

# "位置"扫描用的基准数字尺寸与偏移。
# 尺寸 150 是刻意的：210px 的数字放在 280 画布上，中心只能在 [105,175] 之间移动，
# 偏一点就会被裁切，测不出"写歪"本身的影响。150px 留出了 ±60px 的安全偏移空间。
BASE_SWEEP_SIZE = 150
BASE_SWEEP_OFFSET = 60

# 位置/大小扫描统一用 MNIST 测试集的同一张图，保证各项之间可比
BASE_DIGIT_INDEX = 0


# ===========================================================================
# 画布合成：生成与前端等价的黑底白字图
# ===========================================================================
def _disk(radius: int) -> np.ndarray:
    """生成半径为 radius 的圆形结构元素，用于形态学膨胀。"""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def mnist_stroke_mask(index: int, target_size: int, brush: int = BRUSH_WIDTH):
    """取一张 MNIST 数字，缩放到指定尺寸，并把笔画加粗到接近 brush 像素。

    为什么要"加粗"：真人不论把数字写得多小，用的都是同一支 20px 的笔，
    笔画粗细不随字号变化；而直接把 MNIST 图缩小会让笔画一起变细，
    那就成了"写细"而不是"写小"，测不出真实情况。

    笔画自然宽度按 "约等于外接框长边的 1/10" 估算（MNIST 笔画约 2px / 外接框约 20px），
    再按差值做圆形膨胀补足。这是**近似**，真实的笔宽还与书写速度、压感有关。

    Args:
        index: MNIST 测试集下标。
        target_size: 数字外接框长边缩放到的像素数。
        brush: 目标笔画宽度（像素）。

    Returns:
        (布尔掩码, 真实标签)
    """
    from torchvision import datasets

    dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    image, label = dataset[index]

    mask = np.asarray(image, dtype=np.uint8) > 127

    # 裁到数字外接框，再等比缩放到 target_size
    rows, cols = np.nonzero(mask)
    mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    height, width = mask.shape
    scale = target_size / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    resized = np.asarray(resized) > 127

    # 补足笔画宽度
    natural_width = target_size / 10.0
    radius = max(0, int(round((brush - natural_width) / 2)))
    if radius > 0:
        resized = ndimage.binary_dilation(resized, structure=_disk(radius))

    return resized, int(label)


def is_clipped(mask: np.ndarray, center: tuple[int, int]) -> bool:
    """判断按给定中心点摆放时，数字是否会超出 280x280 画布而被裁切。"""
    height, width = mask.shape
    top = int(round(center[1] - height / 2))
    left = int(round(center[0] - width / 2))
    return (top < 0 or left < 0
            or top + height > CANVAS_SIZE or left + width > CANVAS_SIZE)


def paste_on_canvas(mask: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """把数字掩码按**中心点**贴到 280x280 的黑底画布上，超出部分裁掉。"""
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    height, width = mask.shape

    top = int(round(center[1] - height / 2))
    left = int(round(center[0] - width / 2))

    # 目标区域与源区域求交，实现裁剪式粘贴
    dst_top, dst_left = max(0, top), max(0, left)
    dst_bottom = min(CANVAS_SIZE, top + height)
    dst_right = min(CANVAS_SIZE, left + width)
    if dst_bottom <= dst_top or dst_right <= dst_left:
        return canvas

    src_top, src_left = dst_top - top, dst_left - left
    canvas[dst_top:dst_bottom, dst_left:dst_right] = (
        mask[src_top:src_top + (dst_bottom - dst_top),
             src_left:src_left + (dst_right - dst_left)] * 255
    )
    return canvas


def draw_polylines(polylines: list[list[tuple[int, int]]],
                   brush: int = BRUSH_WIDTH) -> np.ndarray:
    """按折线画笔画，圆头圆角——与 Canvas 的 lineCap/lineJoin = 'round' 等价。

    PIL 的 ``draw.line`` 默认是平头，所以在每条线的端点额外补一个圆点来模拟圆头。
    """
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    drawer = ImageDraw.Draw(image)

    for points in polylines:
        drawer.line(points, fill=255, width=brush, joint="curve")
        for point in (points[0], points[-1]):
            x, y = point
            r = brush / 2
            drawer.ellipse([x - r, y - r, x + r, y + r], fill=255)

    return (np.asarray(image) > 127).astype(np.uint8) * 255


# ===========================================================================
# 用例定义
# ===========================================================================
def _mask_case(name: str, mask: np.ndarray, center: tuple[int, int],
               expected: int, note: str) -> dict:
    """把一个"数字掩码 + 摆放中心"打包成用例，自动判定是否被裁切。"""
    return {
        "name": name,
        "canvas": paste_on_canvas(mask, center),
        "expected": expected,
        "clipped": is_clipped(mask, center),
        "note": note,
    }


def build_cases() -> list[dict]:
    """构造全部边界用例（共 16 项，正好排成 4x4 对照图）。

    每个 case 是 ``{name, canvas, expected, clipped, note}``。
    """
    cases: list[dict] = []
    center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)

    sweep_mask, sweep_label = mnist_stroke_mask(BASE_DIGIT_INDEX, BASE_SWEEP_SIZE)
    sweep_note = f"基准数字：MNIST test[{BASE_DIGIT_INDEX}]，标签 {sweep_label}，{BASE_SWEEP_SIZE}px"

    # ---- 1~4. 写歪：上下左右（150px 数字留足偏移空间，不会被裁切）----
    offset = BASE_SWEEP_OFFSET
    for name, pos in [
        ("偏左", (center[0] - offset, center[1])),
        ("偏右", (center[0] + offset, center[1])),
        ("偏上", (center[0], center[1] - offset)),
        ("偏下", (center[0], center[1] + offset)),
    ]:
        cases.append(_mask_case(
            name, sweep_mask, pos, sweep_label,
            f"{sweep_note}，中心偏移 {offset}px 到 {pos}",
        ))

    # ---- 5~9. 字号扫描：找"写多小会失效"的临界点 ----                ----
    for name, size, note in [
        ("写小(50px)", 50, "占画布约 1/6，20px 笔刷相对字高已经很粗"),
        ("写小(70px)", 70, "占画布 1/4"),
        ("写小(90px)", 90, "占画布约 1/3"),
        ("中号(130px)", 130, "占画布约 1/2"),
        ("写满(210px)", 210, "占画布 3/4，仍在画布内"),
    ]:
        mask, label = mnist_stroke_mask(BASE_DIGIT_INDEX, size)
        cases.append(_mask_case(name, mask, center, label, note))

    # ---- 10. 写在角落，其余大片留白 ----                             ----
    corner_mask, corner_label = mnist_stroke_mask(BASE_DIGIT_INDEX, 90)
    cases.append(_mask_case(
        "角落+大片留白", corner_mask, (58, 58), corner_label,
        "数字贴左上角，其余全空，验证裁剪能否正确定位",
    ))

    # ---- 11~12. 笔画粗细 ----                                        ----
    thick_mask, thick_label = mnist_stroke_mask(BASE_DIGIT_INDEX, 210, brush=34)
    cases.append(_mask_case(
        "粗笔画(34px)", thick_mask, center, thick_label,
        "笔画加粗到 34px，模拟反复描",
    ))
    thin_mask, thin_label = mnist_stroke_mask(BASE_DIGIT_INDEX, 210, brush=8)
    cases.append(_mask_case(
        "细笔画(8px)", thin_mask, center, thin_label,
        "笔画细到 8px，验证二值化后信息量是否仍足够",
    ))

    # ---- 13. 明知会被画布裁切（记录 GIGO 行为）----                   ----
    # 210px 的数字中心放在 y=68，上沿必然超出画布被切掉——真人把数字写得太大、
    # 顶到画布边缘时就是这个效果。列出来是为了说明"识别错在这里是合理的"。
    big_mask, big_label = mnist_stroke_mask(BASE_DIGIT_INDEX, 210)
    cases.append(_mask_case(
        "超出画布被截断", big_mask, (140, 68), big_label,
        "210px 数字中心放在 y=68，上沿被画布切掉——预期的 GIGO，非缺陷",
    ))

    # ---- 14~16. 连笔 / 简化写法（手工折线，圆头 20px）----            ----
    cases.append({
        "name": "连笔: 1 单竖线",
        "canvas": draw_polylines([[(140, 70), (140, 215)]]),
        "expected": 1,
        "clipped": False,
        "note": "一笔到底的竖线，无起笔顿点、无斜度",
    })
    cases.append({
        "name": "连笔: 7 无横杠",
        "canvas": draw_polylines([[(88, 80), (192, 82)], [(192, 82), (118, 218)]]),
        "expected": 7,
        "clipped": False,
        "note": "顶横 + 一撇，中间没有那一小横",
    })
    cases.append({
        "name": "连笔: 4 开口",
        "canvas": draw_polylines([
            [(142, 74), (82, 178)],        # 左上斜
            [(82, 178), (196, 176)],       # 中横
            [(158, 120), (158, 232)],      # 竖笔起点低于顶部 -> 上方不闭合
        ]),
        "expected": 4,
        "clipped": False,
        "note": "顶部不闭合的 4，竖笔不与斜笔相连",
    })

    return cases


# ===========================================================================
# 执行与出图
# ===========================================================================
def to_data_url(canvas: np.ndarray) -> str:
    """把 280x280 灰度数组编码成与前端等价的 base64 PNG data URL。"""
    buffer = io.BytesIO()
    Image.fromarray(canvas, mode="L").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def run_all_cases(client: TestClient) -> list[dict]:
    """跑完全部用例，返回带预测结果的结果列表。"""
    results = []

    for case in build_cases():
        response = client.post("/predict", json={"image": to_data_url(case["canvas"])})

        if response.status_code == 200:
            body = response.json()
            prediction, confidence = body["prediction"], body["confidence"]
            detail = ""
        else:
            prediction, confidence = None, None
            detail = response.json().get("detail", "")

        results.append({
            **case,
            "prediction": prediction,
            "confidence": confidence,
            "detail": detail,
            "ok": (prediction == case["expected"]) if case["expected"] is not None else None,
        })

    return results


def save_montage(results: list[dict], save_path: Path) -> None:
    """把所有用例的输入图拼成 4x4 对照图，标题标注 case 名 / 预测 / 真实。"""
    cols = 4
    rows = (len(results) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.9, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, results):
        ax.imshow(item["canvas"], cmap="gray", vmin=0, vmax=255)

        if item["prediction"] is None:
            title, color = f"{item['name']}\n预测失败", "red"
        else:
            expected = "?" if item["expected"] is None else item["expected"]
            title = (f"{item['name']}\n预测:{item['prediction']} / 真实:{expected}"
                     f"  {item['confidence'] * 100:.1f}%")
            # 被画布裁切导致出错的标成橙色：输入本身残缺，不算是模型的问题
            if item["ok"]:
                color = "green"
            else:
                color = "darkorange" if item["clipped"] else "red"
        ax.set_title(title, fontsize=9.5, color=color)
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle("边界情况探测（绿=正确；红=错误；橙=输入被画布裁切导致，不属模型缺陷）",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    font_name = setup_chinese_font()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("边界情况探测（画布参数与 static/app.js 一致：280x280 黑底，白色圆头笔刷 20px）")
    print(f"chinese font: {font_name}")
    print("=" * 92)

    with TestClient(app) as client:
        if client.app.state.model is None:
            print("[ERROR] 模型未加载，请先运行 python train.py")
            return

        results = run_all_cases(client)

    # ---- 控制台表格 ----
    print(f"{'case':<18} {'预测':>4} {'真实':>4} {'置信度':>9} {'裁切':>4}  结果")
    print("-" * 92)
    for item in results:
        predicted = "-" if item["prediction"] is None else str(item["prediction"])
        expected = "?" if item["expected"] is None else str(item["expected"])
        confidence = "-" if item["confidence"] is None else f"{item['confidence'] * 100:.2f}%"
        verdict = "N/A" if item["ok"] is None else ("通过" if item["ok"] else "★不符★")
        clipped = "是" if item["clipped"] else "否"
        print(f"{item['name']:<18} {predicted:>4} {expected:>4} {confidence:>9} "
              f"{clipped:>4}  {verdict}" + (f"  {item['detail']}" if item["detail"] else ""))

    # ---- 统计：把"输入被裁切"的项单独拎出来，避免冤枉模型 ----
    valid = [item for item in results if item["ok"] is not None and not item["clipped"]]
    passed = sum(1 for item in valid if item["ok"])
    print("-" * 92)
    print(f"输入完整的用例：{passed}/{len(valid)} 通过")

    clipped_cases = [item for item in results if item["clipped"]]
    if clipped_cases:
        print("输入被画布裁切的用例（结果仅供参考，不计入上面统计）：")
        for item in clipped_cases:
            state = "仍识别正确" if item["ok"] else "识别错误（输入残缺，符合预期）"
            print(f"  - {item['name']}: 预测 {item['prediction']} / 真实 {item['expected']}，{state}")

    failed = [item for item in valid if not item["ok"]]
    if failed:
        print("\n★ 输入完整但识别失败的项（可作为阶段6 pytest 回归用例的种子）：")
        for item in failed:
            print(f"  - {item['name']}: 预测 {item['prediction']} / 真实 {item['expected']}"
                  f"，置信度 {item['confidence'] * 100:.2f}%（{item['note']}）")
    else:
        print("输入完整的用例无失败项。")

    # ---- 出图 ----
    montage_path = REPORTS_DIR / "edge_cases.png"
    save_montage(results, montage_path)
    print(f"\n对照图已保存: {montage_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
