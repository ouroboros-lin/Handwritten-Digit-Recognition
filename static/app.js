/* ==========================================================================
   手写数字识别 · 前端逻辑（原生 JS，无框架）
   --------------------------------------------------------------------------
   文件结构：
     1. 常量（画布尺寸、笔刷、接口路径、动画时长、柱状图最小宽度）
     2. DOM 引用与运行时状态
     3. 画布：初始化 / 清空 / 坐标换算 / 绘制 / 判定空白 / 导出 base64
     4. 接口调用：POST /predict、GET /sample
     5. 渲染：结果数字、置信度滚动动画、概率柱状图
     6. 状态：忙碌、加载、错误
     7. 实时识别：按固定间隔自动触发识别（可选开关，默认关闭）
     8. 事件绑定与启动

   为多数字扩展预留的口子（TASK0 第六节）：
     - canvasToDataURL()  只负责"画布 -> base64"，与识别逻辑解耦
     - requestPredict()   只负责"发请求 -> 拿结果对象"，不认识 UI
     - renderResult()     只接受一个结果对象；将来后端返回 {digits:[...], text:"..."} 时，
                          在这里改成遍历 digits 渲染即可，其余代码不用动
   ========================================================================== */

/* ==========================================================================
   1. 常量
   ========================================================================== */

/** 画布的逻辑边长（CSS 像素）。所有绘制坐标都在 0..CANVAS_SIZE 之间。 */
const CANVAS_SIZE = 280;

/** 笔刷线宽（逻辑像素）。20px 在 280 的画布上相当于 1/14，与 MNIST 笔画粗细接近。 */
const BRUSH_WIDTH = 20;

/** 笔刷颜色：白色，配合纯黑背景即"黑底白字"，与 MNIST 极性一致。 */
const BRUSH_COLOR = '#ffffff';

/** 画布背景色：必须是纯黑，导出的 PNG 才是黑底，后端才能正确二值化。 */
const CANVAS_BG = '#000000';

/** 数字类别数，与后端 NUM_CLASSES 对应。 */
const DIGIT_COUNT = 10;

/** 判定"画布空白"的灰度阈值：任一通道大于该值即认为画过。 */
const BLANK_THRESHOLD = 8;

/**
 * 概率条形的**视觉最小宽度**（CSS 像素）。
 *
 * 这是纯视觉处理，**不是数据修正**：
 * 模型自信时（MNIST 上很常见）最高那根概率接近 100%，其余 9 根按真实概率算出来
 * 宽度不到 1px，看上去和"零"没区别。给非零概率一个最小宽度，只是让人一眼看出
 * "这根不是零"，**右侧百分比文字始终显示真实值**，一个字都不改。
 *
 * 注意：本项目的柱状图是**横向**的（数字 | 条形 | 百分比），所以这里控制的是宽度。
 * 概率为**真 0** 时不画任何条形——真零画个 3px 才是真正的误导。
 */
const MIN_BAR_WIDTH = 3;

/** 后端接口路径。集中在这里，改路径只动一行。 */
const API = {
  PREDICT: '/predict',
  SAMPLE: '/sample',
};

/** 动画时长，与 style.css 里的 CSS 变量保持一致。 */
const ANIM = {
  NUMBER_MS: 620,   // 置信度数字从 0 滚到目标值的时长
};

/** 窗口 resize 的防抖延迟（毫秒）。 */
const RESIZE_DEBOUNCE_MS = 150;

/**
 * 实时识别模式的触发间隔（毫秒）。
 *
 * 取值依据是「让上一次动画基本播完再触发下一次」，所以要看最长的那条动画：
 *   - 参考 CSS 中 .result-digit 的 pop-in 动画 380ms（--dur-pop，style.css:465）
 *   - 参考 CSS 中 .bar-fill 的宽度过渡 520ms（--dur-bar，style.css:424）
 *   - 参考本文件 ANIM.NUMBER_MS = 620ms 的置信度滚动（三者中最长）
 * 620 + 约 280ms 余量 = 900ms，动画不会和下一次触发打架。
 *
 * 想调快/调慢改这里即可。调得比 620ms 还小的话，置信度滚动会被打断重来。
 */
const REALTIME_INTERVAL_MS = 900;

/* ==========================================================================
   2. DOM 引用与运行时状态
   ========================================================================== */
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');

const canvasHint = document.getElementById('canvas-hint');
const btnPredict = document.getElementById('btn-predict');
const btnClear = document.getElementById('btn-clear');
const btnSample = document.getElementById('btn-sample');
const realtimeToggle = document.getElementById('realtime-toggle');

const resultDigit = document.getElementById('result-digit');
const resultConfidence = document.getElementById('result-confidence');
const statusEl = document.getElementById('status');
const barsContainer = document.getElementById('bars');

/** 是否正在落笔（按下未抬起）。 */
let isDrawing = false;

/** 上一个采样点，用于连线。 */
let lastPoint = null;

/** 是否处于请求中，防止重复提交。 */
let isBusy = false;

/** 柱状图每一行的 DOM 引用，下标即数字 0-9。 */
let barRows = [];

/** 最近一次渲染的概率数组；窗口尺寸变化时用它重算像素宽度。 */
let lastProbabilities = null;

/** 窗口 resize 防抖句柄。 */
let resizeTimerId = 0;

/** 实时识别是否开启。 */
let isRealtimeEnabled = false;

/** 实时识别的 setInterval 句柄；0 表示没有在跑的定时器。 */
let realtimeTimerId = 0;

/**
 * 实时识别的请求是否正在飞行中。
 *
 * 单独用一个标志而不是复用 isBusy：isBusy 会禁用按钮、显示加载态，
 * 而实时模式要「保持安静」。两者分开，手动识别按钮在实时模式下也能正常点。
 */
let isRealtimeRequestInFlight = false;

/**
 * 上一次渲染结果的指纹（`prediction|confidence|probabilities`）。
 *
 * 实时模式每 900ms 触发一次，如果结果没变还照常重渲染，数字的入场动画和置信度
 * 滚动就会每 900ms 重放一次——看起来一直在闪。拿指纹比一下，没变就什么都不做。
 */
let lastRenderedResultKey = null;

/**
 * 当前正在运行的置信度滚动动画的 requestAnimationFrame 句柄。
 *
 * 记下来是为了在开新动画前取消旧的：请求结束后按钮会重新可用，
 * 用户若立刻再识别一次，旧动画还没播完就会和新动画抢同一个 DOM 文本，
 * 结果是数字来回跳。取消掉旧句柄即可避免。
 */
let confidenceRafId = 0;

/* ==========================================================================
   3. 画布
   ========================================================================== */

/**
 * 计算 canvas 的物理像素尺寸并做 devicePixelRatio 缩放。
 *
 * 高分屏（devicePixelRatio > 1）下如果不放大 backing store，笔画会发虚。
 * 做法是：canvas.width 设为 逻辑尺寸 * dpr，再用 setTransform(dpr,...) 把
 * 绘制指令统一放大，这样**业务代码始终按 0..CANVAS_SIZE 的逻辑坐标书写**，
 * 不用到处乘 dpr。
 */
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.round(CANVAS_SIZE * dpr);
  canvas.height = Math.round(CANVAS_SIZE * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.lineCap = 'round';       // 圆头笔刷，笔画衔接自然
  ctx.lineJoin = 'round';
  ctx.lineWidth = BRUSH_WIDTH;

  resetCanvas();
}

/**
 * 把画布重置为纯黑背景。
 *
 * 刻意用 fillRect 填满而不是 clearRect：clearRect 会留下透明像素，
 * 导出的 PNG 就是透明的；虽然后端 convert("L") 也会把透明区当黑色，
 * 但显式填充黑底更直观、也避免后续换预处理方式时踩坑。
 */
function resetCanvas() {
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillStyle = BRUSH_COLOR;
  ctx.strokeStyle = BRUSH_COLOR;
}

/**
 * 把指针事件的屏幕坐标换算成画布逻辑坐标。
 *
 * 这是 Canvas 最容易出错的地方：画布在 CSS 里可能被拉伸/压缩
 * （窄屏上宽度小于 280px），必须按 getBoundingClientRect() 的实际显示尺寸
 * 做比例换算，否则会出现"笔迹跟手偏移"。
 *
 * @param {PointerEvent} event 指针事件
 * @returns {{x: number, y: number}} 画布逻辑坐标
 */
function toCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE,
    y: ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE,
  };
}

/** 在指定位置画一个圆点（单击不拖动时也能留下笔迹）。 */
function drawDot(point) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
  ctx.fill();
}

/** 从上一个点到当前点连一条线段。 */
function drawLine(from, to) {
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function hideHint() {
  canvasHint.hidden = true;
}

function showHint() {
  canvasHint.hidden = false;
}

/**
 * 判断画布上是否一点笔迹都没有。
 *
 * 用读取像素而不是维护一个"画过没有"的布尔标记：标记容易被某条代码路径漏改，
 * 而像素是唯一的事实来源。280x280 的画布读一遍只要几毫秒，代价可以忽略。
 */
function isCanvasBlank() {
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > BLANK_THRESHOLD || data[i + 1] > BLANK_THRESHOLD || data[i + 2] > BLANK_THRESHOLD) {
      return false;
    }
  }
  return true;
}

/**
 * 把当前画布导出成 base64 PNG（与浏览器地址栏里 canvas.toDataURL() 的格式一致）。
 *
 * 前端**不做任何图像处理**：灰度化、二值化、裁剪、缩放、居中全部由后端
 * app/preprocess.py 统一完成，避免前后端各写一份导致算法漂移。
 *
 * @returns {string} 形如 "data:image/png;base64,...."
 */
function canvasToDataURL() {
  return canvas.toDataURL('image/png');
}

/**
 * 把一张 data URL 图片铺满画布（供「随机样例」使用）。
 *
 * @param {string} dataUrl 图片 data URL
 * @returns {Promise<void>} 图片加载完成并绘制后 resolve
 */
function drawImageToCanvas(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      resetCanvas();
      ctx.drawImage(image, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      resolve();
    };
    image.onerror = () => reject(new Error('样例图片加载失败'));
    image.src = dataUrl;
  });
}

/* ==========================================================================
   4. 接口调用
   ========================================================================== */

/**
 * 统一处理 fetch 响应：HTTP 非 2xx 时抛出带后端 detail 的错误。
 *
 * @param {Response} response fetch 返回的响应对象
 * @returns {Promise<any>} 解析后的 JSON
 */
async function parseResponse(response) {
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;     // 后端返回了非 JSON（例如 500 的 HTML 错误页）
  }

  if (!response.ok) {
    const detail = body && body.detail
      ? body.detail
      : `请求失败（HTTP ${response.status}）`;
    throw new Error(detail);
  }
  return body;
}

/**
 * POST /predict，识别单个手写数字。
 *
 * @param {string} imageDataUrl 画布导出的 base64 PNG
 * @returns {Promise<{prediction: number, confidence: number, probabilities: number[]}>}
 */
async function requestPredict(imageDataUrl) {
  const response = await fetch(API.PREDICT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  return parseResponse(response);
}

/**
 * GET /sample，取一张带真实标签的 MNIST 测试图。
 *
 * @returns {Promise<{image: string, label: number}>}
 */
async function requestSample() {
  const response = await fetch(API.SAMPLE);
  return parseResponse(response);
}

/* ==========================================================================
   5. 渲染
   ========================================================================== */

/** 把 0-1 的概率格式化成百分比文本。极小的值显示为 "<0.01%" 而不是 "0.00%"。 */
function formatPercent(value) {
  const percent = value * 100;
  if (percent > 0 && percent < 0.01) {
    return '<0.01%';
  }
  return `${percent.toFixed(2)}%`;
}

/**
 * 数字从 0 平滑滚动到目标值（easeOutCubic）。
 *
 * @param {HTMLElement} element 目标元素
 * @param {number} target 目标数值
 * @param {{decimals?: number, suffix?: string}} [options]
 */
function animateNumber(element, target, options = {}) {
  const { decimals = 2, suffix = '' } = options;
  const duration = ANIM.NUMBER_MS;
  const startAt = performance.now();

  // 取消上一次还没播完的动画，避免两次动画抢同一个 DOM 文本
  cancelAnimationFrame(confidenceRafId);

  function frame(now) {
    const progress = Math.min(1, (now - startAt) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);   // easeOutCubic
    element.textContent = `${(target * eased).toFixed(decimals)}${suffix}`;
    if (progress < 1) {
      confidenceRafId = requestAnimationFrame(frame);
    }
  }

  confidenceRafId = requestAnimationFrame(frame);
}

/** 生成 10 行概率柱状图（只在初始化时执行一次，之后只改宽度和文字）。 */
function buildBarRows() {
  const fragment = document.createDocumentFragment();
  barRows = [];

  for (let digit = 0; digit < DIGIT_COUNT; digit += 1) {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const label = document.createElement('span');
    label.className = 'bar-digit';
    label.textContent = String(digit);

    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    track.appendChild(fill);

    const percent = document.createElement('span');
    percent.className = 'bar-pct';
    percent.textContent = '—';

    row.append(label, track, percent);
    fragment.appendChild(row);

    barRows.push({ row, fill, percent });
  }

  barsContainer.appendChild(fragment);
}

/**
 * 读取条形轨道的实际像素宽度。
 *
 * 10 行共用同一套网格列，宽度一致，所以只读第一行即可——
 * 读 clientWidth 会强制浏览器重算布局，放在循环里读 10 次是没必要的开销。
 *
 * @returns {number} 轨道宽度（CSS 像素）；布局尚未就绪时可能为 0
 */
function measureTrackWidth() {
  const firstRow = barRows[0];
  if (!firstRow) {
    return 0;
  }
  return firstRow.fill.parentElement.clientWidth;
}

/**
 * 把概率换算成条形的像素宽度。
 *
 * 真实概率决定宽度，只对**非零**概率套一个视觉下限（见 MIN_BAR_WIDTH 注释）。
 *
 * @param {number} value 概率（0-1）
 * @param {number} trackWidth 轨道像素宽度
 * @returns {number} 条形像素宽度
 */
function probabilityToWidth(value, trackWidth) {
  if (value <= 0) {
    return 0;                    // 真·零就真的不画
  }
  if (trackWidth <= 0) {
    return MIN_BAR_WIDTH;        // 布局还没算出来时的兜底：至少让人看见"非零"
  }
  return Math.max(value * trackWidth, MIN_BAR_WIDTH);
}

/**
 * 渲染 10 个数字的概率分布。
 *
 * 柱宽由**真实概率**换算，仅对非零概率施加视觉最小宽度（不修改任何数值）；
 * 具体数值由右侧百分比文字给出，两者互补。
 *
 * @param {number[]} probabilities 长度为 10 的概率数组
 * @param {{animate?: boolean}} [options] animate=false 用于窗口缩放时静默重排
 */
function renderProbabilities(probabilities, options = {}) {
  const { animate = true } = options;

  lastProbabilities = probabilities;

  const maxValue = Math.max(...probabilities);
  const trackWidth = measureTrackWidth();

  if (!animate) {
    // 缩放窗口时重排：临时关掉过渡，否则 10 根条形会各自滑动一下，很突兀
    barsContainer.classList.add('is-static');
  }

  barRows.forEach(({ row, fill, percent }, digit) => {
    const value = probabilities[digit] ?? 0;

    row.classList.toggle('is-top', maxValue > 0 && value === maxValue);

    const targetWidth = probabilityToWidth(value, trackWidth);

    if (animate) {
      // 先把宽度归零并强制一次回流，这样再设成目标值才会触发 CSS 过渡；
      // 否则连续两次识别若宽度相同，浏览器认为没变化，动画不会重放。
      fill.style.width = '0px';
      void fill.offsetWidth;
    }
    fill.style.width = `${targetWidth}px`;

    percent.textContent = formatPercent(value);
  });

  if (!animate) {
    void barsContainer.offsetWidth;      // 让浏览器把新宽度提交，再恢复过渡
    barsContainer.classList.remove('is-static');
  }
}

/**
 * 生成一次识别结果的指纹，用于判断"结果有没有变"。
 *
 * @param {{prediction: number, confidence: number, probabilities: number[]}} result
 * @returns {string} 可比较的字符串指纹
 */
function resultKey(result) {
  return `${result.prediction}|${result.confidence}|${result.probabilities.join(',')}`;
}

/**
 * 渲染一次识别的结果。
 *
 * @param {{prediction: number, confidence: number, probabilities: number[]}} result
 */
function renderResult(result) {
  lastRenderedResultKey = resultKey(result);

  // 先移除动画类 -> 强制回流 -> 再加回来，否则动画只在第一次生效
  resultDigit.classList.remove('is-animating');
  void resultDigit.offsetWidth;
  resultDigit.textContent = String(result.prediction);
  resultDigit.classList.add('is-animating');

  animateNumber(resultConfidence, result.confidence * 100, { decimals: 2, suffix: '%' });
  renderProbabilities(result.probabilities);

  // ---- 多数字扩展点 ----
  // 后端将来返回 {digits: [{prediction, confidence}, ...], text: "2026"} 时，
  // 只需在这里改成遍历 digits 逐个渲染，本文件的其它函数都不用改。
}

/** 把结果区恢复成未识别的初始状态。 */
function renderEmptyResult() {
  // 若置信度还在滚动，先停掉，否则它会继续把 '—' 覆盖成数字
  cancelAnimationFrame(confidenceRafId);

  resultDigit.textContent = '—';
  resultDigit.classList.remove('is-animating');
  resultConfidence.textContent = '—';

  // 清空后不能再用旧概率重排，否则缩放窗口会让已清空的条形"复活"
  lastProbabilities = null;

  // 指纹也一起清掉：清空后再写出一模一样的结果时应当正常渲染，不能被判成"没变"
  lastRenderedResultKey = null;

  barRows.forEach(({ row, fill, percent }) => {
    row.classList.remove('is-top');
    fill.style.width = '0px';
    percent.textContent = '—';
  });
}

/* ==========================================================================
   6. 状态（忙碌 / 加载 / 错误）
   ========================================================================== */

/**
 * 更新状态行文字。
 *
 * @param {string} message 文案
 * @param {'' | 'loading' | 'ok' | 'error'} [kind] 样式类型
 */
function setStatus(message, kind = '') {
  statusEl.textContent = message;
  statusEl.className = kind ? `status is-${kind}` : 'status';
}

/**
 * 切换忙碌状态：请求期间禁用全部按钮并显示 loading 指示器。
 *
 * 除了禁用按钮，handlePredict / handleSample 开头还有 `if (isBusy) return;` 兜底，
 * 所以即使通过键盘或脚本触发也只会发出一次请求。
 *
 * @param {boolean} busy 是否忙碌
 * @param {string} [message] 忙碌时显示的状态文案
 */
function setBusy(busy, message) {
  isBusy = busy;

  [btnPredict, btnClear, btnSample].forEach((button) => {
    button.disabled = busy;
  });

  btnPredict.classList.toggle('is-loading', busy);

  if (busy && message) {
    setStatus(message, 'loading');
  }
}

/* ==========================================================================
   7. 实时识别
   --------------------------------------------------------------------------
   开启后按固定间隔（REALTIME_INTERVAL_MS）自动把画布送去识别，不需要点按钮。
   设计上有意保持克制：
     - 空画布**静默跳过**：不提示、不报错、不发请求
     - 结果和上一次完全一样就**什么都不做**（避免动画每 900ms 重放一次）
     - 出错也不写状态行，否则会每 900ms 闪一次红字
     - 不复用 setBusy()，所以不会禁用按钮、不会出现"识别中…"的连续闪烁
   ========================================================================== */

/**
 * 实时模式的一次触发：识别当前画布并更新结果。
 *
 * 用 setInterval 驱动，但每一步都可能提前返回，所以单次执行非常轻。
 * 注意这里**不调用 setBusy()**——实时模式要保持安静，
 * 手动识别按钮在此期间仍然可用。
 */
async function realtimeTick() {
  if (!isRealtimeEnabled) {
    return;
  }
  // 手动识别正在进行，或上一次实时请求还没回来 → 跳过这一拍
  if (isBusy || isRealtimeRequestInFlight) {
    return;
  }
  // 空画布静默跳过（区别于手动识别会显示红字提示）
  if (isCanvasBlank()) {
    return;
  }

  isRealtimeRequestInFlight = true;
  try {
    const result = await requestPredict(canvasToDataURL());

    // 请求期间用户可能已经把开关关掉了，此时不应再更新界面
    if (!isRealtimeEnabled) {
      return;
    }
    // 结果没变就什么都不做：否则入场动画和置信度滚动会每 900ms 重放一次
    if (resultKey(result) === lastRenderedResultKey) {
      return;
    }
    renderResult(result);
    // 刻意不写状态行：实时模式下保持安静
  } catch (error) {
    // 同样刻意不写状态行——每 900ms 闪一次红字比不提示更糟
  } finally {
    isRealtimeRequestInFlight = false;
  }
}

/** 启动实时识别的定时器（重复调用安全，不会叠加出多个定时器）。 */
function startRealtime() {
  if (realtimeTimerId) {
    return;
  }
  realtimeTimerId = window.setInterval(realtimeTick, REALTIME_INTERVAL_MS);
}

/** 停止实时识别的定时器。必须真的清掉，否则关掉开关后请求还在继续发。 */
function stopRealtime() {
  window.clearInterval(realtimeTimerId);
  realtimeTimerId = 0;
}

/**
 * 开关切换的处理。
 *
 * 状态不持久化（刷新后回到默认的关闭状态）——这是个临时的使用模式，
 * 存进 localStorage 反而会让人下次打开页面莫名其妙地自动识别。
 *
 * @param {Event} event change 事件
 */
function handleRealtimeToggle(event) {
  isRealtimeEnabled = event.target.checked;

  if (isRealtimeEnabled) {
    startRealtime();
  } else {
    stopRealtime();
  }

  // 只在切换时给一次反馈。之后每个周期都不再动状态行，避免闪烁
  setStatus(isRealtimeEnabled ? '实时识别已开启，书写后将自动识别' : '实时识别已关闭');
}

/* ==========================================================================
   8. 事件处理与启动
   ========================================================================== */

/** 指针按下：开始一笔。 */
function onPointerDown(event) {
  if (isBusy) {
    return;
  }
  event.preventDefault();

  // 捕获指针：手指/鼠标移出画布后仍能收到 move 事件，不会突然断笔
  canvas.setPointerCapture(event.pointerId);

  isDrawing = true;
  lastPoint = toCanvasPoint(event);
  drawDot(lastPoint);
  hideHint();
}

/** 指针移动：连线。 */
function onPointerMove(event) {
  if (!isDrawing) {
    return;
  }
  event.preventDefault();

  const point = toCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
}

/** 指针抬起 / 取消：结束这一笔。 */
function onPointerUp(event) {
  if (!isDrawing) {
    return;
  }
  isDrawing = false;
  lastPoint = null;

  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

/** 清空画布并复位结果区。 */
function handleClear() {
  if (isBusy) {
    return;
  }
  resetCanvas();
  renderEmptyResult();
  showHint();
  setStatus('');
}

/** 识别当前画布上的数字。 */
async function handlePredict() {
  if (isBusy) {
    return;
  }
  if (isCanvasBlank()) {
    setStatus('画布是空的，请先写一个数字', 'error');
    return;
  }

  setBusy(true, '识别中…');
  const startedAt = performance.now();

  try {
    const result = await requestPredict(canvasToDataURL());
    renderResult(result);
    // 这里刻意提交的是**画布导出的图**而不是内部数据，
    // 走的就是用户手写时完全相同的链路，能真实反映端到端表现。
    setStatus(`识别完成 · 耗时 ${Math.round(performance.now() - startedAt)} ms`, 'ok');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

/** 取一张随机样例贴到画布上，并自动识别。 */
async function handleSample() {
  if (isBusy) {
    return;
  }

  setBusy(true, '获取样例中…');
  try {
    const sample = await requestSample();
    await drawImageToCanvas(sample.image);
    hideHint();

    setStatus(`已载入随机样例（真实标签：${sample.label}），识别中…`, 'loading');

    const startedAt = performance.now();
    // 提交画布导出的图：这样"贴图 -> 画布 -> 导出 -> 预处理 -> 推理"整条链路都被验证
    const result = await requestPredict(canvasToDataURL());
    renderResult(result);

    const elapsed = Math.round(performance.now() - startedAt);
    const matched = result.prediction === sample.label
      ? '与真实标签一致'
      : `与真实标签 ${sample.label} 不一致`;
    setStatus(`识别完成 · 耗时 ${elapsed} ms · ${matched}`,
      result.prediction === sample.label ? 'ok' : 'error');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

/**
 * 窗口尺寸变化：条形的像素宽度是按当时的轨道宽度算出来的，
 * 轨道变宽变窄后必须重算，否则条形长度会和百分比对不上。
 * 防抖处理，避免拖动窗口时高频触发。
 */
function handleResize() {
  window.clearTimeout(resizeTimerId);
  resizeTimerId = window.setTimeout(() => {
    if (lastProbabilities) {
      renderProbabilities(lastProbabilities, { animate: false });
    }
  }, RESIZE_DEBOUNCE_MS);
}

/** 绑定所有事件。 */
function bindEvents() {
  // 用 Pointer Events 统一处理鼠标 / 触摸 / 触控笔，无需分别写 mouse 与 touch 监听。
  // 触屏上禁止页面滚动由 CSS 的 touch-action: none 负责（见 style.css）。
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);

  // 阻止右键菜单，避免在画布上误触弹窗打断书写
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());

  btnPredict.addEventListener('click', handlePredict);
  btnClear.addEventListener('click', handleClear);
  btnSample.addEventListener('click', handleSample);

  // 用 change 而不是 click：键盘空格切换 checkbox 也会触发 change，
  // 只监听 click 的话键盘操作会漏掉（虽然浏览器通常也会补一个 click，但不保险）
  realtimeToggle.addEventListener('change', handleRealtimeToggle);

  window.addEventListener('resize', handleResize);
}

/** 页面入口。 */
function init() {
  setupCanvas();
  buildBarRows();
  renderEmptyResult();
  bindEvents();
  setStatus('在画布上写一个数字，然后点击「识别」');
}

init();
