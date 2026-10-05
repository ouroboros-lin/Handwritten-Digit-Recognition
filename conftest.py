# -*- coding: utf-8 -*-
"""pytest 根配置文件。

放在项目根目录的目的是让 pytest 把**项目根目录**加入 ``sys.path``，
这样测试文件里可以直接 ``from app.main import app``，
无需安装为包、也不依赖 ``python -m pytest`` 才有的 cwd 注入行为。

本文件故意不放任何 fixture，保持职责单一。
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
