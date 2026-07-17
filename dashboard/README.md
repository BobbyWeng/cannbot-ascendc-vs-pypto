# Cannbot 中文算子 Dashboard v3

用于查看 Torch、Ascend C、PyPTO 三条路线的发布状态、正确性、性能和数据证据。

## 生成数据

在仓库根目录运行：

```bash
python3 dashboard/dashboard.py --release reports/release/current_release.json
python3 tests/regression/check_dashboard.py dashboard/dashboard.json
```

生成器只更新 `dashboard/dashboard.json`。HTML、CSS 和 JavaScript 是独立、可维护的静态源码，不会在生成时被覆盖。

## 启动

```bash
python3 -m http.server 8765 --directory dashboard
```

浏览器打开 `http://127.0.0.1:8765/index.html`。

## 数据口径

1. 状态和正确性以 `reports/release/current_release.json` 为唯一权威源。
2. 性能优先使用 `current_release.routes.*.profiler` 的结构化数据。
3. release 缺少结构化性能时，读取 `operators/*/reports/parsed`，同时核验算子 `SHA256SUMS`。
4. 哈希不匹配的数据直接排除。
5. 未登记到 SHA 清单的数据可以展示，但明确标黄且不参与性能排名。
6. 性能排名仅使用 correctness=PASS、msprof 且来源可信的 `primary_compute_kernel_us`。
7. `reports/release/performance_matrix.csv` 未参与生成，避免旧数据和 Event/msprof 混用。

## 文件结构

```text
dashboard/
├── dashboard.py      # 数据生成器
├── dashboard.json    # 生成的数据模型
├── index.html        # 中文页面结构
├── dashboard.css     # 页面样式
├── dashboard.js      # 交互与渲染
└── README.md
```

## 页面功能

- 中文发布总览与 Post-RC3 更新摘要
- 算子搜索和状态筛选
- 三路线正确性与原始覆盖声明
- 按 batch 展示主计算核、全设备、AICPU、TFLOPS、kernel 类型等指标
- 正确性门禁和跨测量方法保护
- 每条数据的来源、SHA256 与完整性状态
- 限制、更新与待审计项集中展示
