# event_stacked_plotter.py — 事件级堆叠图绘制工具

独立的命令行工具，用于对 `--event-selection-output` 产生的事件级 PKL / ROOT 输出文件绘制 CMS 风格的对数坐标堆叠背景图，可选叠加数据黑线。

---

## 目录

1. [功能特性](#功能特性)
2. [环境配置](#环境配置)
3. [输入文件格式](#输入文件格式)
4. [用法](#用法)
5. [参数说明](#参数说明)
6. [典型示例](#典型示例)
7. [输出说明](#输出说明)
8. [注意事项](#注意事项)

---

## 功能特性

- 读取事件级 PKL 文件（结构：`n_events` / `events` / `objects`）和 ROOT 文件（含 `Events` 树与 `Metadata` 树）
- 自动合并每个输入文件夹内的多个 PKL / ROOT 文件
- 支持多个背景文件夹，每个文件夹作为一个独立的堆叠分量（标签取文件夹名）
- 自动从 `objects` 字典中提取所有可绘图数值变量，包含嵌套对象（如 `jets_pt`、`muons_eta` 等）
- 所有直方图按 `1/n_events × lumi` 归一化
- 背景分量按总产额从小到大从下到上叠加
- 数据样本用黑色阶梯线叠加
- CMS 标准样式输出（`CMS Simulation` 标注、对数 Y 轴、网格线、图例）

---

## 环境配置

```bash
# 1. 获取 LCG 环境（CERN lxplus 或兼容节点）
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh

# 2. 配置项目环境变量
source start.sh
```

---

## 输入文件格式

### PKL 文件（`--event-selection-output` 输出）

文件为 Python pickle，包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `n_events` | `int` | 分析开始前读取的总原始事件数（用于归一化） |
| `events` | `list/array` | 通过预选的事件列表 |
| `objects` | `dict` | 每个物理对象字段（见下） |

`objects` 字段示例（支持的嵌套结构）：

```python
objects = {
    "PFMET_pt":     [float, ...],           # 扁平数值列表
    "jets":         [[{"pt": ..., "eta": ..., "phi": ..., "mass": ...}, ...], ...],  # 嵌套对象列表
    "n_bjets":      [int, ...],             # 整数变量（自动用整数分箱）
}
```

### ROOT 文件

含 `Events` 树（各分支存储物理变量）和可选的 `Metadata` 树（含 `n_events` 分支）。

### 文件夹结构

每个背景（或数据）文件夹下直接放置 PKL / ROOT 文件（脚本**不递归**子目录）：

```
outputs/
├── newDIBOSON_2022_EVENTSELECTION/
│   └── 2022/              <-- 指向此层
│       ├── abc123.pkl
│       ├── def456.pkl
│       └── ...
├── newTop_2022_EVENTSELECTION/
│   └── 2022/
│       └── ...
└── data_folder/
    └── *.pkl
```

---

## 用法

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders <bkg_folder1> [<bkg_folder2> ...] \
    [--data-folder <data_folder>] \
    [--data-folders <data_folder1> <data_folder2> ...] \
    --output-dir <output_dir> \
    [OPTIONS]
```

---

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--background-folders` | ✅ | — | 一个或多个背景文件夹，每个文件夹合并为一个堆叠分量 |
| `--output-dir` | ✅ | — | 输出 PNG 图片存放目录（不存在时自动创建） |
| `--data-folder` | ❌ | 无 | 数据文件夹，合并后用黑色线段叠加；不指定则仅画背景 |
| `--data-folders` | ❌ | 无 | 多个数据文件夹，会先合并再叠加；适合同时使用 2022C 和 2022D |
| `--variables` | ❌ | 全部 | 只绘制指定变量，如 `--variables PFMET_pt jets_pt n_bjets` |
| `--bins` | ❌ | `40` | 连续变量的分箱数（整数变量自动使用整数分箱） |
| `--max-variables` | ❌ | 无 | 最多绘制的变量数（调试 / 快速预览用） |
| `--lumi` | ❌ | `1.0` | 积分亮度（单位 fb⁻¹），用于直方图纵轴缩放 |
| `--file-type` | ❌ | `auto` | 文件类型：`auto`（优先 ROOT）、`pkl`、`root` |

---

## 典型示例

### 1. 多背景 + 亮度归一化（最常用）

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders \
        outputs/3-16/2022/newZto2Nu-2Jets_2022_EVENTSELECTION \
        outputs/3-16/2022/newTop_2022_EVENTSELECTION \
        outputs/3-16/2022/newWtoLNU-2Jets_2022_EVENTSELECTION \
        outputs/3-16/2022/newDYto2L-2Jets_2022_EVENTSELECTION \
        outputs/3-16/2022/newDIBOSON_2022_EVENTSELECTION \
    --output-dir outputs/3-16/stacked_plots_2022 \
    --variables PFMET_pt PFMET_phi electron_pt electron_eta muon_pt muon_eta jets_pt jets_eta \
    --lumi 7.99
```

### 2. 仅绘制指定变量

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders \
        outputs/newDIBOSON_2022_EVENTSELECTION/2022 \
        outputs/newTop_2022_EVENTSELECTION/2022 \
    --output-dir outputs/stacked_met \
    --variables PFMET_pt PFMET_phi n_bjets \
    --lumi 7.99
```

### 3. 加入数据样本叠加

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders \
        outputs/newDIBOSON_2022_EVENTSELECTION/2022 \
        outputs/newTop_2022_EVENTSELECTION/2022 \
    --data-folders \
        outputs/4-2/2022/Run2022C_met_22Sep2023-v1_EVENTSELECTION \
        outputs/4-2/2022/Run2022D-22Sep2023-v1_EVENTSELECTION \
    --output-dir outputs/stacked_data_mc \
    --lumi 7.99
```

### 4. 同时使用 2022C 和 2022D 的完整 2022 图

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders \
        outputs/4-2/2022/newDIBOSON_2022_EVENTSELECTION \
        outputs/4-2/2022/newDYto2L-2Jets_2022_EVENTSELECTION \
        outputs/4-2/2022/newTop_2022_EVENTSELECTION \
        outputs/4-2/2022/newWtoLNU-2Jets_2022_EVENTSELECTION \
        outputs/4-2/2022/newZto2Nu-2Jets_2022_EVENTSELECTION \
    --data-folders \
        outputs/4-2/2022/Run2022C_met_22Sep2023-v1_EVENTSELECTION \
        outputs/4-2/2022/Run2022D-22Sep2023-v1_EVENTSELECTION \
    --output-dir outputs/4-2/2022_full_physics_plots_2022CD_7p99 \
    --xsection-json scripts/xsection_results.json \
    --year 2022 \
    --lumi 7.99
```

### 5. 快速预览（限制变量数）

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders outputs/newTop_2022_EVENTSELECTION/2022 \
    --output-dir outputs/quick_check \
    --max-variables 5
```

### 6. 强制使用 PKL 文件（忽略同目录 ROOT 文件）

```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders outputs/newTop_2022_EVENTSELECTION/2022 \
    --output-dir outputs/stacked_pkl_only \
    --file-type pkl \
    --lumi 7.99
```

---

## 输出说明

输出目录下每个变量生成一个 PNG 文件，命名格式：

```
stacked_<variable_name>.png
```

示例：

```
outputs/stacked_plots_2022/
├── stacked_PFMET_pt.png
├── stacked_PFMET_phi.png
├── stacked_n_bjets.png
├── stacked_jets_pt.png
├── stacked_muons_eta.png
└── ...
```

图像特征：
- **Y 轴**：对数坐标，单位为 `Events × lumi / n_events`
- **CMS 标注**：左上角 `CMS Simulation`，右上角显示积分亮度
- **背景堆叠**：按总事件数从小（底）到大（顶）排布，标签自动简化（去除年份、前缀等）
- **数据覆盖**：若提供 `--data-folder` 或 `--data-folders`，用黑色阶梯线叠加
- **固定分箱**：`PFMET_pt` / `MET` / `Recoil` 使用 `[250, 300, 400, 550, 1000]`，`ctsValue` 使用 `[0, 0.25, 0.50, 0.75, 1.0]`
- **颜色方案**：背景颜色已对齐参考脚本的 process palette

---

## 注意事项

1. **路径层级**：输出文件夹通常含年份子目录（如 `2022/`），需将路径指向直接含 PKL/ROOT 文件的层级。

2. **`*.awk_raw.pkl` 自动跳过**：合并时自动忽略以 `.awk_raw.pkl` 或 `raw.pkl` 结尾的中间文件。

3. **`n_events` 归一化**：脚本依赖 PKL 中的 `n_events` 字段（由 `--event-selection-output` 自动写入）进行归一化。若 `n_events=0` 则该样本直方图为零。

4. **MET 过滤**：变量名末尾为 `met_pt` 的分布自动过滤 `< 100 GeV` 部分，以隐去触发效率不完整区域。

5. **整数变量分箱**：若变量值全为整数且不超过 20 个唯一值（如多重数），自动使用整数分箱而非等宽连续分箱。

### 常用代码
python3 scripts/event_stacked_plotter.py --background-folders outputs/4-2/2022/newDIBOSON_2022_EVENTSELECTION outputs/4-2/2022/newDYto2L-2Jets_2022_EVENTSELECTION outputs/4-2/2022/newTop_2022_EVENTSELECTION outputs/4-2/2022/newWtoLNU-2Jets_2022_EVENTSELECTION outputs/4-2/2022/newZto2Nu-2Jets_2022_EVENTSELECTION --data-folders outputs/4-2/2022/Run2022C_met_22Sep2023-v1_EVENTSELECTION outputs/4-2/2022/Run2022D-22Sep2023-v1_EVENTSELECTION --output-dir outputs/4-2/2022_full_physics_plots_2022CD_7p99 --xsection-json scripts/xsection_results.json --year 2022 --lumi 7.99