# Cross-Section Support in event_stacked_plotter.py

## 快速开始

### 基础命令（有交叉截面）
```bash
python3 scripts/event_stacked_plotter.py \
    --background-folders \
        outputs/3-23/2022/newDIBOSON_2022_EVENTSELECTION \
        outputs/3-23/2022/newDYto2L-2Jets_2022_EVENTSELECTION \
        outputs/3-23/2022/newTop_2022_EVENTSELECTION \
        outputs/3-23/2022/newWtoLNU-2Jets_2022_EVENTSELECTION \
        outputs/3-23/2022/newZto2Nu-2Jets_2022_EVENTSELECTION \
    --data-folder outputs/3-23/2022/Run2022C_met_22Sep2023-v1_EVENTSELECTION \
    --output-dir outputs/test_xsection_2022 \
    --xsection-json scripts/xsection_results.json \
    --year 2022 \
    --lumi 5.02
```

## 新增参数说明

### --xsection-json PATH
- **类型**：文件路径（可选）
- **默认值**：None
- **说明**：指定JSON文件，包含样本的交叉截面数据（单位：pb）
- **格式**：
  ```json
  {
    "ProcessName": [
      {
        "year": "2022",
        "process": "...",
        "xsection": 123.45,
        "full_dataset": "..."
      },
      ...
    ]
  }
  ```

### --year YEAR_STRING
- **类型**：字符串（可选）
- **默认值**：None
- **说明**：用于从JSON中查找交叉截面的年份标签
- **可用值**：2022, 2022EE, 2023, 2024
- **必须与--xsection-json一起使用**

## 物理含义

### 归一化公式

当指定了交叉截面时：
$$\text{weight} = \frac{\text{luminosity} \times \sigma_{\text{fb}}}{N_{\text{events}}}$$

其中：
- luminosity: 集成亮度（fb⁻¹）
- σ_fb: 交叉截面（飞巴，fb = 10⁻³ pb）
- N_events: 模拟事件总数

### 单位转换
- JSON中的交叉截面单位：**pb**（皮巴）
- 内部转换为：**fb**（飞巴）
- 转换因子：1 fb = 1000 pb

## 样本名称匹配规则

系统自动从样本文件夹名称中提取过程名称：

1. 移除"new"前缀
   - newDIBOSON_2022_EVENTSELECTION → DIBOSON_2022_EVENTSELECTION

2. 移除年份和后缀
   - DIBOSON_2022_EVENTSELECTION → DIBOSON

3. 进行大小写不敏感匹配
   - 匹配JSON中的过程类别

### 示例
- newDIBOSON_2022_EVENTSELECTION → matches "DIBOSON" in JSON
- newWtoLNU-2Jets_2022_EVENTSELECTION → matches "WtoLNu-2Jets" （大小写不敏感）
- newDYto2L-2Jets_2023_EVENTSELECTION → matches "DYto2L-2Jets" with year 2023

## 都支持的过程

根据xsection_results.json：

| 过程名称 | 说明 |
|--------|------|
| DIBOSON | 双重玻色子过程 |
| DYto2L-2Jets | Drell-Yan到轻子对+2喷流 |
| Top | 顶夸克过程 |
| WtoLNu-2Jets | W→lν+2喷流 |
| Zto2Nu-2Jets | Z→νν+2喷流 |

## 错误处理

### 交叉截面未找到
- 自动降级到无交叉截面模式
- 使用原始归一化：weight = luminosity / n_events
- 打印警告信息

### 无效的年份
- 返回None，使用原始归一化方式
- 检查输入的year与JSON中可用的年份是否匹配

## 向后兼容性

对于不使用--xsection-json参数的命令：
- 行为完全与之前相同
- 所有现有的脚本继续正常工作
- 无任何破坏性改动

## 验证输出

生成的直方图应该显示：
- Y轴标签："MC: Events × lumi / n_events, Data: Events"
- 正确的相对大小（考虑了各过程的交叉截面）
- 所有背景样本按总产额排序堆叠

## 故障排除

### 1. 交叉截面未找到
```
没有看到 "Found cross-section for ..." 的消息
```
**解决**：
- 检查--year参数是否正确
- 验证样本文件夹名称是否匹配JSON中的过程
- 检查JSON文件是否有效

### 2. 生成的图与预期不符
- 检查luminosity值是否正确
- 确认JSON中的xsection值单位是pb
- 验证--year是否在JSON中存在

### 3. JSON加载失败
```
Warning: Could not load cross-section JSON
```
**解决**：
- 检查JSON文件路径是否正确
- 验证JSON格式是否有效
- 确保文件可读

## 相关文件

- 主脚本：`scripts/event_stacked_plotter.py`
- 交叉截面数据：`scripts/xsection_results.json`
- 文档：`scripts/README_event_stacked_plotter.md`

---

**版本**：1.0  
**更新日期**：2026-03-27  
**状态**：生产就绪
