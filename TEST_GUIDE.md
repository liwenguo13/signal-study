# 测试步骤文档

## 一、程序用途说明

| 程序 | 用途 | 依赖 |
|------|------|------|
| `wav_signal.py` | WAV文件加载工具，解码PCM数据 | 无 |
| `analyze_session_a.py` | 事件检测，检测录音中的敲击事件 | `wav_signal.py` |
| `estimate_distance.py` | 主程序，使用互相关计算A/B端距离 | `analyze_session_a.py`, `wav_signal.py` |

---

## 二、执行命令

### 1. analyze_session_a.py（事件检测）

**分析单个文件**：
```bash
python analyze_session_a.py session_A.wav
```

**分析两个文件**：
```bash
python analyze_session_a.py session_A.wav session_B.wav
```

**输出JSON结果**：
```bash
python analyze_session_a.py session_A.wav session_B.wav --output-json result.json
```

**自定义参数**：
```bash
python analyze_session_a.py session_A.wav session_B.wav \
    --block-ms 10 \
    --min-score-ratio 0.002 \
    --min-event-gap-ms 300 \
    --edge-margin-ms 200 \
    --max-candidates 128
```

### 2. estimate_distance.py（测距）

**空气测距（默认）**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json
```

**钢管测距**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json --medium iron
```

**水管测距**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json --medium water
```

**木材测距**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json --medium wood
```

**混凝土测距**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json --medium concrete
```

**自定义声速**：
```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json --speed-mps 3500
```

---

## 三、介质声速表

| 介质 | 声速 (m/s) | 命令参数 |
|------|-----------|----------|
| 空气 20°C | 340 | `--medium air` |
| 水 | 1500 | `--medium water` |
| 铁/钢管 | 5000 | `--medium iron` |
| 木材 | 3500 | `--medium wood` |
| 混凝土 | 3200 | `--medium concrete` |

---

## 四、可调参数及后果

### analyze_session_a.py 参数

| 参数 | 默认值 | 调大后果 | 调小后果 | 何时调 |
|------|--------|----------|----------|--------|
| `--block-ms` | 10.0 | 灵敏度降，噪声少，事件被合并 | 灵敏度高，噪声多，事件被拆开 | 事件检测不准时 |
| `--min-score-ratio` | 0.002 | 门槛高，漏检多，只留强事件 | 门槛低，误检多，弱事件也能进 | 漏检或误检时 |
| `--min-event-gap-ms` | 300.0 | 合并激进，快速敲击被合并 | 能分开近事件，但可能多检 | 快速敲击场景 |
| `--edge-margin-ms` | 200.0 | 忽略区大，边界事件被忽略 | 忽略区小，边界伪峰可能进入 | 边界有伪峰时 |
| `--max-candidates` | 128 | 候选上限高，内存占用多 | 候选上限低，可能丢事件 | 事件特别多时 |

### estimate_distance.py 参数

| 参数 | 默认值 | 调大后果 | 调小后果 | 何时调 |
|------|--------|----------|----------|--------|
| `--medium` | air | 声速快，距离大 | 声速慢，距离小 | 介质不同时 |
| `--speed-mps` | 无 | 自定义声速 | 自定义声速 | 需要精确声速时 |

---

## 五、现场调试优先级

### 偏差过大（测出距离比实际大）

1. **优先检查介质选择**：是否选对了空气/水/铁/木/混凝土
2. **调整 `--speed-mps`**：使用实际介质的精确声速
3. **检查事件检测**：运行 `analyze_session_a.py` 查看事件是否准确

### 偏差过小（测出距离比实际小）

1. **检查事件配对**：运行 `analyze_session_a.py` 查看事件数是否一致
2. **增大 `--min-score-ratio`**：从 0.002 调到 0.005，减少误检
3. **增大 `--min-event-gap-ms`**：从 300 调到 400，避免回响干扰

### 事件检测不准

1. **调整 `--block-ms`**：
   - 事件被拆开 → 调大（如 15ms）
   - 事件被忽略 → 调小（如 8ms）

2. **调整 `--min-score-ratio`**：
   - 漏检 → 调小（如 0.001）
   - 误检 → 调大（如 0.005）

---

## 六、测试流程

### 第一步：事件检测

```bash
python analyze_session_a.py session_A.wav session_B.wav
```

检查输出的事件数是否一致（应该都是13个）

### 第二步：测距

```bash
python estimate_distance.py session_A.wav session_B.wav \
    --json-a session_A.json --json-b session_B.json
```

查看输出的距离值

### 第三步：调整参数（如有需要）

根据偏差情况调整参数，重复第一步和第二步

---

## 七、快速参考

### 空气测距
```bash
python estimate_distance.py session_A.wav session_B.wav --json-a session_A.json --json-b session_B.json
```

### 钢管测距
```bash
python estimate_distance.py session_A.wav session_B.wav --json-a session_A.json --json-b session_B.json --medium iron
```

### 水管测距
```bash
python estimate_distance.py session_A.wav session_B.wav --json-a session_A.json --json-b session_B.json --medium water
```

### 木材测距
```bash
python estimate_distance.py session_A.wav session_B.wav --json-a session_A.json --json-b session_B.json --medium wood
```

### 混凝土测距
```bash
python estimate_distance.py session_A.wav session_B.wav --json-a session_A.json --json-b session_B.json --medium concrete
```
