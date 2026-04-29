# `analyze_session_a.py` 公式说明表

这个文档只做一件事：  
把 [analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:1) 里关键计算公式的**输入、输出、单位、为什么这么算、调大调小的后果**讲清楚。

注意两点：

1. 这里解释的是“算法内部为什么这么设计”，不是数学上唯一正确答案。
2. 这些公式大多是**工程经验公式**，目标是让“冲击类事件检测”在真实噪声环境下更稳。

---

## 1. 毫秒和样本数怎么换

### 公式
```python
block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:160)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:310)

输入：
- `sample_rate`：采样率，单位是“样本/秒”
- `block_ms`：希望的时间块长度，单位是“毫秒”

输出：
- `block_size`：一个时间块包含多少个样本，单位是“样本”

为什么这么算：
- `sample_rate` 表示 1 秒里有多少个采样点
- `block_ms` 是毫秒，不是秒
- `1 秒 = 1000 毫秒`
- 所以先把 `block_ms` 换成秒：`block_ms / 1000.0`
- 再乘 `sample_rate`，得到这段时间里大约有多少个样本

例子：
- `sample_rate = 192000`
- `block_ms = 10`
- `block_size = 192000 * 10 / 1000 = 1920`

为什么还要 `round` 和 `int`：
- 因为样本数必须是整数
- `round` 是为了尽量接近真实值，而不是直接向下截断

为什么还要 `max(1, ...)`：
- 防止极端情况下算出来是 0
- 一个时间块最少也得包含 1 个样本

如果 `block_ms` 调大：
- 一个 block 会覆盖更长时间
- 更不容易把一次敲击后的回响拆成多次
- 但多个靠得很近的小事件会更容易被揉在一起

如果 `block_ms` 调小：
- 时间定位更细
- 更容易抓到弱小或很快的变化
- 但也更容易多检

---

## 2. 短窗口和长窗口为什么要除 `block_ms`

### 公式
```python
short_window = max(1, int(round(10.0 / block_ms)))
long_window = max(short_window + 1, int(round(120.0 / block_ms)))
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:165)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:168)

输入：
- `block_ms`：每个 block 多长，单位毫秒

输出：
- `short_window`：短窗口长度，单位是“block 个数”
- `long_window`：长窗口长度，单位是“block 个数”

为什么这么算：
- 这里想要的是“大约 10ms 的短窗口”和“大约 120ms 的长窗口”
- 但 `moving_average()` 接受的不是毫秒，而是“多少个点”
- 因为现在数组已经是按 block 压缩过的，所以一个点对应 `block_ms` 毫秒
- 所以：
  - `10.0 / block_ms` = 10ms 约等于多少个 block
  - `120.0 / block_ms` = 120ms 约等于多少个 block

为什么 `long_window` 要 `max(short_window + 1, ...)`：
- 长窗口必须真的比短窗口长
- 如果两者一样长，就没法区分“局部突然增强”和“慢变化背景”

---

## 3. 为什么 `feature = novelty * (1 + rise / rise_scale)`

### 公式
```python
novelty = np.maximum(smooth_short - smooth_long, 0.0)
rise = np.maximum(np.diff(smooth_short, prepend=smooth_short[0]), 0.0)
rise_scale = np.percentile(rise, 95) + 1.0
feature = novelty * (1.0 + rise / rise_scale)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:175)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:178)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:181)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:187)

输入：
- `smooth_short`：短窗口平滑后的局部包络
- `smooth_long`：长窗口平滑后的慢变化背景

输出：
- `feature`：每个 block 的“像不像新事件”的分数

分步解释：

### `novelty = max(smooth_short - smooth_long, 0)`
含义：
- 只有当“局部短时能量明显高于背景”时，才给正分
- 如果只是背景慢慢变大，或者根本没高出背景，就不给分

为什么 `max(..., 0)`：
- 我们只关心“突然增强”
- 不关心“突然变弱”

### `rise = max(diff(smooth_short), 0)`
含义：
- 看短时包络是不是在往上冲
- 上升越陡，越像一次短促冲击

### `rise_scale = 95分位数 + 1`
含义：
- 95 分位数代表“比较大的上升沿典型值”
- 用它做归一化，比直接拿最大值更稳
- `+1` 是防止分母为 0

### `feature = novelty * (1 + rise / rise_scale)`
含义：
- `novelty` 决定“是不是明显高于背景”
- `rise / rise_scale` 决定“上升沿够不够陡”
- 两者相乘表示：既要高于背景，又要像冲击

为什么不用相加：
- 如果只是简单相加，某一项特别大时可能掩盖另一项不足
- 相乘更像“必须同时满足两个条件”

---

## 4. 为什么用 `threshold = np.percentile(diff, 80)`

### 公式
```python
threshold = np.percentile(diff, 80)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:222)

输入：
- `diff`：峰值前局部信号的一阶差分

输出：
- `threshold`：局部起点搜索时的涨幅门槛

为什么这样算：
- 这里不是要找“最强跳变”
- 而是想找“开始明显抬头”的较早位置
- 用 80 分位数可以理解成：
  - 只保留局部里偏大的那一批上升
  - 但又不会像最大值那样太苛刻

如果分位数太高：
- 起点会偏晚

如果分位数太低：
- 起点会偏早，容易被小抖动提前触发

---

## 5. 为什么峰宽用 `peak * 0.25`

### 公式
```python
threshold = peak * 0.25
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:236)

输入：
- `peak`：这个候选峰的 feature 峰值

输出：
- `threshold`：定义峰宽边界的高度门槛

为什么这么算：
- 宽度不是看整个拖尾，而是看“这个峰明显成立的主体部分”
- 用峰值的 25% 当边界，是经验折中：
  - 太高：会把峰宽算得过窄
  - 太低：会把长拖尾、背景起伏也算进来

---

## 6. 为什么 `attack_ratio = peak_energy / pre_energy`

### 公式
```python
attack_ratio = peak_energy / pre_energy
decay_ratio = post_energy / peak_energy
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:351)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:353)

输入：
- `pre_energy`：峰前局部总量
- `peak_energy`：峰中心块大小
- `post_energy`：峰后拖尾总量

输出：
- `attack_ratio`：上升突然程度
- `decay_ratio`：峰后拖尾相对长度

为什么这样定义：
- 冲击类事件的典型特征是：
  - 峰前比较安静
  - 突然冒出一个强峰
  - 峰后有一定拖尾

所以：
- `peak / pre` 大，表示更像“突然冒出来”
- `post / peak` 大，表示峰后拖尾更明显

为什么都要 `+1.0`：
- 防止分母接近 0 时除零
- 也避免极小数值导致比值夸张到不稳定

---

## 7. 为什么候选阈值取两者 `max`

### 公式
```python
threshold = max(
    np.percentile(feature, percentile_threshold),
    np.max(feature) * min_score_ratio,
)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:386)

输入：
- `feature`
- `percentile_threshold`
- `min_score_ratio`

输出：
- 候选检测阈值

为什么要两套阈值：

### 第一项：`np.percentile(feature, percentile_threshold)`
作用：
- 从全局分布角度控制阈值
- 防止整体都很弱时，阈值掉得太低

### 第二项：`np.max(feature) * min_score_ratio`
作用：
- 从最强峰尺度角度控制阈值
- 防止有极强峰时，分位数阈值与主峰尺度脱节

为什么取 `max`：
- 取更严格的那一个
- 目的是“候选不要太滥”

---

## 8. 为什么 `candidate_gap_ms = max(80.0, block_ms * 8.0)`

### 公式
```python
candidate_gap_ms = max(80.0, block_ms * 8.0)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:468)

含义：
- 如果用户没给候选分组间隔，就按 block 大小自动给一个候选级分组尺度

为什么乘 8：
- 一个 block 太短，不足以代表“事件之间最小分组距离”
- 用 8 个 block 作为起步，相当于给一个“稍宽松但不是无限大”的分组间隔

为什么还要 `max(80.0, ...)`：
- 保底 80ms，防止 block_ms 很小时分组尺度也跟着小得离谱

---

## 9. 自动学习参数为什么乘这些比例

### 公式
```python
candidate_gap_ms = min(interval_p10_ms * 0.18, interval_median_ms * 0.15)
min_event_gap_ms = min(interval_p10_ms * 0.45, interval_median_ms * 0.50)
echo_gap_ms = min(interval_p10_ms * 0.34, interval_median_ms * 0.30)
recover_search_radius_ms = max(140.0, min_event_gap_ms * 0.55)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:569)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:572)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:573)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:576)

输入：
- `interval_p10_ms`：候选间隔的 10 分位数
- `interval_median_ms`：候选间隔的中位数

为什么用候选间隔：
- 它是在回答“这次录音里，事件彼此大概隔多久”
- 一旦知道这个尺度，就能反推很多内部间隔参数

为什么是这些比例：

### `candidate_gap_ms`
目标：
- 候选分组时就先把明显太近的峰归到一起

为什么是 `0.18 / 0.15`：
- 要远小于真实事件间隔
- 否则不同真实事件会过早并在一起

### `min_event_gap_ms`
目标：
- 最终确认阶段，尽量按“人耳数事件”的尺度合并

为什么是 `0.45 / 0.50`：
- 要明显大于候选分组间隔
- 但又不能大到把两个真实相邻敲击并掉

### `echo_gap_ms`
目标：
- 专门消掉一次敲击后的近邻反射

为什么比 `min_event_gap_ms` 略小：
- 回响通常比真实相邻事件更近
- 所以应先在更小的时间尺度上压掉

### `recover_search_radius_ms`
目标：
- 双文件复核时，在预测时间附近找弱候选

为什么是 `min_event_gap_ms * 0.55`：
- 搜索半径要与本次事件节奏成比例
- 但不能太小，否则时差一点就找不到
- 也不能太大，否则会搜进太多噪声

为什么还要 `clip(...)`：
- 给这些经验量加安全上下限
- 防止个别样本把自动参数拉飞

---

## 10. 为什么模板聚类阈值先 0.82，再退到 0.74

### 公式
```python
cluster_threshold = 0.82
if max_cluster_size < 3:
    cluster_threshold = 0.74
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:658)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:661)

含义：
- 模板学习时，先假设“同类事件应该很像”，所以先用较严格相似度门槛
- 如果严格门槛下根本聚不成团，就说明这次录音更散，需要放宽一点

为什么不是一次性固定一个值：
- 固定太高：弱端同类事件进不来
- 固定太低：不同形状的噪声也可能混入模板

---

## 11. 为什么模板向量加权平均用 `sqrt(score)`

### 公式
```python
weights = np.sqrt(score)
template_vector = np.average(..., weights=weights)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:682)

为什么不直接用 `score`：
- 直接用 `score` 会让超强峰权重过大
- 一两个特别强的事件会把模板“拽歪”

为什么不全都等权：
- 强事件通常更稳定、更可靠
- 完全等权又会让弱噪声成员影响过大

为什么取平方根：
- 这是折中办法
- 让强事件更重要，但重要程度不至于失控

---

## 12. 为什么确认阈值这样推

### 公式
```python
confirm_similarity = clip(percentile20(cluster_similarities) - 0.05, 0.70, 0.90)
recover_similarity = clip(confirm_similarity - 0.10, 0.60, 0.84)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:699)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:701)

为什么用 20 分位数：
- 它代表模板簇里“偏弱但仍然属于同类”的那一截成员
- 比最小值稳，比中位数更宽容

为什么再减 `0.05`：
- 给正式确认留一点余量
- 避免模板簇刚好边缘的真实事件被卡掉

为什么复核阈值再低 `0.10`：
- 弱端本地证据天然更弱
- 复核阶段本来就是“给弱端机会”

为什么还要 `clip`：
- 防止阈值太低导致滥检
- 也防止阈值太高导致只认最强峰

---

## 13. 为什么能量门槛这样推

### 公式
```python
min_confirm_energy = clip(percentile10(cluster_energies) * 0.35, 120000.0, 420000.0)
min_recover_energy = clip(min_confirm_energy * 0.55, 70000.0, 260000.0)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:707)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:711)

为什么用 10 分位数：
- 它代表模板簇里“比较弱但还可信”的那一截能量

为什么乘 `0.35`：
- 正式确认门槛不能等于模板簇弱端本身
- 要再往下留一些空间给环境衰减、远端、安装方式变化

为什么复核再乘 `0.55`：
- 复核阶段比直接确认更宽容

为什么要上下限：
- 上限防止强端把弱端全压死
- 下限防止门槛低到什么噪声都能进

---

## 14. 为什么宽度上限用 `max(40.0, median_width * 2.5)`

### 公式
```python
width_limit = max(40.0, template.median_width_ms * 2.5)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:822)

为什么这么算：
- 模板宽度可以有波动，但不能宽得离谱
- `模板中位宽度 * 2.5` 是相对约束
- `40.0ms` 是绝对下限保护，避免模板本身很窄时宽度上限也窄得太夸张

---

## 15. 为什么最终合并间隔用 `min_event_gap_ms / 1000 * 0.70`

### 公式
```python
merge_gap_s = max(0.16, min_event_gap_ms / 1000.0 * 0.70)
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:825)

输入：
- `min_event_gap_ms`：毫秒

输出：
- `merge_gap_s`：秒

为什么要除以 `1000`：
- 因为时间戳 `onset_s` / `peak_s` 全是“秒”
- 这里必须统一单位

为什么乘 `0.70`：
- 最终合并尺度通常略小于“最小事件间隔”
- 否则两个真实相邻事件也会被并掉

为什么再和 `0.16` 取 `max`：
- 给一个保底合并尺度
- 避免自动学习值过小导致仍然多检

---

## 16. 为什么 `pairwise = target - reference`

### 公式
```python
pairwise = target_times[:, None] - reference_times[None, :]
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:860)

含义：
- `pairwise[i, j]` 表示：
  - `target` 第 `i` 个事件
  - 比 `reference` 第 `j` 个事件
  - 晚了多少秒

为什么这样做：
- 如果两边录的是同一串事件，那么很多“正确配对”的时间差应该相近
- 所以先把所有可能配对的时间差都列出来，再用投票法找出最常见那个

---

## 17. 为什么偏移量要量化到 10ms

### 公式
```python
rounded = np.round(pairwise / 0.01) * 0.01
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:865)

为什么除以 `0.01`：
- `0.01 秒 = 10ms`
- 相当于先把时间差映射到“多少个 10ms 单位”

为什么再 `round` 再乘回来：
- 把所有时间差吸附到 10ms 网格
- 这样原本 `0.543s`、`0.547s`、`0.551s` 这种很接近的偏移，就都可能落到同一票箱里

为什么要这么做：
- 现实录音里事件定位总有小抖动
- 如果完全按精确浮点值投票，几乎每个偏移都不相同，投票就没意义

---

## 18. 为什么 `distance_penalty` 这样写

### 公式
```python
distance_penalty = abs(candidate.onset_s - center_time_s) / max(radius_ms / 1000.0, 1e-6)
rank = candidate.template_similarity - 0.20 * distance_penalty
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:970)  
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:974)

输入：
- `candidate.onset_s`：候选起点，单位秒
- `center_time_s`：预测起点，单位秒
- `radius_ms`：搜索半径，单位毫秒

输出：
- `distance_penalty`：相对偏离比例，无单位
- `rank`：用于排序的综合分数

分步解释：

### 分子
```python
abs(candidate.onset_s - center_time_s)
```
表示：
- 候选与预测时刻相差多少秒

### 分母
```python
radius_ms / 1000.0
```
表示：
- 把搜索半径从毫秒换成秒

### 整体含义
```python
distance_penalty = 时间偏差 / 搜索半径
```
得到的是一个“相对偏离比例”：
- 接近 `0`：非常接近预测位置
- 接近 `1`：已经偏到搜索半径边缘

### 为什么要 `max(..., 1e-6)`
- 防止半径极小或异常时分母变 0
- `1e-6` 只是一个数值保险丝，不是业务阈值

### 为什么 `rank = 相似度 - 0.20 * penalty`
- 模板相似度仍然是主要依据
- 距离只做轻度扣分
- `0.20` 是经验权重，表示“近一点更好”，但不能让时间距离压倒形状相似度

如果这个权重太大：
- 会过度偏好“更近但不太像”的候选

如果这个权重太小：
- 会过度偏好“更像但偏得有点远”的候选

---

## 19. 为什么 `0.05 秒` 用来判定同一个漏检点

### 公式
```python
abs(predicted_time - current) <= 0.05
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:984)

含义：
- 如果两个疑似漏检时间点相差不超过 50ms，就当成同一个位置

为什么这么做：
- 双文件复核时，同一个缺失点可能被多条逻辑反复触发
- 不去重的话，输出会出现一串几乎相同的疑似漏检时间

---

## 20. 为什么先侦察，再正式分析

### 流程
```python
scout_results -> infer_auto_parameters -> results -> build_template -> classify -> recover
```

位置：
[analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:1098) 之后

为什么不一次做完：
- 如果一开始就用固定参数正式检测，往往会带入场景偏见
- 侦察阶段的作用是先粗看这次录音：
  - 事件大概多宽
  - 间隔大概多大
  - 候选大概强弱分布怎样
- 有了这些统计量，正式阶段才能更接近“本次介质、本次安装、本次环境”

这就是“自动自学习”的核心。

---

## 21. 这份算法说明怎么和源码配合看

推荐顺序：

1. 先看 [analyze_session_a.py](/home/chao/workspace/signal-study/analyze_session_a.py:1) 文件顶部总说明  
2. 再看这个文档里的公式 1、3、7、9  
3. 然后读源码里的：
   - `build_feature()`
   - `collect_candidate_indices()`
   - `infer_auto_parameters()`
   - `build_template()`
   - `search_candidate_near_time()`
4. 遇到不懂的公式，再回这个文档按编号查

---

## 22. 最后一句实话

这些公式不是“物理定律”，而是为了把“人耳数得清的冲击事件”自动检测出来而设计的工程折中。  
所以你以后如果要改，不要先问“数学上能不能换”，而要先问：

1. 这个改动会不会让弱端更容易漏检？
2. 这个改动会不会让回响更容易多检？
3. 这个改动会不会让强端把弱端门槛拉得太高？
4. 这个改动是不是还适合“冲击类事件”这个前提？

只要你沿着这四个问题想，后面自己改算法就会稳很多。


起始时刻除为起始时刻=1.978979秒/起始时刻=2.580781秒