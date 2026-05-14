# 自动测距系统：完整操作手册

本手册覆盖从零开始部署整个系统的所有步骤。最终效果：在笔记本电脑上运行一条命令，自动控制两台树莓派录音、拉取文件、计算距离并显示结果。

---

## 目录

1. [系统拓扑与角色](#1-系统拓扑与角色)
2. [笔记本电脑部署](#2-笔记本电脑部署)
   - 2.1 [克隆项目](#21-克隆项目)
   - 2.2 [安装 Python 虚拟环境](#22-安装-python-虚拟环境)
   - 2.3 [配置 SSH 免密登录（极其重要）](#23-配置-ssh-免密登录极其重要)
   - 2.4 [验证项目文件](#24-验证项目文件)
3. [树莓派部署](#3-树莓派部署)
4. [PTP 高精度对时](#4-ptp-高精度对时)
5. [执行测距流程](#5-执行测距流程)
6. [目录结构总览](#6-目录结构总览)
7. [故障排查](#7-故障排查)
   - 7.1 [网络与连接](#71-网络与连接问题)
   - 7.2 [SSH 免密登录](#72-ssh-免密登录问题)
   - 7.3 [Python 环境](#73-python-环境问题)
   - 7.4 [录音](#74-录音问题)
   - 7.5 [SCP 文件拉取](#75-scp-文件拉取问题)
   - 7.6 [get_distance.py 测距](#76-get_distancepy-测距问题)
   - 7.7 [PTP 对时](#77-ptp-对时问题)
   - 7.8 [录音文件 / JSON 元数据](#78-录音文件--json-元数据问题)
   - 7.9 [磁盘空间](#79-磁盘空间问题)
   - 7.10 [其他意外问题](#710-其他意外问题)
8. [快速命令速查卡](#8-快速命令速查卡)

---

## 1. 系统拓扑与角色

```
┌──────────────────────────────────────────────────────────────────┐
│                       笔记本电脑 (你的开发机)                       │
│                                                                  │
│  ~/workspace/signal-study/                                       │
│  ├── run_measurement.py    ← 主控程序（你只运行这一个）               │
│  ├── get_distance.py       ← 测距算法                              │
│  ├── record_A/                                                    │
│  │   ├── 1/session_A.wav + session_A.json                         │
│  │   ├── 2/session_A.wav + session_A.json                         │
│  │   └── N/...                                                    │
│  └── record_B/                                                    │
│      ├── 1/session_B.wav + session_B.json                         │
│      ├── 2/session_B.wav + session_B.json                         │
│      └── N/...                                                    │
│                                                                  │
│  执行: python run_measurement.py --num-sessions 5                 │
│         │                                                        │
│         ├── SSH ──► Master Pi ── 录音 A ──┐                      │
│         │                                  │                      │
│         └── SSH ──► Slave Pi  ── 录音 B ──┤                      │
│                                            │                      │
│         SCP ◄──── Master Pi ── 拉取文件 ───┘                      │
│         SCP ◄──── Slave Pi  ── 拉取文件 ───┘                      │
│                                            │                      │
│         运行 get_distance.py ◄─────────────┘                      │
│         显示: 距离 = X.XX m                                       │
└──────────────────────────────────────────────────────────────────┘
        │                                   │
        ▼                                   ▼
┌──────────────────────┐    ┌──────────────────────┐
│ 树莓派 Master (136)   │    │ 树莓派 Slave (154)    │
│ 192.168.1.136        │    │ 192.168.1.154        │
│ Node ID: A           │    │ Node ID: B (声源端)   │
│                      │    │                      │
│ ~/shuiting-project/  │    │ ~/shuiting-project/  │
│ ├── recorder/        │    │ ├── recorder/        │
│ │   └── record_      │    │ │   └── record_      │
│ │       session.py   │    │ │       session.py   │
│ └── sessions/        │    │ └── sessions/        │
│     └── test01/      │    │     └── test01/      │
│         ├── 1/       │    │         ├── 1/       │
│         │   ├── session_A.wav    │   ├── session_B.wav
│         │   └── session_A.json   │   └── session_B.json
│         ├── 2/       │    │         ├── 2/       │
│         └── 3/       │    │         └── 3/       │
└──────────────────────┘    └──────────────────────┘

  两声听器分别接在管道两端，B端旁敲击，A端接收。
```

| 角色 | IP | Node ID | 职责 |
|------|-----|---------|------|
| **笔记本电脑** | 任意 | — | 总控：触发录音、拉取文件、计算距离、显示结果 |
| **Master Pi** | `192.168.1.136` | A | 接收端录音 |
| **Slave Pi** | `192.168.1.154` | B | 声源端录音（敲击端） |

---

## 2. 笔记本电脑部署

本节在新电脑上从头部署本项目。

### 2.1 克隆项目

```bash
cd ~/workspace
git clone <你的仓库地址> signal-study
cd signal-study
```

### 2.2 安装 Python 虚拟环境

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 重新打开终端使 uv 生效

# 创建 venv 并安装依赖
cd ~/workspace/signal-study
uv venv .venv
source .venv/bin/activate
uv pip install numpy scipy
```

验证安装：

```bash
source .venv/bin/activate
python -c "import numpy; import scipy; print('OK')"
```

输出 `OK` 即成功。

### 2.3 配置 SSH 免密登录（极其重要）

> **这是整个系统最关键的配置步骤。** SSH 免密登录不通，`run_measurement.py` 完全无法工作。

#### 2.3.1 检查是否已有 SSH 密钥

```bash
# 查看已有密钥
ls -la ~/.ssh/
```

如果看到 `id_rsa` + `id_rsa.pub` 或 `id_ed25519` + `id_ed25519.pub`，说明已有密钥，跳到 2.3.3。

#### 2.3.2 生成新 SSH 密钥

```bash
# 方式一：RSA 4096 位（兼容性最好）
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 方式二：Ed25519（更安全更快，较新的系统支持）
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

参数说明：
| 参数 | 含义 |
|------|------|
| `-t rsa` / `-t ed25519` | 密钥类型 |
| `-b 4096` | 密钥位数（仅 RSA） |
| `-f ~/.ssh/id_rsa` | 输出文件路径 |
| `-N ""` | 空密码短语（不设密码，否则脚本无法自动使用） |

#### 2.3.3 复制公钥到树莓派

```bash
# 方法一：ssh-copy-id（推荐，自动处理权限）
ssh-copy-id ici@192.168.1.136
ssh-copy-id ici@192.168.1.154
```

**如果 `ssh-copy-id` 命令不存在**（某些系统不带此工具），用手动方法：

```bash
# 方法二：手动复制公钥
cat ~/.ssh/id_rsa.pub | ssh ici@192.168.1.136 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
cat ~/.ssh/id_rsa.pub | ssh ici@192.168.1.154 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**如果使用 ed25519 密钥**：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub ici@192.168.1.136
ssh-copy-id -i ~/.ssh/id_ed25519.pub ici@192.168.1.154
```

#### 2.3.4 验证免密登录

```bash
ssh ici@192.168.1.136 "echo 'Master OK'"
ssh ici@192.168.1.154 "echo 'Slave OK'"
```

应该**不提示任何密码**直接输出 `Master OK` / `Slave OK`。

**如果第一次 SSH 提示 `yes/no`**，输入 `yes` 回车（这是正常的 host key 确认，只出现一次）。

#### 2.3.5 配置 SSH Config（可选但推荐）

创建 `~/.ssh/config` 简化后续操作：

```bash
cat >> ~/.ssh/config << 'EOF'
Host master
    HostName 192.168.1.136
    User ici
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no

Host slave
    HostName 192.168.1.154
    User ici
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
EOF

chmod 600 ~/.ssh/config
```

配置后可以用更简短的方式访问：

```bash
ssh master "echo OK"
ssh slave "echo OK"
```

#### 2.3.6 SSH 免密常见故障速查

| 症状 | 原因 | 解决 |
|------|------|------|
| `Permission denied (publickey)` | 公钥没复制成功 | 重做 2.3.3 |
| 仍然提示输入密码 | Pi 上 `authorized_keys` 权限不对 | SSH 到 Pi，`chmod 600 ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 755 ~` |
| `ssh: connect to host 192.168.1.136 port 22: No route to host` | 笔记本不在同一网段 | 笔记本也要设置 `192.168.1.x` 静态 IP 或用网线直连 |
| `Host key verification failed.` | Pi 重装过系统，host key 变了 | `ssh-keygen -R 192.168.1.136` 清除旧 key |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` | 中间人攻击风险或 IP 换了设备 | 确认设备无误后：`ssh-keygen -R 192.168.1.136` |
| SSH 连接极慢（5-10 秒才连上） | Pi 上 DNS 反查超时 | SSH 到 Pi，`sudo sed -i 's/#UseDNS yes/UseDNS no/' /etc/ssh/sshd_config && sudo systemctl restart sshd` |

### 2.4 验证项目文件

```bash
cd ~/workspace/signal-study

# 确认核心文件存在
ls -la run_measurement.py get_distance.py analyze_session_a.py wav_signal.py

# 确认 venv 可用
.venv/bin/python --version
```

---

## 3. 树莓派部署

本节在**两台树莓派上都执行**（除非特别指明）。

### 3.1 系统环境确认

```bash
# 确认 Python 版本 >= 3.10
python3 --version

# 确认 ALSA 工具
which arecord || sudo apt install -y alsa-utils

# 确认树莓派型号（必须是 Pi 5 才有 PTP 硬件支持）
cat /proc/device-tree/model
```

### 3.2 项目目录结构

两台树莓派上的目录结构相同（但录音时生成的文件不同）：

```
~/
└── shuiting-project/
    ├── recorder/
    │   └── record_session.py    ← 录音程序（已存在）
    └── sessions/
        └── <session_id>/         ← 由 record_session.py 自动创建
            └── <N>/
                ├── session_A.wav    (Master 上) 或 session_B.wav  (Slave 上)
                └── session_A.json   (Master 上) 或 session_B.json (Slave 上)
```

> 确保 `~/shuiting-project/recorder/record_session.py` 存在。如果不存在，需要先把录音程序部署到两台 Pi 上。

### 3.3 确认录音设备

```bash
arecord -l
```

期望输出示例：

```
**** List of CAPTURE Hardware Devices ****
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
```

记下 `card` 和 `device` 编号。例如 `card 1, device 0` 对应 `plughw:1,0`。

| 角色 | 设备参数 |
|------|----------|
| Master (A) | `plughw:1,0` |
| Slave (B) | `plughw:0,0` |

如果设备号不同，需要在 `run_measurement.py` 中修改 `trigger_recording` 里的 `--device` 参数。

### 3.4 确认项目路径

```bash
cd ~/shuiting-project
ls recorder/record_session.py
```

应输出 `recorder/record_session.py`，否则路径不对。

---

## 4. PTP 高精度对时

PTP 对时确保两台 Pi 的系统时钟同步到微秒级，是测距精度的基础。

### 4.1 硬件支持验证（两台都做）

```bash
sudo ethtool -T eth0
```

**必须包含以下行**：

```
hardware-transmit
hardware-receive
hardware-raw-clock
PTP Hardware Clock: 0
```

如果缺少任一行，说明不是树莓派 5 或内核不支持，无法继续。

### 4.2 配置静态 IP（两台都做）

Pi 之间用网线直连，需要设置静态 IP。

```bash
# 编辑 dhcpcd 配置
sudo nano /etc/dhcpcd.conf
```

**在 Master (192.168.1.136) 上**，文件末尾追加：

```
interface eth0
static ip_address=192.168.1.136/24
```

**在 Slave (192.168.1.154) 上**，文件末尾追加：

```
interface eth0
static ip_address=192.168.1.154/24
```

配置完成后：

```bash
sudo systemctl restart dhcpcd
```

验证连通性：

```bash
# Master 上 ping Slave
ping -c 3 192.168.1.154

# Slave 上 ping Master
ping -c 3 192.168.1.136
```

期望 3 个包全部返回，延迟 < 1ms。

### 4.3 安装 linuxptp（两台都做）

```bash
sudo apt update
sudo apt install -y linuxptp ethtool iproute2
```

### 4.4 禁用中断聚合（两台都做）

```bash
sudo ethtool -C eth0 rx-usecs 0
```

### 4.5 创建 PTP 配置文件（两台都做）

**Master (192.168.1.136)**，创建 `/etc/linuxptp/ptp4l-master.conf`：

```bash
sudo mkdir -p /etc/linuxptp
sudo tee /etc/linuxptp/ptp4l-master.conf > /dev/null << 'EOF'
[global]
serverOnly             1
time_stamping          hardware
network_transport      UDPv4
delay_mechanism        E2E
logging_level          6
EOF
```

**Slave (192.168.1.154)**，创建 `/etc/linuxptp/ptp4l-slave.conf`：

```bash
sudo mkdir -p /etc/linuxptp
sudo tee /etc/linuxptp/ptp4l-slave.conf > /dev/null << 'EOF'
[global]
clientOnly             1
time_stamping          hardware
network_transport      UDPv4
delay_mechanism        E2E
logging_level          6
EOF
```

### 4.6 部署 systemd 服务（从笔记本电脑执行）

项目已准备好 systemd 服务文件，让 PTP 开机自启。

```bash
cd ~/workspace/signal-study
./systemd-services/install.sh
```

按提示选择部署目标：

```
1) Master   — ptp4l-master.service
2) Slave    — ptp4l-slave.service + phc2sys.service
3) 两台都部署  ← 推荐选这个
```

### 4.7 验证 PTP 同步状态

```bash
# 在 Master 上查看
ssh ici@192.168.1.136 "systemctl status ptp4l-master"

# 在 Slave 上查看
ssh ici@192.168.1.154 "systemctl status ptp4l-slave"
ssh ici@192.168.1.154 "systemctl status phc2sys"
```

所有服务应显示 `active (running)`。

查看 Slave 的 phc2sys 日志确认同步质量：

```bash
ssh ici@192.168.1.154 "journalctl -u phc2sys -n 10 --no-pager"
```

期望看到 `master offset` 在几十~几百纳秒范围内。

---

## 5. 执行测距流程

### 5.1 准备工作检查清单

在笔记本电脑上执行：

- [ ] SSH 免密登录两台 Pi 正常
- [ ] 两台 Pi 的 PTP 服务运行中
- [ ] 两台 Pi 的录音设备已接好
- [ ] 声源（B 端）旁准备好锤子/敲击工具
- [ ] 管道两端已就位

### 5.2 运行命令

```bash
cd ~/workspace/signal-study

python run_measurement.py \
    --num-sessions 5 \
    --duration-sec 10 \
    --min-distance 0 \
    --max-distance 3 \
    --sound-speed 1400
```

### 5.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-sessions` | `1` | 录音次数 N。程序会做 N 轮"录音→拉取→测距" |
| `--duration-sec` | `10` | 每次录音时长（秒）。敲击节奏应在此时间内完成 |
| `--session-id` | `test01` | 会话标识，树莓派上保存目录名的一部分 |
| `--min-distance` | `0` | 最小预估距离（米），缩小搜索范围 |
| `--max-distance` | `3` | 最大预估距离（米），缩小搜索范围 |
| `--sound-speed` | `1400` | 声速（米/秒），水中约 1400-1500 |
| `--master-ip` | `192.168.1.136` | Master Pi IP |
| `--slave-ip` | `192.168.1.154` | Slave Pi IP |
| `--ssh-user` | `ici` | SSH 用户名 |

### 5.4 执行过程解读

运行后你会看到：

```
──────────────────────────────────────────────────
Session 1/5
──────────────────────────────────────────────────
  [A] 192.168.1.136 开始录音...
  [B] 192.168.1.154 开始录音...
  [A] 录音完成                    ← 录音结束
  [B] 录音完成
  等待文件落盘...
  拉取文件到 record_A/ 和 record_B/ ...
  SCP 完成 [A] -> .../record_A/1/session_A.wav
  SCP 完成 [A] -> .../record_A/1/session_A.json
  SCP 完成 [B] -> .../record_B/1/session_B.wav
  SCP 完成 [B] -> .../record_B/1/session_B.json
  运行 get_distance.py ...
  距离: 2.45 m                    ← 测距结果

──────────────────────────────────────────────────
Session 2/5
  ...（同上）...
```

**关键时机**：看到 `[A] 开始录音...` 和 `[B] 开始录音...` 后，立即在 B 端（154）水听器旁边开始敲击。约 8-15 次，间隔 1-3 秒略随机，在录音结束前完成。

### 5.5 最终结果汇总

```
============================================================
  测距结果汇总
============================================================
Session  Distance(m)  Status          Reason
------------------------------------------------------------
1        2.45         ok              ok
2        2.51         ok              ok
3        2.38         ok              ok
4        2.42         ok              ok
5        2.48         ok              ok
------------------------------------------------------------
  中位数 (median): 2.45 m
  MAD:             0.04 m
  一致性: 良好
  有效 session: 5/5
============================================================
```

| 指标 | 含义 |
|------|------|
| **中位数 (median)** | N 次测距的中位数，即最终距离估计 |
| **MAD** | 中位绝对偏差，反映 N 次结果的一致性。越小越好 |
| **一致性 良好** | MAD < 0.10m |
| **一致性 一般** | MAD 0.10~0.50m |
| **一致性 较差** | MAD > 0.50m，需排查 |

### 5.6 仅测距模式（跳过 SSH 录音）

当 `record_A/` 和 `record_B/` 中已有录音文件时，可跳过 SSH 直接测距：

```bash
python run_measurement.py --no-ssh --num-sessions 1
```

> 前提：`record_A/1/session_A.wav`、`record_A/1/session_A.json`、`record_B/1/session_B.wav`、`record_B/1/session_B.json` 已存在。

### 5.7 输出 JSON（手动解析）

`get_distance.py` 的输出也可以直接查看：

```bash
source .venv/bin/activate
python get_distance.py \
    --a-wav record_A/1/session_A.wav \
    --a-json record_A/1/session_A.json \
    --b-wav record_B/1/session_B.wav \
    --b-json record_B/1/session_B.json \
    --min-distance 0 \
    --max-distance 3 \
    --sound-speed 1400
```

---

## 6. 目录结构总览

```
笔记本电脑:
~/workspace/signal-study/
├── run_measurement.py          ← 主控程序
├── get_distance.py             ← 测距算法 (GCC-PHAT)
├── analyze_session_a.py        ← 事件检测引擎
├── wav_signal.py               ← WAV 解码器
├── .venv/                      ← Python 虚拟环境
├── record_A/                   ← 从 Master Pi 拉取的 A 端文件
│   ├── 1/
│   │   ├── session_A.wav
│   │   └── session_A.json
│   ├── 2/
│   └── ...
├── record_B/                   ← 从 Slave Pi 拉取的 B 端文件
│   ├── 1/
│   │   ├── session_B.wav
│   │   └── session_B.json
│   ├── 2/
│   └── ...
└── systemd-services/           ← PTP 服务文件 + 部署脚本


Master Pi (192.168.1.136):
~/shuiting-project/
├── recorder/
│   └── record_session.py       ← 录音程序
└── sessions/
    └── test01/
        ├── 1/
        │   ├── session_A.wav   ← 录音输出
        │   └── session_A.json  ← 录音元数据
        ├── 2/
        └── ...


Slave Pi (192.168.1.154):
~/shuiting-project/
├── recorder/
│   └── record_session.py       ← 录音程序
└── sessions/
    └── test01/
        ├── 1/
        │   ├── session_B.wav   ← 录音输出
        │   └── session_B.json  ← 录音元数据
        ├── 2/
        └── ...
```

---

## 7. 故障排查

### 7.1 网络与连接问题

#### 7.1.1 笔记本 ping 不通树莓派

```
症状: ping 192.168.1.136 无响应 或 "No route to host"

原因: 笔记本和树莓派不在同一子网。

解决:
  1. 笔记本也配静态 IP:
     sudo ip addr add 192.168.1.100/24 dev eth0
     (或其他 192.168.1.x 地址)
  2. 或者用网线直连笔记本到树莓派所在的交换机/路由器
  3. 确认笔记本与 Pi 之间物理连接正确
```

#### 7.1.2 SSH 连接超时 / 拒绝连接

```
症状: ssh: connect to host 192.168.1.136 port 22: Connection timed out
      ssh: connect to host ... port 22: Connection refused

原因和解决:
  Connection timed out:
    → Pi 未开机/未联网/IP 地址变了
    → ping 确认网络是否通
    → 在 Pi 上: ip addr show eth0 确认 IP

  Connection refused:
    → Pi 上 SSH 服务未运行
    → 在 Pi 上: sudo systemctl start ssh && sudo systemctl enable ssh
```

#### 7.1.3 笔记本有 WiFi 和有线两个网络

```
症状: SSH 走了 WiFi 而非有线网络，导致无法连接 192.168.1.x

解决: 检查路由表
  ip route show

  如果有 WiFi 的默认路由优先级高于有线:
  sudo ip route add 192.168.1.0/24 dev eth0

  这会强制访问 192.168.1.x 的流量走有线网卡。
```

#### 7.1.4 防火墙阻止 SSH

```
症状: 能 ping 通但 SSH 无响应 (Connection refused)

检查:
  # 在 Pi 上
  sudo iptables -L -n
  sudo ufw status

关闭防火墙:
  sudo ufw disable          # 或
  sudo iptables -F
```

---

### 7.2 SSH 免密登录问题

#### 7.2.1 Permission denied (publickey)

```
症状: ssh 提示 Permission denied (publickey)

逐步排查:
  1. 检查本地密钥: ls -la ~/.ssh/
  2. 检查公钥内容: cat ~/.ssh/id_rsa.pub
  3. SSH 到 Pi (用密码): ssh -o PreferredAuthentications=password ici@192.168.1.136
  4. 在 Pi 上检查:
     cat ~/.ssh/authorized_keys     # 是否包含笔记本的公钥
     ls -la ~/.ssh/                 # 权限是否正确
     chmod 700 ~/.ssh
     chmod 600 ~/.ssh/authorized_keys
     chmod 755 ~
  5. 在 Pi 上查看日志: sudo tail -50 /var/log/auth.log
```

#### 7.2.2 仍然提示输入密码（公钥已复制）

```
原因: 最常见的是文件权限问题。

在 Pi 上执行:
  chmod 755 ~
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys

  # 如果还是不行，检查 SELinux (极少见):
  restorecon -R -v ~/.ssh
```

#### 7.2.3 Host key verification failed

```
症状: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!

原因: Pi 重装了系统或 IP 被分配给了另一台设备。

解决:
  ssh-keygen -R 192.168.1.136
  ssh-keygen -R 192.168.1.154
  然后重新 SSH 连接，输入 yes 确认新 key。
```

#### 7.2.4 SSH 连接很慢（延迟 5+ 秒）

```
原因: Pi 的 SSH 服务尝试对连接 IP 做 DNS 反向解析，但 DNS 不可用。

在 Pi 上执行:
  sudo sed -i 's/#UseDNS yes/UseDNS no/' /etc/ssh/sshd_config
  sudo systemctl restart sshd
```

---

### 7.3 Python 环境问题

#### 7.3.1 uv 命令不存在

```
症状: uv: command not found

解决:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # 关闭终端重新打开，或:
  source ~/.bashrc    # bash 用户
  source ~/.zshrc     # zsh 用户
```

#### 7.3.2 uv venv 创建失败

```
症状: error: No Python installation found

解决:
  # 确认系统有 Python
  python3 --version

  # 如果没有，安装:
  sudo apt install -y python3 python3-venv python3-pip  # Ubuntu/Debian

  # 指定 Python 路径创建 venv:
  uv venv .venv --python python3.12
```

#### 7.3.3 uv pip install 失败（网络问题）

```
症状: error sending request for url ... 或 timeout

解决:
  # 设置代理（如果有）
  export HTTPS_PROXY=http://your-proxy:port
  uv pip install numpy scipy

  # 或者不使用 uv，直接用 pip:
  source .venv/bin/activate
  pip install numpy scipy
```

#### 7.3.4 numpy 或 scipy 安装编译失败

```
症状: error: can't find Rust compiler 或漫长的编译过程

原因: 某些平台需要从源码编译 scipy，缺少编译器。

解决:
  # 安装编译依赖
  sudo apt install -y build-essential gfortran python3-dev

  # 或安装预编译版本（锁定版本号避免从源码编译）:
  uv pip install numpy==1.26.4 scipy==1.13.0
```

#### 7.3.5 run_measurement.py 找不到 .venv

```
症状: No such file or directory: '.../.venv/bin/python'

原因: run_measurement.py 硬编码了 .venv 路径，必须先在项目目录创建 venv。

解决:
  cd ~/workspace/signal-study
  uv venv .venv
  source .venv/bin/activate
  uv pip install numpy scipy
```

#### 7.3.6 NumPy 1.x / 2.x 版本冲突

```
症状: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.3

原因: 系统装了一个版本的 scipy（链接 NumPy 1.x），但 venv 里装的是 NumPy 2.x。

解决:
  source .venv/bin/activate
  uv pip uninstall scipy numpy -y 2>/dev/null
  uv pip install numpy scipy
  # 重新安装确保版本兼容
```

---

### 7.4 录音问题

#### 7.4.1 arecord: device or resource busy

```
症状: arecord: main:831: audio open error: Device or resource busy

原因: ALSA 设备被另一个进程占用。

解决:
  # 查看谁在用
  sudo fuser /dev/snd/*

  # 杀掉占用进程
  sudo fuser -k /dev/snd/*

  # 或者重启
  sudo reboot
```

#### 7.4.2 arecord: command not found

```
症状: arecord: command not found

解决:
  sudo apt update
  sudo apt install -y alsa-utils
```

#### 7.4.3 录音文件大小为 0

```
症状: WAV 文件大小 0 字节

检查:
  # 确认录音设备号正确
  arecord -l

  # 确认设备支持指定参数
  arecord -D plughw:1,0 -c 2 -r 192000 -f S24_3LE -d 3 /tmp/test.wav
  ls -la /tmp/test.wav     # 应有内容

  # 如果设备号是 hw:1,0 而不是 plughw:1,0，用 plughw 做重采样兼容
```

#### 7.4.4 录音程序崩溃 (rc ≠ 0)

```
症状: [A] 录音失败 (rc=1)

排查步骤:
  1. 手动在 Pi 上运行录音命令:
     ssh ici@192.168.1.136 "cd ~/shuiting-project && python3 recorder/record_session.py --node-id A --duration-sec 5 --sample-rate 192000 --channels 2 --format S24_3LE --device plughw:1,0 --output-dir sessions/test01/manual_test"

  2. 查看完整错误信息:
     ssh ici@192.168.1.136 "cd ~/shuiting-project && python3 recorder/record_session.py ... 2>&1"

  3. 常见原因:
     - device 参数错误（plughw:1,0 vs plughw:0,0）
     - Python 版本太旧（< 3.10）
     - record_session.py 文件损坏
     - 磁盘已满: df -h ~
```

#### 7.4.5 录音文件不完整 / 时长远不到预期

```
症状: total_samples 远小于 expected（如预期 1920000，实际只有 500000）

原因:
  1. 录音过程中 ALSA 缓冲区溢出 (xrun)
  2. SD 卡写入速度不够
  3. USB 声卡不稳定

解决:
  1. 降低采样率或声道数（牺牲精度）
  2. 使用更快的 SD 卡 (Class 10 / U3)
  3. 检查 SD 卡: sudo fsck /dev/mmcblk0p2
```

#### 7.4.6 录音设备号因重启改变

```
症状: 上次录音正常的设备号，重启后不对了

原因: USB 声卡每次插入可能分配不同的 card 编号。

解决:
  # 用设备名代替数字（更稳定）
  arecord -l      # 记下设备名称
  arecord -L      # 查看可用 PCM 名称

  # 然后用名称如 plughw:Device 而非 plughw:1,0
```

---

### 7.5 SCP 文件拉取问题

#### 7.5.1 SCP 返回 No such file or directory

```
症状: SCP 失败 [A] ... No such file or directory

排查:
  # 检查远程文件是否存在
  ssh ici@192.168.1.136 "ls -la ~/shuiting-project/sessions/test01/1/"

  # 检查目录名是否正确
  ssh ici@192.168.1.136 "ls -la ~/shuiting-project/sessions/test01/"

  # 可能原因:
  # - session_id 不对（check run_measurement.py 的 --session-id）
  # - record_session.py 生成的文件名不是 session_A.wav
  # - 录音目录路径拼写错误
```

#### 7.5.2 SCP 传输很慢

```
症状: 单个 WAV 文件 (~11MB) SCP 超过 30 秒

原因:
  1. WiFi 速度慢（如果笔记本用 WiFi）
  2. 网线质量问题

解决:
  1. 笔记本也接有线网络
  2. 检查网线是否 Cat5e 以上
  3. 增加 SCP 超时: 修改 run_measurement.py 中 scp_pull 的 timeout 值
```

#### 7.5.3 本地目录权限不足

```
症状: SCP 失败 Permission denied (本地)

解决:
  chmod 755 ~/workspace/signal-study/record_A
  chmod 755 ~/workspace/signal-study/record_B
  # 或确认当前用户有写权限
```

---

### 7.6 get_distance.py 测距问题

#### 7.6.1 输出 "status: error / reason: no A-side knock events found"

```
症状: get_distance.py 在 A 端录音中找不到敲击事件

原因:
  1. 敲击声太弱（麦克风灵敏度不够）
  2. A 端和 B 端接反了（敲击在 A 端但 A 端文件其实是 B 端的）
  3. --event-threshold-z 阈值太高

解决:
  1. 降低阈值: --event-threshold-z 5.0
  2. 减小帧长: --event-frame-ms 3.0
  3. 检查录音文件是否正确: 用 Audacity 打开 WAV 看波形
  4. 确认敲击在 B 端 (Slave)，A 端录音文件来自 Master
```

#### 7.6.2 所有 event status 都是 weak 或 unstable

```
症状: 输出中 status 列全是 "weak" 或 "unstable"

原因:
  - B 端信号太弱，GCC-PHAT 找不到可靠延迟
  - 搜索窗不够大
  - 频率段不匹配信号特征

解决:
  1. 扩大搜索范围: --min-distance 0 --max-distance 10
  2. 调整频段: --bands "100-500,500-2000,2000-8000"
  3. 降低最小同意频段数: --min-bands-agree 2
  4. 降低峰值阈值: --band-peak-threshold 0.3
```

#### 7.6.3 测距结果始终为 0 或非常小

```
症状: median_distance_m: 0.00 或接近 0

原因:
  1. --sound-speed 太大导致时延换算距离 ≈ 0
  2. PTP 对时没工作，两端时间戳实际对齐但偏移量计算错误
  3. --min-distance / --max-distance 设置太小

解决:
  1. 检查 --sound-speed 值
  2. 验证 PTP 同步: ssh ici@192.168.1.154 "journalctl -u phc2sys -n 5"
  3. 手动检查 start_time_realtime_ns 差值是否合理:
     cat record_A/1/session_A.json | python3 -c "import json,sys; print(json.load(sys.stdin)['start_time_realtime_ns'])"
     cat record_B/1/session_B.json | python3 -c "import json,sys; print(json.load(sys.stdin)['start_time_realtime_ns'])"
```

#### 7.6.4 MAD 很大（一致性差）

```
症状: MAD > 0.50 m

原因:
  1. 每次敲击力度/位置不一致
  2. 录音中有背景噪声
  3. 部分敲击的 GCC-PHAT 匹配错误

解决:
  1. 统一敲击位置和力度
  2. 确保录音环境安静
  3. 用 --debug 查看每个事件的 band_candidates 输出，确认多频段是否一致
```

#### 7.6.5 get_distance.py 运行极慢（超过 60 秒）

```
症状: get_distance.py 超时

原因: WAV 文件太大（192kHz 10秒 = 1,920,000 采样点），GCC-PHAT 对每帧做 FFT 开销大。

解决:
  1. 减少 max-events: --max-events 5（只分析前 5 个事件）
  2. 检查是否 event 太多: 先用 --debug 看 detected_a_events 数量
  3. 增加超时: 修改 run_measurement.py 中 subprocess.run 的 timeout 参数
```

---

### 7.7 PTP 对时问题

#### 7.7.1 ethtool -T eth0 无 PTP Hardware Clock

```
症状: PTP Hardware Clock: none

原因: 不是树莓派 5，或内核版本太旧。

解决:
  # 确认硬件: cat /proc/device-tree/model
  # 升级内核: sudo apt update && sudo apt upgrade
  # 非 Pi 5 无法使用硬件 PTP，只能用软件 PTP（精度差，不推荐）
```

#### 7.7.2 ptp4l: interface eth0 is down

```
症状: port 1: interface eth0 is down

解决:
  sudo ip link set eth0 up
```

#### 7.7.3 Slave 一直 UNCALIBRATED 状态

```
症状: Slave 一直显示 UNCALIBRATED，不变成 SLAVE

原因:
  1. Master 的 ptp4l 未启动
  2. 网络不通
  3. 中间经过不支持 PTP 的交换机

解决:
  1. 确认 Master ptp4l 在运行: ssh ici@192.168.1.136 "systemctl status ptp4l-master"
  2. 确认网络: ping 192.168.1.136
  3. 必须网线直连！普通交换机不转发 PTP 包
```

#### 7.7.4 master offset 持续很大（> 10000 ns）

```
症状: phc2sys 日志中 master offset 在微秒甚至毫秒级别

解决:
  # 重新禁用中断聚合（重启后可能恢复默认值）
  ssh ici@192.168.1.136 "sudo ethtool -C eth0 rx-usecs 0"
  ssh ici@192.168.1.154 "sudo ethtool -C eth0 rx-usecs 0"

  # 重启 PTP 服务
  ssh ici@192.168.1.136 "sudo systemctl restart ptp4l-master"
  ssh ici@192.168.1.154 "sudo systemctl restart ptp4l-slave"

  # 等 10 秒后检查
  ssh ici@192.168.1.154 "journalctl -u phc2sys -n 5 --no-pager"
```

#### 7.7.5 PTP 服务没有开机自启

```
症状: Pi 重启后 PTP 服务没运行

检查:
  ssh ici@192.168.1.136 "systemctl is-enabled ptp4l-master"
  ssh ici@192.168.1.154 "systemctl is-enabled ptp4l-slave"

如果输出 disabled 或 not-found:
  # 重新运行部署脚本
  cd ~/workspace/signal-study && ./systemd-services/install.sh
```

#### 7.7.6 重启后 eth0 静态 IP 丢失

```
症状: Pi 重启后 IP 变成了自动分配的，不再是 192.168.1.136/154

原因: 静态 IP 配置在 /etc/dhcpcd.conf 中但 dhcpcd 服务未启用。

解决:
  sudo systemctl enable dhcpcd
  sudo systemctl restart dhcpcd
  ip addr show eth0   # 检查 IP
```

---

### 7.8 录音文件 / JSON 元数据问题

#### 7.8.1 session_X.json 缺少 start_time_realtime_ns

```
症状: KeyError: 'start_time_realtime_ns'

原因: record_session.py 版本太旧，没有记录时间戳。

解决: 更新树莓派上的 record_session.py 到最新版本。
      确保 JSON 包含 "start_time_realtime_ns" 字段。
```

#### 7.8.2 两次录音的 start_time_realtime_ns 差值异常

```
症状: --debug 输出 start_delta_s_B_minus_A: 180.xxx

原因: PTP 对时未工作，导致两端时间戳相差很大（180 秒意味着实际录音时间差了 3 分钟）。

解决: 重新启动 PTP 服务，等待同步完成再开始录音。
  ssh ici@192.168.1.154 "journalctl -u phc2sys -n 3 --no-pager"
  # 确认 master offset 在百纳秒级
```

#### 7.8.3 WAV 采样率与预期不符

```
症状: sample rate mismatch: A=44100, B=192000

原因: record_session.py 调用 arecord 时传错了参数。

解决: 检查 record_session.py 的参数是否正确。
      或者手动指定: 在 get_distance.py 中不用 --a-wav，直接传正确的 WAV。
```

---

### 7.9 磁盘空间问题

#### 7.9.1 树莓派磁盘空间不足

```
症状: record_session.py 写入失败 / WAV 文件截断

检查:
  ssh ici@192.168.1.136 "df -h ~"
  ssh ici@192.168.1.154 "df -h ~"

每次录音约产生 11 MB WAV 文件（10s × 192kHz × 2ch × 24bit）。
如果多次录音未清理，累积很快。

清理:
  rm -rf ~/shuiting-project/sessions/test01/
  # 或移动到外接存储
```

#### 7.9.2 笔记本电脑磁盘空间不足

```
症状: SCP 失败 / record_A 不增长

检查:
  df -h ~/workspace/signal-study/

每次 N 次录音会存储 N × 22 MB (A+B 两路)。
```

---

### 7.10 其他意外问题

#### 7.10.1 两个树莓派的 r.sh / record_session.py 设备参数不同

```
症状: Master 用 plughw:1,0 能录，Slave 用的 plughw:0,0 失败

原因: 两台 Pi 的 USB 声卡插在不同端口，card 编号不同。

解决: 在每台 Pi 上单独运行 arecord -l，确认正确的 card 号。
      如果需要不同的 device 参数，目前需要直接修改 run_measurement.py
      中 trigger_recording 函数的 device 参数。
```

#### 7.10.2 git clone 失败（网络问题）

```
症状: fatal: unable to access '...': Could not resolve host

解决:
  # 如果有 U 盘，直接复制整个文件夹
  cp -r /media/usb/signal-study ~/workspace/

  # 或配置 git 代理
  git config --global http.proxy http://proxy:port
```

#### 7.10.3 Python 版本太低

```
症状: SyntaxError / unsupported Python version

解决:
  python3 --version

  如果 < 3.10:
  sudo apt install -y python3.12 python3.12-venv
  uv venv .venv --python python3.12
```

#### 7.10.4 笔记本与树莓派时间不同步

```
症状: get_distance.py 的 --debug 显示 b_start_offset_ms 很大

说明: 笔记本的系统时间不影响测距精度（测距只用 Pi 的 PTP 时间），
      但 SCP 文件的时间戳可能不准确。这不影响功能。
```

#### 7.10.5 run_measurement.py 中途退出，Pi 上还有残留录音进程

```
症状: 重新运行 run_measurement.py 时录音失败 (device busy)

解决:
  # 杀掉 Pi 上可能残留的 arecord 或 python 录音进程
  ssh ici@192.168.1.136 "pkill -f record_session.py; pkill arecord" 2>/dev/null
  ssh ici@192.168.1.154 "pkill -f record_session.py; pkill arecord" 2>/dev/null
```

#### 7.10.6 MAC 地址变化导致网卡名称变化

```
症状: eth0 不存在，变成了 enx... 格式的名称

原因: 新版 Linux 使用 predictable network interface names。

解决:
  # 查看实际网卡名
  ip link show

  # 如果不是 eth0，修改 /etc/dhcpcd.conf 中的接口名
  # 并修改 systemd 服务和 ptp4l 命令中的 -i 参数
```

---

## 8. 快速命令速查卡

### 笔记本电脑

```bash
# 激活 venv
cd ~/workspace/signal-study && source .venv/bin/activate

# 完整测距流程（5 次录音）
python run_measurement.py \
    --num-sessions 5 --duration-sec 10 \
    --min-distance 0 --max-distance 3 --sound-speed 1400

# 仅测距（不录音）
python run_measurement.py --no-ssh --num-sessions 1

# 单独运行测距查看详情
python get_distance.py \
    --a-wav record_A/1/session_A.wav --a-json record_A/1/session_A.json \
    --b-wav record_B/1/session_B.wav --b-json record_B/1/session_B.json \
    --min-distance 0 --max-distance 3 --sound-speed 1400 --debug

# 验证 SSH 连通性
ssh ici@192.168.1.136 "echo OK"
ssh ici@192.168.1.154 "echo OK"

# 查看 Pi 上的文件
ssh ici@192.168.1.136 "ls -la ~/shuiting-project/sessions/test01/"
```

### Master Pi (192.168.1.136)

```bash
# PTP 服务状态
systemctl status ptp4l-master

# 查看录音设备
arecord -l

# 手动测试录音
arecord -D plughw:1,0 -c 2 -r 192000 -f S24_3LE -d 3 /tmp/test.wav

# 查看已保存的录音文件
ls -la ~/shuiting-project/sessions/
```

### Slave Pi (192.168.1.154)

```bash
# PTP 服务状态
systemctl status ptp4l-slave
systemctl status phc2sys

# 查看 PTP 同步日志
journalctl -u phc2sys -n 10 --no-pager

# 查看录音设备
arecord -l

# 手动测试录音
arecord -D plughw:0,0 -c 2 -r 192000 -f S24_3LE -d 3 /tmp/test.wav

# 查看已保存的录音文件
ls -la ~/shuiting-project/sessions/
```

---

**以上就是完整操作手册。从新电脑部署到最终测距结果，每一步都有明确命令。**
