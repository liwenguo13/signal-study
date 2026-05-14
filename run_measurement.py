#!/usr/bin/env python3
"""自动录音 + 拉取 + 测距，支持 N 次 session 并汇总结果。

用法:
    python run_measurement.py --num-sessions 5 --duration-sec 10
    python run_measurement.py --num-sessions 3 --duration-sec 10 --min-distance 0 --max-distance 3
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MASTER_IP = "192.168.1.136"
SLAVE_IP = "192.168.1.154"
SSH_USER = "ici"
RECORDER = "recorder/record_session.py"
PI_WORK_DIR = "shuiting-project"
SESSIONS_BASE = "sessions"

SCRIPT_DIR = Path(__file__).resolve().parent
RECORD_A_BASE = SCRIPT_DIR / "record_A"
RECORD_B_BASE = SCRIPT_DIR / "record_B"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
DISTANCE_SCRIPT = SCRIPT_DIR / "get_distance.py"


@dataclass
class SessionResult:
    index: int
    distance_m: float | None = None
    status: str = "pending"
    reason: str = ""
    raw_output: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动录音 + 拉取 + 测距")
    parser.add_argument("--num-sessions", type=int, default=1, help="录音次数 N")
    parser.add_argument("--duration-sec", type=int, default=10, help="每次录音时长 (秒)")
    parser.add_argument("--session-id", type=str, default="test01", help="会话标识")
    parser.add_argument("--min-distance", type=float, default=0.0, help="最小预估距离 (米)")
    parser.add_argument("--max-distance", type=float, default=3.0, help="最大预估距离 (米)")
    parser.add_argument("--sound-speed", type=float, default=1400.0, help="声速 (米/秒)")
    parser.add_argument("--master-ip", type=str, default=MASTER_IP)
    parser.add_argument("--slave-ip", type=str, default=SLAVE_IP)
    parser.add_argument("--ssh-user", type=str, default=SSH_USER)
    parser.add_argument("--no-ssh", action="store_true", help="跳过 SSH，直接运行 get_distance.py")
    return parser


def ssh(ip: str, command: str, user: str) -> list[str]:
    return ["ssh", f"{user}@{ip}", f"cd ~/{PI_WORK_DIR} && {command}"]


def scp_pull(ip: str, remote_path: str, local_path: Path, user: str) -> list[str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return ["scp", f"{user}@{ip}:{remote_path}", str(local_path)]


def trigger_recording(ip: str, node_id: str, duration: int, output_dir: str, device: str, user: str) -> subprocess.Popen:
    command = (
        f"python3 {RECORDER} "
        f"--node-id {node_id} "
        f"--duration-sec {duration} "
        f"--sample-rate 192000 "
        f"--channels 2 "
        f"--format S24_3LE "
        f"--device {device} "
        f"--output-dir {output_dir}"
    )
    cmd_list = ssh(ip, command, user)
    print(f"  [{node_id}] {ip} 开始录音...")
    return subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_recording(proc_a: subprocess.Popen, proc_b: subprocess.Popen, timeout: int) -> dict:
    results: dict = {}
    for label, proc in (("A", proc_a), ("B", proc_b)):
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            results[label] = {"rc": proc.returncode, "stdout": stdout.strip(), "stderr": stderr.strip()}
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            results[label] = {"rc": -1, "stdout": "", "stderr": f"录音超时 ({timeout}s)"}
    return results


def pull_files(ip: str, node_id: str, session_id: str, session_num: int, local_dir: Path, user: str) -> bool:
    remote_subdir = f"{SESSIONS_BASE}/{session_id}/{session_num}"
    wav_name = f"session_{node_id}.wav"
    json_name = f"session_{node_id}.json"

    session_local_dir = local_dir / str(session_num)
    session_local_dir.mkdir(parents=True, exist_ok=True)
    local_wav = session_local_dir / wav_name
    local_json = session_local_dir / json_name

    ok = True
    for remote_name, local_path in [(wav_name, local_wav), (json_name, local_json)]:
        remote_path = f"~/{PI_WORK_DIR}/{remote_subdir}/{remote_name}"
        cmd = scp_pull(ip, remote_path, local_path, user)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"  SCP 失败 [{node_id}] {remote_path}: {result.stderr.strip()}")
                ok = False
            else:
                print(f"  SCP 完成 [{node_id}] -> {local_path}")
        except subprocess.TimeoutExpired:
            print(f"  SCP 超时 [{node_id}] {remote_path}")
            ok = False
    return ok


def run_distance_for_session(args: argparse.Namespace, session_num: int) -> tuple[bool, str]:
    a_wav = RECORD_A_BASE / str(session_num) / "session_A.wav"
    a_json = RECORD_A_BASE / str(session_num) / "session_A.json"
    b_wav = RECORD_B_BASE / str(session_num) / "session_B.wav"
    b_json = RECORD_B_BASE / str(session_num) / "session_B.json"

    cmd = [
        str(VENV_PYTHON), str(DISTANCE_SCRIPT),
        "--a-wav", str(a_wav),
        "--a-json", str(a_json),
        "--b-wav", str(b_wav),
        "--b-json", str(b_json),
        "--min-distance", str(args.min_distance),
        "--max-distance", str(args.max_distance),
        "--sound-speed", str(args.sound_speed),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "get_distance.py 超时"
    except FileNotFoundError:
        return False, f"get_distance.py 未找到 ({DISTANCE_SCRIPT})"


def parse_distance(output: str) -> tuple[float | None, str]:
    match = re.search(r"median_distance_m:\s*([\d.]+)", output)
    if match:
        return float(match.group(1)), "ok"

    match = re.search(r"status:\s*(\S+)", output)
    status = match.group(1) if match else "unknown"

    match = re.search(r"reason:\s*(.+)", output)
    reason = match.group(1).strip() if match else "无法解析输出"

    return None, f"{status}: {reason}"


def print_summary(results: list[SessionResult]) -> None:
    print()
    print("=" * 60)
    print("  测距结果汇总")
    print("=" * 60)
    print(f"{'Session':<8} {'Distance(m)':<12} {'Status':<15} Reason")
    print("-" * 60)

    valid_distances = []
    for r in results:
        dist_str = f"{r.distance_m:.2f}" if r.distance_m is not None else "-"
        print(f"{r.index:<8} {dist_str:<12} {r.status:<15} {r.reason}")
        if r.distance_m is not None:
            valid_distances.append(r.distance_m)

    print("-" * 60)

    if len(valid_distances) >= 1:
        import numpy as np
        median = float(np.median(valid_distances))
        if len(valid_distances) >= 2:
            mad = float(np.median(np.abs(np.array(valid_distances) - median)))
            print(f"  中位数 (median): {median:.2f} m")
            print(f"  MAD:             {mad:.2f} m")
            if mad < 0.10:
                print("  一致性: 良好")
            elif mad < 0.50:
                print("  一致性: 一般")
            else:
                print("  一致性: 较差")
        else:
            print(f"  距离: {median:.2f} m")
        print(f"  有效 session: {len(valid_distances)}/{len(results)}")
    else:
        print("  无有效测距结果")

    print("=" * 60)


def run_session(args: argparse.Namespace, session_num: int) -> SessionResult:
    result = SessionResult(index=session_num)
    session_subdir = f"{SESSIONS_BASE}/{args.session_id}/{session_num}"

    print(f"\n{'─' * 50}")
    print(f"Session {session_num}/{args.num_sessions}")
    print(f"{'─' * 50}")

    timeout = args.duration_sec + 15

    proc_a = trigger_recording(args.master_ip, "A", args.duration_sec, session_subdir, "plughw:1,0", args.ssh_user)
    proc_b = trigger_recording(args.slave_ip, "B", args.duration_sec, session_subdir, "plughw:0,0", args.ssh_user)

    rec_results = wait_recording(proc_a, proc_b, timeout)

    for label in ("A", "B"):
        r = rec_results[label]
        if r["rc"] != 0:
            print(f"  [{label}] 录音失败 (rc={r['rc']}): {r['stderr'][:200]}")
        else:
            print(f"  [{label}] 录音完成")

    if rec_results["A"]["rc"] != 0 or rec_results["B"]["rc"] != 0:
        result.status = "recording_failed"
        result.reason = "一台或两台录音失败"
        return result

    print("  等待文件落盘...")
    time.sleep(2)

    print(f"  拉取文件到 {RECORD_A_BASE}/ 和 {RECORD_B_BASE}/ ...")
    ok_a = pull_files(args.master_ip, "A", args.session_id, session_num, RECORD_A_BASE, args.ssh_user)
    ok_b = pull_files(args.slave_ip, "B", args.session_id, session_num, RECORD_B_BASE, args.ssh_user)

    if not ok_a or not ok_b:
        result.status = "scp_failed"
        result.reason = "文件拉取失败"
        return result

    print("  运行 get_distance.py ...")
    ok, output = run_distance_for_session(args, session_num)
    distance, status_reason = parse_distance(output)

    result.distance_m = distance
    result.raw_output = output

    if distance is not None:
        result.status = "ok"
        result.reason = "ok"
        print(f"  距离: {distance:.2f} m")
    else:
        result.status = status_reason.split(":")[0] if ":" in status_reason else status_reason
        result.reason = status_reason.split(":", 1)[1].strip() if ":" in status_reason else status_reason
        print(f"  测距失败: {result.reason}")
        if output.strip():
            print(f"  [debug] get_distance.py 原始输出:")
            for line in output.strip().splitlines():
                print(f"    | {line}")

    return result


def run_no_ssh(args: argparse.Namespace) -> list[SessionResult]:
    results = []
    for i in range(1, args.num_sessions + 1):
        result = SessionResult(index=i)
        print(f"\n{'─' * 50}")
        print(f"Session {i}/{args.num_sessions} (skip ssh)")
        print(f"{'─' * 50}")

        ok, output = run_distance_for_session(args, i)
        distance, status_reason = parse_distance(output)

        result.distance_m = distance
        result.raw_output = output

        if distance is not None:
            result.status = "ok"
            result.reason = "ok"
            print(f"  距离: {distance:.2f} m")
        else:
            result.status = status_reason.split(":")[0] if ":" in status_reason else status_reason
            result.reason = status_reason.split(":", 1)[1].strip() if ":" in status_reason else status_reason
            print(f"  测距失败: {result.reason}")
            if output.strip():
                print(f"  [debug] get_distance.py 原始输出:")
                for line in output.strip().splitlines():
                    print(f"    | {line}")
        results.append(result)
    return results


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_sessions < 1:
        print("错误: --num-sessions 必须 >= 1", file=sys.stderr)
        return 1

    RECORD_A_BASE.mkdir(parents=True, exist_ok=True)
    RECORD_B_BASE.mkdir(parents=True, exist_ok=True)

    if args.no_ssh:
        results = run_no_ssh(args)
    else:
        results = []
        for i in range(1, args.num_sessions + 1):
            result = run_session(args, i)
            results.append(result)

    print_summary(results)

    valid = [r for r in results if r.distance_m is not None]
    return 0 if len(valid) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
