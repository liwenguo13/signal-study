#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_IP="192.168.1.136"
SLAVE_IP="192.168.1.154"
REMOTE_USER="ici"
REMOTE_PATH="/tmp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_banner() {
    echo "=========================================="
    echo "  PTP 对时服务 一键部署脚本"
    echo "=========================================="
    echo ""
    echo "  Master : ${MASTER_IP}  (ptp4l-master)"
    echo "  Slave  : ${SLAVE_IP}   (ptp4l-slave + phc2sys)"
    echo ""
}

choose_target() {
    >&2 echo "请选择要部署到哪台机器:"
    >&2 echo "  1) Master (${MASTER_IP})  — ptp4l-master.service"
    >&2 echo "  2) Slave  (${SLAVE_IP})   — ptp4l-slave.service + phc2sys.service"
    >&2 echo "  3) 两台都部署"
    >&2 echo "  q) 退出"
    >&2 echo ""
    read -r -p "请输入 [1/2/3/q]: " choice
    echo "$choice"
}

install_services() {
    local target_ip="$1"
    local target_name="$2"
    shift 2
    local service_files=("$@")

    echo ""
    echo -e "${YELLOW}>>> 部署到 ${target_name} (${target_ip})${NC}"

    # Check connectivity
    echo "检查连通性..."
    if ! ping -c 1 -W 2 "${target_ip}" &>/dev/null; then
        echo -e "${RED}✗ 无法 ping 通 ${target_ip}，跳过${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ 连通性正常${NC}"

    local services=""
    for sf in "${service_files[@]}"; do
        services="${services} ${sf}"
    done

    # Copy service files
    echo "上传 service 文件..."
    scp ${services} "${REMOTE_USER}@${target_ip}:${REMOTE_PATH}/"

    # Install and enable
    echo "安装并启用服务..."
    local install_cmds=""
    for sf in "${service_files[@]}"; do
        local sname
        sname="$(basename "$sf")"
        install_cmds="${install_cmds}
sudo cp ${REMOTE_PATH}/${sname} /etc/systemd/system/${sname} && \
sudo systemctl daemon-reload && \
sudo systemctl enable ${sname} && \
sudo systemctl restart ${sname} && \
echo '  ✓ ${sname} 已启用'"
    done

    ssh "${REMOTE_USER}@${target_ip}" "bash -s" <<< "${install_cmds}"

    echo ""
    echo -e "${GREEN}>>> ${target_name} 部署完成${NC}"
}

check_status() {
    local target_ip="$1"
    local target_name="$2"
    shift 2
    local service_names=("$@")

    echo ""
    echo -e "${YELLOW}>>> ${target_name} 服务状态${NC}"
    for sn in "${service_names[@]}"; do
        ssh "${REMOTE_USER}@${target_ip}" "systemctl is-active ${sn} 2>/dev/null || echo 'not-found'"
        ssh "${REMOTE_USER}@${target_ip}" "systemctl is-enabled ${sn} 2>/dev/null || echo 'not-found'"
    done
}

# --- Main ---
show_banner

choice=$(choose_target)

case "$choice" in
    1)
        install_services "$MASTER_IP" "Master" \
            "$SCRIPT_DIR/ptp4l-master.service"
        ;;
    2)
        install_services "$SLAVE_IP" "Slave" \
            "$SCRIPT_DIR/ptp4l-slave.service" \
            "$SCRIPT_DIR/phc2sys.service"
        ;;
    3)
        install_services "$MASTER_IP" "Master" \
            "$SCRIPT_DIR/ptp4l-master.service"
        install_services "$SLAVE_IP" "Slave" \
            "$SCRIPT_DIR/ptp4l-slave.service" \
            "$SCRIPT_DIR/phc2sys.service"
        ;;
    q|Q)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  部署完成！"
echo ""
echo "  常用管理命令:"
echo "    systemctl status ptp4l-master"
echo "    systemctl status ptp4l-slave"
echo "    systemctl status phc2sys"
echo "    journalctl -u ptp4l-slave -f"
echo "    journalctl -u phc2sys -f"
echo "=========================================="
