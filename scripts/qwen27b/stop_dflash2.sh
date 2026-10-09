#!/usr/bin/env bash
#
# 停止 DFlash2 服务 (cuda2/3, 端口 8092).
# 只按 --port 8092 精确匹配目标进程, 不用 pkill -f, 也不影响别的实例(如 cuda0/1 调试实例).
#
set -euo pipefail

PORT=${PORT:-8092}
PIDFILE=${PIDFILE:-/tmp/ftllm_dflash2.pid}

targets=""
for p in $(pgrep -x ftllm || true); do
    if tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -qx -- "$PORT"; then
        targets="$targets $p"
    fi
done

if [ -z "$targets" ]; then
    echo "端口 $PORT 上没有 ftllm 进程"
    rm -f "$PIDFILE"
    exit 0
fi

echo "停止 ftllm (port=$PORT, pid:$targets)"
kill $targets
for _ in $(seq 1 30); do
    alive=""
    for p in $targets; do
        kill -0 "$p" 2>/dev/null && alive="$alive $p"
    done
    [ -z "$alive" ] && { echo "已停止"; rm -f "$PIDFILE"; exit 0; }
    sleep 1
done

echo "30s 仍未退出, SIGKILL:$targets"
kill -9 $targets 2>/dev/null || true
rm -f "$PIDFILE"
echo "已强制停止"
