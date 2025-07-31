#!/bin/bash
# monitor_and_notify.sh - 监控进程并在完成时发送通知
# ====================================================
#
# 使用方法：
#   ./monitor_and_notify.sh <PID> [进程描述]
#
# 示例：
#   ./monitor_and_notify.sh 12345 "理论生成"

PID=$1
DESCRIPTION=${2:-"进程"}

if [ -z "$PID" ]; then
    echo "使用方法: $0 <PID> [进程描述]"
    exit 1
fi

# 检查进程是否存在
if ! ps -p $PID > /dev/null; then
    echo "进程 $PID 不存在！"
    exit 1
fi

echo "开始监控进程 $PID ($DESCRIPTION)..."
echo "按 Ctrl+C 停止监控"

# 监控循环
while ps -p $PID > /dev/null; do
    sleep 10
done

# 进程结束，发送通知
echo "进程 $PID 已结束！"

# macOS 通知
if command -v osascript &> /dev/null; then
    osascript -e "display notification \"$DESCRIPTION (PID: $PID) 已完成！\" with title \"进程监控\""
fi

# 播放提示音
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true

echo "✅ $DESCRIPTION 已完成！"