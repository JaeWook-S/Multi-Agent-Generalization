#!/bin/bash

# Resolve relative to this file; works with the spaces and Korean in the path.
launcher_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)" || exit 1
player_dir="$launcher_dir/diagnostics/mac410_compat"
player_python="$player_dir/.venv/bin/python"

echo "HiSSD — 이 Mac에서 SC2 4.10 리플레이 보기"
echo

player_status=0
# Keep the launcher alive while Python handles Ctrl+C and closes SC2. A caught
# (not ignored) signal is reset for the child, whose controller installs its
# own handler. The terminal prompt still appears after cancellation.
trap ':' INT
if [[ ! -x "$player_python" ]]; then
    echo "전용 Python 환경을 찾을 수 없습니다: $player_python"
    echo "diagnostics/mac410_compat/README_KO.md의 복구 방법을 확인해 주세요."
    player_status=1
else
    "$player_python" -u "$player_dir/play_replay.py" "$@" || player_status=$?
fi
trap - INT

echo
if [[ -t 0 && -t 1 ]]; then
    # Terminal opens a double-clicked .command as "script; exit". Returning
    # directly would end that session and leave a dead window. Replace this
    # launcher with an interactive login shell so the user gets a prompt.
    echo "재생 실행기를 마쳤습니다. 이제 이 창에서 터미널 명령을 입력할 수 있습니다."
    exec /bin/zsh -l
fi
exit "$player_status"
