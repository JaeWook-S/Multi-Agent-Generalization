"""Manual integration check: real SC2 + a PTY-generated Ctrl+C (no mocks)."""
import argparse
import errno
import json
import os
from pathlib import Path
import pty
import select
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("startup", "loading", "playback"))
args = parser.parse_args()
root = Path(__file__).resolve().parent
launcher = root.parent.parent / "Play_SC2_Replay.command"
replay = root.parent.parent / "marine-hard-expert/30000_replays/3m_2026-10-27-08-24-05.SC2Replay"
game_logs = Path.home() / "Library/Application Support/Blizzard/StarCraft II/GameLogs"


def reports():
    return {p.name for p in game_logs.iterdir()
            if p.is_dir() and (" Crash" in p.name or " Error" in p.name)}


before = reports()
output = bytearray()
started = time.monotonic()
scheduled = None
sent = 0
shell_exit = False
marker = {"startup": "SC2 기동 대기: 0초", "loading": "맵:",
          "playback": "재생 진행: 224"}[args.phase].encode()
pid, fd = pty.fork()
if pid == 0:
    os.execv("/bin/bash", ["/bin/bash", str(launcher), str(replay)])
os.set_blocking(fd, False)
try:
    while time.monotonic() - started < 150:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            output.extend(chunk)
            print(chunk.decode(errors="replace"), end="", flush=True)
        now = time.monotonic()
        if scheduled is None and marker in output:
            scheduled = now + 0.3
        if scheduled is not None and sent < 2 and now >= scheduled + sent * 0.4:
            os.write(fd, b"\x03")
            sent += 1
        if not shell_exit and "이제 이 창에서 터미널 명령".encode() in output:
            os.write(fd, b"unset HISTFILE\nexit\n")
            shell_exit = True
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
    else:
        raise RuntimeError("PTY interrupt test timed out")
finally:
    os.close(fd)
result = json.loads((root / "last_playback_result.json").read_text())
version = result.get("player_version", "unknown").split("-")[0].replace(".", "")
prefix = root / f"validation_v{version}_{args.phase}_ctrlc"
prefix.with_suffix(".terminal.log").write_bytes(output)
prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
summary = {
    "phase": args.phase, "sent_terminal_interrupts": sent,
    "interrupt_count": result.get("interrupt_count"),
    "interrupted_request": result.get("interrupted_request"),
    "process_exit": result.get("process_exit"),
    "graceful_exit": result.get("graceful_exit"),
    "forced_shutdown": result.get("forced_shutdown"),
    "returned_to_shell": shell_exit, "new_reports": sorted(reports() - before),
}
prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print("\nCONTROL_C_VALIDATION", json.dumps(summary, ensure_ascii=False), flush=True)
if not (sent and summary["graceful_exit"] and shell_exit and not summary["new_reports"]):
    raise SystemExit(1)
