"""Local macOS SC2 4.10 replay player using the tested process-local ICU shim."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import zlib

import mpyq
import websocket
from s2clientprotocol import sc2api_pb2 as sc


class ReplayStopped(Exception):
    """A user stop request handled between complete API transactions."""


@contextmanager
def intercept_interrupt(handler):
    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def parse_start_time(value):
    """Parse seconds or [hours:]minutes:seconds in the native replay clock."""
    try:
        parts = [float(part) for part in value.strip().split(":")]
        if not 1 <= len(parts) <= 3 or any(not math.isfinite(p) or p < 0 for p in parts):
            raise ValueError
        if len(parts) > 1 and any(p >= 60 for p in parts[1:]):
            raise ValueError
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    except ValueError:
        raise argparse.ArgumentTypeError("시작 시간은 90 또는 1:30처럼 입력해 주세요.")


def save_png(image, destination):
    if image.bits_per_pixel != 24 or not image.data:
        return False
    width, height = image.size.x, image.size.y
    pixels = image.data
    if len(pixels) != width * height * 3:
        raise RuntimeError("Unexpected RGB frame length")
    # The API can return a black buffer before the first rendered frame.
    # Retry on a later observation rather than saving that buffer as evidence.
    if not any(pixels):
        return False
    rows = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3]
                    for y in range(height))
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    destination.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", nargs="?")
    parser.add_argument("--sc2", default="/Applications/StarCraft II")
    parser.add_argument("--probe", action="store_true", help="Run quickly and save validation frames.")
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--compact-ui", action="store_true", help="Use the compact side-message replay interface.")
    parser.add_argument("--default-ui", dest="compact_ui", action="store_false",
                        help="Use the original SC2 message placement.")
    parser.set_defaults(compact_ui=True)
    parser.add_argument("--start-at", type=parse_start_time,
                        help="Start at replay time, e.g. 90 or 1:30; fast-forwards through earlier events.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    replay_root = root.parent.parent
    candidates = sorted(
        replay_root.rglob("*.SC2Replay"),
        key=lambda p: [int(part) if part.isdigit() else part
                       for part in re.split(r"(\d+)", str(p.relative_to(replay_root)))])
    if args.replay:
        replay_path = Path(args.replay).expanduser().resolve()
    elif candidates:
        for index, path in enumerate(candidates, 1):
            print(f"{index:2}. {path.relative_to(replay_root)}")
        while True:
            choice = input("재생할 리플레이 번호 [Enter: 3m]: ").strip()
            if not choice:
                replay_path = next((p for p in candidates if p.name.startswith("3m_")), candidates[0])
                break
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                replay_path = candidates[int(choice) - 1]
                break
            print(f"1부터 {len(candidates)} 사이의 번호를 입력해 주세요.")
    else:
        parser.error("Specify a .SC2Replay path")
    start_seconds = args.start_at
    if start_seconds is None and args.replay is None:
        while True:
            value = input("시작 시간 [Enter: 처음부터 / 예: 1:30]: ").strip()
            try:
                start_seconds = parse_start_time(value) if value else 0.0
                break
            except argparse.ArgumentTypeError as error:
                print(error)
    start_loop = round((start_seconds or 0.0) * 22.4)
    metadata = json.loads(mpyq.MPQArchive(str(replay_path)).read_file("replay.gamemetadata.json"))
    if metadata["GameVersion"] != "4.10.0.75689":
        raise RuntimeError("This compatibility player is restricted to SC2 4.10.0.75689")
    sc2 = Path(args.sc2)
    executable = sc2 / "Versions/Base75689/SC2.app/Contents/MacOS/SC2"
    map_path = sc2 / "Maps" / metadata["MapName"]
    shim = root / "libsc2_icu_compat.dylib"
    for path in (executable, map_path, shim):
        if not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = os.environ.copy()
    environment["DYLD_INSERT_LIBRARIES"] = str(shim)
    command = [str(executable), "-dataDir", str(sc2) + "/", "-dataVersion", metadata["DataVersion"],
               "-listen", "127.0.0.1", "-port", str(port), "-displayMode", "0",
               "-windowwidth", "1280", "-windowheight", "720",
               "-displayunitstatus", "All", "-verbose"]
    if args.compact_ui:
        if not (sc2 / "Interfaces/HiSSDCompact.SC2Interface").is_file():
            raise FileNotFoundError("SC2 Interfaces 폴더에 HiSSDCompact.SC2Interface가 없습니다.")
        command.extend(["-replayinterface", "HiSSDCompact.SC2Interface",
                        "-observerinterface", "HiSSDCompact.SC2Interface"])
    run_dir = root / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    logfile = run_dir / "sc2.log"
    connection = None
    sequence = 0
    result = {"replay": str(replay_path), "metadata": metadata, "observations": 0,
              "player_version": "2.4-health-bars", "runtime_log": str(logfile),
              "render_data_requested": args.probe, "requested_start_loop": start_loop,
              "isolated_game_session": True, "compact_messages": args.compact_ui,
              "unit_status_bars": "All"}
    playback_started = None
    stop_requested = False
    ready_to_stop = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        result["interrupt_count"] = result.get("interrupt_count", 0) + 1
        if not stop_requested:
            stop_requested = True
            # Do not raise in the middle of a websocket send/receive. Complete
            # that transaction, then send SC2 its normal quit request.
            message = ("중지 요청을 받았습니다. 게임을 정상 종료하는 중입니다." if ready_to_stop
                       else "중지 요청을 받았습니다. 게임 준비가 끝나면 정상 종료합니다.")
            os.write(1, ("\n" + message + "\n").encode())

    def check_stop():
        if stop_requested and ready_to_stop:
            raise ReplayStopped

    def rpc(field, payload):
        nonlocal sequence
        check_stop()
        result["last_request"] = field
        sequence += 1
        request = sc.Request(id=sequence)
        getattr(request, field).CopyFrom(payload)
        connection.send_binary(request.SerializeToString())
        response = sc.Response.FromString(connection.recv())
        check_stop()
        if response.error:
            raise RuntimeError("; ".join(response.error))
        if not response.HasField(field):
            raise RuntimeError(f"Unexpected API response for {field}: {response.WhichOneof('response')}")
        return response
    print("리플레이:", replay_path.name, flush=True)
    print("Mac 호환 실행기 v2.4 — 체력 바 기본 표시", flush=True)
    print("게임 안 타임라인·일시정지/재생 버튼은 지원하지 않습니다.", flush=True)
    print("시작 위치는 실행기의 시작 시간으로 지정하고, 중지는 터미널의 Control+C를 사용하세요.", flush=True)
    print("SC2 창이 열릴 때까지 기다려 주세요. 보통 10~60초 걸립니다.", flush=True)
    with intercept_interrupt(request_stop), logfile.open("w") as log:
        # A terminal sends Ctrl+C to its foreground process group. SC2 treats
        # SIGINT as a crash, so only this Python controller may receive it.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        result["pid"] = process.pid
        try:
            for attempt in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"SC2 exited ({process.returncode}); see {logfile}")
                try:
                    connection = websocket.create_connection(
                        f"ws://127.0.0.1:{port}/sc2api", timeout=3,
                        http_no_proxy=["127.0.0.1", "localhost"])
                    break
                except (OSError, websocket.WebSocketException):
                    if attempt % 10 == 0:
                        print(f"SC2 기동 대기: {attempt}초", flush=True)
                    time.sleep(1)
            if connection is None:
                raise RuntimeError("SC2 API startup timed out")
            connection.settimeout(120)
            ping = rpc("ping", sc.RequestPing()).ping
            result["engine_version"] = ping.game_version
            print("엔진:", ping.game_version, flush=True)
            info = rpc("replay_info", sc.RequestReplayInfo(replay_data=replay_path.read_bytes())).replay_info
            print("맵:", metadata["MapName"], "총 game loop:", info.game_duration_loops, flush=True)
            if start_loop >= info.game_duration_loops and start_loop:
                raise ValueError(f"시작 시간이 리플레이 길이({info.game_duration_loops / 22.4:.1f}초)를 벗어났습니다.")
            if args.max_loops and start_loop >= args.max_loops:
                raise ValueError("--max-loops는 --start-at으로 정한 시작 loop보다 커야 합니다.")
            interface = sc.InterfaceOptions(raw=True, score=True)
            # The native Mac window renders independently. RGB API frames are
            # only needed for diagnostics; requesting them in normal playback
            # adds a large payload to every observation and can slow playback.
            if args.probe:
                interface.render.resolution.x = 960
                interface.render.resolution.y = 720
                interface.render.minimap_resolution.x = 128
                interface.render.minimap_resolution.y = 128
            start = sc.RequestStartReplay(replay_data=replay_path.read_bytes(), map_data=map_path.read_bytes(),
                                         options=interface, observed_player_id=1, disable_fog=True)
            started = rpc("start_replay", start).start_replay
            if started.HasField("error"):
                raise RuntimeError(f"Replay start failed: {started.error} {started.error_details}")
            print("리플레이 재생 시작 — SC2 게임 창에서 볼 수 있습니다.", flush=True)
            playback_started = time.monotonic()
            if not args.probe:
                print("종료하려면 이 터미널에서 Control+C를 누르세요.", flush=True)
            step_size = 112 if args.probe else 1
            screenshot_saved = False
            camera_centered = False
            next_progress_loop = start_loop + 224
            start_position_reached = False
            if start_loop:
                print(f"시작 위치까지 빠르게 진행 중: {start_seconds:g}초 ({start_loop} loop)", flush=True)
            while True:
                tick = time.monotonic()
                reply = rpc("observation", sc.RequestObservation())
                observation = reply.observation.observation
                loop = observation.game_loop
                result["observations"] += 1
                result["last_game_loop"] = loop
                result["last_unit_count"] = len(observation.raw_data.units)
                # SC2 4.10 can hang on quit before its initial rendering and
                # simulation setup settles. Keep servicing the normal replay
                # path briefly, then honour a stop queued during loading.
                if loop > 0 and time.monotonic() - playback_started >= 2:
                    ready_to_stop = True
                check_stop()
                moved_camera = False
                if not camera_centered:
                    units = [u for u in observation.raw_data.units
                             if u.alliance in (1, 4) and u.health > 0]
                    if units:
                        move = sc.ObserverAction()
                        move.camera_move.world_pos.x = sum(u.pos.x for u in units) / len(units)
                        move.camera_move.world_pos.y = sum(u.pos.y for u in units) / len(units)
                        move.camera_move.distance = 24
                        rpc("obs_action", sc.RequestObserverAction(actions=[move]))
                        camera_centered = moved_camera = True
                if loop < start_loop and reply.status != sc.ended and not reply.observation.player_result:
                    # Keep replay state continuous. Do not use the native UI's
                    # seek/reload path while API stepping is active.
                    rpc("step", sc.RequestStep(count=min(224, start_loop - loop)))
                    result["fast_forward_requests"] = result.get("fast_forward_requests", 0) + 1
                    continue
                if not start_position_reached:
                    result["first_playback_loop"] = loop
                    start_position_reached = True
                    if start_loop:
                        print(f"시작 위치 도착: {loop / 22.4:.1f}초. 일반 속도로 재생합니다.", flush=True)
                if args.probe and not screenshot_saved and loop >= 16 and camera_centered and not moved_camera:
                    screenshot_saved = save_png(
                        observation.render_data.map, root / f"replay_{map_path.stem}_validation.png")
                    result["rgb_frame_saved"] = screenshot_saved
                if args.probe and (result["observations"] <= 2 or result["observations"] % 10 == 0):
                    print("OBS", loop, "units", len(observation.raw_data.units), flush=True)
                if not args.probe and loop >= next_progress_loop:
                    print(f"재생 진행: {loop} / {info.game_duration_loops} game loop", flush=True)
                    next_progress_loop = loop + 224
                if reply.status == sc.ended or reply.observation.player_result:
                    result["ended"] = True
                    break
                if args.max_loops and loop >= args.max_loops:
                    result["stopped_at_limit"] = True
                    break
                if info.game_duration_loops and loop >= info.game_duration_loops:
                    result["reached_recorded_end"] = True
                    break
                rpc("step", sc.RequestStep(count=step_size))
                if not args.probe:
                    time.sleep(max(0, step_size / 22.4 - (time.monotonic() - tick)))
            print("재생 종료", flush=True)
        except ReplayStopped:
            result["interrupted"] = True
            result["interrupted_request"] = result.get("last_request")
        except Exception as error:
            result["error"] = str(error)
            result["failed_request"] = result.get("last_request")
            raise
        finally:
            if connection:
                try:
                    if process.poll() is None:
                        sequence += 1
                        result["last_request"] = "quit"
                        connection.settimeout(5)
                        # Like PySC2's quit(), write without waiting for an
                        # acknowledgement: SC2 closes the API while exiting.
                        request = sc.Request(id=sequence, quit=sc.RequestQuit())
                        connection.send_binary(request.SerializeToString())
                        result["quit_requested"] = True
                except (OSError, websocket.WebSocketException) as error:
                    result["quit_send_error"] = str(error)
                # SC2's shutdown joins its API worker. Complete the websocket
                # close handshake now so that worker is not left reading.
                connection.close(timeout=3)
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    # This SC2 build treats SIGTERM as another reportable
                    # crash. If its normal quit really hangs, reap only our
                    # replay process without invoking that signal handler.
                    result["forced_shutdown"] = "kill_after_quit_timeout"
                    process.kill()
                    process.wait()
            result["process_exit"] = process.returncode
            result["graceful_exit"] = process.returncode == 0 and "forced_shutdown" not in result
            if playback_started is not None:
                result["playback_seconds"] = round(time.monotonic() - playback_started, 2)
            result_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            (run_dir / "result.json").write_text(result_text)
            (root / "last_playback_result.json").write_text(result_text)
            log.flush()
            shutil.copyfile(logfile, root / "player_runtime.log")
            print("결과:", json.dumps({k: v for k, v in result.items() if k != "metadata"}, ensure_ascii=False), flush=True)
            if "error" in result:
                print("오류 기록:", run_dir / "result.json", flush=True)
    if result.get("interrupted"):
        if result.get("forced_shutdown"):
            print("게임이 정상 종료 요청에 응답하지 않아 해당 게임 프로세스를 강제 종료했습니다.", flush=True)
        else:
            print("리플레이 재생을 중지했습니다.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n리플레이 재생을 중지했습니다.")
    except Exception as error:
        print(f"\n재생 중 오류: {error}", file=sys.stderr)
        sys.exit(1)
