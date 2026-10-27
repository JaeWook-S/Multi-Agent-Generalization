# Mac SC2 4.10 리플레이 실행기 — v2.4

Apple M4 / macOS 26.6.2 / SC2 4.10.0.75689에서 확인한 로컬 실행기입니다. 저장된 리플레이를 재생하며 모델 rollout 명령은 저장소 최상위 README의 **Rollout and save replays**를 참고하세요.

## 사용법

1. `.SC2Replay` 파일이나 폴더를 `output/replay/` 아래에 넣습니다. 실행기가 하위 폴더까지 검색합니다.
2. `output/replay/Play_SC2_Replay.command`를 Finder에서 더블클릭합니다.
3. 리플레이 번호를 선택합니다. Enter는 `3m`을 선택합니다.
4. 시작 시간을 입력합니다. Enter는 처음부터, `1:30` 또는 `90`은 90초부터입니다.
5. 열리는 SC2 창에서 봅니다. 체력 바는 항상 표시되고 `CHEAT ...` 메시지는 왼쪽 위에 작게 표시됩니다.
6. 종료하려면 실행기 터미널에서 Control+C를 누릅니다. 게임이 닫히면 같은 창에 zsh 프롬프트가 나타납니다.

로딩 중 중지 요청은 초기 화면 준비를 마친 뒤 처리하므로 잠시 기다릴 수 있습니다. 게임 안 타임라인 탐색과 일시정지·재생 버튼은 지원하지 않습니다. 다른 구간을 보려면 종료 후 시작 시간을 바꿔 다시 실행하세요.

저장소 최상위 폴더에서 특정 파일을 바로 지정할 수도 있습니다.

```bash
./output/replay/Play_SC2_Replay.command "/절대/경로/파일.SC2Replay" --start-at 1:30
```

`--default-ui`를 붙이면 원래 메시지 배치를 사용합니다. 체력 바 항상 표시 설정은 유지됩니다.

## 필요한 구성

- `play_replay.py`: SC2 기동, 버전 검사, API 재생, 시작 시간 이동, 정상 종료와 로그 기록.
- `.venv/`, `requirements-player.txt`: 실행 전용 Python 환경과 의존성 목록.
- `libsc2_icu_compat.dylib`: 이 Mac의 구형 SC2 초기화 충돌을 우회하는 라이브러리.
- `icu_new_compat.cpp`: 호환 라이브러리의 재빌드용 소스.
- `HiSSDCompact.SC2Interface`, `compact_ui/`, `build_compact_interface.py`: 작은 사이드 메시지 UI와 원본/생성 도구.
- `validate_stop.py`: 실제 SC2와 터미널 신호를 이용하는 수동 종료 회귀 검사 도구.

`runs/`, `player_runtime.log`, `last_playback_result.json`은 실행 기록이며 재생에 필요한 입력 파일이 아닙니다. 삭제해도 다음 실행 시 다시 생성됩니다. 오류 기록 코드는 유지됩니다.

SC2의 기본 설치 경로는 `/Applications/StarCraft II`입니다. 다른 위치라면 실행기 인자에 `--sc2 "/설치/경로"`를 추가하세요. 설치 폴더에는 `Versions/Base75689/SC2.app`와 해당 맵이 `Maps/SMAC_Maps/`에 있어야 합니다.

## Mac 호환 처리

이 Mac에서는 SC2 4.10이 초기화 중 다음 순환 호출로 종료됐습니다.

```text
SC2 초기화 → NSRegularExpression → ICU → SC2 메모리 할당 → 초기화 재진입
libc++abi: __cxa_guard_acquire detected recursive initialization
```

전용 라이브러리는 ICU 내부 정규식 처리에 한정해 이 순환을 피합니다. 실행기가 SC2 프로세스에만 라이브러리를 전달하고 로드 후 주입 환경 변수를 지웁니다. 셸 설정 파일에 DYLD_INSERT_LIBRARIES를 전역으로 추가하지 마세요. Blizzard 공식 패치는 아니며 위 환경에서 검증한 임시 호환 방법입니다.

Control+C가 SC2에도 직접 전달되면 블리자드 충돌 보고가 생성되므로, 게임은 별도 프로세스 세션에서 실행합니다. Python이 중지 요청을 받아 API의 RequestQuit로 종료하고 웹소켓을 정리합니다. 정상 종료 요청에 응답하지 않을 때만 해당 게임 프로세스를 강제 종료하며 `forced_shutdown`에 기록합니다.

## 환경 복구

아래 명령은 저장소 최상위 폴더에서 실행합니다. 실행 중인 게임을 먼저 종료하세요.

### 전용 Python 환경

```bash
python3 -m venv output/replay/diagnostics/mac410_compat/.venv
output/replay/diagnostics/mac410_compat/.venv/bin/python -m pip install \
  -r output/replay/diagnostics/mac410_compat/requirements-player.txt
```

### 호환 라이브러리

Xcode Command Line Tools가 있는 Mac에서 재빌드할 수 있습니다.

```bash
xcrun clang++ -arch x86_64 -std=c++17 -fsized-deallocation \
  -mmacosx-version-min=10.15 -O2 -Wall -Wextra -dynamiclib \
  output/replay/diagnostics/mac410_compat/icu_new_compat.cpp \
  -licucore \
  -o output/replay/diagnostics/mac410_compat/libsc2_icu_compat.dylib
```

### 사이드 메시지 UI

```bash
output/replay/diagnostics/mac410_compat/.venv/bin/python \
  output/replay/diagnostics/mac410_compat/build_compact_interface.py
cp output/replay/diagnostics/mac410_compat/HiSSDCompact.SC2Interface \
  "/Applications/StarCraft II/Interfaces/HiSSDCompact.SC2Interface"
```

### SMAC 맵

```bash
mkdir -p "/Applications/StarCraft II/Maps/SMAC_Maps"
cp smac_patch/SMAC_Maps/*.SC2Map "/Applications/StarCraft II/Maps/SMAC_Maps/"
```

참고: [Blizzard SC2 버전 설명](https://github.com/Blizzard/s2client-proto/blob/master/docs/protocol.md#versioning), [PySC2 종료 구현](https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/remote_controller.py#L328).
