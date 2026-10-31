# 2단계: 행동과 관측 진단

`4s5z`에서 고정한 HiSSD 정책으로 100판을 실행하고 첫 손실 양상, Stalker/Zealot의 비공격 선택, 첫 아군 사망 직전 동료 가시성을 비교합니다. HiSSD 논문의 공식 협력 점수가 아니라 이번 연구를 위한 관찰 지표입니다.

## 수집 및 자동 분석

기존 학습용 Ubuntu `hissd` 환경, 저장소 최상위 폴더에서 실행합니다. 별도의 추가 패키지는 필요하지 않습니다.

```bash
export SC2PATH="$PWD/3rdparty/StarCraftII"

python experiment_analysis/stage2_behavior_diagnostics/collect_diagnostics.py \
  --task-config stalker-zealot-expert \
  --checkpoint-path "/path/to/run/models" \
  --load-step 30000 \
  --map 4s5z \
  --episodes 100 \
  --seed 1 \
  --device cuda
```

- 학습 실행 폴더의 `config.json`으로 모델을 구성하고 `agent.th`를 불러옵니다. 데이터셋 로딩과 추가 학습은 없습니다.
- 기본 seed 1에 전체 시험 목록의 인덱스 11을 더하므로 **실제 환경 seed와 맵별 정책 RNG seed는 12**입니다. 맵 하나만 선택해도 1단계와 같습니다.
- `--device cuda`에서 GPU가 없으면 CPU로 전환하지 않고 중단합니다.
- 먼저 `--episodes 2 --bootstrap-samples 100`으로 확인할 수 있습니다.
- 이전 평가 결과 없이 실행할 수 있습니다. 결과를 보관한 경우에만 `--reference-run "/path/to/stage1/run"`을 추가하면 기존 모델 SHA-256, 실제 맵 seed, 환경·관측 및 정책 입력/추론 설정을 대조합니다. 다르면 첫 게임 전에 중단합니다.
- 성공/실패 라벨은 새 수집의 실제 종료 정보로 붙입니다. 이전 100판의 라벨이나 50승·50패 개수를 복사하지 않습니다.

## 코드와 기록 순서

| 파일 | 역할 |
| --- | --- |
| `collect_diagnostics.py` | 기존 평가 루프에 기록기를 연결하고 완료한 에피소드 저장 |
| `recording_env.py` | 원래 환경 호출을 전달하면서 실제 입력·출력의 값을 복사 |
| `diagnostic_metrics.py` | 사망 사건·비공격 선택·실제 관측의 가시성 계산 |
| `analyze_diagnostics.py` | 원본 파일을 읽어 성공/패배 비교와 조건별 표 재생성 |

```text
obs[t] + avail_actions[t] + units[t] → 실제 actions[t]
                                   → 기존 env.step()
                                   → units[t+1] + reward[t] + 종료 정보
```

기존 runner/MAC 순서를 유지합니다. 기록을 위해 모델을 추가 호출하거나 난수를 소비하지 않습니다. `_seed` 쓰기도 실제 환경에 위임합니다. 종료 후 미실행 행동은 `step()`을 통과하지 않아 기록하지 않습니다. 추가 SC2 관측 요청도 만들지 않습니다. 주요 함수와 중요한 처리에는 한글 주석을 달았습니다.

## 결과 파일

기본 위치는 `stage2_behavior_diagnostics/results/4s5z/<step>_<timestamp>/`입니다. `--output-dir`로 새 폴더를 지정할 수 있습니다.

| 파일/폴더 | 내용 |
| --- | --- |
| `step_logs/4s5z/episode_*.npz` | 에피소드별 압축 원본 배열. pickle 없이 로드 |
| 같은 이름의 `.json` | 종료 유형, 모델 해시, 실제 seed, 관측 구조, 유닛 ID/태그/종류 매핑 |
| `episodes.csv`, `summary.*` | 1단계와 같은 성과 기록 |
| `run_metadata.json`, `effective_config.json`, `run.log` | 설정·장치·진행·오류 기록 |
| `analysis/window_5/episode_diagnostics.csv` | 에피소드별 진단 값과 분자·분모 |
| `analysis/window_5/nonattack_context.csv` | 공격 명령 기회가 있던 행동 직전 상황·선택 |
| `analysis/window_5/visibility_windows.csv` | 대상–관찰자–스텝별 실제 관측·마스크·행동 |
| `analysis/window_5/events.csv` | 동시 사망을 포함한 사건 목록 |
| `analysis/window_5/context_comparison.csv` | 유닛 종류·체력·실드·쿨다운·전투 구간별 비교 |
| `analysis/window_5/comparison.md`, `comparison.json` | 종료 유형별 비교표, 평균·중앙값·유효 표본 수 |
| `analysis/window_5/source_manifest.json` | 재집계에 사용한 원본 파일의 SHA-256 |

관측·상태·마스크와 유닛 상태는 `T+1`개, 실제 행동·보상은 `T`개입니다. 유닛 태그는 `uint64`, 종류는 문자열로 저장합니다. 매판 새 매핑을 기록하고 객체 참조가 아닌 값을 복사하므로 나중에 체력이 바뀌어도 이전 기록은 유지됩니다.

## 지표 정의

1. **첫 손실:** `health[t]>0 → health[t+1]<=0`을 사망으로 기록합니다. 첫 진영은 아군/적군/같은 전이 양쪽/없음입니다. 첫 아군 사망자는 모두 보존하고 두 종류가 섞이면 `mixed`입니다. 적 처치는 첫 아군 사망 전, 같은 전이, 해당 전이까지 누적으로 나눕니다. 같은 전이는 현재 `step_mul=8` 구간에서 함께 발견됐다는 뜻이며 실제 순간 순서는 알 수 없습니다.
2. **비공격 선택:** 행동 직전 살아 있고 공격 마스크가 하나 이상 허용된 경우 중 이동/정지(1..5)를 선택한 비율입니다. Stalker와 Zealot을 분리합니다. 죽은 유닛의 noop, 패딩, 종료 후 미실행 행동을 제외합니다. 공격 기회가 없으면 0이 아닌 N/A입니다.
3. **사망 직전 비가시성:** 첫 아군 사망이 결과 스냅샷 `d`에서 확인되면 `max(0,d-window)..d-1`의 행동 직전 관측만 봅니다. 살아 있던 관찰자만 포함하고 대상 자신은 제외합니다. 실제 obs의 ally visible 열과 자기 자신이 빠진 슬롯 순서를 사용합니다. 보이지 않을 때 해석된 위치/체력/실드는 `null`로 두고 원본 obs의 0과 실제 체력 0을 구별합니다. 동시 사망 대상은 대상별 비가시 비율을 구한 뒤 에피소드 안에서 동일 가중 평균합니다.

## 비교 방법

- 승리/전투 패배/시간 초과/동시 전멸을 분리합니다. 지표는 에피소드별로 먼저 계산하고 비율 지표에는 에피소드를 동일 가중으로 재표집한 bootstrap 95% 구간을 저장합니다. 연속 스텝을 독립 표본으로 세지 않습니다.
- 조건별 표는 체력/실드 비율 `[0,0.3]`, `(0.3,0.7]`, `(0.7,1]`, 쿨다운 0/양수, 첫 아군 사망 전/이후/사망 없음으로 나눕니다. 연구용 고정 기준이며 연속 원본 값도 보존합니다.
- 각 조건에서도 에피소드별 비율을 먼저 계산합니다. 기회 수와 유효 에피소드 수를 함께 보고, 에피소드가 하나뿐인 조건에는 CI를 표시하지 않습니다.
- 원시 쿨다운/global state는 분석용이며 actor에게 새로 주지 않습니다. 현재 local obs에 쿨다운이 명시적으로 제공된다고 해석하지 않습니다.
- 비공격 비율은 DPS 손실이 아닙니다. 가시성 차이도 실패 원인을 확정하지 않습니다. 후속 개입 실험의 가설을 좁히는 자료입니다.

## 게임 재실행 없이 관측 창 변경

```bash
python experiment_analysis/stage2_behavior_diagnostics/analyze_diagnostics.py \
  --input-dir "/path/to/stage2/run" --window 3

python experiment_analysis/stage2_behavior_diagnostics/analyze_diagnostics.py \
  --input-dir "/path/to/stage2/run" --window 10
```

각각 `analysis/window_3/`, `analysis/window_10/`에 생성되며 원본을 바꾸지 않습니다. `--bootstrap-samples`와 `--bootstrap-seed`는 분석 전용이고 게임 seed와 별개입니다. 1단계 성과 CSV만으로는 이 분석을 할 수 없습니다.

## 중단과 검증

정상 완료한 판만 성과 CSV와 원본 궤적에 등록합니다. 연결 오류·강제 재시작·누락된 관측은 패배로 세지 않고 중단합니다. 앞서 완료한 판은 보존하고 중단 상태를 표시합니다. 기존 수집 폴더와 원본 파일은 덮어쓰지 않습니다.

```bash
python -m unittest discover -s experiment_analysis -p 'test_*.py' -v
```

대역 환경으로 기록 전후 정책 입력·행동·환경 호출 순서·성과 CSV가 일치하는지 검증합니다. 실제 GPU/SC2 평가는 Ubuntu에서 실행해야 합니다.

참고: [SMAC 관측·행동·종료 구현](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/starcraft2.py).
