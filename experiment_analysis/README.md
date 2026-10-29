# Checkpoint Performance Analysis

고정한 HiSSD 체크포인트로 맵마다 100회 평가합니다. `src/`를 수정하지 않고 기존 runner/MAC의 추론 경로를 재사용하며, 데이터셋 로딩이나 재학습은 하지 않습니다.

## 실행

기존 학습용 Ubuntu `hissd` 환경에서 저장소 최상위 폴더를 기준으로 실행합니다.

```bash
export SC2PATH="$PWD/3rdparty/StarCraftII"

python experiment_analysis/evaluate_performance.py \
  --task-config stalker-zealot-expert \
  --checkpoint-path "/path/to/run/models" \
  --load-step 30000 \
  --episodes 100 \
  --seed 1 \
  --device cuda
```

- `--checkpoint-path`에는 `models` 또는 `models/30000`을 지정합니다. `models` 경로에서 `--load-step 0`은 최신 체크포인트를 선택합니다. 지정한 step이 없으면 오류로 중단합니다.
- 기본값은 `test_tasks`의 모든 맵입니다. `stalker-zealot-expert`는 **13개 맵 × 100회 = 1,300판**입니다. 일부 맵만 보려면 `--maps 2s4z 1s5z 3s3z`를 추가합니다.
- `--device cuda`는 CUDA를 사용할 수 없으면 중단합니다. CPU 평가를 원하면 `--device cpu`를 사용합니다.
- 학습 실행 폴더의 `config.json`을 자동으로 불러옵니다. 별도 위치라면 `--training-config "/path/to/config.json"`으로 지정하세요. 해당 파일이 없으면 경고 후 현재 YAML 설정을 사용합니다.
- `--seed`는 평가 seed입니다. 맵별로 `seed + test_tasks에서의 인덱스`를 사용하고, 한 SC2 환경에서 100에피소드를 연속 실행합니다. 매판 정책의 hidden state는 초기화합니다.

먼저 동작만 확인하려면 `--episodes 2 --maps 2s4z 1s5z`로 실행하세요.

## 저장 결과

결과는 `experiment_analysis/results/<task>/<checkpoint>_<timestamp>/`에 저장합니다. `--output-dir`로 새 경로를 지정할 수도 있습니다.

| 파일 | 내용 |
| --- | --- |
| `episodes.csv` | 매판 승패, 시간 초과, 생존 적 수/비율, 길이, 보상, 사망 수, 평가 seed |
| `summary.csv`, `summary.json` | 맵별 지표의 원본 수치. 비율은 0~1 |
| `summary.md` | 맵별 성과표와 학습/미학습/전체 승률 표 |
| `effective_config.json` | 실제 사용한 모델·환경 설정 |
| `run_metadata.json` | 체크포인트 경로와 SHA-256, 설정 출처, 장치, seed, 완료 여부 |
| `run.log` | 10판마다 출력되는 진행 로그와 오류 |

에피소드 CSV는 매판 기록합니다. 중단/오류 시 완료된 판의 결과를 보존하고 `interrupted`/`failed`로 표시합니다. SC2가 강제 재시작된 판은 패배로 세지 않고 실행을 중단합니다. 기존 결과 디렉터리는 덮어쓰지 않습니다.

## 지표 정의

| 지표 | 계산 |
| --- | --- |
| 승률 | 승리 수 / 평가 수, 맵별 95% Wilson 신뢰구간 포함 |
| 시간 초과율 | SMAC의 `timeouts` 카운터가 증가한 판 수 / 평가 수 |
| 패배 시 적 잔존 비율 | 비승리 판마다 `(초기 적 수 - 죽은 적 수) / 초기 적 수`를 계산한 뒤 평균 |
| 승리 시 전투 길이 | 승리한 판만의 환경 step 수 평균과 중앙값 |

여기서 비승리는 시간 초과와 동시 전멸도 포함합니다. 승리가 없거나 비승리가 없으면 해당 조건부 지표는 `N/A`로 둡니다. JSON은 `null`, CSV는 빈칸으로 저장합니다. 시간 초과는 길이만으로 판단하지 않아 마지막 스텝에서 난 정상 승패와 구분됩니다.

신뢰구간은 고정한 모델의 에피소드 평가 불확실성입니다. 여러 학습 seed에서의 안정성을 의미하지 않습니다. 이 도구는 첨부한 분석 계획의 **1번 성과 지표**만 수집하며, 스텝별 진단 로그나 리플레이는 저장하지 않습니다.

참고: [SMAC 종료 및 시간 초과 처리](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/starcraft2.py), [Wilson 신뢰구간](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html).

## 코드 검증

```bash
python -m unittest experiment_analysis.test_evaluate_performance -v
```

집계/종료 처리 검증은 SC2와 PyTorch 없이 실행할 수 있습니다.
