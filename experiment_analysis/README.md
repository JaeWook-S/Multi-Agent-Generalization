# Experiment Analysis

| 단계 | 폴더 | 분석 목적 |
| --- | --- | --- |
| 1 | [stage1_map_selection](stage1_map_selection/README.md) | 맵별 성과 평가: 승률·시간 초과율·적 잔존 비율·승리 길이로 분석 대상 선정 |
| 2 | [stage2_behavior_diagnostics](stage2_behavior_diagnostics/README.md) | 선택한 맵의 첫 사망·유형별 비공격 선택·사망 직전 동료 가시성 분석 |

두 단계 모두 고정 체크포인트로 추론하며 `src/` 학습·정책 코드는 수정하지 않습니다.

## Seed 확인

`--seed 1`은 기본 평가 seed입니다. 실제 맵 seed는 **기본 seed + 전체 `test_tasks`의 0부터 세는 인덱스**입니다. `4s5z`는 인덱스 11이므로 환경 seed는 **12**입니다. 앞서 확인한 `2s5z`는 인덱스 5이므로 **6**입니다. SMAC의 자동 변환이 아니라 공통 함수 `map_seed()`의 규칙입니다. 2단계에서 맵 하나만 선택해도 인덱스를 다시 0으로 매기지 않습니다.

기존 `30000_20261031-190414-979608` 실행의 메타데이터, 콘솔 로그, 에피소드 CSV 모두 이 값과 일치합니다. 두 단계 모두 모델 SHA-256과 실제 seed를 기록합니다. 100판은 한 환경의 연속 에피소드이며 매판 다른 학습 seed를 뜻하지 않습니다.

## 1단계 실행

```bash
python experiment_analysis/stage1_map_selection/evaluate_performance.py \
  --task-config stalker-zealot-expert \
  --checkpoint-path "/path/to/run/models" \
  --load-step 30000 --episodes 100 --seed 1 --device cuda
```

결과는 `stage1_map_selection/results/`에 저장합니다. 최상위 호환용 파일과 결과 이동 도구는 제거했으므로 위 경로로 실행합니다.

## 2단계 실행

기존 Ubuntu `hissd` 환경, 저장소 최상위 폴더에서 실행합니다.

```bash
export SC2PATH="$PWD/3rdparty/StarCraftII"

python experiment_analysis/stage2_behavior_diagnostics/collect_diagnostics.py \
  --task-config stalker-zealot-expert \
  --checkpoint-path "/path/to/run/models" \
  --load-step 30000 --map 4s5z --episodes 100 --seed 1 --device cuda
```

이전 평가 결과 없이 실행할 수 있으며 모델 SHA-256과 실제 환경 seed를 새로 기록합니다. `stage1_map_selection`의 평가 코드와 지표 모듈은 2단계가 재사용하므로 유지합니다.

성과 기록과 원본 스텝 로그를 함께 저장하고 진단 표를 자동 생성합니다. 수집 원본은 `stage2_behavior_diagnostics/results/4s5z/<step>_<timestamp>/`, 분석 표는 그 아래 `analysis/window_5/comparison.md`에 저장합니다. 상세 지표·재분석 방법은 [2단계 README](stage2_behavior_diagnostics/README.md)를 참고하세요.
