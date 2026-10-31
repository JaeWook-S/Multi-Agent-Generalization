"""고정한 HiSSD 정책으로 한 맵의 실제 스텝 궤적을 수집하고 세 진단 지표를 계산한다."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiment_analysis.stage1_map_selection.evaluate_performance import resolve_checkpoint, run


class DiagnosticObserver:
    """1단계 평가 흐름의 관찰자. 정책 호출 없이 환경 입출력만 기록한다."""

    def __init__(self, reference_run=None, window=5):
        self.reference_run = Path(reference_run).expanduser().resolve() if reference_run else None
        self.window = window
        self.count = 0
        self.output_dir = None
        self.model_hash = None
        self.checkpoint = None

    def wrap_env(self, task, env):
        """위임 환경은 _seed 설정도 원래 환경에 전달하므로 맵별 seed 규칙을 바꾸지 않는다."""
        from experiment_analysis.stage2_behavior_diagnostics.recording_env import RecordingEnv
        return RecordingEnv(env)

    def on_start(self, output_dir, config, metadata):
        """실제 로드 모델과 seed를 기록하고, 기준 실행을 지정했다면 서로 일치하는지 검사한다."""
        self.output_dir = output_dir
        self.model_hash = metadata["agent_sha256"]
        self.checkpoint = metadata["checkpoint"]
        if self.reference_run:
            reference = json.loads((self.reference_run / "run_metadata.json").read_text(encoding="utf-8"))
            if reference.get("status") != "complete":
                raise ValueError("비교 기준은 정상 완료한 1단계 실행이어야 합니다.")
            if reference.get("agent_sha256") != self.model_hash:
                raise ValueError("기준 실행과 agent.th 해시가 다릅니다. 같은 체크포인트를 지정하세요.")
            for task, seed in metadata["map_seeds"].items():
                if reference.get("map_seeds", {}).get(task) != seed:
                    raise ValueError(f"{task}의 실제 환경 seed가 기준 실행과 다릅니다.")
            # 평가 UI/출력 위치를 제외한 환경·관측 설정도 고정한다.
            baseline = json.loads((self.reference_run / "effective_config.json").read_text(encoding="utf-8"))
            ignored = {"seed", "map_name", "replay_dir", "replay_prefix"}
            old_env = {k: v for k, v in baseline["env_args"].items() if k not in ignored}
            new_env = {k: v for k, v in config["env_args"].items() if k not in ignored}
            if old_env != new_env:
                raise ValueError("기준 실행과 환경/관측 설정이 다릅니다.")
            # env의 관측 옵션과 MAC이 덧붙이는 이전 행동/ID는 별도 설정이므로 둘 다 검사한다.
            policy_keys = ("agent", "mac", "agent_output_type", "action_selector", "test_greedy",
                           "obs_last_action", "obs_agent_id", "c_step", "num_stack_frames", "noise_weight", "mask")
            differences = [key for key in policy_keys if baseline.get(key) != config.get(key)]
            if differences:
                raise ValueError(f"기준 실행과 정책 입력/추론 설정이 다릅니다: {differences}")
        return {
            "analysis_stage": "stage2_behavior_diagnostics", "diagnostic_schema_version": 1,
            "visibility_window": self.window,
            "reference_run": str(self.reference_run) if self.reference_run else None,
            "seed_rule": "base_seed + index_in_full_test_tasks",
        }

    def episode_finished(self, runner, record):
        """완료된 한 판의 궤적을 검증한 뒤 원자적으로 저장한다. 오류 판은 통계에 넣지 않는다."""
        import numpy as np
        arrays, metadata = runner.env.episode_data()
        length = record["episode_length_steps"]
        if metadata["transition_count"] != length:
            raise ValueError("기록한 전이 수와 기존 runner의 실제 실행 길이가 다릅니다.")
        if not np.isclose(np.sum(arrays["rewards"], dtype=np.float64), record["episode_return"], atol=1e-6):
            raise ValueError("원본 스텝 보상과 에피소드 보상의 합이 다릅니다.")
        for side, key in (("ally", "dead_allies"), ("enemy", "dead_enemies")):
            if int(np.count_nonzero(arrays[f"{side}_health"][-1] <= 0)) != record[key]:
                raise ValueError(f"{side}의 원시 종료 상태와 SMAC 사망 수가 다릅니다.")
        if int(metadata["env_seed"]) != record["env_seed"]:
            raise ValueError("원래 환경에 적용된 seed와 평가 기록의 seed가 다릅니다.")
        destination = self.output_dir / "step_logs" / record["map"] / f"episode_{record['episode']:06d}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        runner.env.save_episode(destination, extra_metadata={
            **record, "checkpoint": self.checkpoint, "agent_sha256": self.model_hash})
        self.count += 1

    def final_metadata(self):
        """초기화 실패 시에도 단계 정보와 실제 저장 건수를 남긴다."""
        return {"analysis_stage": "stage2_behavior_diagnostics", "diagnostic_schema_version": 1,
                "recorded_diagnostic_episodes": self.count, "visibility_window": self.window}


def parse_args():
    """기본 seed와 실제 환경 seed를 구분하고, 한 맵만 선택해 진단 로그를 수집한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--load-step", type=int, default=30000)
    parser.add_argument("--task-config", default="stalker-zealot-expert")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--map", default="4s5z", dest="map_name")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1,
                        help="기본 평가 seed. 4s5z의 실제 환경 seed는 기본 설정에서 1+11=12")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sc2-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-run", type=Path, help="모델 해시·환경 seed·관측 설정을 대조할 1단계 결과 폴더")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="분석 전용 난수 seed. 정책/SC2 난수열과 독립")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if min(args.episodes, args.window, args.bootstrap_samples, args.progress_every) < 1:
        parser.error("episodes/window/bootstrap-samples/progress-every는 양수여야 합니다.")
    if args.load_step < 0 or args.bootstrap_seed < 0 or not 0 <= args.seed < 2**31 - 10000:
        parser.error("load-step/bootstrap-seed는 0 이상이고 seed는 허용된 양의 범위여야 합니다.")
    args.maps = [args.map_name]
    return args


def main():
    """수집은 기존 평가 함수에 관찰자만 붙이고, 재집계는 별도 오프라인 함수에 맡긴다."""
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint_path, args.load_step)
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "experiment_analysis/stage2_behavior_diagnostics/results" / args.map_name / (
            f"{checkpoint.name}_{datetime.now():%Y%m%d-%H%M%S-%f}")
    args.output_dir = args.output_dir.expanduser().resolve()
    observer = DiagnosticObserver(args.reference_run, args.window)
    code = run(args, diagnostic_observer=observer)
    # 수집이 중간에 실패했어도 이미 완료한 판의 진단 결과를 생성한다.
    from experiment_analysis.stage2_behavior_diagnostics.analyze_diagnostics import analyze_run
    analyze_run(args.output_dir, window=args.window, bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
