"""데이터 이력을 유지한 HiSSD 행동 점수 검사. 학습과 SC2 게임 실행은 하지 않는다."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment_analysis.stage1_map_selection.evaluate_performance import resolve_checkpoint
from experiment_analysis.stage3_retreat_analysis.trajectory import map_spec, sha256
from experiment_analysis.stage4_policy_scores.episode_prefix import (
    check_hash, load_bundle, read_h5_prefix, save_bundle, validate_prefix, write_json,
)


def read_training_config(checkpoint, explicit=None):
    """구조가 다른 현재 YAML로 대체하지 않고 학습 당시 설정을 요구한다."""
    path = Path(explicit).expanduser().resolve() if explicit else checkpoint.parent.parent / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"학습 config.json이 필요합니다. --training-config로 지정하세요: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


class CausalBatch:
    """현재 관측까지만 공개하고 정답 행동은 forward 이후에만 이력에 넣는다.

    HISSDSMAC이 사용하는 EpisodeBatch 인터페이스만 제공한다. state는 이 배치에
    존재하지 않으므로 분석용 global state가 실행 정책에 유입되지 않는다.
    """

    def __init__(self, arrays, device, torch):
        self.torch, self.batch_size, self.device = torch, 1, device
        self.max_seq_length = arrays["obs"].shape[0]
        self.n_agents, self.n_actions = arrays["avail_actions"].shape[1:]
        self.data = {
            "obs": torch.zeros((1,) + arrays["obs"].shape, dtype=torch.float32, device=device),
            "avail_actions": torch.zeros((1,) + arrays["avail_actions"].shape, dtype=torch.int64, device=device),
            "actions": torch.zeros((1, self.max_seq_length, self.n_agents, 1), dtype=torch.long, device=device),
            "actions_onehot": torch.zeros((1, self.max_seq_length, self.n_agents, self.n_actions),
                                           dtype=torch.float32, device=device),
        }
        self.revealed = -1

    def __getitem__(self, key):
        return self.data[key]

    def reveal(self, step, arrays):
        """0,1,2,... 순서만 허용한다. 12스텝에서 hidden state를 새로 시작할 수 없다."""
        if step != self.revealed + 1:
            raise ValueError("관측은 0스텝부터 순서대로 공개해야 합니다.")
        self.data["obs"][0, step] = self.torch.as_tensor(arrays["obs"][step], device=self.device)
        self.data["avail_actions"][0, step] = self.torch.as_tensor(arrays["avail_actions"][step], device=self.device)
        self.revealed = step

    def commit_recorded_action(self, step, actions):
        """모델 예측으로 원본 이력을 바꾸지 않고 실제 데이터 행동을 다음 입력에 사용한다."""
        if step != self.revealed:
            raise ValueError("관측을 공개한 스텝의 행동만 기록할 수 있습니다.")
        indices = self.torch.as_tensor(actions, dtype=self.torch.long, device=self.device)
        self.data["actions"][0, step, :, 0] = indices
        self.data["actions_onehot"][0, step] = self.torch.nn.functional.one_hot(
            indices, num_classes=self.n_actions).to(self.torch.float32)


def create_mac(config, arrays, metadata, checkpoint, device):
    """원래 controller/agent/decomposer를 생성하되 runner/env/mto는 불러오지 않는다."""
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from controllers.multi_task.mt_hissd_controller import HISSDSMAC
    from modules.decomposers.sc2_decomposer import SC2Decomposer

    effective = copy.deepcopy(config)
    if (effective.get("env") != "sc2" or effective.get("mac") != "mt_hissd_mac"
            or effective.get("agent") != "mt_hissd"):
        raise ValueError("SC2 HiSSD의 학습 설정이 필요합니다.")
    if effective.get("agent_output_type") != "q":
        raise ValueError("현재 도구는 agent_output_type=q인 HiSSD 원시 행동 점수를 검사합니다.")
    effective.update(device=str(device), use_cuda=device.type == "cuda", batch_size_run=1,
                     test_greedy=True, evaluate=True, save_replay=False, save_model=False,
                     n_tasks=len(effective["train_tasks"]))
    tasks = list(dict.fromkeys(effective["train_tasks"] + effective["test_tasks"] +
                              (effective.get("pretrain_tasks", []) if effective.get("pretrain") else [])))
    if metadata["map"] not in tasks:
        raise ValueError("원본 맵이 학습 설정의 과제 목록에 없습니다.")
    main_args = SimpleNamespace(**effective)
    task_args, schemes = {}, {}
    for task in tasks:
        args = copy.deepcopy(main_args)
        args.env_args["map_name"] = task
        spec = map_spec(task)
        # 이 생성자는 정적 맵/feature 규격만 읽는다. StarCraft2Env는 만들지 않는다.
        decomposer = SC2Decomposer(args)
        if (decomposer.n_agents != spec["n_agents"] or decomposer.n_enemies != spec["n_enemies"]
                or decomposer.unit_type_bits != spec["unit_type_bits"]):
            raise ValueError(f"설치된 SMAC의 맵 규격이 저장소 패치와 다릅니다: {task}")
        args.n_agents, args.n_enemies = decomposer.n_agents, decomposer.n_enemies
        args.n_actions, args.episode_limit = decomposer.n_actions, decomposer.episode_limit
        args.obs_shape, args.state_shape = decomposer.obs_dim, decomposer.state_dim
        task_args[task] = args
        schemes[task] = {"obs": {"vshape": decomposer.obs_dim},
                         "actions_onehot": {"vshape": (decomposer.n_actions,)}}
    task = metadata["map"]
    if (task_args[task].obs_shape != arrays["obs"].shape[-1]
            or task_args[task].n_actions != arrays["avail_actions"].shape[-1]):
        raise ValueError("원본 H5/prefix의 입력 크기가 모델의 실제 관측 규격과 다릅니다.")
    mac = HISSDSMAC(tasks, schemes, task_args, main_args)
    mac.load_models(str(checkpoint))
    mac.agent.to(device)
    mac.agent.eval()
    return mac, effective


def replay_recorded_history(mac, arrays, metadata, last_step, device, torch):
    """원본 관측·이전 행동을 고정한 조건부 추론이다. 새로운 정책 rollout이 아니다."""
    validate_prefix(arrays, metadata)
    if not 0 <= last_step < len(arrays["actions"]):
        raise ValueError("요청 스텝이 준비한 prefix를 벗어났습니다.")
    batch = CausalBatch(arrays, device, torch)
    mac.init_params()
    mac.init_hidden(batch_size=1, task=metadata["map"])
    # 새 에피소드의 t=0에서 덮어쓸 캐시도 명시적으로 비워 재사용 시 혼입을 막는다.
    for name in ("last_out_h", "last_h_plan"):
        if hasattr(mac.agent, name):
            setattr(mac.agent, name, None)
    outputs = []
    with torch.no_grad():
        for step in range(last_step + 1):
            batch.reveal(step, arrays)
            scores = mac.forward(batch, step, metadata["map"], test_mode=True)
            shape = (1, batch.n_agents, batch.n_actions)
            if tuple(scores.shape) != shape or not torch.isfinite(scores).all().item():
                raise ValueError(f"행동 점수 차원/수치가 유효하지 않습니다: {tuple(scores.shape)}, expected={shape}")
            outputs.append(scores.detach().cpu().numpy()[0].copy())
            # 현재 정답을 넣기 전에 점수를 복사했다. 미래 t+1부터 이 행동이 이전 행동이 된다.
            batch.commit_recorded_action(step, arrays["actions"][step])
    return np.stack(outputs)


def action_label(index):
    return ("noop", "stop", "move_north", "move_south", "move_east", "move_west")[index] if index < 6 else f"attack_E{index - 6}"


def summarize_scores(outputs, arrays, steps, agents):
    """허용 행동만 비교한다. 정지와 이동, 공격을 구분하고 동점·점수 차이도 보존한다."""
    decisions, action_rows = [], []
    for step in sorted(steps):
        for agent in agents:
            scores = outputs[step, agent]
            allowed = arrays["avail_actions"][step, agent].astype(bool)
            valid = np.flatnonzero(allowed)
            greedy = int(valid[np.argmax(scores[valid])])
            recorded = int(arrays["actions"][step, agent])
            move = valid[(valid >= 2) & (valid <= 5)]
            attack = valid[valid >= 6]

            def best(indices):
                if not len(indices):
                    return None, None
                index = int(indices[np.argmax(scores[indices])])
                return index, float(scores[index])

            move_id, move_score = best(move)
            attack_id, attack_score = best(attack)
            ranked = np.sort(scores[valid])[::-1]
            gap = float(ranked[0] - ranked[1]) if len(ranked) > 1 else None
            decisions.append({
                "step": step, "agent_id": agent, "unit_type": str(arrays["unit_type"][agent]),
                "alive": bool(arrays["health_ratio"][step, agent] > 0),
                "health_ratio": float(arrays["health_ratio"][step, agent]),
                "shield_ratio": float(arrays["shield_ratio"][step, agent]),
                "cooldown_normalized": float(arrays["cooldown_normalized"][step, agent]),
                "recorded_action": recorded, "recorded_label": action_label(recorded),
                "recorded_score": float(scores[recorded]),
                "recorded_valid_rank": int((scores[valid] > scores[recorded]).sum()) + 1,
                "recorded_tied_for_best": bool(scores[recorded] == scores[greedy]),
                "greedy_action": greedy, "greedy_label": action_label(greedy),
                "greedy_matches_recorded": greedy == recorded,
                "best_move_action": move_id, "best_move_score": move_score,
                "best_attack_action": attack_id, "best_attack_score": attack_score,
                "move_minus_attack": move_score - attack_score if move_score is not None and attack_score is not None else None,
                "top1_margin": gap, "valid_action_count": len(valid),
            })
            for action, score in enumerate(scores):
                action_rows.append({"step": step, "agent_id": agent, "action_id": action,
                                    "action_label": action_label(action), "available": bool(allowed[action]),
                                    "raw_score": float(score),
                                    "valid_rank": int((scores[valid] > score).sum()) + 1 if allowed[action] else None,
                                    "is_greedy": action == greedy, "is_recorded": action == recorded})
    return decisions, action_rows


def write_csv(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summary_markdown(metadata, decisions):
    lines = ["# Teacher-forced action scores", "",
             f"Map: `{metadata['map']}` · H5 episode: {metadata['episode']} (row {metadata['h5_row_index']})", "",
             f"Model SHA-256: `{metadata['agent_sha256']}`", "",
             "0스텝부터 데이터의 관측과 이전 행동으로 hidden state를 재구성했습니다. "
             "새 게임이나 행동 개입의 결과가 아닙니다.", "",
             "| Step | Agent | Data action | Greedy action | Data rank | Best move − best attack |",
             "| ---: | ---: | --- | --- | ---: | ---: |"]
    for row in decisions:
        margin = row["move_minus_attack"]
        lines.append(f"| {row['step']} | A{row['agent_id']} | {row['recorded_label']} | {row['greedy_label']} | "
                     f"{row['recorded_valid_rank']} | {margin:.6f} |" if margin is not None else
                     f"| {row['step']} | A{row['agent_id']} | {row['recorded_label']} | {row['greedy_label']} | "
                     f"{row['recorded_valid_rank']} | N/A |")
    lines += ["", "- 점수는 Decoder의 행동 분류 점수입니다. 기대 보상으로 해석하지 않습니다.",
              "- 차이가 양수이면 허용된 최선의 이동이 최선의 공격보다 높습니다. 정지가 더 높을 수도 있어 greedy action도 함께 확인하세요.",
              "- 공격 명령이 없으면 차이는 N/A입니다. 정답 순위는 허용 행동 안의 순위이며 동점은 같은 순위를 가집니다.",
              "- 입력 이력은 모델의 예측이 아니라 실제 데이터 행동입니다. 모델이 다른 행동을 선택한 뒤의 상태·승패는 계산하지 않았습니다.",
              "- 모델이 이동을 고르면 데이터 이력에서 재현할 수 있다는 뜻입니다. 실제 rollout에서 같은 상황을 만나는지와는 별개입니다.",
              "- 이동 점수가 낮다는 사실만으로 잠재공간에 전략 정보가 없다고 확정할 수 없습니다.", ""]
    return "\n".join(lines)


def execute(args, mac_factory=None):
    """파일 정체성→입력 검증→순차 추론→점수 저장 순서로 실행한다."""
    if args.episode < 1:
        raise ValueError("episode는 1 이상이어야 합니다.")
    if args.command == "export":
        config = json.loads(args.training_config.expanduser().read_text(encoding="utf-8"))
        arrays, meta = read_h5_prefix(args.dataset_file, config, args.map, args.episode,
                                      args.through_step, args.expected_dataset_sha256)
        meta["training_config_original_sha256"] = sha256(args.training_config.expanduser())
        output = args.output_dir or default_output("prefix")
        output = save_bundle(output, arrays, meta, config)
        print(f"prefix 저장: {output}")
        return output

    steps, agents = args.steps, args.agents
    if not steps or min(steps) < 0 or len(set(steps)) != len(steps):
        raise ValueError("steps는 중복 없는 0 이상의 정수여야 합니다.")
    if not agents or min(agents) < 0 or len(set(agents)) != len(agents):
        raise ValueError("agents는 중복 없는 0 이상의 ID여야 합니다.")
    checkpoint = resolve_checkpoint(args.checkpoint_path, args.load_step)
    model_hash = sha256(checkpoint / "agent.th")
    check_hash(model_hash, args.expected_model_sha256, "모델")
    if args.bundle_dir:
        arrays, meta, config = load_bundle(args.bundle_dir, args.expected_dataset_sha256)
        if meta["map"] != args.map or meta["episode"] != args.episode:
            raise ValueError("bundle의 맵/에피소드와 지정값이 다릅니다.")
        source = checkpoint.parent.parent / "config.json"
        if args.training_config or source.is_file():
            check_config, source = read_training_config(checkpoint, args.training_config)
            if check_config != config:
                raise ValueError("bundle과 체크포인트의 학습 config가 다릅니다. 같은 실행의 설정을 준비하세요.")
        config_source = "bundle/config.json"
    else:
        config, source = read_training_config(checkpoint, args.training_config)
        config_source = str(source)
        arrays, meta = read_h5_prefix(args.dataset_file, config, args.map, args.episode,
                                      max(steps), args.expected_dataset_sha256)
    _, n, _ = validate_prefix(arrays, meta)
    if max(steps) > meta["through_step"] or max(agents) >= n:
        raise ValueError("요청 스텝 또는 agent ID가 원본 범위를 벗어났습니다.")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("점수 추론에는 PyTorch가 필요합니다. 기존 Ubuntu hissd 환경에서 실행하세요.") from error
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행할 수 있습니다.")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    factory = mac_factory or create_mac
    mac, effective = factory(config, arrays, meta, checkpoint, device)
    output = (args.output_dir or default_output("scores")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = dict(meta, status="running", checkpoint=str(checkpoint), agent_sha256=model_hash,
               model_hash_verified=args.expected_model_sha256 is not None,
               dataset_hash_verified=args.expected_dataset_sha256 is not None,
               config_source=config_source, requested_steps=sorted(steps), agent_ids=agents,
               device=str(device), torch_version=torch.__version__, seed=args.seed,
               torch_cuda=torch.version.cuda, history_mode="recorded_actions_from_step_zero",
               started_at=datetime.now().astimezone().isoformat(),
               source_code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                    ("probe_action_scores.py", "episode_prefix.py")})
    write_json(output / "metadata.json", run)
    # 장치 등 실행용 값을 덮기 전의 학습 설정도 보존한다. bundle과 직접 H5 입력을 비교할 수 있다.
    write_json(output / "training_config.json", config)
    run["training_config_sha256"] = sha256(output / "training_config.json")
    write_json(output / "effective_config.json", effective)
    try:
        outputs = replay_recorded_history(mac, arrays, meta, max(steps), device, torch)
        decisions, action_rows = summarize_scores(outputs, arrays, steps, agents)
        np.savez_compressed(output / "raw_scores.npz", scores=outputs,
                            avail_actions=arrays["avail_actions"][:len(outputs)],
                            recorded_actions=arrays["actions"][:len(outputs)])
        write_csv(output / "decisions.csv", decisions)
        write_csv(output / "action_scores.csv", action_rows)
        run.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
        write_json(output / "metadata.json", run)
        (output / "summary.md").write_text(summary_markdown(run, decisions), encoding="utf-8")
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "metadata.json", run)
        raise
    print(f"점수 확인 완료: {output / 'summary.md'}")
    return output


def default_output(label):
    return Path(__file__).resolve().parent / "results" / f"{label}_{datetime.now():%Y%m%d-%H%M%S-%f}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "export"):
        p = commands.add_parser(command)
        p.add_argument("--map", default="3s5z")
        p.add_argument("--episode", type=int, default=809, help="지정 H5 파일 안의 1부터 세는 번호")
        p.add_argument("--training-config", type=Path, required=command == "export")
        p.add_argument("--expected-dataset-sha256")
        p.add_argument("--output-dir", type=Path, help="새 폴더만 허용")
        if command == "export":
            p.add_argument("--dataset-file", type=Path, required=True)
            p.add_argument("--through-step", type=int, default=15)
        else:
            inputs = p.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--dataset-file", type=Path)
            inputs.add_argument("--bundle-dir", type=Path)
            p.add_argument("--checkpoint-path", type=Path, required=True)
            p.add_argument("--load-step", type=int, default=30000)
            p.add_argument("--expected-model-sha256")
            p.add_argument("--steps", nargs="+", type=int, default=[12, 13, 14, 15])
            p.add_argument("--agents", nargs="+", type=int, default=[2], help="0부터 세는 agent ID")
            p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
            p.add_argument("--seed", type=int, default=1, help="추론 초기화 seed; 데이터 환경 seed가 아님")
    args = parser.parse_args(argv)
    if getattr(args, "seed", 0) < 0:
        parser.error("seed는 0 이상이어야 합니다.")
    return args


def main(argv=None):
    try:
        execute(parse_args(argv))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"점수 확인 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
