"""체크포인트를 고정해 맵마다 100회 평가하고 성과 지표를 저장한다. 재학습은 하지 않는다."""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiment_analysis.stage1_map_selection.metrics import episode_record, markdown_report, summarize


def parse_args():
    """분석 전용 CLI. 기본값은 task-config의 모든 test_tasks를 각각 100회 평가한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                        help="models 디렉터리 또는 agent.th가 있는 숫자 체크포인트 디렉터리")
    parser.add_argument("--load-step", type=int, default=0,
                        help="models 경로에서 선택할 정확한 step. 0은 가장 큰 step")
    parser.add_argument("--task-config", required=True, help="예: stalker-zealot-expert")
    parser.add_argument("--training-config", type=Path,
                        help="학습 당시 config.json. 생략하면 체크포인트에서 자동 탐색")
    parser.add_argument("--episodes", type=int, default=100, help="맵별 에피소드 수")
    parser.add_argument("--maps", nargs="+", help="생략하면 모든 test_tasks 평가")
    parser.add_argument("--seed", type=int, default=1, help="학습 seed가 아닌 평가용 기본 seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sc2-path", type=Path, help="SC2 설치 경로. 기본값은 SC2PATH")
    parser.add_argument("--output-dir", type=Path, help="새 결과 디렉터리. 기존 경로 덮어쓰기 금지")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.episodes <= 0 or args.progress_every <= 0 or args.load_step < 0:
        parser.error("episodes/progress-every는 양수, load-step은 0 이상이어야 합니다.")
    if not 0 <= args.seed < 2**31 - 10000:
        parser.error("seed는 0 이상 2**31-10000 미만이어야 합니다.")
    return args


def resolve_checkpoint(path, step):
    """숫자 폴더를 확정한다. 지정 step이 없으면 가까운 모델로 바꾸지 않고 실패한다."""
    path = path.expanduser().resolve()
    if (path / "agent.th").is_file():
        if step and path.name != str(step):
            raise ValueError("체크포인트 폴더 이름과 --load-step이 다릅니다.")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"체크포인트 디렉터리가 없습니다: {path}")
    candidates = {int(p.name): p for p in path.iterdir()
                  if p.is_dir() and p.name.isdigit() and (p / "agent.th").is_file()}
    if not candidates:
        raise FileNotFoundError(f"agent.th가 있는 숫자 하위 폴더가 없습니다: {path}")
    selected = step or max(candidates)
    if selected not in candidates:
        raise FileNotFoundError(f"정확한 step {selected}이 없습니다. 사용 가능: {sorted(candidates)}")
    return candidates[selected]


def load_config(checkpoint, task_name, explicit_config=None):
    """학습 때 저장된 설정을 우선 사용해 모델 구조·관측 설정이 바뀌는 것을 방지한다."""
    source = explicit_config.expanduser().resolve() if explicit_config else checkpoint.parent.parent / "config.json"
    if source.is_file():
        config = json.loads(source.read_text(encoding="utf-8"))
        if config.get("task") != task_name:
            raise ValueError(f"학습 config의 task={config.get('task')}와 --task-config={task_name}이 다릅니다.")
        return config, str(source)
    if explicit_config:
        raise FileNotFoundError(source)
    # config.json을 별도로 복사하지 않은 모델도 현재 저장소의 YAML 설정으로 실행할 수 있다.
    import yaml
    config = {}

    def merge(target, incoming):
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    for name in ("default.yaml", "algs/hissd.yaml", "envs/sc2_offline.yaml", f"tasks/{task_name}.yaml"):
        with (REPO_ROOT / "src/config" / name).open(encoding="utf-8") as stream:
            merge(config, yaml.safe_load(stream))
    return config, "repository YAML (training config.json not found)"


def write_json(path, value):
    """중단 중 파일이 깨지지 않도록 같은 폴더의 임시 파일을 원자적으로 교체한다."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_hash(path):
    """Python 3.10에서도 동작하도록 모델 파일을 나눠 읽으며 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_reports(output_dir, records, maps, train_tasks, episodes, checkpoint, status):
    """이미 끝난 판만 집계한다. 중단된 실행도 CSV/JSON/Markdown으로 남긴다."""
    summaries = []
    for task in maps:
        subset = [record for record in records if record["map"] == task]
        if subset:
            summaries.append(summarize(subset, task, "seen" if task in train_tasks else "unseen", episodes))
    write_json(output_dir / "summary.json", {"status": status, "maps": summaries})
    if summaries:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            # CSV의 None은 빈칸, JSON에서는 null, Markdown에서는 N/A로 저장된다.
            writer.writerows(summaries)
    (output_dir / "summary.md").write_text(markdown_report(summaries, checkpoint, status), encoding="utf-8")


def collect_episode(runner, episode, env_seed, split):
    """기존 정책 실행 경로를 그대로 사용하고, 종료 정보만 분석용으로 꺼낸다."""
    env = runner.env
    runner.test_stats.clear()
    runner.test_returns.clear()
    timeouts_before, restarts_before = env.timeouts, env.force_restarts
    # test_mode=True: 탐험 없이 평가. nolog=True: 기존 평균 로그가 원본 정보를 지우지 않게 한다.
    # runner.run()은 매판 hidden state를 초기화하며 obs/이전 행동/행동 마스크를 기존 MAC에 전달한다.
    runner.run(test_mode=True, nolog=True)
    if len(runner.test_returns) != 1:
        raise RuntimeError("에피소드 보상이 정확히 한 개가 아닙니다.")
    return episode_record(
        runner.task, split, episode, env_seed, dict(runner.test_stats), runner.test_returns[0],
        env.n_agents, env.n_enemies, runner.episode_limit,
        timeouts_before, env.timeouts, restarts_before, env.force_restarts)


def seed_policy(seed, torch, numpy):
    """맵별 난수열을 고정한다. 이전 맵의 길이/실행 순서가 다음 맵에 영향을 주지 않게 한다."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_seed(base_seed, task, test_tasks):
    """평가할 맵의 부분집합·순서와 무관하게 전체 시험 맵 목록의 인덱스로 seed를 정한다."""
    return base_seed + list(test_tasks).index(task)


def run(args, diagnostic_observer=None):
    """모델 하나를 고정하고 평가한다. learner/optimizer 생성과 offline dataset 로딩은 하지 않는다."""
    checkpoint = resolve_checkpoint(args.checkpoint_path, args.load_step)
    config, config_source = load_config(checkpoint, args.task_config, args.training_config)
    if config.get("env") != "sc2" or config.get("mac") != "mt_hissd_mac":
        raise ValueError("이 도구는 SC2의 HiSSD MAC 평가용입니다.")
    maps = args.maps or list(config["test_tasks"])
    if len(set(maps)) != len(maps) or not maps or set(maps) - set(config["test_tasks"]):
        raise ValueError("--maps는 test_tasks에 있는 서로 다른 맵이어야 합니다.")
    sc2_path = (args.sc2_path or Path(os.environ.get("SC2PATH", REPO_ROOT / "3rdparty/StarCraftII"))).expanduser().resolve()
    if not sc2_path.is_dir():
        raise FileNotFoundError(f"SC2 설치 경로가 없습니다. --sc2-path로 지정하세요: {sc2_path}")
    # 상대 SC2PATH로 인한 AppErrors.txt/언어 데이터 초기화 실패를 피한다.
    os.environ["SC2PATH"] = str(sc2_path)
    output_dir = (args.output_dir or REPO_ROOT / "experiment_analysis/stage1_map_selection/results" / args.task_config /
                  f"{checkpoint.name}_{datetime.now():%Y%m%d-%H%M%S-%f}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("performance_analysis")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    metadata = {"status": "initializing", "checkpoint": str(checkpoint), "config_source": config_source,
                "task_config": args.task_config, "training_seed": config.get("seed"), "evaluation_seed": args.seed,
                "episodes_per_map": args.episodes, "maps": maps, "sc2_path": str(sc2_path),
                "started_at": datetime.now().astimezone().isoformat(), "completed_episodes": 0}
    records, runners = [], {}
    status, error = "failed", None
    try:
        # 무거운 의존성은 --help나 집계 테스트에서는 불러오지 않고 실제 평가 때만 불러온다.
        import numpy as np
        import torch
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from mto import init_tasks
        from controllers.multi_task import REGISTRY as mac_registry
        from utils.logging import Logger

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. 드라이버/PyTorch 호환성을 확인하거나 --device cpu를 사용하세요.")
        device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
        if device == "cpu" and args.device == "auto":
            logger.warning("CUDA를 사용할 수 없어 CPU로 평가합니다. GPU를 필수로 하려면 --device cuda를 지정하세요.")
        metadata.update({"device": device, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda})
        if device == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["agent_sha256"] = checkpoint_hash(checkpoint / "agent.th")
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["git_commit"] = None
        config.update({"device": device, "use_cuda": device == "cuda", "evaluate": True,
                       "save_replay": False, "save_model": False, "use_tensorboard": False,
                       "batch_size_run": 1, "test_nepisode": args.episodes,
                       "seed": args.seed, "checkpoint_path": str(checkpoint.parent),
                       "load_step": int(checkpoint.name) if checkpoint.name.isdigit() else 0,
                       "test_greedy": True, "n_tasks": len(config["train_tasks"])})
        config["env_args"]["seed"] = args.seed
        config["env_args"]["heuristic_ai"] = False
        write_json(output_dir / "effective_config.json", config)
        if config_source.startswith("repository YAML"):
            logger.warning("학습 config.json이 없어 현재 YAML 설정을 사용합니다. 학습 때 구조/관측 설정과 같은지 확인하세요.")
        main_args = SimpleNamespace(**config)
        all_tasks = list(dict.fromkeys(config["train_tasks"] + config["test_tasks"] +
                                      (config.get("pretrain_tasks", []) if config.get("pretrain") else [])))
        seed_policy(args.seed, torch, np)
        # 기존 init_tasks()의 관측 규격/전처리만 재사용한다. SC2는 각 맵의 첫 reset에서 시작된다.
        task_args, runners, buffers, schemes, groups, preprocess = init_tasks(all_tasks, main_args, Logger(logger))
        mac = mac_registry[main_args.mac](
            train_tasks=all_tasks, task2scheme={task: buffers[task].scheme for task in all_tasks},
            task2args=task_args, main_args=main_args)
        mac.load_models(str(checkpoint))  # 행동 추론에는 agent.th만 필요하며 opt.th/mixer.th는 읽지 않는다.
        if device == "cuda":
            mac.cuda()
        mac.agent.eval()
        for task in maps:
            runners[task].setup(scheme=schemes[task], groups=groups[task], preprocess=preprocess[task], mac=mac)
            # 2단계에서만 읽기용 기록 환경을 덧씌운다. 원래 runner/MAC 호출 순서는 유지한다.
            if diagnostic_observer is not None:
                runners[task].env = diagnostic_observer.wrap_env(task, runners[task].env)
        del buffers
        map_seeds = {task: map_seed(args.seed, task, config["test_tasks"]) for task in maps}
        metadata.update({"status": "running", "map_seeds": map_seeds})
        if diagnostic_observer is not None:
            metadata.update(diagnostic_observer.on_start(output_dir, config, metadata))
        write_json(output_dir / "run_metadata.json", metadata)
        logger.info("Checkpoint: %s | device: %s | %d maps x %d episodes", checkpoint, device, len(maps), args.episodes)
        logger.info("Results: %s", output_dir)
        with (output_dir / "episodes.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = None
            # 정책 가중치는 고정하고 autograd도 끈다. 실제 신경망 갱신은 한 번도 수행하지 않는다.
            with torch.no_grad():
                for task in maps:
                    runner = runners[task]
                    env_seed = map_seeds[task]
                    runner.env._seed = env_seed
                    runner.t_env = int(checkpoint.name) if checkpoint.name.isdigit() else 0
                    seed_policy(env_seed, torch, np)
                    split = "seen" if task in config["train_tasks"] else "unseen"
                    logger.info("[%s] split=%s | env_seed=%d", task, split, env_seed)
                    for episode in range(1, args.episodes + 1):
                        record = collect_episode(runner, episode, env_seed, split)
                        # 원본 관측/행동의 일관성 검사를 통과하고 저장한 판만 완료 목록에 넣는다.
                        if diagnostic_observer is not None:
                            diagnostic_observer.episode_finished(runner, record)
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        stream.flush()  # 한 판이 끝날 때마다 디스크에 기록해 중단 시 앞선 결과를 보존한다.
                        records.append(record)
                        if episode % args.progress_every == 0 or episode == args.episodes:
                            recent = [r for r in records if r["map"] == task]
                            logger.info("[%s] %d/%d | win=%.1f%% | timeout=%.1f%%", task, episode, args.episodes,
                                        100 * sum(r["battle_won"] for r in recent) / episode,
                                        100 * sum(r["timeout"] for r in recent) / episode)
                            save_reports(output_dir, records, maps, config["train_tasks"], args.episodes, checkpoint, "running")
                    runner.close_env()  # 다음 맵으로 넘어가기 전에 이전 SC2 프로세스를 닫는다.
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        logger.warning("사용자가 중단했습니다. 완료된 에피소드만 저장합니다.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("평가 실패. 완료된 에피소드는 보존합니다.")
    finally:
        for task, runner in runners.items():
            try:
                runner.close_env()
            except Exception:
                logger.exception("[%s] 환경 종료 실패", task)
        metadata.update({"status": status, "completed_episodes": len(records), "error": error,
                         "finished_at": datetime.now().astimezone().isoformat()})
        if diagnostic_observer is not None:
            metadata.update(diagnostic_observer.final_metadata())
        write_json(output_dir / "run_metadata.json", metadata)
        save_reports(output_dir, records, maps, config["train_tasks"], args.episodes, checkpoint, status)
        logger.info("Status: %s | completed episodes: %d | %s", status, len(records), output_dir / "summary.md")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    return 0 if status == "complete" else 130 if status == "interrupted" else 1


def main():
    """기존 명령과 단계별 명령에서 같은 인자 검사·종료 코드를 사용한다."""
    try:
        sys.exit(run(parse_args()))
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
