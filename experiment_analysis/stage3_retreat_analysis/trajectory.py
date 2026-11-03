"""SMAC H5/NPZ를 같은 분석 구조로 읽는다. SC2/PyTorch를 실행하지 않는다."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    """대용량 H5 전체를 메모리에 올리지 않고 원본 식별 해시를 계산한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def map_spec(name):
    """패치의 리터럴 맵 정보를 읽되 pysc2를 import하거나 파일을 실행하지 않는다."""
    tree = ast.parse((ROOT / "smac_patch/smac_maps.py").read_text(encoding="utf-8"))
    registry = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "map_param_registry"
                            for target in node.targets))
    spec = registry.get(name)
    if not spec or spec["map_type"] != "stalkers_and_zealots":
        raise ValueError(f"현재 분석은 Stalker–Zealot 맵만 지원합니다: {name}")
    return spec


@dataclass
class Trajectory:
    """위치 단위는 시야 거리=1, 체력·실드는 최대치 대비 비율이다."""

    meta: dict
    actions: np.ndarray
    available: np.ndarray
    rewards: np.ndarray
    ally_hp: np.ndarray
    ally_shield: np.ndarray
    ally_cd: np.ndarray
    ally_xy: np.ndarray
    enemy_hp: np.ndarray
    enemy_xy: np.ndarray
    ally_types: np.ndarray
    calibration: dict

    @property
    def steps(self):
        return len(self.actions)


def decode_episode(data, config, meta):
    """global state 배치와 실제 local obs를 대조한 후 좌표 단위를 복원한다.

    SMAC state 위치는 맵 크기, obs 상대 위치는 시야 거리로 나눈 값이다.
    두 값의 관계를 검증하여 x/y를 각각 환산한다. 맵 크기를 32 등으로
    추측하거나 다른 맵의 환산값을 가져오지 않는다.
    """
    env = config["env_args"]
    if env.get("obs_instead_of_state", False):
        raise ValueError("obs_instead_of_state=True 자료는 global state 분석에 사용할 수 없습니다.")
    if not env.get("obs_all_health", False):
        raise ValueError("좌표/체력 검증을 위해 obs_all_health=True인 자료가 필요합니다.")
    spec = map_spec(meta["map"])
    n, e, bits = spec["n_agents"], spec["n_enemies"], spec["unit_type_bits"]
    if bits != 2 or spec["a_race"] != "P" or spec["b_race"] != "P":
        raise ValueError("현재 decoder는 실드와 두 종류 one-hot이 있는 자료만 지원합니다.")
    actions = np.asarray(data["actions"])
    if actions.ndim == 3 and actions.shape[-1] == 1:
        actions = actions[..., 0]
    t = len(actions)
    if t < 1 or actions.shape != (t, n) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("실행 행동은 정수 [T, N]이어야 합니다.")
    state, obs = np.asarray(data["state"]), np.asarray(data["obs"])
    available, rewards = np.asarray(data["avail_actions"]), np.asarray(data["rewards"])
    expected_state = n * 7 + e * 6 + n * (6 + e) * bool(env.get("state_last_action", True))
    expected_state += bool(env.get("state_timestep_number", False))
    move = 4 + 8 * bool(env.get("obs_pathing_grid", False)) + 9 * bool(env.get("obs_terrain_height", False))
    ally_width = 8 + (6 + e) * bool(env.get("obs_last_action", False))
    own_width = 4 + bool(env.get("obs_timestep_number", False))
    expected_obs = move + e * 8 + (n - 1) * ally_width + own_width
    if state.shape != (t + 1, expected_state) or obs.shape != (t + 1, n, expected_obs):
        raise ValueError(f"state/obs 차원이 설정과 다릅니다: {state.shape}, {obs.shape}; "
                         f"예상 {(t + 1, expected_state)}, {(t + 1, n, expected_obs)}")
    if available.shape != (t + 1, n, 6 + e) or rewards.shape != (t,):
        raise ValueError("마스크 T+1/보상 T 차원이 일치하지 않습니다.")
    for values in (state, obs, available, rewards):
        if not np.isfinite(values).all():
            raise ValueError("원본 배열에 유한하지 않은 수치가 있습니다.")
    if not np.isin(available, [0, 1]).all() or ((actions < 0) | (actions >= 6 + e)).any():
        raise ValueError("행동 인덱스/마스크가 유효하지 않습니다.")
    if not available[np.arange(t)[:, None], np.arange(n)[None, :], actions].all():
        raise ValueError("실제로 기록한 행동이 해당 시점의 행동 마스크와 다릅니다.")
    al = state[:, :n * 7].reshape(t + 1, n, 7)
    en = state[:, n * 7:n * 7 + e * 6].reshape(t + 1, e, 6)
    own_start = move + e * 8 + (n - 1) * ally_width
    own = obs[:, :, own_start:own_start + 4]
    for values in (al[..., 0], al[..., 4], en[..., 0], en[..., 3]):
        if ((values < -1e-5) | (values > 1.00001)).any():
            raise ValueError("state의 체력/실드 비율이 0..1 범위를 벗어났습니다.")
    if (al[..., 1] < 0).any():
        raise ValueError("아군 쿨다운은 음수일 수 없습니다.")
    for values in (al[..., 0], en[..., 0]):
        if ((values[:-1] <= 0) & (values[1:] > 0)).any():
            raise ValueError("에피소드 안에서 사망 유닛이 다시 나타납니다.")
    alive = al[..., 0] > 0
    for values in (al, en):
        active = values[..., 0] > 0
        onehot = values[..., -2:][active]
        if not np.isin(onehot, [0, 1]).all() or not (onehot.sum(axis=-1) == 1).all():
            raise ValueError("살아 있는 유닛의 종류 one-hot이 유효하지 않습니다.")
        initial = values[0, :, -2:]
        if not (values[0, :, 0] > 0).all() or not np.array_equal(
                values[..., -2:][active], np.broadcast_to(initial, values[..., -2:].shape)[active]):
            raise ValueError("초기 유닛 종류를 확인할 수 없거나 도중에 종류가 바뀌었습니다.")
    if not np.allclose(own[..., :2][alive], al[..., [0, 4]][alive], atol=2e-5):
        raise ValueError("local own obs와 global state의 체력·실드가 다릅니다.")
    if not np.array_equal(own[..., 2:4][alive], al[..., -2:][alive]):
        raise ValueError("local own obs와 global state의 유닛 순서/종류가 다릅니다.")
    # ally와 enemy의 one-hot 순서는 서로 반대다. 정렬을 추정하지 않고 규약을 적용한다.
    ally_types = np.array(["Stalker", "Zealot"])[al[0, :, -2:].argmax(axis=-1)]
    enemy_types = np.array(["Zealot", "Stalker"])[en[0, :, -2:].argmax(axis=-1)]
    match = re.fullmatch(r"(\d+)s(\d+)z", meta["map"])
    if match:
        expected = (int(match[1]), int(match[2]))
        for types in (ally_types, enemy_types):
            if tuple(int((types == name).sum()) for name in ("Stalker", "Zealot")) != expected:
                raise ValueError("맵 구성과 one-hot으로 해석한 유닛 수가 다릅니다.")
    enemy_obs = obs[:, :, move:move + e * 8].reshape(t + 1, n, e, 8)
    visible = (enemy_obs[..., 4] > 0) & alive[:, :, None] & (en[:, None, :, 0] > 0)
    delta = en[:, None, :, 1:3] - al[:, :, None, 2:4]
    scale, residuals = [], []
    for axis in range(2):
        x, y = delta[..., axis][visible], enemy_obs[..., 2 + axis][visible]
        denominator = np.dot(x, x)
        if denominator < 1e-12:
            raise ValueError("가시 적 상대 좌표가 부족하여 시야 거리 단위로 환산할 수 없습니다.")
        factor = float(np.dot(x, y) / denominator)
        residual = float(np.max(np.abs(x * factor - y)))
        if not np.isfinite(factor) or factor <= 0 or residual > 2e-4:
            raise ValueError("state/obs 위치 인코딩이 일치하지 않습니다. 다른 SMAC 스키마인지 확인하세요.")
        scale.append(factor)
        residuals.append(residual)
    # NPZ에는 raw 상태도 있다. H5와 공통 decoder를 쓰면서 종류/체력 대응을 한 번 더 검사한다.
    if "ally_health" in data:
        for side, block, hp_column, shield_column, types in (
                ("ally", al, 0, 4, ally_types), ("enemy", en, 0, 3, enemy_types)):
            hp, maximum = np.asarray(data[f"{side}_health"]), np.asarray(data[f"{side}_health_max"])
            active = hp > 0
            ratio = np.divide(hp, maximum, out=np.zeros_like(hp, dtype=float), where=maximum > 0)
            shield, shield_max = np.asarray(data[f"{side}_shield"]), np.asarray(data[f"{side}_shield_max"])
            shield_ratio = np.divide(shield, shield_max, out=np.zeros_like(shield, dtype=float), where=shield_max > 0)
            if not np.allclose(block[..., hp_column], ratio, atol=2e-5) or not np.allclose(
                    block[..., shield_column][active], shield_ratio[active], atol=2e-5):
                raise ValueError("NPZ raw 체력/실드와 state 해석이 다릅니다.")
            if not np.array_equal(data[f"{side}_type"], types):
                raise ValueError("NPZ raw 유닛 종류와 state 해석이 다릅니다.")
    return Trajectory(dict(meta), actions, available, rewards,
                      al[..., 0], al[..., 4], al[..., 1], al[..., 2:4] * scale,
                      en[..., 0], en[..., 1:3] * scale, ally_types,
                      {"coordinate_units": "sight_range=1", "state_to_sight_xy": scale,
                       "max_obs_coordinate_residual_xy": residuals,
                       "cooldown_units": "normalized_by_unit_max_cooldown",
                       "enemy_cooldown_used": False})


def policy_episodes(run_dir, limit=None):
    """완료 CSV에 등록된 원본만 읽고 모델·seed·종료 라벨의 혼입을 막는다."""
    run_dir = Path(run_dir).expanduser().resolve()
    run = read_json(run_dir / "run_metadata.json")
    config = read_json(run_dir / "effective_config.json")
    if run.get("analysis_stage") != "stage2_behavior_diagnostics":
        raise ValueError("2단계 수집 결과 폴더를 지정하세요.")
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != run["completed_episodes"] or not records:
        raise ValueError("완료 에피소드 CSV와 실행 메타데이터가 다릅니다.")
    seen = set()
    for row in records[:limit]:
        episode, task = int(row["episode"]), row["map"]
        if (task, episode) in seen:
            raise ValueError("완료 CSV에 중복 에피소드가 있습니다.")
        seen.add((task, episode))
        path = run_dir / "step_logs" / task / f"episode_{episode:06d}.npz"
        meta = read_json(path.with_suffix(".json"))
        if (meta.get("status") != "valid" or meta.get("map") != task or meta.get("episode") != episode
                or meta.get("agent_sha256") != run["agent_sha256"]
                or meta.get("env_seed") != int(row["env_seed"])
                or meta.get("env_seed") != run["map_seeds"][task]
                or meta.get("outcome") != row["outcome"]
                or meta.get("transition_count") != int(row["episode_length_steps"])):
            raise ValueError(f"모델/맵/seed/종료 정보가 섞인 로그입니다: {path}")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if len(data["actions"]) != meta["transition_count"]:
            raise ValueError("NPZ 행동 수와 완료 정보의 길이가 다릅니다.")
        meta.update(source_kind="policy", source_id=str(run_dir), source_file=str(path),
                    source_index=episode, source_sha256=sha256(path),
                    metadata_sha256=sha256(path.with_suffix(".json")),
                    outcome_source="stage2_terminal_env_info")
        try:
            yield decode_episode(data, config, meta)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error


def h5_episode(data, config, meta):
    """filled의 마지막 상태를 남기고, 그 상태에서 고른 미실행 행동은 제외한다."""
    filled = np.asarray(data["filled"]).reshape(-1)
    if not np.isin(filled, [0, 1]).all():
        raise ValueError("filled는 이진 마스크여야 합니다.")
    length = int(filled.sum())
    if length < 2 or not np.array_equal(filled > 0, np.arange(len(filled)) < length):
        raise ValueError("filled가 연속된 유효 상태와 뒤쪽 패딩의 형태가 아닙니다.")
    t = length - 1
    terminated = np.asarray(data["terminated"]).reshape(-1)[:length]
    if not np.isin(terminated, [0, 1]).all() or terminated[:max(0, t - 1)].any() or terminated[t]:
        raise ValueError("종료 마스크와 마지막 유효 전이가 일치하지 않습니다.")
    packed = {key: data[key][:length] for key in ("state", "obs", "avail_actions")}
    packed.update(actions=data["actions"][:t], rewards=np.asarray(data["reward"][:t]).reshape(t))
    episode = decode_episode(packed, config, meta)
    a, e = int((episode.ally_hp[-1] > 0).sum()), int((episode.enemy_hp[-1] > 0).sum())
    # reward/terminated만으로 timeout을 추정하지 않는다. 살아 있는 양쪽이 남으면 unknown이다.
    outcome = "win" if a > 0 and e == 0 else "defeat" if a == 0 and e > 0 else "draw" if a == e == 0 else "unknown"
    episode.meta.update(outcome=outcome, outcome_source="terminal_global_state",
                        terminal_marker=bool(terminated[t - 1]),
                        unknown_outcome_reason="both_sides_alive_timeout_or_truncation" if outcome == "unknown" else None)
    return episode


def dataset_plan(config, dataset_root, maps=None, audit_all=False, overrides=None):
    """실제 OfflineBuffer의 폴더 선택·part 순서·소스별 뒤쪽 제한을 재현한다.

    무작위 부분집합은 seed만으로 과거 선택을 복구할 수 없다. 그런 설정에서는
    명시적으로 audit_all을 요청해야 하며 결과에도 전체 소스 조사라고 표시한다.
    """
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("H5 분석에는 h5py가 필요합니다. 학습용 hissd 환경에서 실행하세요.") from error
    tasks = list(dict.fromkeys(config.get("pretrain_tasks", []) + config.get("train_tasks", [])))
    selected = tasks if maps is None else list(dict.fromkeys(maps))
    if not selected or not set(selected).issubset(tasks):
        raise ValueError("--maps에는 학습/pretrain에 사용한 맵을 지정하세요.")
    if config.get("offline_data_shuffle", False) and not audit_all:
        raise ValueError("무작위 학습 부분집합은 사후 복원이 불가능합니다. 전체 소스 조사에는 --audit-all을 쓰세요.")
    root, overrides = Path(dataset_root).expanduser().resolve(), overrides or {}
    plans = []
    for task in selected:
        map_spec(task)
        qualities = {config[key][task] for key in ("train_tasks_data_quality", "pretrain_tasks_data_quality")
                     if task in config.get(key, {})}
        if len(qualities) != 1:
            raise ValueError(f"pretrain/train의 데이터 품질이 다르거나 누락됐습니다: {task}")
        quality = qualities.pop()
        sources = ["medium", "expert"] if quality == "medium-expert" else [quality]
        size = int(config.get("offline_data_size", 2000))
        cap = (size if size > 0 else 100000000) // len(sources)
        if cap < 1:
            raise ValueError("소스별 offline_data_size가 0입니다.")
        for source in sources:
            override = overrides.get(f"{task}/{source}")
            if override:
                directory = Path(override).expanduser().resolve()
                resolution = "explicit_source_dir"
            else:
                parent = root / task / source
                folder = config.get("offline_data_name", "")
                if not folder:
                    folders = sorted(p for p in parent.iterdir() if p.is_dir())
                    if not folders:
                        raise FileNotFoundError(f"데이터 소스 폴더가 없습니다: {parent}")
                    directory = folders[-1]
                    resolution = "current_lexicographic_last_folder"
                else:
                    directory, resolution = parent / folder, "config_named_folder"
            files = list(directory.iterdir())
            parts = [p for p in files if "part" in p.name]
            if parts:
                if any(not re.fullmatch(r"part_\d+\.(h5|hdf5)", p.name) for p in parts):
                    raise ValueError(f"OfflineBuffer 규약에 없는 part 파일이 있습니다: {directory}")
                indexed = {int(p.stem.split("_")[1]): p for p in parts}
                if len(indexed) != len(parts) or set(indexed) != set(range(max(indexed) + 1)):
                    raise ValueError(f"part 번호가 중복되거나 연속되지 않습니다: {directory}")
                if len({p.suffix for p in parts}) != 1:
                    raise ValueError("part 확장자가 혼합되어 원래 loader의 파일 선택을 재현할 수 없습니다.")
                files = [indexed[i] for i in range(len(indexed))]
            elif len(files) != 1:
                raise ValueError("part가 없는 다중 파일은 원래 loader의 순서가 불명확합니다. part_N.h5 소스를 지정하세요.")
            counts = []
            for path in files:
                if not path.is_file() or path.suffix not in (".h5", ".hdf5"):
                    raise ValueError(f"H5 이외의 소스는 자동 해석하지 않습니다: {path}")
                with h5py.File(path, "r") as file:
                    keys = ("state", "obs", "actions", "avail_actions", "reward", "filled", "terminated")
                    if not set(keys).issubset(file):
                        raise ValueError(f"필수 H5 dataset이 누락됐습니다: {path}")
                    count = file["state"].shape[0]
                    if any(file[key].shape[0] != count for key in keys):
                        raise ValueError("H5 key별 에피소드 수가 다릅니다.")
                    counts.append(count)
            total, begin, cursor = sum(counts), 0, 0
            if not audit_all:
                begin = max(0, total - cap)
            for path, count in zip(files, counts):
                start = max(0, begin - cursor)
                if start < count:
                    plans.append({"map": task, "quality": source, "path": str(path),
                                  "start": start, "stop": count, "source_total_episodes": total,
                                  "folder_resolution": resolution})
                cursor += count
    return plans


def dataset_episodes(plan, config, limit=None):
    """H5를 읽기 전용으로 열고 한 에피소드씩 처리해 메모리 사용을 제한한다."""
    import h5py
    counts = {}
    for source in plan:
        task, path = source["map"], Path(source["path"])
        if limit is not None and counts.get(task, 0) >= limit:
            continue
        stop = source["stop"] if limit is None else min(
            source["stop"], source["start"] + limit - counts.get(task, 0))
        with h5py.File(path, "r") as file:
            datasets = {key: file[key] for key in
                        ("state", "obs", "actions", "avail_actions", "reward", "filled", "terminated")}
            # 작은 블록으로 읽으면 큰 압축 chunk를 에피소드마다 다시 푸는 비용을 줄인다.
            # 전체 2,000판을 한꺼번에 메모리에 복사하지 않는다.
            for begin in range(source["start"], stop, 16):
                end = min(begin + 16, stop)
                block = {key: dataset[begin:end] for key, dataset in datasets.items()}
                for index in range(begin, end):
                    counts[task] = counts.get(task, 0) + 1
                    data = {key: values[index - begin] for key, values in block.items()}
                    meta = {"source_kind": "dataset", "source_id": str(path.parent),
                            "source_file": str(path), "source_index": index, "map": task,
                            "quality": source["quality"], "episode": counts[task], "env_seed": None,
                            "agent_sha256": None}
                    try:
                        yield h5_episode(data, config, meta)
                    except ValueError as error:
                        raise ValueError(f"{path}, H5 row {index}: {error}") from error
