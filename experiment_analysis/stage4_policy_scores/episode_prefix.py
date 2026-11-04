"""H5 한 에피소드를 검증하고 추론에 필요한 앞부분만 읽기 전용으로 추출한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_analysis.stage3_retreat_analysis.trajectory import h5_episode, map_spec, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_hash(actual, expected, label):
    """같은 이름의 다른 모델/데이터로 분석하는 일을 막는다."""
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(f"{label} SHA-256이 기준과 다릅니다: actual={actual}, expected={expected}")


def validate_prefix(arrays, metadata):
    """T개 정답 행동과 T+1개 상태의 경계를 검사한다. 마지막 상태의 행동은 넣지 않는다."""
    if metadata.get("schema_version") != 1 or metadata.get("start_step") != 0:
        raise ValueError("0스텝부터 시작하는 schema_version=1 prefix가 필요합니다.")
    spec = map_spec(metadata["map"])
    n, e = spec["n_agents"], spec["n_enemies"]
    actions = np.asarray(arrays["actions"])
    t = metadata.get("prefix_transitions")
    if not isinstance(t, int) or t < 1 or metadata.get("through_step") != t - 1:
        raise ValueError("prefix 길이와 마지막 실행 스텝이 일치하지 않습니다.")
    if metadata.get("original_transitions", 0) < t:
        raise ValueError("원본보다 긴 prefix입니다.")
    if actions.shape != (t, n) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions는 실제 실행한 정수 [T,N]이어야 합니다.")
    obs, avail = np.asarray(arrays["obs"]), np.asarray(arrays["avail_actions"])
    if obs.ndim != 3 or obs.shape[:2] != (t + 1, n) or avail.shape != (t + 1, n, 6 + e):
        raise ValueError("관측·마스크의 T+1/N/행동 수가 일치하지 않습니다.")
    for name in ("obs", "avail_actions", "state", "health_ratio", "shield_ratio", "cooldown_normalized"):
        values = np.asarray(arrays[name])
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{name}에는 유한한 수치만 허용됩니다.")
    for name in ("health_ratio", "shield_ratio", "cooldown_normalized"):
        if arrays[name].shape != (t + 1, n):
            raise ValueError(f"{name}의 상태 수가 다릅니다.")
    if arrays["state"].ndim != 2 or len(arrays["state"]) != t + 1:
        raise ValueError("state에 T+1개 스냅샷이 필요합니다.")
    if arrays["unit_type"].shape != (n,) or not set(arrays["unit_type"].tolist()).issubset({"Stalker", "Zealot"}):
        raise ValueError("유닛 종류 매핑이 유효하지 않습니다.")
    if not np.isin(avail, (0, 1)).all() or ((actions < 0) | (actions >= 6 + e)).any():
        raise ValueError("행동 인덱스 또는 마스크가 유효하지 않습니다.")
    if not avail[np.arange(t)[:, None], np.arange(n)[None, :], actions].all():
        raise ValueError("기록 행동과 실제 행동 마스크가 다릅니다.")
    if not avail.any(axis=-1).all():
        raise ValueError("허용 행동이 하나도 없는 유닛 상태가 있습니다.")
    return t, n, 6 + e


def read_h5_prefix(dataset_file, config, task, episode, through_step, expected_hash=None):
    """episode는 지정 H5 안의 1부터 세는 번호다. 809번은 H5 row 808이다."""
    if episode < 1 or through_step < 0:
        raise ValueError("episode는 1 이상, through-step은 0 이상이어야 합니다.")
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("H5 추출에는 h5py가 필요합니다. 기존 학습 환경에서 실행하세요.") from error
    path = Path(dataset_file).expanduser().resolve()
    source_hash = sha256(path)
    check_hash(source_hash, expected_hash, "데이터")
    required = ("obs", "state", "actions", "avail_actions", "reward", "filled", "terminated")
    with h5py.File(path, "r") as file:
        if not set(required).issubset(file):
            raise ValueError("원본 H5의 필수 key가 누락됐습니다.")
        count = file["obs"].shape[0]
        if not 1 <= episode <= count or any(file[key].shape[0] != count for key in required):
            raise ValueError(f"episode 또는 key별 에피소드 수가 유효하지 않습니다. 범위: 1..{count}")
        data = {key: file[key][episode - 1] for key in required}
        stored_onehot = file["actions_onehot"][episode - 1] if "actions_onehot" in file else None
    # 기존 3단계 decoder로 패딩/관측/종류/좌표 및 행동 정합성을 같은 기준으로 검사한다.
    decoded = h5_episode(data, config, {"map": task, "episode": episode})
    full_t, n = decoded.actions.shape
    n_actions = decoded.available.shape[-1]
    if through_step >= full_t:
        raise ValueError(f"스텝 {through_step}은 미실행/없는 행동입니다. 실행 스텝 범위: 0..{full_t - 1}")
    onehot = np.eye(n_actions, dtype=np.float32)[decoded.actions]
    if stored_onehot is not None and not np.array_equal(stored_onehot[:full_t], onehot):
        raise ValueError("H5 actions_onehot과 정수 actions가 다릅니다.")
    if config["env_args"].get("state_last_action", True):
        spec = map_spec(task)
        offset = n * 7 + spec["n_enemies"] * 6
        last = data["state"][1:full_t + 1, offset:offset + n * n_actions].reshape(full_t, n, n_actions)
        if not np.array_equal(last, onehot):
            raise ValueError("state[t+1]의 이전 행동과 actions[t]가 다릅니다. 시간 정렬을 먼저 확인하세요.")
    length = through_step + 1
    arrays = {"obs": np.array(data["obs"][:length + 1], copy=True),
              "state": np.array(data["state"][:length + 1], copy=True),
              "avail_actions": np.array(decoded.available[:length + 1], copy=True),
              "actions": np.array(decoded.actions[:length], copy=True),
              "health_ratio": np.array(decoded.ally_hp[:length + 1], copy=True),
              "shield_ratio": np.array(decoded.ally_shield[:length + 1], copy=True),
              "cooldown_normalized": np.array(decoded.ally_cd[:length + 1], copy=True),
              "unit_type": np.array(decoded.ally_types, dtype="U7")}
    metadata = {"schema_version": 1, "map": task, "episode": episode, "h5_row_index": episode - 1,
                "source_file": str(path), "source_h5_sha256": source_hash, "start_step": 0,
                "through_step": through_step, "prefix_transitions": length, "original_transitions": full_t,
                "source_outcome": decoded.meta["outcome"], "calibration": decoded.calibration,
                "state_last_action_alignment_checked": bool(config["env_args"].get("state_last_action", True)),
                "actor_input_keys": ["obs", "recorded_previous_actions", "agent_id", "reconstructed_hidden_state"],
                "global_state_used_by_actor": False}
    validate_prefix(arrays, metadata)
    return arrays, metadata


def save_bundle(output_dir, arrays, metadata, config):
    """Mac으로 옮길 작은 prefix를 저장한다. 모델 가중치는 별도로 준비한다."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "prefix.npz", **arrays)
    write_json(output / "config.json", config)
    metadata = dict(metadata, prefix_npz_sha256=sha256(output / "prefix.npz"),
                    bundle_config_sha256=sha256(output / "config.json"))
    write_json(output / "metadata.json", metadata)
    return output


def load_bundle(bundle_dir, expected_dataset_hash=None):
    """원본 전체 H5 없이도 prefix·설정의 체크섬과 배열 경계를 검사한다."""
    root = Path(bundle_dir).expanduser().resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    check_hash(sha256(root / "prefix.npz"), meta["prefix_npz_sha256"], "prefix")
    check_hash(sha256(root / "config.json"), meta["bundle_config_sha256"], "bundle config")
    check_hash(meta["source_h5_sha256"], expected_dataset_hash, "원본 데이터")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    with np.load(root / "prefix.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_prefix(arrays, meta)
    return arrays, meta, config
