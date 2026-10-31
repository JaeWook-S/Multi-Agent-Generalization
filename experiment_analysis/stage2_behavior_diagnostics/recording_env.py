"""기존 SMAC runner의 호출만 복사하는 투명한 에피소드 기록기.

관측을 얻기 위한 추가 호출, 정책 호출, RNG 호출, SC2 요청을 만들지 않는다.
따라서 runner가 마지막 상태에서 고른 미실행 행동은 기록에 들어가지 않는다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
_UNIT_FIELDS = ("health", "health_max", "shield", "shield_max", "cooldown", "x", "y")


def _array_copy(value: Any) -> np.ndarray:
    """torch를 import하지 않고도 실제 runner의 Tensor 값을 즉시 복사한다."""
    for method in ("detach", "cpu", "numpy"):
        conversion = getattr(value, method, None)
        if callable(conversion):
            value = conversion()
    result = np.array(value, copy=True)
    if result.dtype.hasobject:
        raise ValueError("object arrays cannot be recorded safely")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class RecordingEnv:
    """환경을 대체하지 않고 동일한 입력·출력 경로에 기록만 덧붙인다.

    ``_seed`` 같은 환경 필드 쓰기도 원본 환경으로 위임한다. 기록기 내부
    상태는 단 하나의 ``_recording`` 필드에만 보관한다. 불완전한 기록은
    게임 진행을 바꾸지 않고 무효로 표시한 뒤 저장 시 거절한다.
    """

    def __init__(self, env: Any):
        object.__setattr__(self, "_recording", {"env": env, "started": False,
                                               "errors": [], "last_env_info": {}})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._recording["env"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_recording":
            object.__setattr__(self, name, value)
        else:
            setattr(self._recording["env"], name, value)

    @property
    def wrapped_env(self) -> Any:
        return self._recording["env"]

    @property
    def invalid_reasons(self) -> list[str]:
        return list(self._validation_errors())

    @property
    def validity_error(self) -> str | None:
        errors = self._validation_errors()
        return "; ".join(errors) if errors else None

    @property
    def episode_valid(self) -> bool:
        return not self._validation_errors()

    @property
    def last_env_info(self) -> dict[str, Any]:
        return dict(self._recording["last_env_info"])

    def _invalidate(self, message: str) -> None:
        errors = self._recording["errors"]
        if message not in errors:
            errors.append(message)

    def _record_safely(self, operation: Any, *args: Any) -> None:
        # 기록 실패 때문에 원래 정책의 reset/step 반환 경로가 바뀌면 안 된다.
        try:
            operation(*args)
        except Exception as error:
            self._invalidate(f"recording failed: {type(error).__name__}: {error}")

    def _restart_count(self) -> int | None:
        value = getattr(self.wrapped_env, "force_restarts", None)
        return int(value) if value is not None else None

    def _check_restarts(self, before: int | None, after: int | None, where: str) -> None:
        if before != after:
            self._invalidate(f"force_restarts changed during {where}: {before} -> {after}")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        recording = self._recording
        before = self._restart_count()
        recording.update(started=False, errors=[], last_env_info={}, actions=[], rewards=[],
                         terminated=[], inputs={"obs": {}, "state": {}, "avail_actions": {}},
                         units={}, metadata={}, mapping={}, finished=False,
                         force_restarts_before=before, force_restarts_last=before)
        try:
            result = self.wrapped_env.reset(*args, **kwargs)
        except Exception:
            self._invalidate("environment reset raised an exception")
            raise
        recording["started"] = True
        after = self._restart_count()
        self._check_restarts(before, after, "reset")
        recording["force_restarts_last"] = after
        self._record_safely(self._start_episode, result)
        return result

    def _canonical_type(self, unit: Any, ally: bool) -> str:
        env = self.wrapped_env
        unit_type = int(unit.unit_type)
        if ally:
            for attribute, name in (("stalker_id", "Stalker"), ("zealot_id", "Zealot")):
                if getattr(env, attribute, None) == unit_type:
                    return name
        else:
            if unit_type in (74, 73):
                return {74: "Stalker", 73: "Zealot"}[unit_type]
        # SMAC의 소유자별 변환을 사용한다. allied custom ID를 enemy ID와 섞지 않는다.
        canonical = env.get_unit_type_id(unit, ally=ally)
        if canonical in (0, 1):
            # stalkers_and_zealots의 canonical 순서는 소유자마다 다르다.
            names = ("Stalker", "Zealot") if ally else ("Zealot", "Stalker")
            return names[int(canonical)]
        raise ValueError(f"unsupported {'ally' if ally else 'enemy'} unit type {unit_type}")

    def _build_mapping(self, units: Any, count: int, ally: bool) -> list[dict[str, Any]]:
        mapping = []
        for index in range(count):
            unit = units[index]  # dict 삽입 순서가 아닌 SMAC action/observation 인덱스가 기준이다.
            tag = int(unit.tag)
            if not 0 <= tag <= np.iinfo(np.uint64).max:
                raise ValueError(f"unit tag outside uint64 range: {tag}")
            mapping.append({"agent_id" if ally else "enemy_id": index,
                            "unit_tag": tag, "canonical_type": self._canonical_type(unit, ally),
                            "raw_unit_type": int(unit.unit_type)})
        if len({item["unit_tag"] for item in mapping}) != count:
            raise ValueError("duplicate unit tags")
        return mapping

    def _feature_size(self, name: str) -> Any:
        method = getattr(self.wrapped_env, f"get_{name}_size", None)
        return _json_value(method() if callable(method) else getattr(self.wrapped_env, name, None))

    def _start_episode(self, reset_result: Any) -> None:
        env, recording = self.wrapped_env, self._recording
        map_type = getattr(env, "map_type", None)
        if map_type is not None and map_type != "stalkers_and_zealots":
            raise ValueError(f"unsupported map_type for unit identity: {map_type}")
        mapping = {"allies": self._build_mapping(env.agents, int(env.n_agents), True),
                   "enemies": self._build_mapping(env.enemies, int(env.n_enemies), False)}
        recording["mapping"] = mapping
        recording["metadata"] = {
            "schema_version": SCHEMA_VERSION,
            "n_agents": int(env.n_agents), "n_enemies": int(env.n_enemies),
            "n_actions": int(env.n_actions),
            "n_actions_no_attack": int(env.n_actions_no_attack),
            "obs_move_feats": self._feature_size("obs_move_feats"),
            "obs_enemy_shape": self._feature_size("obs_enemy_feats"),
            "obs_ally_shape": self._feature_size("obs_ally_feats"),
            "obs_own_feats": self._feature_size("obs_own_feats"),
            "obs_all_health": bool(getattr(env, "obs_all_health", False)),
            "obs_own_health": bool(getattr(env, "obs_own_health", False)),
            "shield_bits_ally": int(getattr(env, "shield_bits_ally", 0)),
            "shield_bits_enemy": int(getattr(env, "shield_bits_enemy", 0)),
            "state_source": "obs" if getattr(env, "obs_instead_of_state", False) else "global_state",
            "cooldown_analysis_only": True,
            "step_mul": int(getattr(env, "_step_mul", getattr(env, "step_mul", None))),
            "env_seed": _json_value(getattr(env, "_seed", None)), "unit_mapping": mapping,
        }
        self._snapshot_units(0)
        # SMAC reset은 내부에서 obs/state를 읽고 그 값을 반환한다. 이를 재사용한다.
        # 반환값이 없는 환경도 이후 runner의 getter 호출만으로 같은 행을 채운다.
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            self._capture_input("obs", reset_result[0])
            self._capture_input("state", reset_result[1])

    def _snapshot_units(self, row: int) -> None:
        env, recording = self.wrapped_env, self._recording
        snapshot = {}
        for prefix, attr, mapping_key in (("ally", "agents", "allies"),
                                          ("enemy", "enemies", "enemies")):
            units = getattr(env, attr)
            mapping = recording["mapping"][mapping_key]
            values = {field: [] for field in _UNIT_FIELDS}
            for index, identity in enumerate(mapping):
                unit = units[index]
                if int(unit.tag) != identity["unit_tag"]:
                    raise ValueError(f"{prefix} {index} unit tag changed inside episode")
                # protobuf 참조를 저장하면 다음 step이 이전 health까지 바꾼다. 즉시 값으로 복사한다.
                values["health"].append(float(unit.health))
                values["health_max"].append(float(unit.health_max))
                values["shield"].append(float(getattr(unit, "shield", 0)))
                values["shield_max"].append(float(getattr(unit, "shield_max", 0)))
                values["cooldown"].append(float(unit.weapon_cooldown))
                values["x"].append(float(unit.pos.x))
                values["y"].append(float(unit.pos.y))
            snapshot.update({f"{prefix}_{key}": np.asarray(value, dtype=np.float64)
                             for key, value in values.items()})
        recording["units"][row] = snapshot

    def _capture_input(self, name: str, value: Any) -> None:
        recording = self._recording
        if recording["started"]:
            recording["inputs"][name][len(recording["actions"])] = _array_copy(value)

    def get_obs(self, *args: Any, **kwargs: Any) -> Any:
        result = self.wrapped_env.get_obs(*args, **kwargs)
        self._record_safely(self._capture_input, "obs", result)
        return result

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        result = self.wrapped_env.get_state(*args, **kwargs)
        self._record_safely(self._capture_input, "state", result)
        return result

    def get_avail_actions(self, *args: Any, **kwargs: Any) -> Any:
        result = self.wrapped_env.get_avail_actions(*args, **kwargs)
        self._record_safely(self._capture_input, "avail_actions", result)
        return result

    def step(self, actions: Any, *args: Any, **kwargs: Any) -> Any:
        recording = self._recording
        if not recording["started"]:
            raise RuntimeError("RecordingEnv.step requires reset first")
        if recording["finished"]:
            self._invalidate("step called after terminal transition")
        row = len(recording["actions"])
        before = self._restart_count()
        self._check_restarts(recording["force_restarts_last"], before, "before step")
        self._record_safely(self._snapshot_units, row)
        captured_actions = None
        try:
            captured_actions = _array_copy(actions).reshape(-1)
            if captured_actions.shape != (int(self.wrapped_env.n_agents),):
                raise ValueError("action count does not match n_agents")
            if not np.issubdtype(captured_actions.dtype, np.integer):
                raise ValueError("actions must have an integer dtype")
            captured_actions = captured_actions.astype(np.int64, copy=True)
        except Exception as error:
            self._invalidate(f"action recording failed: {error}")
        try:
            result = self.wrapped_env.step(actions, *args, **kwargs)
        except Exception:
            self._invalidate("environment step raised an exception")
            raise
        after = self._restart_count()
        self._check_restarts(before, after, "step")
        recording["force_restarts_last"] = after
        reward, terminated, info = result
        recording["actions"].append(captured_actions)
        recording["rewards"].append(float(reward))
        recording["terminated"].append(bool(terminated))
        recording["last_env_info"] = _json_value(dict(info))
        recording["finished"] = bool(terminated)
        if terminated and "battle_won" not in info:
            self._invalidate("terminal env_info is missing battle_won; outcome is unknown")
        self._record_safely(self._snapshot_units, row + 1)
        return result

    def _validation_errors(self) -> list[str]:
        recording = self._recording
        errors = list(recording["errors"])
        if not recording["started"]:
            return errors + ["episode has not been reset"]
        count = len(recording["actions"])
        if not count or not recording["finished"]:
            errors.append("episode has no completed terminal transition")
        for name, rows in recording["inputs"].items():
            missing = [row for row in range(count + 1) if row not in rows]
            if missing:
                errors.append(f"missing {name} rows: {missing}")
        missing_units = [row for row in range(count + 1) if row not in recording["units"]]
        if missing_units:
            errors.append(f"missing unit snapshot rows: {missing_units}")
        current = self._restart_count()
        if current != recording["force_restarts_last"]:
            errors.append("force_restarts changed after the final recorded call")
        return errors

    def episode_data(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """완전한 T개 전이와 T+1개 상태만 반환하며 누락값을 추정하지 않는다."""
        errors = self._validation_errors()
        if errors:
            raise ValueError("Invalid recorded episode: " + "; ".join(errors))
        recording = self._recording
        count = len(recording["actions"])
        try:
            arrays = {name: np.stack([rows[row] for row in range(count + 1)])
                      for name, rows in recording["inputs"].items()}
            arrays.update(actions=np.stack(recording["actions"]),
                          rewards=np.asarray(recording["rewards"], dtype=np.float64),
                          terminated=np.asarray(recording["terminated"], dtype=np.bool_))
            for name in recording["units"][0]:
                arrays[name] = np.stack([recording["units"][row][name] for row in range(count + 1)])
            for prefix, key in (("ally", "allies"), ("enemy", "enemies")):
                mapping = recording["mapping"][key]
                arrays[f"{prefix}_tag"] = np.asarray([item["unit_tag"] for item in mapping], dtype=np.uint64)
                arrays[f"{prefix}_type"] = np.asarray([item["canonical_type"] for item in mapping], dtype="U7")
            self._validate_shapes(arrays, count)
        except Exception as error:
            self._invalidate(f"episode array validation failed: {error}")
            raise ValueError(f"Invalid recorded episode: {error}") from error
        metadata = dict(recording["metadata"])
        metadata.update(transition_count=count, valid=True, status="valid",
                        battle_won=bool(recording["last_env_info"]["battle_won"]),
                        terminal_env_info=dict(recording["last_env_info"]),
                        force_restarts_before=recording["force_restarts_before"],
                        force_restarts_after=recording["force_restarts_last"])
        return arrays, _json_value(metadata)

    def _validate_shapes(self, arrays: dict[str, np.ndarray], count: int) -> None:
        env = self.wrapped_env
        for name, shape in (("obs", (count + 1, int(env.n_agents))),
                            ("avail_actions", (count + 1, int(env.n_agents), int(env.n_actions))),
                            ("actions", (count, int(env.n_agents)))):
            if arrays[name].shape[:len(shape)] != shape or (name != "obs" and arrays[name].ndim != len(shape)):
                raise ValueError(f"unexpected {name} shape {arrays[name].shape}, expected {shape}")
        if arrays["obs"].ndim != 3 or arrays["state"].ndim != 2:
            raise ValueError("obs and state must have shapes [T+1,N,O] and [T+1,S]")
        if not arrays["terminated"][-1] or arrays["terminated"][:-1].any():
            raise ValueError("only the last executed transition may terminate")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("object arrays are forbidden")

    def save_episode(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> tuple[Path, Path]:
        """검증한 배열과 JSON을 원자적으로 생성하며 기존 파일과 pickle을 쓰지 않는다."""
        arrays, metadata = self.episode_data()
        if extra_metadata:
            extra_metadata = _json_value(extra_metadata)
            conflicts = [key for key in set(metadata) & set(extra_metadata)
                         if metadata[key] != extra_metadata[key]]
            if conflicts:
                raise ValueError(f"extra_metadata cannot replace recorder fields: {sorted(conflicts)}")
            metadata.update(extra_metadata)
        serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        sidecar = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        for destination in (path, sidecar):
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing episode file: {destination}")
        temporary_paths = []
        installed_paths = []
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False) as stream:
                temporary_paths.append(Path(stream.name))
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.stem}-", suffix=".json", delete=False) as stream:
                temporary_paths.append(Path(stream.name))
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            # 같은 디렉터리의 완성된 임시 파일을 link하면 독자가 부분 파일을
            # 보지 않고, 동시 저장이 일어나도 이미 존재하는 파일은 덮어쓰지 않는다.
            for temporary, destination in zip(temporary_paths, (path, sidecar)):
                os.link(temporary, destination)
                installed_paths.append(destination)
        except Exception:
            for installed in installed_paths:
                installed.unlink(missing_ok=True)
            raise
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
        return path, sidecar
