"""고정 정책의 SMAC 행동을 원본 궤적으로부터 진단한다. SC2/torch는 필요 없다.

스냅샷 t에서 고른 행동이 전이 t를 만들고, 그 결과는 스냅샷 t+1이다.
동일 전이의 사망은 같은 의사결정 구간이라는 뜻이며 물리적으로 같은 순간이라는
뜻은 아니다. 비공격 선택률은 행동 빈도일 뿐 인과 효과나 DPS 손실의 추정치가 아니다.
"""

from collections import Counter
from numbers import Real

import numpy as np


OUTCOMES = ("win", "defeat", "timeout", "draw")
UNIT_TYPES = ("Stalker", "Zealot")
_SNAPSHOT_FIELDS = ("health", "health_max", "shield", "shield_max", "cooldown", "x", "y")
_IDENTIFIERS = {"episode", "env_seed"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}은 정수여야 합니다.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name}은 {minimum} 이상이어야 합니다.")
    return value


def _ratio(value, maximum):
    """분모가 없는 체력/실드는 0으로 대체하지 않고 결측으로 남긴다."""
    return float(value / maximum) if maximum > 0 else None


def _validated_arrays(arrays, metadata):
    """실제로 실행한 T개 전이만 읽는다. 버퍼의 패딩/종료 후 행동은 분석하지 않는다."""
    if metadata.get("status") != "valid":
        raise ValueError("status='valid'인 정상 종료 에피소드만 분석할 수 있습니다.")
    if metadata.get("outcome") not in OUTCOMES:
        raise ValueError(f"outcome은 {OUTCOMES} 중 하나여야 합니다.")
    transitions = _integer(metadata.get("transition_count"), "transition_count")
    actions = np.asarray(arrays["actions"])
    if actions.ndim != 2 or len(actions) < transitions or actions.shape[1] < 1:
        raise ValueError("actions는 실제 T개 전이를 포함한 [T, N] 배열이어야 합니다.")
    n_allies = actions.shape[1]
    enemy_types = np.asarray(arrays["enemy_type"])
    if enemy_types.ndim != 1 or len(enemy_types) < 1:
        raise ValueError("enemy_type은 비어 있지 않은 [E] 배열이어야 합니다.")
    n_enemies = len(enemy_types)
    result = {}
    for side, count in (("ally", n_allies), ("enemy", n_enemies)):
        for field in _SNAPSHOT_FIELDS:
            key = f"{side}_{field}"
            if key not in arrays and key == "enemy_cooldown":
                continue
            values = np.asarray(arrays[key])
            if values.ndim != 2 or values.shape[0] < transitions + 1 or values.shape[1] != count:
                raise ValueError(f"{key}는 T+1개 스냅샷을 포함한 [T+1, {count}] 배열이어야 합니다.")
            values = values[:transitions + 1]
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"{key}에 유한하지 않은 값이 있습니다.")
            if field in ("health", "health_max", "shield", "shield_max", "cooldown") and (values < 0).any():
                raise ValueError(f"{key}는 음수일 수 없습니다.")
            result[key] = values
        for field in ("type", "tag"):
            key = f"{side}_{field}"
            values = np.asarray(arrays[key])
            if values.shape != (count,):
                raise ValueError(f"{key}의 유닛 수가 일치하지 않습니다.")
            if field == "type" and not set(values.tolist()).issubset(UNIT_TYPES):
                raise ValueError(f"{key}에는 정규화한 Stalker/Zealot 이름이 필요합니다.")
            if field == "tag" and (not np.issubdtype(values.dtype, np.integer) or (values < 0).any()):
                raise ValueError(f"{key}는 음수가 아닌 정수 tag여야 합니다.")
            result[key] = values
    for key, rank in (("obs", 3), ("state", 2), ("avail_actions", 3)):
        values = np.asarray(arrays[key])
        if values.ndim != rank or values.shape[0] < transitions + 1:
            raise ValueError(f"{key}에 T+1개 스냅샷이 필요합니다.")
        if rank == 3 and values.shape[1] != n_allies:
            raise ValueError(f"{key}의 아군 수가 일치하지 않습니다.")
        values = values[:transitions + 1]
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{key}에 유한하지 않은 값이 있습니다.")
        result[key] = values
    rewards = np.asarray(arrays["rewards"])
    if rewards.ndim != 1 or len(rewards) < transitions or not np.isfinite(rewards[:transitions]).all():
        raise ValueError("rewards는 유한한 [T] 배열이어야 합니다.")
    result["rewards"] = rewards[:transitions]
    actions = actions[:transitions]
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions는 정수 행동 인덱스여야 합니다.")
    offset = _integer(metadata.get("n_actions_no_attack", 6), "n_actions_no_attack")
    if offset != 6:
        raise ValueError("이 진단은 SMAC의 noop=0, stop/move=1..5, attack=6.. 규약을 사용합니다.")
    available = result["avail_actions"]
    n_actions = available.shape[2]
    if n_actions != offset + n_enemies or not np.isin(available, (0, 1)).all():
        raise ValueError("avail_actions는 SMAC 행동 수에 맞는 이진 마스크여야 합니다.")
    if ((actions < 0) | (actions >= n_actions)).any():
        raise ValueError("실행한 행동 인덱스가 행동 공간을 벗어났습니다.")
    if transitions:
        chosen = available[np.arange(transitions)[:, None], np.arange(n_allies)[None, :], actions]
        if not chosen.all():
            step, unit = np.argwhere(chosen == 0)[0]
            raise ValueError(f"마스크가 금지한 행동입니다: step={int(step)}, unit_id={int(unit)}.")
    result["actions"] = actions
    return result, transitions, n_allies, n_enemies, offset


def _observation_layout(data, metadata, n_allies, n_enemies):
    move = _integer(metadata.get("obs_move_feats"), "obs_move_feats")
    own = _integer(metadata.get("obs_own_feats"), "obs_own_feats")
    try:
        enemy_shape = tuple(metadata["obs_enemy_shape"])
        ally_shape = tuple(metadata["obs_ally_shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError("관측 블록의 실제 obs_enemy_shape/obs_ally_shape가 필요합니다.") from exc
    if len(enemy_shape) != 2 or len(ally_shape) != 2:
        raise ValueError("관측 shape는 [유닛 수, 특징 수]여야 합니다.")
    enemy_width = _integer(enemy_shape[1], "obs_enemy_shape[1]", 4)
    ally_width = _integer(ally_shape[1], "obs_ally_shape[1]", 4)
    if enemy_shape[0] != n_enemies or ally_shape[0] != n_allies - 1:
        raise ValueError("관측 블록의 유닛 수가 궤적과 일치하지 않습니다.")
    base = move + n_enemies * enemy_width
    if data["obs"].shape[2] != base + (n_allies - 1) * ally_width + own:
        raise ValueError("관측 벡터 길이가 기록한 SMAC 블록 크기와 다릅니다.")
    all_health = metadata.get("obs_all_health")
    if not isinstance(all_health, (bool, np.bool_)):
        raise ValueError("실제 obs_all_health 설정이 필요합니다.")
    shield_bits = _integer(metadata.get("shield_bits_ally"), "shield_bits_ally")
    if shield_bits not in (0, 1) or (all_health and ally_width < 5 + shield_bits):
        raise ValueError("체력/실드 관측 열 설정과 ally 특징 수가 일치하지 않습니다.")
    return base, ally_width, bool(all_health), shield_bits


def _identity(metadata):
    return {key: metadata.get(key) for key in ("map", "episode", "env_seed", "outcome")}


def _unit_state(data, side, step, unit):
    return {
        "health_ratio": _ratio(data[f"{side}_health"][step, unit], data[f"{side}_health_max"][step, unit]),
        "shield_ratio": _ratio(data[f"{side}_shield"][step, unit], data[f"{side}_shield_max"][step, unit]),
        "cooldown": float(data[f"{side}_cooldown"][step, unit]) if f"{side}_cooldown" in data else None,
        "x": float(data[f"{side}_x"][step, unit]),
        "y": float(data[f"{side}_y"][step, unit]),
    }


def analyze_episode(arrays, metadata, window=5):
    """정상 한 판의 최초 사망·비공격 선택·사망 직전 아군 관측을 계산한다.

    사망 snapshot d는 h[d-1]>0, h[d]<=0인 결과 상태다. 관측 창은
    max(0,d-window)..d-1이며 실제 관측의 ally 가시성 열만 사용한다. 다중 최초
    아군 사망은 각 대상의 관측률을 구한 뒤 동일 가중 평균한다. 자기 관측과 이미
    죽은 관측자는 분모에서 뺀다. 종료 후 미실행 행동은 T개 전이에 포함하지 않는다.
    """
    window = _integer(window, "window", 1)
    data, transitions, n_allies, n_enemies, attack_offset = _validated_arrays(arrays, metadata)
    base, ally_width, all_health, shield_bits = _observation_layout(data, metadata, n_allies, n_enemies)
    ally_alive = data["ally_health"] > 0
    enemy_alive = data["enemy_health"] > 0
    ally_deaths = ally_alive[:-1] & ~ally_alive[1:]
    enemy_deaths = enemy_alive[:-1] & ~enemy_alive[1:]
    ally_times = np.flatnonzero(ally_deaths.any(axis=1))
    any_times = np.flatnonzero(ally_deaths.any(axis=1) | enemy_deaths.any(axis=1))
    first_ally_snapshot = int(ally_times[0] + 1) if len(ally_times) else None
    first_snapshot = int(any_times[0] + 1) if len(any_times) else None
    targets = np.flatnonzero(ally_deaths[first_ally_snapshot - 1]).tolist() if first_ally_snapshot else []
    target_types = [str(data["ally_type"][unit]) for unit in targets]
    first_ally_type = (target_types[0] if len(set(target_types)) == 1 else "mixed") if targets else None
    first_side = "none"
    if first_snapshot is not None:
        first_side = "both" if ally_deaths[first_snapshot - 1].any() and enemy_deaths[first_snapshot - 1].any() else (
            "ally" if ally_deaths[first_snapshot - 1].any() else "enemy")
    metrics = {
        **_identity(metadata), "transition_count": transitions,
        "first_death_snapshot": first_snapshot,
        "first_death_transition": first_snapshot - 1 if first_snapshot is not None else None,
        "first_death_side": first_side,
        "first_ally_death_snapshot": first_ally_snapshot,
        "first_ally_death_transition": first_ally_snapshot - 1 if first_ally_snapshot is not None else None,
        "first_ally_death_type": first_ally_type, "first_ally_death_count": len(targets),
        "enemy_kills_before_first_ally_death": int(enemy_deaths[:first_ally_snapshot - 1].sum()) if targets else None,
        "enemy_kills_same_transition_as_first_ally_death": int(enemy_deaths[first_ally_snapshot - 1].sum()) if targets else None,
        "enemy_kills_through_first_ally_death": int(enemy_deaths[:first_ally_snapshot].sum()) if targets else None,
        "first_ally_death_visibility_window_steps": min(window, first_ally_snapshot) if targets else None,
    }
    events = []
    for transition in any_times.tolist():
        allies = np.flatnonzero(ally_deaths[transition]).tolist()
        enemies = np.flatnonzero(enemy_deaths[transition]).tolist()
        events.append({
            **_identity(metadata), "event": "death", "snapshot": transition + 1,
            "transition": transition, "side": "both" if allies and enemies else "ally" if allies else "enemy",
            "ally_ids": allies, "ally_types": [str(data["ally_type"][unit]) for unit in allies],
            "ally_tags": [int(data["ally_tag"][unit]) for unit in allies],
            "enemy_ids": enemies, "enemy_types": [str(data["enemy_type"][unit]) for unit in enemies],
            "enemy_tags": [int(data["enemy_tag"][unit]) for unit in enemies],
            "is_first_death": transition + 1 == first_snapshot,
            "is_first_ally_death": transition + 1 == first_ally_snapshot,
        })
    context = []
    for step in range(transitions):
        for unit in range(n_allies):
            if not ally_alive[step, unit] or not data["avail_actions"][step, unit, attack_offset:].any():
                continue
            action = int(data["actions"][step, unit])
            context.append({
                **_identity(metadata), "step": step, "unit_id": unit,
                "unit_type": str(data["ally_type"][unit]), "unit_tag": int(data["ally_tag"][unit]),
                "action": action, "is_nonattack": 1 <= action <= 5, "attack_available": True,
                "avail_actions": data["avail_actions"][step, unit].astype(int).tolist(),
                **_unit_state(data, "ally", step, unit),
                "ally_alive_count": int(ally_alive[step].sum()), "enemy_alive_count": int(enemy_alive[step].sum()),
                "relative_step_to_first_ally_death": step - first_ally_snapshot if targets else None,
                "steps_to_first_ally_death": first_ally_snapshot - step if targets else None,
            })
    for unit_type in UNIT_TYPES:
        rows = [row for row in context if row["unit_type"] == unit_type]
        numerator = sum(row["is_nonattack"] for row in rows)
        suffix = unit_type.lower()
        metrics[f"nonattack_numerator_{suffix}"] = numerator
        metrics[f"nonattack_denominator_{suffix}"] = len(rows)
        metrics[f"nonattack_ratio_{suffix}"] = numerator / len(rows) if rows else None
    visibility = []
    target_ratios, visible_total, eligible_total = [], 0, 0
    for target in targets:
        visible_count = 0
        eligible_count = 0
        for step in range(max(0, first_ally_snapshot - window), first_ally_snapshot):
            for observer in range(n_allies):
                if observer == target or not ally_alive[step, observer]:
                    continue
                # SMAC ally 블록은 자기 자신을 빼고 원래 agent ID 순서로 배치한다.
                slot = target if target < observer else target - 1
                features = data["obs"][step, observer, base + slot * ally_width:base + (slot + 1) * ally_width]
                visible = bool(features[0] > 0)
                visible_count += int(visible)
                eligible_count += 1
                visibility.append({
                    **_identity(metadata), "step": step, "death_snapshot": first_ally_snapshot,
                    "target_id": target, "target_type": str(data["ally_type"][target]),
                    "target_tag": int(data["ally_tag"][target]),
                    "observer_id": observer, "observer_type": str(data["ally_type"][observer]),
                    "observer_tag": int(data["ally_tag"][observer]), "ally_obs_slot": slot,
                    "target_visible": visible,
                    "observed_relative_x": float(features[2]) if visible else None,
                    "observed_relative_y": float(features[3]) if visible else None,
                    "observed_health_ratio": float(features[4]) if visible and all_health else None,
                    "observed_shield_ratio": float(features[5]) if visible and all_health and shield_bits else None,
                    "observer_action": int(data["actions"][step, observer]),
                    "observer_avail_actions": data["avail_actions"][step, observer].astype(int).tolist(),
                    "observer_state": _unit_state(data, "ally", step, observer),
                    "target_state": _unit_state(data, "ally", step, target),
                    "ally_alive_count": int(ally_alive[step].sum()), "enemy_alive_count": int(enemy_alive[step].sum()),
                    "relative_step_to_first_ally_death": step - first_ally_snapshot,
                })
        ratio = visible_count / eligible_count if eligible_count else None
        if ratio is not None:
            target_ratios.append(ratio)
        visible_total += visible_count
        eligible_total += eligible_count
        events.append({
            **_identity(metadata), "event": "first_ally_death_target_visibility", "target_id": target,
            "target_type": str(data["ally_type"][target]), "target_tag": int(data["ally_tag"][target]),
            "death_snapshot": first_ally_snapshot, "visible_count": visible_count,
            "eligible_count": eligible_count, "visibility_ratio": ratio,
            "invisible_count": eligible_count - visible_count, "observer_step_count": eligible_count,
            "invisibility_ratio": 1.0 - ratio if ratio is not None else None,
        })
    metrics.update({
        "first_ally_death_visibility_ratio": float(np.mean(target_ratios)) if target_ratios else None,
        "first_ally_death_invisibility_ratio": float(np.mean([1.0 - value for value in target_ratios])) if target_ratios else None,
        "first_ally_death_visibility_visible_count": visible_total if targets else None,
        "first_ally_death_visibility_eligible_count": eligible_total if targets else None,
        "first_ally_death_invisible_count": eligible_total - visible_total if targets else None,
        "first_ally_death_observer_step_count": eligible_total if targets else None,
        "first_ally_death_visibility_valid_target_count": len(target_ratios) if targets else None,
    })
    return {"episode_metrics": metrics, "nonattack_context": context,
            "visibility_windows": visibility, "events": events}


def _aggregate(values, bootstrap_samples, rng, interval):
    values = np.asarray(values, dtype=float)
    result = {"mean": None, "median": None, "valid_n": len(values), "mean_ci95": None}
    if not len(values):
        return result
    result.update(mean=float(values.mean()), median=float(np.median(values)))
    if interval:
        # 각 값은 한 에피소드의 비율이다. agent-step을 재표집하지 않는다.
        means = np.empty(bootstrap_samples)
        for start in range(0, bootstrap_samples, 256):
            count = min(256, bootstrap_samples - start)
            means[start:start + count] = values[rng.integers(len(values), size=(count, len(values)))].mean(axis=1)
        result["mean_ci95"] = np.quantile(means, (0.025, 0.975)).tolist()
    return result


def summarize_diagnostics(episode_metrics_list, bootstrap_samples=1000, seed=0):
    """승/패/시간초과/동시전멸을 나누어 에피소드 단위 평균과 bootstrap CI를 낸다.

    분모가 없는 비율은 None으로 유지하고 valid_n에 포함하지 않는다. 비율의 CI는
    유효 에피소드를 동일 가중 재표집한 percentile 구간이며 학습 seed 변동을 뜻하지
    않는다. 함수 내부 Generator를 써서 numpy 전역 난수 상태를 바꾸지 않는다.
    """
    bootstrap_samples = _integer(bootstrap_samples, "bootstrap_samples", 1)
    seed = _integer(seed, "seed")
    records = list(episode_metrics_list)
    for row in records:
        if row.get("outcome") not in OUTCOMES:
            raise ValueError("정상 outcome이 없는 행은 진단 요약에 포함할 수 없습니다.")
    metric_keys = sorted({
        key for row in records for key, value in row.items()
        if key not in _IDENTIFIERS and (value is None or isinstance(value, Real))
        and key not in {"first_ally_death_type"}
    })
    # 값이 전부 None인 비수치 metadata도 자동 metric으로 들어가지 않게 제한한다.
    metric_keys = [key for key in metric_keys if key == "transition_count" or key.startswith(
        ("first_death_snapshot", "first_death_transition", "first_ally_death_", "enemy_kills_", "nonattack_"))]
    rng = np.random.default_rng(seed)
    groups = {}
    for outcome in OUTCOMES:
        rows = [row for row in records if row["outcome"] == outcome]
        aggregates = {}
        for key in metric_keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), Real)
                      and not isinstance(row[key], (bool, np.bool_)) and np.isfinite(row[key])]
            aggregates[key] = _aggregate(values, bootstrap_samples, rng, "ratio" in key)
        groups[outcome] = {
            "n": len(rows), "metrics": aggregates,
            "first_death_side_counts": dict(Counter(row.get("first_death_side", "none") for row in rows)),
            "first_ally_death_type_counts": dict(Counter(row.get("first_ally_death_type") or "none" for row in rows)),
        }
    return {"groups": groups, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed}
