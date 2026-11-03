"""명령·실제 이동·이후 경과를 분리하는 관찰 지표. 생존의 인과 효과는 계산하지 않는다."""

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """연구용 판정 기준이며 사거리/SC2 공식 위험도를 뜻하지 않는다."""

    near_distance: float = 0.75
    low_health: float = 0.30
    recent_window: int = 3
    min_move: float = 0.01
    min_separation: float = 0.01
    horizons: tuple = (3, 5, 10)

    def validate(self):
        numbers = (self.near_distance, self.low_health, self.min_move, self.min_separation)
        if not all(np.isfinite(value) and value > 0 for value in numbers) or self.low_health > 1:
            raise ValueError("거리/이동 기준은 양수, low-health는 0 초과 1 이하여야 합니다.")
        if not isinstance(self.recent_window, int) or self.recent_window < 1:
            raise ValueError("recent-window는 양의 정수여야 합니다.")
        if not self.horizons or any(not isinstance(h, int) or h < 1 for h in self.horizons):
            raise ValueError("horizons에는 양의 정수를 지정하세요.")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons에 중복 값이 있습니다.")
        return self

    def dictionary(self):
        return asdict(self)


def followup(trajectory, step, unit, horizon):
    """종료 직전의 짧은 관측을 h스텝 생존으로 잘못 세지 않도록 검열을 기록한다."""
    end = min(step + horizon, trajectory.steps)
    hp = trajectory.ally_hp[step:end + 1, unit]
    shield = trajectory.ally_shield[step:end + 1, unit]
    alive = bool(hp[-1] > 0)
    complete = end == step + horizon
    return {
        f"h{horizon}_observed_steps": end - step,
        f"h{horizon}_window_complete": complete,
        f"h{horizon}_alive_at_observed_end": alive,
        f"h{horizon}_survived_full_horizon": (alive if complete else False if not alive else None),
        # 회복 후 다시 입은 피해를 net 차이로 상쇄하지 않고 양의 감소량을 더한다.
        f"h{horizon}_health_loss_ratio": float(np.maximum(hp[:-1] - hp[1:], 0).sum()),
        f"h{horizon}_shield_loss_ratio": float(np.maximum(shield[:-1] - shield[1:], 0).sum()),
        f"h{horizon}_shield_gain_ratio": float(np.maximum(shield[1:] - shield[:-1], 0).sum()),
        f"h{horizon}_team_return": float(trajectory.rewards[step:end].sum()),
        f"h{horizon}_enemy_deaths": int(((trajectory.enemy_hp[step] > 0)
                                           & (trajectory.enemy_hp[end] <= 0)).sum()),
        f"h{horizon}_ally_deaths": int(((trajectory.ally_hp[step] > 0)
                                          & (trajectory.ally_hp[end] <= 0)).sum()),
    }


def contexts(trajectory, parameters):
    """모든 생존 agent-step을 판정하고, 승패/쿨다운/공격 마스크로 제외하지 않는다."""
    p = parameters.validate()
    a_hp, e_hp = trajectory.ally_hp, trajectory.enemy_hp
    a_alive, e_alive = a_hp > 0, e_hp > 0
    losses = np.maximum(a_hp[:-1] - a_hp[1:], 0) + np.maximum(
        trajectory.ally_shield[:-1] - trajectory.ally_shield[1:], 0)
    cumulative = np.concatenate((np.zeros((1, losses.shape[1])), np.cumsum(losses, axis=0)))
    death_rows = np.flatnonzero((a_alive[:-1] & ~a_alive[1:]).any(axis=1))
    first_death = int(death_rows[0] + 1) if len(death_rows) else None
    last_attack = [None] * a_hp.shape[1]
    last_attack_step = [-1] * a_hp.shape[1]
    identity = {key: trajectory.meta.get(key) for key in
                ("source_kind", "source_id", "map", "episode", "source_file", "source_index",
                 "quality", "outcome", "env_seed", "agent_sha256")}
    for step in range(trajectory.steps):
        enemies = np.flatnonzero(e_alive[step])
        begin = max(0, step - p.recent_window)
        phase = "cleanup" if len(enemies) <= 1 else "contested"
        for unit in np.flatnonzero(a_alive[step]):
            unit = int(unit)
            hp, shield = float(a_hp[step, unit]), float(trajectory.ally_shield[step, unit])
            action = int(trajectory.actions[step, unit])
            recent_loss = float(cumulative[step, unit] - cumulative[begin, unit])
            position = trajectory.ally_xy[step, unit]
            distances = np.linalg.norm(trajectory.enemy_xy[step, enemies] - position, axis=1)
            near = enemies[distances <= p.near_distance]
            recent_damage, low_health = recent_loss > 1e-7, hp <= p.low_health
            # 건강한 유닛의 선제 이탈도 놓치지 않도록 가까운 적만 있어도 후보 상황으로 둔다.
            risk = recent_damage or bool(len(near))
            risk_band = ("recent_damage_low_health" if recent_damage and low_health else
                         "recent_damage" if recent_damage else
                         "low_health_near_enemy" if len(near) and low_health else
                         "contact_only" if len(near) else "none")
            previous = last_attack[unit]
            died_recently = (previous is not None and last_attack_step[unit] >= begin
                             and e_alive[begin:step + 1, previous].any()
                             and not e_alive[step, previous])
            previous_dies_now = previous is not None and e_alive[step, previous] and not e_alive[step + 1, previous]
            next_alive = bool(a_alive[step + 1, unit])
            displacement = float(np.linalg.norm(trajectory.ally_xy[step + 1, unit] - position)) if next_alive else None
            # 적 사망으로 nearest enemy가 바뀐 것을 거리 확보로 오인하지 않도록
            # t와 t+1에 모두 생존한 동일 적 집합만 비교한다.
            references = near if len(near) else enemies[np.argsort(distances)[:1]]
            references = references[e_alive[step + 1, references]]
            self_delta = observed_delta = reference_distance = None
            if len(references) and next_alive:
                before = trajectory.enemy_xy[step, references]
                after = trajectory.enemy_xy[step + 1, references]
                next_position = trajectory.ally_xy[step + 1, unit]
                reference_distance = float(np.linalg.norm(before - position, axis=1).min())
                # 적 위치를 t에 고정한 값: 거리 변화 중 우리 유닛 이동의 성분을 분리한다.
                self_delta = float(np.linalg.norm(before - next_position, axis=1).min() - reference_distance)
                observed_delta = float(np.linalg.norm(after - next_position, axis=1).min() - reference_distance)
            family = "move" if 2 <= action <= 5 else "stop" if action == 1 else "attack" if action >= 6 else "noop"
            geometry = family
            if family == "move":
                if not next_alive:
                    geometry = "died_during_transition"
                elif displacement < p.min_move:
                    geometry = "move_without_displacement"
                elif not len(references):
                    geometry = "no_surviving_reference"
                elif self_delta > p.min_separation and observed_delta > p.min_separation:
                    geometry = "separation_increase"
                elif self_delta < -p.min_separation:
                    geometry = "approach"
                else:
                    geometry = "lateral_or_unclear"
            row = {
                **identity, "step": step, "unit_id": unit, "unit_type": str(trajectory.ally_types[unit]),
                "action": action, "action_family": family, "movement_geometry": geometry,
                "separation_candidate": geometry == "separation_increase",
                "risk_context": risk, "risk_band": risk_band,
                "health_ratio": hp, "shield_ratio": shield, "low_health": low_health,
                "cooldown_normalized": float(trajectory.ally_cd[step, unit]),
                "cooldown_band": "ready" if trajectory.ally_cd[step, unit] <= 1e-8 else "waiting",
                "recent_loss_ratio_sum": recent_loss, "recent_damage": recent_damage,
                "nearby_enemy_count": len(near), "nearest_enemy_distance": float(distances.min()) if len(enemies) else None,
                "ally_alive_count": int(a_alive[step].sum()), "enemy_alive_count": len(enemies),
                "combat_phase": phase,
                "first_ally_death_phase": "no_ally_death" if first_death is None else (
                    "before_first_ally_death" if step < first_death else "after_first_ally_death"),
                "attack_command_available": bool(trajectory.available[step, unit, 6:].any()),
                "last_attack_target_id": previous, "target_died_recently": bool(died_recently),
                "previous_target_dies_this_transition": bool(previous_dies_now),
                "reference_enemy_ids": references.tolist(), "reference_distance_before": reference_distance,
                "actual_displacement": displacement, "self_separation_delta": self_delta,
                "observed_separation_delta": observed_delta,
            }
            # 이후 경과는 원래 정책의 후속 행동도 포함한다. 한 번의 이동 효과가 아니다.
            if risk or family == "move":
                for horizon in p.horizons:
                    row.update(followup(trajectory, step, unit, horizon))
            yield row
            if action >= 6:
                last_attack[unit], last_attack_step[unit] = action - 6, step


COUNT_NAMES = ("agent_steps", "move_choices", "stop_choices", "attack_choices",
               "attack_available_steps", "separation_candidates", "move_without_displacement",
               "recent_target_death_steps")
GROUP_NAMES = ("source_kind", "source_id", "map", "outcome", "unit_type",
               "cooldown_band", "scope", "combat_phase")


def episode_counts(rows):
    """횟수를 독립 에피소드 수와 구분한다. 분모는 생존 agent-step 전체/위험 후보 상황이다."""
    counts = defaultdict(lambda: np.zeros(len(COUNT_NAMES), dtype=np.int64))
    for row in rows:
        increment = np.array((1, row["action_family"] == "move", row["action_family"] == "stop",
                              row["action_family"] == "attack", row["attack_command_available"],
                              row["separation_candidate"], row["movement_geometry"] == "move_without_displacement",
                              row["target_died_recently"]), dtype=np.int64)
        # all 층을 함께 저장하여 쿨다운/종료 유형을 제한하지 않은 집계를 직접 확인한다.
        for cooldown in ("all", row["cooldown_band"]):
            for scope in (("all", "risk") if row["risk_context"] else ("all",)):
                for phase in ("all", row["combat_phase"]):
                    key = tuple(row[name] for name in GROUP_NAMES[:5]) + (cooldown, scope, phase)
                    counts[key] += increment
    return counts


def count_records(counts, episode):
    for key, value in sorted(counts.items()):
        row = {**dict(zip(GROUP_NAMES, key)), "episode": episode, **dict(zip(COUNT_NAMES, map(int, value)))}
        row["candidate_per_alive_step"] = row["separation_candidates"] / row["agent_steps"]
        yield row


def summarize_counts(records):
    """먼저 에피소드별 비율을 구한다. 수백 개 스텝을 독립 표본으로 세지 않는다."""
    groups = {}
    for row in records:
        key = tuple(row[name] for name in GROUP_NAMES)
        if key not in groups:
            groups[key] = {**dict(zip(GROUP_NAMES, key)), "episodes_with_context": 0,
                           **{name: 0 for name in COUNT_NAMES}, "episodes_with_candidate": 0,
                           "episode_candidate_ratio_mean": 0.0}
        group = groups[key]
        group["episodes_with_context"] += 1
        group["episodes_with_candidate"] += int(row["separation_candidates"] > 0)
        group["episode_candidate_ratio_mean"] += row["candidate_per_alive_step"]
        for name in COUNT_NAMES:
            group[name] += row[name]
    result = []
    for key, group in sorted(groups.items()):
        group["episode_candidate_ratio_mean"] /= group["episodes_with_context"]
        group["pooled_candidate_ratio"] = group["separation_candidates"] / group["agent_steps"]
        result.append(group)
    return result


def separation_segments(rows):
    """연속된 거리 확보 후보를 묶는다. 5개 연속 명령을 전략 5개로 세지 않는다."""
    by_unit = defaultdict(list)
    for row in rows:
        if row["separation_candidate"]:
            by_unit[row["unit_id"]].append(row)
    result = []
    for unit, candidates in sorted(by_unit.items()):
        segments = []
        for row in candidates:
            if (not segments or row["step"] != segments[-1][-1]["step"] + 1
                    or row["combat_phase"] != segments[-1][-1]["combat_phase"]):
                segments.append([])
            segments[-1].append(row)
        for segment in segments:
            first, last = segment[0], segment[-1]
            result.append({**first, "start_step": first["step"], "end_step": last["step"],
                           "candidate_steps": len(segment),
                           "risk_context_any": any(row["risk_context"] for row in segment),
                           "target_died_recently_any": any(row["target_died_recently"] for row in segment),
                           "sum_actual_displacement": sum(row["actual_displacement"] for row in segment),
                           "followup_anchor": "start_step"})
    return result
