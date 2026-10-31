"""SC2 없이 사망 경계·관측 분모·에피소드 bootstrap을 검증한다."""

import copy
import json
import unittest

import numpy as np

from experiment_analysis.stage2_behavior_diagnostics.diagnostic_metrics import (
    analyze_episode, summarize_diagnostics,
)


def episode(transitions=3, stored_transitions=None, stalkers=2, zealots=5):
    """유닛 수가 다른 Stalker/Zealot 맵의 원본 궤적을 최소 수치로 만든다."""
    stored = transitions if stored_transitions is None else stored_transitions
    n_units, width, move, own = stalkers + zealots, 8, 4, 4
    size = move + n_units * width + (n_units - 1) * width + own
    arrays = {
        "actions": np.full((stored, n_units), 6, dtype=np.int64),
        "rewards": np.zeros(stored),
        "obs": np.zeros((stored + 1, n_units, size)),
        "state": np.zeros((stored + 1, 20)),
        "avail_actions": np.ones((stored + 1, n_units, 6 + n_units), dtype=np.int8),
    }
    for side in ("ally", "enemy"):
        for name, value in (("health", 10), ("health_max", 10), ("shield", 5),
                            ("shield_max", 5), ("cooldown", 0), ("x", 1), ("y", 2)):
            arrays[f"{side}_{name}"] = np.full((stored + 1, n_units), value, dtype=float)
        arrays[f"{side}_type"] = np.asarray(["Stalker"] * stalkers + ["Zealot"] * zealots)
        # SC2 tag는 float로 바꾸면 정밀도가 손실될 수 있다.
        arrays[f"{side}_tag"] = np.arange(2**60, 2**60 + n_units, dtype=np.uint64)
    metadata = {
        "status": "valid", "map": f"{stalkers}s{zealots}z", "episode": 1, "env_seed": 17,
        "outcome": "defeat", "transition_count": transitions,
        "n_actions_no_attack": 6, "obs_move_feats": move,
        "obs_enemy_shape": [n_units, width], "obs_ally_shape": [n_units - 1, width],
        "obs_own_feats": own, "obs_all_health": True, "shield_bits_ally": 1,
        "step_mul": 8,
    }
    return arrays, metadata


def kill(arrays, side, unit, snapshot):
    arrays[f"{side}_health"][snapshot:, unit] = 0
    if side == "ally":
        arrays["avail_actions"][snapshot:, unit] = 0
        arrays["avail_actions"][snapshot:, unit, 0] = 1
        arrays["actions"][snapshot:, unit] = 0


def seen(arrays, metadata, step, observer, target, visible=True,
         relative_x=0.2, relative_y=-0.3, health=0.4, shield=0.6):
    """자기 자신이 빠지는 SMAC ally 블록의 실제 슬롯을 직접 채운다."""
    if observer == target:
        raise ValueError("자기 자신은 ally 관측 슬롯에 없습니다.")
    base = metadata["obs_move_feats"] + np.prod(metadata["obs_enemy_shape"])
    slot = target if target < observer else target - 1
    start = int(base + slot * metadata["obs_ally_shape"][1])
    arrays["obs"][step, observer, start:start + 6] = (
        int(visible), 0.1, relative_x, relative_y, health, shield,
    )


class DeathMetricTests(unittest.TestCase):
    def test_enemy_kills_strictly_before_and_same_transition_are_separate(self):
        arrays, metadata = episode()
        kill(arrays, "enemy", 0, 1)
        kill(arrays, "enemy", 1, 2)
        kill(arrays, "ally", 0, 2)
        kill(arrays, "ally", 2, 2)
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertEqual(metric["first_death_side"], "enemy")
        self.assertEqual(metric["first_death_snapshot"], 1)
        self.assertEqual(metric["first_ally_death_snapshot"], 2)
        self.assertEqual(metric["first_ally_death_transition"], 1)
        self.assertEqual(metric["first_ally_death_type"], "mixed")
        self.assertEqual(metric["first_ally_death_count"], 2)
        self.assertEqual(metric["enemy_kills_before_first_ally_death"], 1)
        self.assertEqual(metric["enemy_kills_same_transition_as_first_ally_death"], 1)
        self.assertEqual(metric["enemy_kills_through_first_ally_death"], 2)
        event = next(row for row in result["events"] if row.get("is_first_ally_death"))
        self.assertEqual(event["side"], "both")
        self.assertEqual(event["ally_ids"], [0, 2])
        self.assertEqual(event["ally_types"], ["Stalker", "Zealot"])
        self.assertEqual(event["ally_tags"], [2**60, 2**60 + 2])
        self.assertTrue(all(not isinstance(value, (list, dict)) for value in metric.values()))

    def test_first_global_death_both_and_same_type_tie(self):
        arrays, metadata = episode()
        kill(arrays, "ally", 0, 1)
        kill(arrays, "ally", 1, 1)
        kill(arrays, "enemy", 0, 1)
        metric = analyze_episode(arrays, metadata)["episode_metrics"]
        self.assertEqual(metric["first_death_side"], "both")
        self.assertEqual(metric["first_ally_death_type"], "Stalker")
        self.assertEqual(metric["enemy_kills_before_first_ally_death"], 0)
        self.assertEqual(metric["enemy_kills_same_transition_as_first_ally_death"], 1)

    def test_no_allied_death_has_missing_conditional_metrics(self):
        arrays, metadata = episode()
        kill(arrays, "enemy", 0, 1)
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertIsNone(metric["first_ally_death_type"])
        self.assertIsNone(metric["enemy_kills_before_first_ally_death"])
        self.assertIsNone(metric["first_ally_death_invisibility_ratio"])
        self.assertIsNone(metric["first_ally_death_observer_step_count"])
        self.assertEqual(result["visibility_windows"], [])
        self.assertEqual(metric["first_ally_death_count"], 0)
        arrays, metadata = episode()
        self.assertEqual(analyze_episode(arrays, metadata)["episode_metrics"]["first_death_side"], "none")


class VisibilityMetricTests(unittest.TestCase):
    def test_4s5z_uses_nine_units_and_last_ally_slot(self):
        # 7명에 고정하지 않고 4s5z의 9명·15개 행동·마지막 ally 슬롯까지 분석한다.
        arrays, metadata = episode(stalkers=4, zealots=5)
        kill(arrays, "ally", 8, 2)
        for step in range(2):
            for observer in range(8):
                seen(arrays, metadata, step, observer, 8)
        arrays["actions"][1, 3] = 1
        arrays["actions"][1, 7] = 5
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertEqual(metric["first_ally_death_type"], "Zealot")
        self.assertEqual(metric["first_ally_death_observer_step_count"], 16)
        self.assertEqual(metric["first_ally_death_invisibility_ratio"], 0)
        self.assertEqual(metric["nonattack_denominator_stalker"], 12)
        self.assertAlmostEqual(metric["nonattack_ratio_stalker"], 1 / 12)
        self.assertEqual(metric["nonattack_denominator_zealot"], 14)
        self.assertAlmostEqual(metric["nonattack_ratio_zealot"], 1 / 14)
        self.assertTrue(all(len(row["observer_avail_actions"]) == 15
                            for row in result["visibility_windows"]))

    def test_early_death_uses_short_window_and_excludes_self(self):
        arrays, metadata = episode()
        kill(arrays, "ally", 2, 1)
        for observer in range(7):
            if observer != 2:
                seen(arrays, metadata, 0, observer, 2)
        result = analyze_episode(arrays, metadata, window=5)
        metric = result["episode_metrics"]
        self.assertEqual(metric["first_ally_death_visibility_window_steps"], 1)
        self.assertEqual(metric["first_ally_death_observer_step_count"], 6)
        self.assertEqual(metric["first_ally_death_invisible_count"], 0)
        self.assertEqual(metric["first_ally_death_invisibility_ratio"], 0)
        self.assertEqual(len(result["visibility_windows"]), 6)
        self.assertEqual({row["step"] for row in result["visibility_windows"]}, {0})
        self.assertNotIn(2, {row["observer_id"] for row in result["visibility_windows"]})

    def test_observer_specific_slots_and_invisible_values_stay_missing(self):
        arrays, metadata = episode()
        kill(arrays, "ally", 2, 2)
        seen(arrays, metadata, 0, 0, 2, relative_x=0.7, relative_y=-0.2)
        seen(arrays, metadata, 1, 3, 2, relative_x=-0.8, relative_y=0.9)
        # 비가시 상태의 저장 값이 0이 아니어도 관측한 특징으로 취급하지 않는다.
        seen(arrays, metadata, 0, 1, 2, visible=False, health=0.99, shield=0.99)
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertEqual(metric["first_ally_death_observer_step_count"], 12)
        self.assertEqual(metric["first_ally_death_invisible_count"], 10)
        self.assertAlmostEqual(metric["first_ally_death_invisibility_ratio"], 10 / 12)
        rows = result["visibility_windows"]
        first = next(row for row in rows if row["step"] == 0 and row["observer_id"] == 0)
        second = next(row for row in rows if row["step"] == 1 and row["observer_id"] == 3)
        self.assertEqual(first["ally_obs_slot"], 1)
        self.assertEqual(second["ally_obs_slot"], 2)
        self.assertEqual(first["observed_relative_x"], 0.7)
        self.assertEqual(second["observed_relative_x"], -0.8)
        self.assertEqual(first["observed_health_ratio"], 0.4)
        self.assertEqual(first["observed_shield_ratio"], 0.6)
        self.assertEqual(first["observer_action"], 6)
        self.assertEqual(len(first["observer_avail_actions"]), 13)
        invisible = next(row for row in rows if row["step"] == 0 and row["observer_id"] == 1)
        for key in ("observed_relative_x", "observed_relative_y", "observed_health_ratio", "observed_shield_ratio"):
            self.assertIsNone(invisible[key])
        self.assertEqual(invisible["target_state"]["health_ratio"], 1)

    def test_multiple_targets_have_auditable_target_ratios(self):
        arrays, metadata = episode()
        kill(arrays, "ally", 0, 2)
        kill(arrays, "ally", 2, 2)
        for step in (0, 1):
            for observer in range(7):
                if observer != 0:
                    seen(arrays, metadata, step, observer, 0)
        result = analyze_episode(arrays, metadata)
        targets = [row for row in result["events"] if row["event"] == "first_ally_death_target_visibility"]
        self.assertEqual([row["target_id"] for row in targets], [0, 2])
        self.assertEqual([row["eligible_count"] for row in targets], [12, 12])
        self.assertEqual([row["visibility_ratio"] for row in targets], [1, 0])
        metric = result["episode_metrics"]
        self.assertEqual(metric["first_ally_death_invisibility_ratio"], 0.5)
        self.assertEqual(metric["first_ally_death_invisible_count"], 12)
        self.assertEqual(metric["first_ally_death_observer_step_count"], 24)

    def test_dead_observers_and_missing_health_features(self):
        arrays, metadata = episode()
        # 처음부터 없는 관측자는 신규 사망 이벤트를 만들지 않는다.
        arrays["ally_health"][:, 6] = 0
        arrays["avail_actions"][:, 6] = 0
        arrays["avail_actions"][:, 6, 0] = 1
        arrays["actions"][:, 6] = 0
        kill(arrays, "ally", 2, 2)
        seen(arrays, metadata, 0, 0, 2)
        metadata["obs_all_health"] = False
        result = analyze_episode(arrays, metadata)
        self.assertEqual(result["episode_metrics"]["first_ally_death_observer_step_count"], 10)
        self.assertTrue(all(row["observer_id"] != 6 for row in result["visibility_windows"]))
        visible = next(row for row in result["visibility_windows"] if row["target_visible"])
        self.assertIsNone(visible["observed_health_ratio"])
        self.assertIsNone(visible["observed_shield_ratio"])
        self.assertEqual(visible["observed_relative_x"], 0.2)

    def test_no_eligible_observer_is_none_not_zero(self):
        arrays, metadata = episode()
        arrays["ally_health"][:, 1:] = 0
        arrays["avail_actions"][:, 1:] = 0
        arrays["avail_actions"][:, 1:, 0] = 1
        arrays["actions"][:, 1:] = 0
        kill(arrays, "ally", 0, 1)
        result = analyze_episode(arrays, metadata)
        self.assertEqual(result["episode_metrics"]["first_ally_death_observer_step_count"], 0)
        self.assertIsNone(result["episode_metrics"]["first_ally_death_invisibility_ratio"])
        self.assertEqual(result["visibility_windows"], [])


class NonattackMetricTests(unittest.TestCase):
    def test_type_denominators_dead_noop_and_raw_context(self):
        arrays, metadata = episode()
        kill(arrays, "ally", 0, 1)
        arrays["actions"][1, 1] = 1
        arrays["actions"][2, 2] = 5
        arrays["ally_health"][1, 1] = 5
        arrays["ally_shield"][1, 1] = 1
        arrays["ally_cooldown"][1, 1] = 9
        arrays["ally_x"][1, 1] = 13
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertEqual(metric["nonattack_denominator_stalker"], 4)
        self.assertEqual(metric["nonattack_numerator_stalker"], 1)
        self.assertEqual(metric["nonattack_ratio_stalker"], 0.25)
        self.assertEqual(metric["nonattack_denominator_zealot"], 15)
        self.assertEqual(metric["nonattack_numerator_zealot"], 1)
        self.assertAlmostEqual(metric["nonattack_ratio_zealot"], 1 / 15)
        context = result["nonattack_context"]
        self.assertEqual(len(context), 19)
        self.assertFalse(any(row["unit_id"] == 0 and row["step"] >= 1 for row in context))
        row = next(row for row in context if row["unit_id"] == 1 and row["step"] == 1)
        self.assertEqual(row["health_ratio"], 0.5)
        self.assertEqual(row["shield_ratio"], 0.2)
        self.assertEqual(row["cooldown"], 9)
        self.assertEqual(row["x"], 13)
        self.assertEqual(row["ally_alive_count"], 6)
        self.assertEqual(row["enemy_alive_count"], 7)
        self.assertEqual(row["relative_step_to_first_ally_death"], 0)

    def test_no_attack_opportunity_is_missing_despite_move_actions(self):
        arrays, metadata = episode()
        arrays["avail_actions"][:, :, 6:] = 0
        arrays["actions"][:] = 1
        result = analyze_episode(arrays, metadata)
        metric = result["episode_metrics"]
        self.assertEqual(result["nonattack_context"], [])
        self.assertEqual(metric["nonattack_denominator_stalker"], 0)
        self.assertIsNone(metric["nonattack_ratio_stalker"])
        self.assertIsNone(metric["nonattack_ratio_zealot"])

    def test_terminal_unexecuted_action_and_padding_are_excluded(self):
        arrays, metadata = episode(transitions=2, stored_transitions=4)
        arrays["actions"][2:] = 999
        arrays["rewards"][2:] = np.nan
        for key, value in arrays.items():
            if key not in ("actions", "rewards") and value.ndim >= 2:
                value[3:] = 0
        arrays["obs"][3:] = np.nan
        arrays["ally_health"][3:, 0] = 0
        result = analyze_episode(arrays, metadata)
        self.assertEqual(len(result["nonattack_context"]), 14)
        self.assertEqual(result["episode_metrics"]["first_death_side"], "none")
        self.assertEqual(result["episode_metrics"]["nonattack_denominator_stalker"], 4)
        self.assertEqual(result["episode_metrics"]["nonattack_ratio_stalker"], 0)

    def test_invalid_executed_action_is_rejected(self):
        for value in (-1, 13):
            arrays, metadata = episode()
            arrays["actions"][0, 0] = value
            with self.assertRaisesRegex(ValueError, "행동 공간"):
                analyze_episode(arrays, metadata)
        arrays, metadata = episode()
        arrays["avail_actions"][0, 0, 6] = 0
        with self.assertRaisesRegex(ValueError, "마스크"):
            analyze_episode(arrays, metadata)


class ValidationAndSummaryTests(unittest.TestCase):
    def test_bad_status_shape_finite_and_layout_are_rejected(self):
        for modify in (
                lambda a, m: m.update(status="failed"),
                lambda a, m: a.update(ally_health=a["ally_health"][:-1]),
                lambda a, m: a["obs"].__setitem__((0, 0, 0), np.nan),
                lambda a, m: m.update(obs_ally_shape=[6, 7]),
                lambda a, m: a.update(actions=a["actions"].astype(float))):
            arrays, metadata = episode()
            modify(arrays, metadata)
            with self.assertRaises(ValueError):
                analyze_episode(arrays, metadata)

    def test_summary_is_episode_weighted_and_separates_outcomes(self):
        arrays, metadata = episode()
        baseline = analyze_episode(arrays, metadata)["episode_metrics"]
        first = {**baseline, "outcome": "win", "nonattack_ratio_stalker": 0.0,
                 "nonattack_numerator_stalker": 0, "nonattack_denominator_stalker": 100}
        second = {**baseline, "episode": 2, "outcome": "win", "nonattack_ratio_stalker": 1.0,
                  "nonattack_numerator_stalker": 1, "nonattack_denominator_stalker": 1}
        missing = {**baseline, "episode": 3, "outcome": "win", "nonattack_ratio_stalker": None}
        defeat = {**baseline, "outcome": "defeat", "nonattack_ratio_stalker": 0.2}
        result = summarize_diagnostics([first, second, missing, defeat], bootstrap_samples=500, seed=17)
        metric = result["groups"]["win"]["metrics"]["nonattack_ratio_stalker"]
        self.assertEqual(result["groups"]["win"]["n"], 3)
        self.assertEqual(metric["valid_n"], 2)
        self.assertEqual(metric["mean"], 0.5)
        self.assertEqual(metric["median"], 0.5)
        self.assertEqual(metric["mean_ci95"], [0, 1])
        self.assertEqual(result["groups"]["defeat"]["n"], 1)
        self.assertEqual(result["groups"]["timeout"]["n"], 0)
        self.assertIsNone(result["groups"]["timeout"]["metrics"]["nonattack_ratio_stalker"]["mean"])
        self.assertEqual(result["groups"]["draw"]["n"], 0)
        json.dumps(result, allow_nan=False)

    def test_bootstrap_is_deterministic_and_does_not_reseed_numpy(self):
        rows = [{"episode": i, "outcome": "win", "nonattack_ratio_stalker": value}
                for i, value in enumerate((0, 0.1, 0.4, 0.7, 1))]
        before = copy.deepcopy(np.random.get_state())
        first = summarize_diagnostics(rows, bootstrap_samples=31, seed=9)
        second = summarize_diagnostics(rows, bootstrap_samples=31, seed=9)
        after = np.random.get_state()
        self.assertEqual(first, second)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        other = summarize_diagnostics(rows, bootstrap_samples=31, seed=13)
        self.assertNotEqual(first["groups"]["win"]["metrics"]["nonattack_ratio_stalker"]["mean_ci95"],
                            other["groups"]["win"]["metrics"]["nonattack_ratio_stalker"]["mean_ci95"])

    def test_empty_summary_and_single_valid_episode(self):
        summary = summarize_diagnostics([], bootstrap_samples=10)
        self.assertTrue(all(group["n"] == 0 for group in summary["groups"].values()))
        summary = summarize_diagnostics([{"outcome": "timeout", "nonattack_ratio_stalker": 0.3}],
                                        bootstrap_samples=10)
        metric = summary["groups"]["timeout"]["metrics"]["nonattack_ratio_stalker"]
        self.assertEqual(metric["mean_ci95"], [0.3, 0.3])
        with self.assertRaises(ValueError):
            summarize_diagnostics([{"outcome": "failed"}])


if __name__ == "__main__":
    unittest.main()
