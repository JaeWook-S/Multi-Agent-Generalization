"""실제 효과를 오인하기 쉬운 거리·사망·패딩·표본 분모를 SC2 없이 검증한다."""

import contextlib
import csv
from dataclasses import replace
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment_analysis.stage3_retreat_analysis.analyze_retreats import parse_args, run_analysis
from experiment_analysis.stage3_retreat_analysis.retreat_metrics import (
    Parameters, contexts, count_records, episode_counts, followup, separation_segments, summarize_counts,
)
from experiment_analysis.stage3_retreat_analysis.trajectory import (
    Trajectory, dataset_episodes, dataset_plan, decode_episode, h5_episode, policy_episodes,
)


def simple_trajectory(steps=4, outcome="win"):
    """1개 아군이 살아 있는 동일 적에게서 멀어지는 작은 궤적이다."""
    return Trajectory(
        {"source_kind": "policy", "source_id": "test", "source_file": "test.npz", "source_index": 1,
         "map": "test", "episode": 1, "outcome": outcome, "outcome_source": "test"},
        np.full((steps, 1), 5, dtype=int), np.ones((steps + 1, 1, 8), dtype=int),
        np.zeros(steps), np.ones((steps + 1, 1)), np.ones((steps + 1, 1)),
        np.ones((steps + 1, 1)) * 0.2,
        np.array([[[-0.1 * t, 0.0]] for t in range(steps + 1)]),
        np.ones((steps + 1, 2)), np.tile([[0.4, 0.0], [0.6, 0.1]], (steps + 1, 1, 1)),
        np.array(["Stalker"]), {})


CONFIG = {"env_args": {"obs_all_health": True, "state_last_action": True},
          "train_tasks": ["2s3z"], "pretrain_tasks": ["2s3z"],
          "train_tasks_data_quality": {"2s3z": "expert"},
          "pretrain_tasks_data_quality": {"2s3z": "expert"},
          "offline_data_size": 2, "offline_data_shuffle": False, "offline_data_name": ""}


def encoded_episode(steps=4):
    """2s3z 규약으로 state/obs를 만들고 서로 다른 x/y 정규화를 넣는다."""
    n, e = 5, 5
    actions = np.full((steps, n), 1, dtype=np.int64)
    actions[:, 0] = 5
    state = np.zeros((steps + 1, n * 7 + e * 6 + n * (6 + e)), dtype=np.float32)
    obs = np.zeros((steps + 1, n, 4 + e * 8 + (n - 1) * 8 + 4), dtype=np.float32)
    available = np.ones((steps + 1, n, 6 + e), dtype=np.int8)
    factors = np.array([2.0, 3.0])
    ally_xy = np.tile(np.array([[0.0, i * 0.025] for i in range(n)]), (steps + 1, 1, 1))
    ally_xy[:, 0, 0] = -np.arange(steps + 1) * 0.1
    enemy_xy = np.tile(np.array([[0.4 + i * 0.02, i * 0.035] for i in range(e)]), (steps + 1, 1, 1))
    for t in range(steps + 1):
        al = state[t, :n * 7].reshape(n, 7)
        en = state[t, n * 7:n * 7 + e * 6].reshape(e, 6)
        for unit in range(n):
            kind = 0 if unit < 2 else 1
            al[unit, [0, 1, 4, 5 + kind]] = [1, .2, 1, 1]
            al[unit, 2:4] = ally_xy[t, unit] / factors
            obs[t, unit, -4:] = [1, 1, 1 - kind, kind]
        for enemy in range(e):
            kind = 1 if enemy < 2 else 0
            en[enemy, [0, 3, 4 + kind]] = [1, 1, 1]
            en[enemy, 1:3] = enemy_xy[t, enemy] / factors
            for unit in range(n):
                relative = enemy_xy[t, enemy] - ally_xy[t, unit]
                base = 4 + enemy * 8
                obs[t, unit, base:base + 8] = [1, np.linalg.norm(relative), *relative, 1, 1, 1 - kind, kind]
    data = {"state": state, "obs": obs, "actions": actions,
            "avail_actions": available, "rewards": np.arange(steps, dtype=float)}
    meta = {"map": "2s3z", "episode": 1, "outcome": "win", "source_id": "test",
            "source_kind": "policy", "outcome_source": "test", "source_file": "test.npz", "source_index": 1}
    return data, meta


def as_h5_row(data, padding=3):
    """최종 상태에 저장된 미실행 행동을 유효 범위 밖 값으로 채워 경계를 검사한다."""
    t = len(data["actions"])
    result = {}
    for key in ("state", "obs", "avail_actions"):
        result[key] = np.pad(data[key], [(0, padding)] + [(0, 0)] * (data[key].ndim - 1))
    result["actions"] = np.pad(data["actions"][..., None], [(0, padding + 1), (0, 0), (0, 0)], constant_values=999)
    result["reward"] = np.pad(data["rewards"][:, None], [(0, padding + 1), (0, 0)], constant_values=999)
    result["filled"] = np.zeros((t + 1 + padding, 1), dtype=int)
    result["filled"][:t + 1] = 1
    result["terminated"] = np.zeros_like(result["filled"])
    result["terminated"][t - 1] = 1
    return result


class MovementTests(unittest.TestCase):
    def test_healthy_waiting_and_no_attack_opportunity_are_not_filtered(self):
        for outcome in ("win", "defeat", "timeout", "draw", "unknown"):
            trajectory = simple_trajectory(outcome=outcome)
            trajectory.available[..., 6:] = 0
            rows = list(contexts(trajectory, Parameters()))
            self.assertEqual(len(rows), trajectory.steps)
            self.assertTrue(rows[0]["separation_candidate"])
            self.assertEqual(rows[0]["cooldown_band"], "waiting")
            self.assertEqual(rows[0]["risk_band"], "contact_only")
            self.assertFalse(rows[0]["attack_command_available"])
            self.assertFalse(rows[0]["low_health"])
            self.assertEqual(rows[0]["first_ally_death_phase"], "no_ally_death")

    def test_recent_damage_without_close_enemy_is_retained(self):
        trajectory = simple_trajectory()
        trajectory.enemy_xy += 3
        trajectory.ally_hp[1:] = 0.9
        row = list(contexts(trajectory, Parameters()))[1]
        self.assertTrue(row["risk_context"])
        self.assertEqual(row["risk_band"], "recent_damage")
        self.assertEqual(row["nearby_enemy_count"], 0)

    def test_enemy_motion_alone_is_not_our_retreat(self):
        trajectory = simple_trajectory()
        trajectory.ally_xy[:] = 0
        trajectory.enemy_xy[:, :, 0] += np.arange(trajectory.steps + 1)[:, None] * .2
        row = list(contexts(trajectory, Parameters()))[0]
        self.assertEqual(row["movement_geometry"], "move_without_displacement")
        self.assertFalse(row["separation_candidate"])

    def test_enemy_chasing_can_cancel_our_distance_gain(self):
        trajectory = simple_trajectory()
        trajectory.enemy_xy[:, :, 0] -= np.arange(trajectory.steps + 1)[:, None] * .15
        row = list(contexts(trajectory, Parameters()))[0]
        self.assertGreater(row["self_separation_delta"], 0)
        self.assertLess(row["observed_separation_delta"], 0)
        self.assertFalse(row["separation_candidate"])

    def test_dead_reference_does_not_create_distance_gain(self):
        trajectory = simple_trajectory()
        trajectory.enemy_hp[1:, 0] = 0
        trajectory.enemy_xy[1:, 0] = 0
        trajectory.ally_xy[:, 0, 0] = np.arange(trajectory.steps + 1) * .05
        row = list(contexts(trajectory, Parameters()))[0]
        self.assertEqual(row["reference_enemy_ids"], [1])
        self.assertEqual(row["movement_geometry"], "approach")
        self.assertFalse(row["separation_candidate"])

    def test_cleanup_and_recent_target_death_are_flagged_not_excluded(self):
        trajectory = simple_trajectory()
        trajectory.actions[0, 0] = 6
        trajectory.enemy_hp[1:, 0] = 0
        row = list(contexts(trajectory, Parameters()))[1]
        self.assertEqual(row["combat_phase"], "cleanup")
        self.assertTrue(row["target_died_recently"])
        self.assertTrue(row["separation_candidate"])

    def test_death_coordinates_are_not_movement(self):
        trajectory = simple_trajectory()
        trajectory.ally_hp[1:] = 0
        trajectory.ally_xy[1:] = 0
        rows = list(contexts(trajectory, Parameters()))
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["actual_displacement"])
        self.assertEqual(rows[0]["movement_geometry"], "died_during_transition")

    def test_followup_censoring_and_gross_loss(self):
        trajectory = simple_trajectory()
        trajectory.ally_hp[:, 0] = [.5, .7, .5, .5, .5]
        measured = followup(trajectory, 0, 0, 3)
        self.assertAlmostEqual(measured["h3_health_loss_ratio"], .2)
        measured = followup(trajectory, 3, 0, 3)
        self.assertFalse(measured["h3_window_complete"])
        self.assertTrue(measured["h3_alive_at_observed_end"])
        self.assertIsNone(measured["h3_survived_full_horizon"])
        trajectory.ally_hp[-1] = 0
        self.assertFalse(followup(trajectory, 3, 0, 3)["h3_survived_full_horizon"])

    def test_segments_and_episode_weighting(self):
        rows = list(contexts(simple_trajectory(), Parameters()))
        segments = separation_segments(rows)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["candidate_steps"], 4)
        first = list(count_records(episode_counts(rows[:1]), 1))
        other = [{**row, "episode": 2, "agent_steps": 9, "separation_candidates": 0,
                  "candidate_per_alive_step": 0.0} for row in first]
        summary = summarize_counts(iter(first + other))
        for row in summary:
            self.assertEqual(row["episodes_with_context"], 2)
            self.assertAlmostEqual(row["episode_candidate_ratio_mean"], .5)
            self.assertAlmostEqual(row["pooled_candidate_ratio"], .1)

    def test_segments_do_not_mix_cleanup_and_contested(self):
        rows = list(contexts(simple_trajectory(), Parameters()))
        for row in rows[2:]:
            row["combat_phase"] = "cleanup"
        segments = separation_segments(rows)
        self.assertEqual(len(segments), 2)
        self.assertEqual([row["combat_phase"] for row in segments], ["contested", "cleanup"])


class DecodeTests(unittest.TestCase):
    def test_coordinate_calibration_and_h5_policy_equivalence(self):
        data, meta = encoded_episode()
        policy = decode_episode(data, CONFIG, meta)
        offline = h5_episode(as_h5_row(data), CONFIG, meta)
        np.testing.assert_allclose(policy.calibration["state_to_sight_xy"], [2, 3], atol=1e-5)
        np.testing.assert_allclose(policy.ally_xy[:, 0, 0], -np.arange(5) * .1, atol=1e-5)
        np.testing.assert_array_equal(policy.actions, offline.actions)
        np.testing.assert_array_equal(policy.rewards, offline.rewards)
        self.assertEqual(offline.steps, 4)
        self.assertEqual(offline.meta["outcome"], "unknown")
        left = list(contexts(policy, Parameters()))
        right = list(contexts(offline, Parameters()))
        self.assertEqual([row["movement_geometry"] for row in left], [row["movement_geometry"] for row in right])

    def test_filled_hole_and_early_terminal_are_rejected(self):
        data, meta = encoded_episode()
        row = as_h5_row(data)
        row["filled"][1] = 0
        with self.assertRaisesRegex(ValueError, "filled"):
            h5_episode(row, CONFIG, meta)
        row = as_h5_row(data)
        row["terminated"][0] = 1
        with self.assertRaisesRegex(ValueError, "종료"):
            h5_episode(row, CONFIG, meta)

    def test_layout_and_coordinate_corruption_are_rejected(self):
        data, meta = encoded_episode()
        data["state"] = data["state"][:, :-1]
        with self.assertRaisesRegex(ValueError, "차원"):
            decode_episode(data, CONFIG, meta)
        data, meta = encoded_episode()
        data["obs"][0, 0, 6] += .5
        with self.assertRaisesRegex(ValueError, "인코딩"):
            decode_episode(data, CONFIG, meta)

    def test_local_own_information_is_checked(self):
        data, meta = encoded_episode()
        data["obs"][0, 0, -4] = .5
        with self.assertRaisesRegex(ValueError, "체력"):
            decode_episode(data, CONFIG, meta)

    def test_h5_terminal_army_counts_define_observed_outcome(self):
        data, meta = encoded_episode()
        # 마지막 적 전멸 상태를 만들되 과거 상태는 그대로 두어 종료 경계만 검사한다.
        data["state"][-1, 35:65] = 0
        data["obs"][-1, :, 4:44] = 0
        self.assertEqual(h5_episode(as_h5_row(data), CONFIG, meta).meta["outcome"], "win")


@unittest.skipUnless(importlib.util.find_spec("h5py"), "H5 검증에는 h5py가 필요합니다.")
class H5IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "dataset/2s3z/expert/qmix_test"
        self.directory.mkdir(parents=True)
        self.data, self.meta = encoded_episode()
        row = as_h5_row(self.data)
        import h5py
        for part, count in [(0, 2), (1, 3)]:
            with h5py.File(self.directory / f"part_{part}.h5", "w") as file:
                for key, value in row.items():
                    file.create_dataset(key, data=np.stack([value] * count))
        self.config = {**CONFIG, "offline_data_size": 4}

    def tearDown(self):
        self.temp.cleanup()

    def test_tail_selection_across_files_and_unknown_outcome(self):
        plan = dataset_plan(self.config, self.root / "dataset")
        self.assertEqual([(Path(p["path"]).name, p["start"], p["stop"]) for p in plan],
                         [("part_0.h5", 1, 2), ("part_1.h5", 0, 3)])
        trajectories = list(dataset_episodes(plan, self.config))
        self.assertEqual(len(trajectories), 4)
        self.assertEqual([t.meta["source_index"] for t in trajectories], [1, 0, 1, 2])
        self.assertTrue(all(t.meta["outcome"] == "unknown" for t in trajectories))
        self.assertEqual(len(list(dataset_episodes(plan, self.config, 2))), 2)

    def test_shuffle_requires_explicit_full_source_scope(self):
        config = {**self.config, "offline_data_shuffle": True}
        with self.assertRaisesRegex(ValueError, "무작위"):
            dataset_plan(config, self.root / "dataset")
        plan = dataset_plan(config, self.root / "dataset", audit_all=True)
        self.assertEqual(sum(p["stop"] - p["start"] for p in plan), 5)

    def test_cli_writes_compressed_reports_and_preserves_sources(self):
        config = self.root / "config.json"
        config.write_text(json.dumps(self.config), encoding="utf-8")
        output = self.root / "result"
        args = parse_args(["dataset", "--training-config", str(config), "--dataset-root", str(self.root / "dataset"),
                           "--output-dir", str(output), "--limit-episodes", "2"])
        with contextlib.redirect_stdout(io.StringIO()):
            run_analysis(args)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["processed_episodes"], 2)
        self.assertEqual(len(manifest["h5_sources"]), 2)
        self.assertFalse(manifest["historical_training_manifest_available"])
        self.assertEqual(manifest["selection_scope"], "reconstructed_loader_tail")
        with gzip.open(output / "movement_steps.csv.gz", "rt") as stream:
            movements = list(csv.DictReader(stream))
        self.assertTrue(movements)
        self.assertEqual({row["outcome"] for row in movements}, {"unknown"})
        self.assertTrue((self.directory / "part_0.h5").is_file())
        with self.assertRaises(FileExistsError):
            run_analysis(args)


class PolicyIntegrationTests(unittest.TestCase):
    def test_mixed_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step_logs/2s3z").mkdir(parents=True)
            (root / "effective_config.json").write_text(json.dumps(CONFIG))
            (root / "run_metadata.json").write_text(json.dumps({
                "analysis_stage": "stage2_behavior_diagnostics", "completed_episodes": 1,
                "agent_sha256": "expected", "map_seeds": {"2s3z": 4}}))
            (root / "episodes.csv").write_text("map,episode,env_seed,outcome,episode_length_steps\n2s3z,1,4,win,4\n")
            (root / "step_logs/2s3z/episode_000001.json").write_text(json.dumps({
                "status": "valid", "map": "2s3z", "episode": 1, "env_seed": 4,
                "agent_sha256": "different", "outcome": "win", "transition_count": 4}))
            with self.assertRaisesRegex(ValueError, "섞인 로그"):
                list(policy_episodes(root))


if __name__ == "__main__":
    unittest.main()
