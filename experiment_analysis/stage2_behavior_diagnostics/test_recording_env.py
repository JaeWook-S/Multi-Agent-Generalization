"""SC2/torch 없이 원본 runner 호출과 기록의 일대일 대응을 검증한다."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from experiment_analysis.stage2_behavior_diagnostics.recording_env import RecordingEnv


class FakeTensor:
    """torch를 설치하지 않아도 runner의 Tensor 전달 경로를 확인한다."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)
        self.conversions = []

    def detach(self):
        self.conversions.append("detach")
        return self

    def cpu(self):
        self.conversions.append("cpu")
        return self

    def numpy(self):
        self.conversions.append("numpy")
        return self.values


class FakeSMAC:
    """동일한 유닛 객체를 제자리 갱신하는 SMAC의 참조 특성을 재현한다."""

    n_agents = n_enemies = 7
    n_actions_no_attack = 6
    n_actions = 13
    stalker_id, zealot_id = 1974, 1973
    obs_all_health = obs_own_health = True
    obs_instead_of_state = False
    shield_bits_ally = shield_bits_enemy = 1
    _step_mul = 8
    map_type = "stalkers_and_zealots"
    episode_limit = 10

    def __init__(self, return_reset_data=True):
        self.force_restarts = 0
        self._seed = 0
        self.t = 0
        self.return_reset_data = return_reset_data
        self.calls = []
        self.last_returns = {}
        self.step_arguments = []
        self.restart_on_reset = self.restart_on_step = False
        self.missing_battle_won = False

    def _unit(self, index, ally):
        return SimpleNamespace(tag=(2**63 + 2**53 + 137 + index + (0 if ally else 100)),
                               unit_type=(self.stalker_id if index < 2 else self.zealot_id) if ally
                               else (74 if index < 2 else 73),
                               health=100.0, health_max=100.0, shield=80.0, shield_max=80.0,
                               weapon_cooldown=5.0, pos=SimpleNamespace(x=float(index), y=2.0))

    def reset(self):
        self.calls.append("reset")
        self.t = 0
        self.force_restarts += int(self.restart_on_reset)
        # ID 순서와 dict 순서를 의도적으로 다르게 둔다.
        self.agents = {i: self._unit(i, True) for i in reversed(range(self.n_agents))}
        self.enemies = {i: self._unit(i, False) for i in reversed(range(self.n_enemies))}
        result = (self.get_obs(), self.get_state())
        return result if self.return_reset_data else None

    def get_obs(self):
        self.calls.append("obs")
        size = 4 + 7 * 8 + 6 * 8 + 4
        result = (np.arange(self.n_agents * size, dtype=np.float32).reshape(self.n_agents, size)
                  + np.float32(self.t / 10))
        self.last_returns["obs"] = result
        return result

    def get_state(self):
        self.calls.append("state")
        result = np.arange(9, dtype=np.float64) + self.t
        self.last_returns["state"] = result
        return result

    def get_avail_actions(self):
        self.calls.append("avail_actions")
        result = np.ones((self.n_agents, self.n_actions), dtype=np.int32)
        for agent_id, unit in self.agents.items():
            if unit.health <= 0:
                result[agent_id] = 0
                result[agent_id, 0] = 1
        self.last_returns["avail_actions"] = result
        return result

    def get_obs_move_feats_size(self):
        return 4

    def get_obs_enemy_feats_size(self):
        return (7, 8)

    def get_obs_ally_feats_size(self):
        return (6, 8)

    def get_obs_own_feats_size(self):
        return 4

    def get_unit_type_id(self, unit, ally=True):
        self.calls.append(("type_id", ally, unit.unit_type))
        if ally:
            return 0 if unit.unit_type == self.stalker_id else 1
        return 1 if unit.unit_type == 74 else 0

    def step(self, actions):
        self.calls.append("step")
        self.step_arguments.append(actions)
        self.t += 1
        self.force_restarts += int(self.restart_on_step)
        self.agents[0].health = 0
        self.agents[0].shield = 0
        self.enemies[2].health = 0
        for unit in list(self.agents.values()) + list(self.enemies.values()):
            unit.weapon_cooldown = max(0, unit.weapon_cooldown - 2)
            unit.pos.x += 0.5
        terminated = self.t == 2
        if terminated:
            for unit in self.enemies.values():
                unit.health = 0
                unit.shield = 0
        info = {} if self.missing_battle_won else {"battle_won": terminated}
        return float(self.t / 10), terminated, info

    def close(self):
        self.calls.append("close")
        return "closed"


def policy_inputs(env):
    """정책 입력의 원본 접근 순서를 그대로 쓴다."""
    return {"state": env.get_state(), "avail_actions": env.get_avail_actions(), "obs": env.get_obs()}


def run_episode(env):
    env.reset()
    observed = []
    for index in range(2):
        observed.append(policy_inputs(env))
        actions = FakeTensor([6 if index == 0 else 0] + [6] * 6)
        env.step(actions)
    observed.append(policy_inputs(env))
    # 실제 runner도 terminal 상태의 행동을 고르지만 env.step에는 보내지 않는다.
    terminal_selected_action = FakeTensor([0] * 7)
    return observed, terminal_selected_action


class RecordingEnvTests(unittest.TestCase):
    def test_exact_original_calls_inputs_and_action_identity(self):
        plain = FakeSMAC()
        plain_inputs, _ = run_episode(plain)
        real = FakeSMAC()
        recording = RecordingEnv(real)
        wrapped_inputs, _ = run_episode(recording)
        arrays, metadata = recording.episode_data()
        self.assertEqual(real.calls, plain.calls)
        self.assertEqual(len(real.step_arguments), 2)
        for row, (plain_data, wrapped_data) in enumerate(zip(plain_inputs, wrapped_inputs)):
            for name in ("obs", "state", "avail_actions"):
                np.testing.assert_array_equal(plain_data[name], wrapped_data[name])
                np.testing.assert_array_equal(arrays[name][row], wrapped_data[name])
        for action in real.step_arguments:
            self.assertIsInstance(action, FakeTensor)
            self.assertEqual(action.conversions, ["detach", "cpu", "numpy"])
        self.assertEqual(metadata["transition_count"], 2)
        self.assertEqual(metadata["status"], "valid")
        self.assertEqual(metadata["step_mul"], 8)
        self.assertEqual(arrays["actions"].shape, (2, 7))
        self.assertEqual(arrays["obs"].shape[0], 3)
        self.assertEqual(arrays["state"].shape[0], 3)
        self.assertEqual(arrays["avail_actions"].shape[0], 3)

    def test_observer_returns_are_original_objects_and_copies_are_immediate(self):
        real = FakeSMAC()
        env = RecordingEnv(real)
        reset_result = env.reset()
        self.assertIs(reset_result[0], real.last_returns["obs"])
        observed = policy_inputs(env)
        for name, value in observed.items():
            self.assertIs(value, real.last_returns[name])
        expected_obs = observed["obs"].copy()
        observed["obs"][:] = -900  # 호출자가 나중에 값을 바꿔도 이미 기록한 값은 유지한다.
        action = FakeTensor([6] * 7)
        result = env.step(action)
        self.assertEqual(result, (0.1, False, {"battle_won": False}))
        action.values[:] = 0
        policy_inputs(env)
        env.step(np.zeros(7, dtype=np.int64))
        policy_inputs(env)
        arrays, _ = env.episode_data()
        np.testing.assert_array_equal(arrays["obs"][0], expected_obs)
        np.testing.assert_array_equal(arrays["actions"][0], np.full(7, 6))

    def test_unit_mapping_is_indexed_dynamic_and_uint64(self):
        env = RecordingEnv(FakeSMAC())
        run_episode(env)
        arrays, metadata = env.episode_data()
        self.assertEqual(arrays["ally_type"].tolist(), ["Stalker"] * 2 + ["Zealot"] * 5)
        self.assertEqual(arrays["enemy_type"].tolist(), ["Stalker"] * 2 + ["Zealot"] * 5)
        for prefix, offset in (("ally", 0), ("enemy", 100)):
            self.assertEqual(arrays[f"{prefix}_tag"].dtype, np.dtype("uint64"))
            self.assertEqual(arrays[f"{prefix}_tag"].tolist(),
                             [2**63 + 2**53 + 137 + offset + i for i in range(7)])
        self.assertEqual(metadata["unit_mapping"]["allies"][0]["agent_id"], 0)
        self.assertEqual(metadata["unit_mapping"]["allies"][0]["raw_unit_type"], 1974)

    def test_owner_specific_canonical_fallback(self):
        real = FakeSMAC()
        real.stalker_id, real.zealot_id = 3000, 3001
        env = RecordingEnv(real)
        env.reset()
        # 이미 생성한 유닛 ID가 dynamic attr에 없을 때 SMAC owner 플래그를 사용한다.
        real.stalker_id, real.zealot_id = None, None
        original = real.get_unit_type_id
        calls = []

        def owner_type(unit, ally=True):
            calls.append(ally)
            return (0 if unit.unit_type == 3000 else 1) if ally else 0

        real.get_unit_type_id = owner_type
        self.assertEqual(env._canonical_type(real.agents[0], True), "Stalker")
        self.assertEqual(env._canonical_type(SimpleNamespace(unit_type=999), False), "Zealot")
        self.assertEqual(calls, [True, False])
        real.get_unit_type_id = original

    def test_mutated_unit_references_preserve_health_death_and_cooldown(self):
        env = RecordingEnv(FakeSMAC())
        run_episode(env)
        arrays, _ = env.episode_data()
        np.testing.assert_array_equal(arrays["ally_health"][:, 0], [100, 0, 0])
        np.testing.assert_array_equal(arrays["enemy_health"][:, 2], [100, 0, 0])
        np.testing.assert_array_equal(arrays["ally_shield"][:, 0], [80, 0, 0])
        np.testing.assert_array_equal(arrays["ally_cooldown"][:, 1], [5, 3, 1])
        np.testing.assert_array_equal(arrays["ally_x"][:, 1], [1, 1.5, 2])
        self.assertEqual(arrays["actions"][1, 0], 0)
        self.assertEqual(arrays["avail_actions"][1, 0, 0], 1)
        np.testing.assert_array_equal(arrays["terminated"], [False, True])

    def test_attribute_reads_writes_seed_and_methods_delegate(self):
        real = FakeSMAC()
        env = RecordingEnv(real)
        env._seed = 6
        env.custom_field = "real environment"
        self.assertEqual(real._seed, 6)
        self.assertEqual(env._seed, 6)
        self.assertEqual(real.custom_field, "real environment")
        self.assertNotIn("_seed", env.__dict__)
        self.assertEqual(env.close(), "closed")
        self.assertIs(env.wrapped_env, real)

    def test_effective_seed_and_private_step_mul_are_saved(self):
        real = FakeSMAC()
        self.assertFalse(hasattr(real, "step_mul"))
        env = RecordingEnv(real)
        env._seed = 6
        run_episode(env)
        _, metadata = env.episode_data()
        self.assertEqual(real._seed, 6)
        self.assertEqual(metadata["env_seed"], 6)
        self.assertEqual(metadata["step_mul"], 8)
        self.assertEqual(metadata["status"], "valid")

    def test_other_map_type_is_rejected_instead_of_guessing_unit_types(self):
        real = FakeSMAC()
        real.map_type = "marines"
        env = RecordingEnv(real)
        run_episode(env)
        self.assertFalse(env.episode_valid)
        with self.assertRaisesRegex(ValueError, "unsupported map_type"):
            env.episode_data()

    def test_reset_without_returned_obs_uses_only_runner_getters(self):
        env = RecordingEnv(FakeSMAC(return_reset_data=False))
        run_episode(env)
        arrays, _ = env.episode_data()
        self.assertEqual(arrays["obs"].shape[0], 3)

    def test_terminal_getters_and_unexecuted_action_do_not_append_transition(self):
        env = RecordingEnv(FakeSMAC())
        _, terminal_action = run_episode(env)
        self.assertEqual(terminal_action.conversions, [])
        policy_inputs(env)
        arrays, _ = env.episode_data()
        self.assertEqual(arrays["actions"].shape[0], 2)
        self.assertEqual(arrays["rewards"].shape, (2,))
        self.assertEqual(arrays["rewards"].dtype, np.dtype("float64"))

    def test_missing_terminal_inputs_are_rejected_instead_of_fabricated(self):
        env = RecordingEnv(FakeSMAC())
        env.reset()
        for _ in range(2):
            policy_inputs(env)
            env.step(np.zeros(7, dtype=np.int64))
        self.assertFalse(env.episode_valid)
        self.assertIn("missing obs rows: [2]", env.validity_error)
        with self.assertRaisesRegex(ValueError, "missing state rows"):
            env.episode_data()

    def test_restarts_on_reset_step_and_after_terminal_are_invalid(self):
        for phase in ("reset", "step", "after"):
            with self.subTest(phase=phase):
                real = FakeSMAC()
                real.restart_on_reset = phase == "reset"
                real.restart_on_step = phase == "step"
                env = RecordingEnv(real)
                run_episode(env)
                if phase == "after":
                    real.force_restarts += 1
                self.assertFalse(env.episode_valid)
                self.assertIn("force_restarts", env.validity_error)
                with self.assertRaisesRegex(ValueError, "force_restarts"):
                    env.episode_data()

    def test_missing_battle_won_is_unknown_and_cannot_be_saved_as_defeat(self):
        real = FakeSMAC()
        real.missing_battle_won = True
        env = RecordingEnv(real)
        run_episode(env)
        self.assertFalse(env.episode_valid)
        self.assertIn("outcome is unknown", env.validity_error)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "battle_won"):
                env.save_episode(Path(directory) / "invalid.npz")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_npz_without_pickle_and_metadata_exact_tags_no_overwrite(self):
        env = RecordingEnv(FakeSMAC())
        run_episode(env)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000001.npz"
            npz, sidecar = env.save_episode(path, {"episode": 1, "battle_won": True, "n_agents": 7})
            self.assertEqual(npz, path)
            with np.load(npz, allow_pickle=False) as loaded:
                self.assertEqual(loaded["ally_tag"].dtype, np.dtype("uint64"))
                self.assertEqual(loaded["ally_tag"][0], 2**63 + 2**53 + 137)
                self.assertEqual(loaded["ally_type"][0], "Stalker")
                self.assertTrue(all(not loaded[name].dtype.hasobject for name in loaded.files))
            metadata = json.loads(sidecar.read_text())
            self.assertEqual(metadata["unit_mapping"]["allies"][0]["unit_tag"], 2**63 + 2**53 + 137)
            self.assertEqual(metadata["obs_enemy_shape"], [7, 8])
            self.assertEqual(metadata["obs_ally_shape"], [6, 8])
            self.assertEqual(metadata["state_source"], "global_state")
            self.assertTrue(metadata["cooldown_analysis_only"])
            self.assertEqual(metadata["episode"], 1)
            original_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                env.save_episode(path)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), sorted([npz.name, sidecar.name]))

    def test_conflicting_extra_metadata_and_sidecar_collision_are_rejected(self):
        env = RecordingEnv(FakeSMAC())
        run_episode(env)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            with self.assertRaisesRegex(ValueError, "cannot replace"):
                env.save_episode(path, {"battle_won": False})
            path.with_suffix(".json").write_text("existing")
            with self.assertRaises(FileExistsError):
                env.save_episode(path)
            self.assertFalse(path.exists())
            self.assertEqual(path.with_suffix(".json").read_text(), "existing")

    def test_new_reset_discards_previous_episode_and_validation_state(self):
        real = FakeSMAC()
        real.missing_battle_won = True
        env = RecordingEnv(real)
        run_episode(env)
        self.assertFalse(env.episode_valid)
        real.missing_battle_won = False
        run_episode(env)
        self.assertTrue(env.episode_valid)
        self.assertIsNone(env.validity_error)
        self.assertEqual(env.episode_data()[0]["actions"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
