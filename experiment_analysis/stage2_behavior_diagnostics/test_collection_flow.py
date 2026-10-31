"""실제 기록기를 붙인 평가 흐름과 오프라인 집계를 SC2 없이 함께 검증한다."""

from contextlib import nullcontext, redirect_stderr
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment_analysis.stage1_map_selection.evaluate_performance import run
from experiment_analysis.stage2_behavior_diagnostics.collect_diagnostics import DiagnosticObserver
from experiment_analysis.stage2_behavior_diagnostics.recording_env import RecordingEnv


class _FixtureSMAC:
    """두 전이 동안 양측 사망과 승리/패배를 만드는 작은 환경 규격이다."""

    n_agents = n_enemies = 2
    n_actions_no_attack, n_actions = 6, 8
    stalker_id, zealot_id = 1974, 1973
    obs_all_health = obs_own_health = True
    obs_instead_of_state = False
    shield_bits_ally = shield_bits_enemy = 1
    _step_mul, episode_limit = 8, 10
    map_type = "stalkers_and_zealots"

    def __init__(self):
        self._seed, self.timeouts, self.force_restarts = 0, 0, 0
        self.episode, self.t, self.closed = 0, 0, False
        self.trace = []
        self.actions = []
        self.reset_seeds = []
        self.restart_on_episode = None

    def _unit(self, index, ally):
        return SimpleNamespace(
            tag=2**63 + 2**53 + 137 + index + (0 if ally else 100),
            unit_type=(self.stalker_id if index == 0 else self.zealot_id) if ally else (74 if index == 0 else 73),
            health=100.0, health_max=100.0, shield=80.0, shield_max=80.0,
            weapon_cooldown=4.0, pos=SimpleNamespace(x=float(index), y=2.0 if ally else 3.0))

    def reset(self):
        self.trace.append("reset")
        self.reset_seeds.append(self._seed)
        self.episode += 1
        self.t = 0
        self.agents = {index: self._unit(index, True) for index in (1, 0)}
        self.enemies = {index: self._unit(index, False) for index in (1, 0)}
        return self.get_obs(), self.get_state()

    def get_obs(self):
        self.trace.append("obs")
        result = np.zeros((2, 26), dtype=np.float32)
        for observer in range(2):
            target = self.agents[1 - observer]
            if self.agents[observer].health > 0 and target.health > 0:
                result[observer, 16:22] = [1, 0.2, 0.1, -0.1,
                                           target.health / target.health_max, target.shield / target.shield_max]
        return result

    def get_state(self):
        self.trace.append("state")
        return np.asarray([self.t, self.agents[0].health, self.agents[1].health,
                           self.enemies[0].health, self.enemies[1].health], dtype=np.float64)

    def get_avail_actions(self):
        self.trace.append("avail_actions")
        result = np.zeros((2, 8), dtype=np.int32)
        for index in range(2):
            if self.agents[index].health <= 0:
                result[index, 0] = 1
            else:
                result[index, 1:6] = 1
                for enemy in range(2):
                    result[index, 6 + enemy] = int(self.enemies[enemy].health > 0)
        return result

    def get_obs_move_feats_size(self):
        return 4

    def get_obs_enemy_feats_size(self):
        return 2, 6

    def get_obs_ally_feats_size(self):
        return 1, 6

    def get_obs_own_feats_size(self):
        return 4

    def get_unit_type_id(self, unit, ally=True):
        return int(unit.unit_type == (self.zealot_id if ally else 73))

    def step(self, actions):
        self.trace.append("step")
        self.actions.append(np.asarray(actions).copy().tolist())
        self.t += 1
        if self.episode == self.restart_on_episode and self.t == 1:
            self.force_restarts += 1
        self.agents[0].health = self.agents[0].shield = 0
        self.enemies[0].health = self.enemies[0].shield = 0
        terminated = self.t == 2
        won = terminated and self.episode % 2 == 1
        if terminated:
            dead = self.enemies[1] if won else self.agents[1]
            dead.health = dead.shield = 0
        for unit in list(self.agents.values()) + list(self.enemies.values()):
            unit.weapon_cooldown = max(0, unit.weapon_cooldown - 1)
            unit.pos.x += 0.25
        return float(self.t), terminated, {"battle_won": won}

    def close(self):
        self.closed = True


class _FixtureMAC:
    """가중치 로드와 실제 정책 선택 횟수를 추적하는 추론 전용 대역이다."""

    def __init__(self, **kwargs):
        self.load_calls, self.policy_inputs, self.eval_calls = [], [], 0
        self.agent = SimpleNamespace(eval=self._eval)

    def _eval(self):
        self.eval_calls += 1

    def load_models(self, path):
        self.load_calls.append(path)

    def select_actions(self, obs, state, available):
        self.policy_inputs.append((obs.copy(), state.copy(), available.copy()))
        result = []
        for index, mask in enumerate(available):
            attacks = np.flatnonzero(mask[6:])
            if mask[0]:
                result.append(0)
            elif index == 1 and state[0] == 0:
                result.append(1)
            else:
                result.append(int(attacks[0]) + 6 if len(attacks) else 1)
        return np.asarray(result, dtype=np.int64)


class _FixtureRunner:
    """기존 runner처럼 T+1번 정책을 선택하고 T번만 환경에서 실행한다."""

    def __init__(self, task):
        self.task, self.env, self.episode_limit = task, _FixtureSMAC(), 10
        self.test_stats, self.test_returns = {}, []

    def setup(self, **kwargs):
        self.mac = kwargs["mac"]

    def _inputs_and_actions(self):
        state = self.env.get_state()
        available = self.env.get_avail_actions()
        obs = self.env.get_obs()
        return self.mac.select_actions(obs, state, available)

    def run(self, test_mode, nolog):
        assert test_mode and nolog
        self.env.reset()
        terminated, total, length = False, 0.0, 0
        while not terminated:
            reward, terminated, info = self.env.step(self._inputs_and_actions())
            total += reward
            length += 1
        self._inputs_and_actions()
        self.test_returns.append(total)
        self.test_stats.update(
            battle_won=info["battle_won"], dead_allies=sum(unit.health <= 0 for unit in self.env.agents.values()),
            dead_enemies=sum(unit.health <= 0 for unit in self.env.enemies.values()), ep_length=length, n_episodes=1)

    def close_env(self):
        self.env.close()


class CollectionFlowTests(unittest.TestCase):
    def _evaluate(self, root, name, observer=None, restart_on_episode=None):
        """학습 코드의 모듈 경계만 바꾸고 평가·기록·파일 생성 구현은 그대로 실행한다."""
        root = root.resolve()
        checkpoint = root / "training/models/30000"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "agent.th").write_bytes(b"fixed diagnostic checkpoint")
        tasks = ["1s3z", "1s4z", "1s5z", "2s3z", "2s4z", "2s5z"]
        config = {"task": "toy", "env": "sc2", "mac": "mt_hissd_mac", "seed": 42,
                  "train_tasks": ["2s3z"], "test_tasks": tasks, "env_args": {"obs_all_health": True}}
        (root / "training/config.json").write_text(json.dumps(config), encoding="utf-8")
        sc2 = root / "sc2"
        sc2.mkdir(exist_ok=True)
        runners, macs = {}, []

        def init_tasks(all_tasks, main_args, logger):
            runners.update({task: _FixtureRunner(task) for task in all_tasks})
            runners["2s5z"].env.restart_on_episode = restart_on_episode
            empty = {task: {} for task in all_tasks}
            return empty, runners, {task: SimpleNamespace(scheme={}) for task in all_tasks}, empty, empty, empty

        def new_mac(**kwargs):
            macs.append(_FixtureMAC(**kwargs))
            return macs[-1]

        def module(name, **attributes):
            result = ModuleType(name)
            result.__dict__.update(attributes)
            return result

        modules = {
            "torch": module("torch", __version__="test", version=SimpleNamespace(cuda=None),
                            cuda=SimpleNamespace(is_available=lambda: False),
                            manual_seed=lambda seed: None, no_grad=nullcontext),
            "mto": module("mto", init_tasks=init_tasks),
            "controllers": module("controllers", __path__=[]),
            "controllers.multi_task": module("controllers.multi_task", REGISTRY={"mt_hissd_mac": new_mac}),
            "utils": module("utils", __path__=[]),
            "utils.logging": module("utils.logging", Logger=lambda logger: logger),
        }
        output = root / name
        args = SimpleNamespace(checkpoint_path=checkpoint.parent, load_step=30000, task_config="toy",
                               training_config=None, maps=["2s5z"], seed=1, sc2_path=sc2, output_dir=output,
                               device="cpu", episodes=2, progress_every=1)
        with patch.dict("sys.modules", modules), patch.dict("os.environ"), redirect_stderr(io.StringIO()):
            code = run(args) if observer is None else run(args, diagnostic_observer=observer)
        self.assertTrue(all(runner.env.closed for runner in runners.values()))
        self.assertEqual((checkpoint / "agent.th").read_bytes(), b"fixed diagnostic checkpoint")
        self.assertEqual(macs[0].load_calls, [str(checkpoint)])
        self.assertEqual(macs[0].eval_calls, 1)
        self.assertEqual(len(macs[0].policy_inputs), 6)
        return code, output, runners["2s5z"], macs[0]

    def test_recording_preserves_policy_and_performance_seed_and_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain_code, plain, plain_runner, plain_mac = self._evaluate(root, "plain")
            self.assertEqual(plain_code, 0, (plain / "run.log").read_text())
            observer = DiagnosticObserver(reference_run=plain)
            code, output, runner, mac = self._evaluate(root, "recorded", observer)
            self.assertEqual(code, 0, (output / "run.log").read_text())
            self.assertIsInstance(runner.env, RecordingEnv)
            self.assertNotIsInstance(plain_runner.env, RecordingEnv)
            self.assertEqual(runner.env.wrapped_env.trace, plain_runner.env.trace)
            self.assertEqual(runner.env.wrapped_env.actions, plain_runner.env.actions)
            for expected_inputs, actual_inputs in zip(plain_mac.policy_inputs, mac.policy_inputs):
                for expected, actual in zip(expected_inputs, actual_inputs):
                    np.testing.assert_array_equal(actual, expected)
            self.assertEqual(runner.env.wrapped_env.reset_seeds, [6, 6])
            self.assertEqual((output / "episodes.csv").read_bytes(), (plain / "episodes.csv").read_bytes())
            self.assertFalse((plain / "step_logs").exists())
            with (output / "episodes.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["outcome"] for row in rows], ["win", "defeat"])
            self.assertEqual([row["env_seed"] for row in rows], ["6", "6"])
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["map_seeds"], {"2s5z": 6})
            self.assertEqual(metadata["recorded_diagnostic_episodes"], 2)
            self.assertEqual(metadata["reference_run"], str(plain))
            expected_hash = hashlib.sha256(b"fixed diagnostic checkpoint").hexdigest()
            self.assertEqual(metadata["agent_sha256"], expected_hash)
            for episode, outcome in ((1, "win"), (2, "defeat")):
                path = output / "step_logs/2s5z" / f"episode_{episode:06d}.npz"
                sidecar = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(sidecar["env_seed"], 6)
                self.assertEqual(sidecar["agent_sha256"], expected_hash)
                self.assertEqual(sidecar["outcome"], outcome)
                self.assertEqual(sidecar["status"], "valid")
                self.assertEqual(sidecar["step_mul"], 8)
                self.assertEqual(sidecar["transition_count"], 2)
                with np.load(path, allow_pickle=False) as arrays:
                    self.assertEqual(arrays["actions"].shape, (2, 2))
                    self.assertEqual(arrays["obs"].shape, (3, 2, 26))
                    self.assertEqual(arrays["ally_tag"][0], 2**63 + 2**53 + 137)
                    self.assertTrue(arrays["terminated"][-1])

    def test_offline_analysis_uses_recorded_steps_without_runtime_imports(self):
        from experiment_analysis.stage2_behavior_diagnostics.analyze_diagnostics import analyze_run

        with TemporaryDirectory() as temporary:
            _, output, _, _ = self._evaluate(Path(temporary), "recorded", DiagnosticObserver())
            originals = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(len(list((output / "step_logs").rglob("*.npz"))), 2,
                             (output / "run.log").read_text())
            with patch.dict("sys.modules", {"torch": None, "mto": None}):
                summary = analyze_run(output, window=5, bootstrap_samples=20, bootstrap_seed=3)
            self.assertTrue(all(path.read_bytes() == value for path, value in originals.items()))
            analysis = output / "analysis/window_5"
            self.assertEqual(summary["analyzed_episodes"], 2)
            self.assertEqual(summary["collection_status"], "complete")
            self.assertEqual(summary["groups"]["win"]["n"], 1)
            self.assertEqual(summary["groups"]["defeat"]["n"], 1)
            for name in ("episode_diagnostics.csv", "nonattack_context.csv", "visibility_windows.csv",
                         "events.csv", "context_comparison.csv", "comparison.json", "comparison.md",
                         "source_manifest.json"):
                self.assertTrue((analysis / name).is_file(), name)
            with (analysis / "episode_diagnostics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["first_death_side"] for row in rows], ["both", "both"])
            self.assertEqual([row["enemy_kills_before_first_ally_death"] for row in rows], ["0", "0"])
            self.assertEqual([row["enemy_kills_same_transition_as_first_ally_death"] for row in rows], ["1", "1"])

    def test_restart_excludes_failed_episode_and_preserves_completed_diagnostics(self):
        from experiment_analysis.stage2_behavior_diagnostics.analyze_diagnostics import analyze_run

        with TemporaryDirectory() as temporary:
            code, output, _, _ = self._evaluate(
                Path(temporary), "recorded", DiagnosticObserver(), restart_on_episode=2)
            self.assertEqual(code, 1)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_episodes"], 1)
            self.assertEqual(metadata["recorded_diagnostic_episodes"], 1)
            self.assertIn("강제 재시작", metadata["error"])
            self.assertTrue((output / "step_logs/2s5z/episode_000001.npz").is_file())
            self.assertFalse((output / "step_logs/2s5z/episode_000002.npz").exists())
            with (output / "episodes.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            summary = analyze_run(output, bootstrap_samples=20)
            self.assertEqual(summary["analyzed_episodes"], 1)
            self.assertEqual(summary["collection_status"], "failed")


if __name__ == "__main__":
    unittest.main()
