"""SC2 실행 없이 성과 지표의 분모와 SMAC 종료 처리 경계를 검증한다."""

import hashlib
from contextlib import nullcontext, redirect_stderr
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_analysis.evaluate_performance import (
    checkpoint_hash, collect_episode, load_config, resolve_checkpoint, run, save_reports)
from experiment_analysis.metrics import episode_record, summarize, wilson_interval


def record(won=False, timeout=False, dead_enemies=2, length=5):
    """지표 검증용으로 종료 상태만 구성한다. 실제 정책 구현을 흉내 내지는 않는다."""
    return episode_record(
        "3m", "unseen", 1, 7,
        {"battle_won": won, "dead_allies": 1 if won else 3,
         "dead_enemies": dead_enemies, "ep_length": length, "n_episodes": 1},
        20.0 if won else 8.0, 3, 3, 10, 4, 5 if timeout else 4, 0, 0)


class MetricsTests(unittest.TestCase):
    def test_timeout_does_not_require_episode_limit_info(self):
        result = record(timeout=True, length=10)
        self.assertTrue(result["timeout"])
        self.assertEqual(result["outcome"], "timeout")

    def test_terminal_battle_on_last_step_is_not_timeout(self):
        self.assertFalse(record(won=True, dead_enemies=3, length=10)["timeout"])
        self.assertFalse(record(won=False, length=10)["timeout"])

    def test_no_wins_has_no_winning_length(self):
        result = summarize([record(), record(timeout=True, length=10)], "3m", "unseen", 2)
        self.assertEqual(result["win_rate"], 0)
        self.assertIsNone(result["win_length_mean_steps"])
        self.assertIsNone(result["win_length_median_steps"])
        self.assertAlmostEqual(result["nonwin_enemy_remaining_ratio_mean"], 1 / 3)
        self.assertEqual(result["timeout_rate"], 0.5)

    def test_all_wins_has_no_nonwin_ratio(self):
        result = summarize([record(True, dead_enemies=3)], "3m", "seen", 1)
        self.assertIsNone(result["nonwin_enemy_remaining_ratio_mean"])
        self.assertEqual(result["win_rate"], 1)

    def test_conditional_denominators_and_median(self):
        rows = [record(True, dead_enemies=3, length=2), record(True, dead_enemies=3, length=4),
                record(False, dead_enemies=0, length=9), record(False, dead_enemies=2, length=7)]
        result = summarize(rows, "3m", "seen", 100)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertAlmostEqual(result["nonwin_enemy_remaining_ratio_mean"], 2 / 3)
        self.assertEqual(result["win_length_mean_steps"], 3)
        self.assertEqual(result["win_length_median_steps"], 3)
        self.assertFalse(result["complete"])

    def test_wilson_known_values_and_extremes(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038315, places=6)
        self.assertAlmostEqual(high, 0.5961685, places=6)
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0)
        self.assertGreater(wilson_interval(0, 100)[1], 0)
        self.assertLess(wilson_interval(100, 100)[0], 1)
        self.assertAlmostEqual(wilson_interval(100, 100)[1], 1)

    def test_invalid_episode_and_crash_are_not_counted_as_defeats(self):
        for stats, after in [({}, 0),
                             ({"battle_won": False, "dead_allies": 3, "dead_enemies": 2,
                               "ep_length": 5, "n_episodes": 1}, 1)]:
            with self.assertRaises(RuntimeError):
                episode_record("3m", "unseen", 1, 7, stats, 0, 3, 3, 10, 0, 0, 0, after)


class IntegrationTests(unittest.TestCase):
    def test_runner_adapter_uses_test_mode_and_keeps_terminal_info(self):
        env = SimpleNamespace(timeouts=0, force_restarts=0, n_agents=3, n_enemies=3)
        runner = SimpleNamespace(env=env, task="3m", episode_limit=10,
                                 test_stats={"old": 100}, test_returns=[999])

        def run(test_mode, nolog):
            self.assertTrue(test_mode and nolog)
            self.assertEqual(runner.test_stats, {})
            self.assertEqual(runner.test_returns, [])
            runner.test_stats.update({"battle_won": False, "dead_allies": 1,
                                      "dead_enemies": 2, "ep_length": 10, "n_episodes": 1})
            runner.test_returns.append(8)
            env.timeouts += 1

        runner.run = run
        result = collect_episode(runner, 3, 7, "unseen")
        self.assertTrue(result["timeout"])
        self.assertEqual(result["episode"], 3)
        self.assertEqual(result["enemies_alive"], 1)

    def test_checkpoint_selection_is_exact(self):
        with TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"
            for step in (27003, 30000):
                folder = models / str(step)
                folder.mkdir(parents=True)
                (folder / "agent.th").write_bytes(b"model")
            self.assertEqual(resolve_checkpoint(models, 0).name, "30000")
            self.assertEqual(resolve_checkpoint(models, 27003).name, "27003")
            self.assertEqual(resolve_checkpoint(models / "30000", 0).name, "30000")
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(models, 29000)
            with self.assertRaises(ValueError):
                resolve_checkpoint(models / "30000", 27003)
            self.assertEqual(checkpoint_hash(models / "30000/agent.th"), hashlib.sha256(b"model").hexdigest())

    def test_saved_config_is_preferred_and_task_mismatch_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "models/30000"
            checkpoint.mkdir(parents=True)
            config = {"task": "toy", "entity_embed_dim": 123}
            (root / "config.json").write_text(json.dumps(config))
            loaded, source = load_config(checkpoint, "toy")
            self.assertEqual(loaded, config)
            self.assertEqual(source, str(root / "config.json"))
            with self.assertRaises(ValueError):
                load_config(checkpoint, "marine-hard-expert")

    def test_interrupted_report_has_actual_count_and_na(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_reports(output, [record()], ["3m"], [], 100, Path("models/30000"), "interrupted")
            data = json.loads((output / "summary.json").read_text())
            self.assertEqual(data["status"], "interrupted")
            self.assertEqual(data["maps"][0]["episodes"], 1)
            self.assertIsNone(data["maps"][0]["win_length_mean_steps"])
            report = (output / "summary.md").read_text()
            self.assertIn("1/100", report)
            self.assertIn("N/A", report)
            self.assertNotIn("nan", report.lower())
            self.assertTrue((output / "summary.csv").is_file())


class EvaluationFlowTests(unittest.TestCase):
    def evaluate_fake_environment(self, root, fail_on_second_episode=False):
        """게임 통신만 대체해 CLI 이후의 100회 반복·저장·오류 종료 경로를 함께 확인한다."""
        root = root.resolve()
        checkpoint = root / "training/models/30000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "agent.th").write_bytes(b"fixed checkpoint")
        config = {"task": "toy", "env": "sc2", "mac": "mt_hissd_mac", "seed": 42,
                  "train_tasks": ["3m"], "test_tasks": ["3m", "4m"], "env_args": {}}
        (root / "training/config.json").write_text(json.dumps(config))
        sc2 = root / "sc2"
        sc2.mkdir()
        output = root / "results"
        runners = {}

        class FakeRunner:
            def __init__(self, task):
                self.task, self.episode_limit = task, 10
                self.test_stats, self.test_returns = {}, []
                self.env = SimpleNamespace(timeouts=0, force_restarts=0, n_agents=3, n_enemies=3)
                self.calls, self.closed = 0, False

            def setup(self, **kwargs):
                self.mac = kwargs["mac"]

            def run(self, test_mode, nolog):
                assert test_mode and nolog
                self.calls += 1
                won = self.task == "3m"
                timeout = not won and self.calls % 2 == 0
                self.env.timeouts += int(timeout)
                if fail_on_second_episode and self.calls == 2:
                    self.env.force_restarts += 1
                self.test_stats.update({"battle_won": won, "dead_allies": 1 if won else 3,
                                        "dead_enemies": 3 if won else 1, "n_episodes": 1,
                                        "ep_length": 10 if timeout else 5})
                self.test_returns.append(20 if won else 8)

            def close_env(self):
                self.closed = True

        class FakeMAC:
            def __init__(self, **kwargs):
                self.agent = SimpleNamespace(eval=lambda: None)

            def load_models(self, path):
                assert Path(path) == checkpoint

        def init_tasks(tasks, main_args, logger):
            runners.update({task: FakeRunner(task) for task in tasks})
            empty = {task: {} for task in tasks}
            return empty, runners, {task: SimpleNamespace(scheme={}) for task in tasks}, empty, empty, empty

        def module(name, **attributes):
            value = ModuleType(name)
            value.__dict__.update(attributes)
            return value

        modules = {
            "torch": module("torch", __version__="test", version=SimpleNamespace(cuda=None),
                            cuda=SimpleNamespace(is_available=lambda: False),
                            manual_seed=lambda seed: None, no_grad=nullcontext),
            "numpy": module("numpy", random=SimpleNamespace(seed=lambda seed: None)),
            "mto": module("mto", init_tasks=init_tasks),
            "controllers": module("controllers", __path__=[]),
            "controllers.multi_task": module("controllers.multi_task", REGISTRY={"mt_hissd_mac": FakeMAC}),
            "utils": module("utils", __path__=[]),
            "utils.logging": module("utils.logging", Logger=lambda logger: logger),
        }
        args = SimpleNamespace(checkpoint_path=checkpoint.parent, load_step=30000, task_config="toy",
                               training_config=None, maps=None, seed=7, sc2_path=sc2, output_dir=output,
                               device="cpu", episodes=100, progress_every=10)
        with patch.dict("sys.modules", modules), patch.dict("os.environ"), redirect_stderr(io.StringIO()):
            result = run(args)
        self.assertTrue(all(r.closed for r in runners.values()))
        self.assertEqual((checkpoint / "agent.th").read_bytes(), b"fixed checkpoint")
        return result, output

    def test_all_100_episodes_are_saved_and_summarized(self):
        with TemporaryDirectory() as temporary:
            code, output = self.evaluate_fake_environment(Path(temporary))
            self.assertEqual(code, 0, (output / "run.log").read_text())
            with (output / "episodes.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 200)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["training_seed"], 42)
            self.assertEqual(metadata["map_seeds"], {"3m": 7, "4m": 8})
            effective_config = json.loads((output / "effective_config.json").read_text())
            self.assertEqual(effective_config["seed"], 7)
            self.assertEqual(effective_config["load_step"], 30000)
            self.assertFalse(effective_config["save_model"])
            summaries = json.loads((output / "summary.json").read_text())["maps"]
            self.assertTrue(all(row["complete"] for row in summaries))
            self.assertEqual(summaries[0]["win_rate"], 1)
            self.assertEqual(summaries[1]["timeout_rate"], 0.5)

    def test_crash_preserves_completed_rows_and_returns_failure(self):
        with TemporaryDirectory() as temporary:
            code, output = self.evaluate_fake_environment(Path(temporary), fail_on_second_episode=True)
            self.assertEqual(code, 1)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_episodes"], 1)
            self.assertIn("강제 재시작", metadata["error"])
            with (output / "episodes.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)


if __name__ == "__main__":
    unittest.main()
