"""정답 유출·시간 정렬·행동 마스크와 실제 controller의 참조 배치 동등성을 검증한다."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment_analysis.stage1_map_selection.evaluate_performance import load_config
from experiment_analysis.stage3_retreat_analysis.test_retreat_analysis import encoded_episode, as_h5_row, CONFIG
from experiment_analysis.stage4_policy_scores.episode_prefix import (
    load_bundle, read_h5_prefix, save_bundle, validate_prefix,
)
from experiment_analysis.stage4_policy_scores.probe_action_scores import (
    CausalBatch, create_mac, execute, parse_args, replay_recorded_history, summarize_scores,
)


HAS_H5 = importlib.util.find_spec("h5py") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_RUNTIME = HAS_TORCH and HAS_H5 and importlib.util.find_spec("smac") is not None and importlib.util.find_spec("pysc2") is not None


def fixture(steps=5):
    """이전 행동 state와 one-hot을 포함한 학습 H5 규약을 만든다."""
    data, _ = encoded_episode(steps)
    n, a = data["actions"].shape[1], data["avail_actions"].shape[-1]
    onehot = np.eye(a, dtype=np.float32)[data["actions"]]
    data["state"][1:, n * 7 + n * 6:] = onehot.reshape(steps, -1)
    row = as_h5_row(data)
    row["actions_onehot"] = np.zeros((len(row["filled"]), n, a), dtype=np.float32)
    row["actions_onehot"][:steps] = onehot
    return row


def write_h5(path, row):
    import h5py
    with h5py.File(path, "w") as stream:
        for key, value in row.items():
            stream.create_dataset(key, data=value[None])


@unittest.skipUnless(HAS_H5, "H5 경계 검증에는 h5py가 필요합니다.")
class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "part_0.h5"
        self.row = fixture()
        write_h5(self.path, self.row)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefix_retains_history_and_discards_padding_and_unexecuted_action(self):
        arrays, meta = read_h5_prefix(self.path, CONFIG, "2s3z", 1, 3)
        self.assertEqual(arrays["actions"].shape, (4, 5))
        self.assertEqual(arrays["obs"].shape[0], 5)
        self.assertEqual(meta["h5_row_index"], 0)
        self.assertEqual(meta["original_transitions"], 5)
        self.assertEqual(meta["prefix_transitions"], 4)
        self.assertTrue(meta["state_last_action_alignment_checked"])
        self.assertFalse(meta["global_state_used_by_actor"])
        self.assertTrue((arrays["actions"] < 11).all())

    def test_wrong_row_or_final_unexecuted_step_is_rejected(self):
        for ep, step in [(0, 1), (2, 1), (1, 5)]:
            with self.assertRaises(ValueError):
                read_h5_prefix(self.path, CONFIG, "2s3z", ep, step)

    def test_model_independent_hash_check_and_time_alignment(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_h5_prefix(self.path, CONFIG, "2s3z", 1, 3, "0" * 64)
        self.row["state"][2, 65:] = 0
        write_h5(self.path, self.row)
        with self.assertRaisesRegex(ValueError, "시간 정렬"):
            read_h5_prefix(self.path, CONFIG, "2s3z", 1, 3)

    def test_stored_onehot_must_agree_with_integer_actions(self):
        self.row["actions_onehot"][0] = 0
        write_h5(self.path, self.row)
        with self.assertRaisesRegex(ValueError, "actions_onehot"):
            read_h5_prefix(self.path, CONFIG, "2s3z", 1, 3)

    def test_bundle_roundtrip_and_corruption_detection(self):
        arrays, meta = read_h5_prefix(self.path, CONFIG, "2s3z", 1, 3)
        root = save_bundle(self.root / "bundle", arrays, meta, CONFIG)
        loaded, metadata, config = load_bundle(root, meta["source_h5_sha256"])
        self.assertEqual(config, CONFIG)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], loaded[key])
        self.assertEqual(metadata["through_step"], 3)
        with self.assertRaises(FileExistsError):
            save_bundle(root, arrays, meta, CONFIG)
        (root / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_bundle(root)


class ScoreInterpretationTests(unittest.TestCase):
    def test_mask_stop_and_ties_are_distinguished(self):
        scores = np.array([[[20., 8., 7., 99., 2., 3., 6., 5.]]])
        arrays = {"avail_actions": np.array([[[0, 1, 1, 0, 0, 0, 1, 1]]]),
                  "actions": np.array([[2]]), "health_ratio": np.ones((1, 1)),
                  "shield_ratio": np.zeros((1, 1)), "cooldown_normalized": np.ones((1, 1)),
                  "unit_type": np.array(["Stalker"])}
        decisions, rows = summarize_scores(scores, arrays, [0], [0])
        row = decisions[0]
        self.assertEqual(row["greedy_action"], 1)  # 정지가 최선이다.
        self.assertEqual(row["best_move_action"], 2)
        self.assertEqual(row["best_attack_action"], 6)
        self.assertEqual(row["move_minus_attack"], 1.)
        self.assertEqual(row["recorded_valid_rank"], 2)
        self.assertIsNone(rows[3]["valid_rank"])  # 99점이어도 금지 행동은 순위에서 뺀다.
        scores[0, 0, 1] = 7
        row = summarize_scores(scores, arrays, [0], [0])[0][0]
        self.assertTrue(row["recorded_tied_for_best"])
        self.assertFalse(row["greedy_matches_recorded"])

    def test_missing_attack_is_not_a_zero_score(self):
        arrays = {"avail_actions": np.array([[[0, 1, 1, 0, 0, 0, 0]]]),
                  "actions": np.array([[2]]), "health_ratio": np.ones((1, 1)),
                  "shield_ratio": np.zeros((1, 1)), "cooldown_normalized": np.zeros((1, 1)),
                  "unit_type": np.array(["Stalker"])}
        row = summarize_scores(np.ones((1, 1, 7)), arrays, [0], [0])[0][0]
        self.assertIsNone(row["best_attack_score"])
        self.assertIsNone(row["move_minus_attack"])


class HistoryMAC:
    """현재 정답과 미래 관측이 숨겨졌는지 확인하는 시간 의존 대역 controller다."""
    def __init__(self, arrays, torch):
        self.arrays, self.torch = arrays, torch
        self.agent = SimpleNamespace(last_out_h=None, last_h_plan=None)
        self.visited = []

    def init_params(self):
        self.visited = []

    def init_hidden(self, batch_size, task):
        self.hidden = self.torch.zeros(self.arrays["actions"].shape[1])

    def forward(self, batch, t, task, test_mode):
        torch = self.torch
        assert test_mode and not torch.is_grad_enabled()
        assert not batch["actions"][:, t].any()
        assert not batch["actions_onehot"][:, t].any()
        assert not batch["obs"][:, t + 1:].any()
        assert "state" not in batch.data
        previous = batch["actions_onehot"][:, t - 1] if t else torch.zeros_like(batch["actions_onehot"][:, 0])
        factors = torch.arange(batch.n_actions).to(torch.float32)
        self.hidden = .5 * self.hidden + batch["obs"][0, t, :, 0] + (previous[0] * factors).sum(-1)
        self.visited.append(t)
        return self.hidden[None, :, None].repeat(1, 1, batch.n_actions)


@unittest.skipUnless(HAS_TORCH and HAS_H5, "순차 추론 검증에는 torch와 h5py가 필요합니다.")
class HistoryTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path = self.root / "part_0.h5"
        write_h5(path, fixture())
        self.arrays, self.meta = read_h5_prefix(path, CONFIG, "2s3z", 1, 4)

    def tearDown(self):
        self.tmp.cleanup()

    def replay(self, arrays, last=3):
        mac = HistoryMAC(arrays, self.torch)
        scores = replay_recorded_history(mac, arrays, self.meta, last, self.torch.device("cpu"), self.torch)
        return scores, mac

    def test_history_begins_at_zero_and_current_label_cannot_leak(self):
        original, mac = self.replay(self.arrays)
        self.assertEqual(mac.visited, [0, 1, 2, 3])
        changed = copy.deepcopy(self.arrays)
        changed["actions"][2, 0] = 4
        after, _ = self.replay(changed)
        np.testing.assert_array_equal(original[:3], after[:3])
        self.assertNotEqual(original[3, 0, 0], after[3, 0, 0])

    def test_future_observation_cannot_change_current_scores(self):
        first, _ = self.replay(self.arrays, 2)
        changed = copy.deepcopy(self.arrays)
        changed["obs"][3:] += 100
        second, _ = self.replay(changed, 2)
        np.testing.assert_array_equal(first, second)

    def test_reveal_cannot_start_at_focus_step(self):
        batch = CausalBatch(self.arrays, self.torch.device("cpu"), self.torch)
        with self.assertRaisesRegex(ValueError, "0스텝"):
            batch.reveal(2, self.arrays)


@unittest.skipUnless(HAS_RUNTIME, "실제 신경망 검증에는 학습 환경 의존성이 필요합니다.")
class RealControllerTests(unittest.TestCase):
    def test_original_episode_batch_and_new_probe_are_equivalent_without_sc2(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 테스트용 무작위 모델의 구조만 YAML로 만든다. 개인 결과 폴더에 의존하지 않는다.
            # 실제 probe는 이 fallback을 쓰지 않고 학습 당시 config.json을 반드시 요구한다.
            config, _ = load_config(root / "missing/models/30000", "stalker-zealot-expert")
            config.update(train_tasks=["2s3z"], test_tasks=["2s3z"], pretrain_tasks=[], pretrain=False)
            h5 = root / "part_0.h5"
            write_h5(h5, fixture())
            arrays, meta = read_h5_prefix(h5, config, "2s3z", 1, 3)
            checkpoint = root / "run/models/30000"
            checkpoint.mkdir(parents=True)
            (root / "run/config.json").write_text(json.dumps(config))
            # 설치된 환경의 생성자가 호출되면 즉시 실패한다. 신경망만 동작해야 한다.
            import sys
            repo = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(repo / "src"))
            from controllers.multi_task.mt_hissd_controller import HISSDSMAC
            from components.episode_buffer import EpisodeBatch
            from components.transforms import OneHot
            with patch("smac.env.starcraft2.starcraft2.StarCraft2Env.__init__", side_effect=AssertionError("SC2 env created")):
                with patch.object(HISSDSMAC, "load_models"):
                    original, _ = create_mac(config, arrays, meta, checkpoint, torch.device("cpu"))
                torch.save(original.agent.state_dict(), checkpoint / "agent.th")
                mac, _ = create_mac(config, arrays, meta, checkpoint, torch.device("cpu"))
                before = {k: v.clone() for k, v in mac.agent.state_dict().items()}
                actual = replay_recorded_history(mac, arrays, meta, 3, torch.device("cpu"), torch)
                n, a = arrays["avail_actions"].shape[1:]
                scheme = {"obs": {"vshape": arrays["obs"].shape[-1], "group": "agents"},
                          "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
                          "avail_actions": {"vshape": (a,), "group": "agents", "dtype": torch.int}}
                batch = EpisodeBatch(scheme, {"agents": n}, 1, 5,
                                     preprocess={"actions": ("actions_onehot", [OneHot(a)])}, device="cpu")
                mac.init_params()
                mac.init_hidden(1, "2s3z")
                reference = []
                with torch.no_grad():
                    for t in range(4):
                        batch.update({"obs": arrays["obs"][None, t],
                                      "avail_actions": arrays["avail_actions"][None, t]}, ts=t)
                        reference.append(mac.forward(batch, t, "2s3z", test_mode=True)[0].cpu().numpy().copy())
                        batch.update({"actions": arrays["actions"][None, t, :, None]}, ts=t)
                np.testing.assert_allclose(actual, np.stack(reference), rtol=1e-6, atol=1e-6)
                for key, value in mac.agent.state_dict().items():
                    torch.testing.assert_close(before[key], value, rtol=0, atol=0)
                args = parse_args(["probe", "--checkpoint-path", str(checkpoint), "--dataset-file", str(h5),
                                   "--map", "2s3z", "--episode", "1", "--agents", "0", "--steps", "1", "2", "3",
                                   "--output-dir", str(root / "scores")])
                with contextlib.redirect_stdout(io.StringIO()):
                    output = execute(args)
                self.assertEqual(json.loads((output / "metadata.json").read_text())["status"], "complete")
                self.assertTrue((output / "decisions.csv").is_file())
                self.assertTrue((output / "raw_scores.npz").is_file())
                # Ubuntu H5 직접 입력과 Mac용 prefix 입력이 같은 점수를 내는지도 검사한다.
                with contextlib.redirect_stdout(io.StringIO()):
                    bundle = execute(parse_args(["export", "--training-config", str(root / "run/config.json"),
                                                "--dataset-file", str(h5), "--map", "2s3z", "--episode", "1",
                                                "--through-step", "3", "--output-dir", str(root / "bundle")]))
                    bundled = execute(parse_args(["probe", "--checkpoint-path", str(checkpoint),
                                                  "--bundle-dir", str(bundle), "--map", "2s3z", "--episode", "1",
                                                  "--steps", "1", "2", "3", "--agents", "0",
                                                  "--output-dir", str(root / "bundle_scores")]))
                with np.load(output / "raw_scores.npz") as direct, np.load(bundled / "raw_scores.npz") as loaded:
                    np.testing.assert_array_equal(direct["scores"], loaded["scores"])


if __name__ == "__main__":
    unittest.main()
