"""원본 상태/행동 정렬과 HTML 뷰어의 안전한 데이터 전달을 검증한다."""

import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiment_analysis.stage2_behavior_diagnostics.view_npz_cases import build_viewer, load_episode


class ViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.path = self.run / "step_logs/3s5z/episode_000001.npz"
        self.path.parent.mkdir(parents=True)
        self.meta = {"status": "valid", "map": "3s5z", "episode": 1,
                     "transition_count": 3, "n_agents": 2, "n_enemies": 2,
                     "n_actions_no_attack": 6, "env_seed": 9, "step_mul": 8,
                     "agent_sha256": "model", "outcome": "win"}
        self.run_meta = {"analysis_stage": "stage2_behavior_diagnostics", "maps": ["3s5z"],
                         "agent_sha256": "model", "map_seeds": {"3s5z": 9}}
        self.arrays = {"actions": np.asarray([[6, 6], [0, 5], [0, 1]]),
                       "avail_actions": np.ones((4, 2, 8), dtype=int)}
        for side in ("ally", "enemy"):
            for name, value in (("x", 10), ("y", 12), ("health", 100), ("health_max", 100),
                                ("shield", 0), ("shield_max", 80), ("cooldown", 0)):
                self.arrays[f"{side}_{name}"] = np.full((4, 2), value, dtype=float)
            self.arrays[f"{side}_type"] = np.asarray(["Stalker", "Stalker"])
            self.arrays[f"{side}_tag"] = np.asarray([2**63 + 1, 2**63 + 3], dtype=np.uint64)
        self.arrays["ally_health"][1:, 0] = 0
        self.arrays["ally_health"][:, 1] = [100, 20, 15, 10]
        self.arrays["ally_x"][:, 1] = [10, 10, 8, 8]
        self.write()

    def write(self):
        np.savez_compressed(self.path, **self.arrays)
        self.path.with_suffix(".json").write_text(json.dumps(self.meta))
        (self.run / "run_metadata.json").write_text(json.dumps(self.run_meta))

    def test_actions_use_pre_action_snapshot_and_exclude_terminal_and_dead_units(self):
        case = load_episode(self.run, "3s5z", 1, self.run_meta)
        self.assertEqual(case["firstDeath"], 1)
        self.assertEqual(case["events"], [{"step": 1, "unit": 1, "action": 5},
                                         {"step": 2, "unit": 1, "action": 1}])
        self.assertEqual(case["allies"]["x"][1][1], 10)
        self.assertEqual(case["allies"]["x"][2][1], 8)
        self.assertEqual(len(case["actions"]), 3)
        self.assertEqual(len(case["allies"]["health"]), 4)
        self.assertEqual(case["allies"]["tags"][0], str(2**63 + 1))

    def test_model_and_seed_mismatch_are_rejected(self):
        for key, value in (("agent_sha256", "other-model"), ("env_seed", 10)):
            with self.subTest(key=key):
                original = self.meta[key]
                self.meta[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    load_episode(self.run, "3s5z", 1, self.run_meta)
                self.meta[key] = original

    def test_shifted_or_nonfinite_snapshots_are_rejected(self):
        for value in (np.zeros((3, 2)), np.full((4, 2), np.nan)):
            self.arrays["ally_x"] = value
            self.write()
            with self.assertRaises(ValueError):
                load_episode(self.run, "3s5z", 1, self.run_meta)

    def test_html_preserves_source_and_escapes_embedded_strings(self):
        self.arrays["enemy_type"] = np.asarray(["</script><b>test", "Stalker"])
        self.write()
        before = {p: p.read_bytes() for p in self.path.parent.iterdir()}
        output = build_viewer(self.run, episodes=[1])
        standalone = output.read_text()
        fragment = output.with_name(output.stem + "-inline.html").read_text()
        payload = re.search(r'<script type="application/json" data-role="data">(.*?)</script>', fragment, re.S)[1]
        data = json.loads(payload)
        self.assertNotIn("</script>", payload)
        self.assertEqual(data["episodes"][0]["enemies"]["types"][0], "</script><b>test")
        self.assertEqual(data["episodes"][0]["sourceSha256"], hashlib.sha256(before[self.path]).hexdigest())
        self.assertTrue(standalone.startswith("<!doctype html>"))
        self.assertNotIn("__CASE_DATA__", standalone)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaises(ValueError):
            build_viewer(self.run, [1], output=self.path.with_suffix(".html"))


if __name__ == "__main__":
    unittest.main()
