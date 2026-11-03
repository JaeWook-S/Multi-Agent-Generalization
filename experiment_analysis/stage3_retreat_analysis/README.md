# Stage 3: Retreat Analysis

## Training data

Run from the repository root in the Ubuntu `hissd` environment:

```bash
python -m experiment_analysis.stage3_retreat_analysis.analyze_retreats dataset \
  --training-config "/path/to/training/run/config.json" \
  --dataset-root dataset \
  --maps 2s3z 2s4z 3s5z
```

Use the training run's `config.json`, next to `models/`. Add `--limit-episodes 2` for a short check. NumPy and h5py are required; SC2, CUDA and model loading are not required.

- Reconstructs `OfflineBuffer` source selection, numeric `part_N.h5` order and per-source tail limits. Records selected H5 rows and file SHA-256 hashes.
- An empty `offline_data_name` selects the **currently** last folder alphabetically. To specify the original training source, add `--source-dir "3s5z/expert=/path/to/original/qmix_folder"`. Check the training log's `Load dataset from ...` line. Historical file identity cannot be established from config alone.
- Random training subsets cannot be recovered from the seed alone. `--audit-all` audits all source episodes and labels that different scope explicitly.
- H5 outcomes are inferred only from the final living units. If both sides survive, outcome is `unknown`; timeout is not guessed.
- Supports the Stalker–Zealot H5 schema with `obs_all_health=True` and global state. Unsupported or inconsistent layouts stop with an error.

## Existing policy logs

Works on Mac or Ubuntu with NumPy. No new rollout:

```bash
python -m experiment_analysis.stage3_retreat_analysis.analyze_retreats policy \
  --run-dir \
  experiment_analysis/stage2_behavior_diagnostics/results/3s5z/30000_20261101-192931-001034 \
  experiment_analysis/stage2_behavior_diagnostics/results/4s5z/30000_20261031-215640-258935
```

All completed episodes and living agent-steps are examined, including losses, timeouts, positive cooldown, healthy units, and steps without an available attack command. Stalker and Zealot remain separate. Source model hashes, map seeds and raw-file hashes are retained.

## Definitions

| Item | Default / meaning |
| --- | --- |
| Coordinates | Sight range = 1; x/y conversion verified against actual local obs for every episode |
| Risk context | Damage observed in the previous 3 decision steps **or** a living enemy within distance 0.75 |
| Low health | HP ratio ≤ 0.30; a label, not an eligibility filter |
| Separation candidate | Move command, displacement ≥ 0.01, and both self-caused and observed separation increase > 0.01 |
| Reference enemies | Same living enemies at both ends of the transition; nearby set, or nearest enemy if no enemy is nearby |
| Self-caused separation | Distance change with enemy positions held at their pre-action values |
| Combat phase | `contested`: ≥ 2 living enemies; `cleanup`: ≤ 1. This is a count, not a danger estimate. |
| Follow-up | 3, 5, 10 decision steps; observed length, censoring, HP/shield loss, shield gain, team return, deaths |

Thresholds are research heuristics, not weapon ranges or proof of danger. Change them with `--near-distance`, `--low-health`, `--recent-window`, `--min-move`, `--min-separation`, and `--horizons`. Results are descriptive associations, not the causal effect or intent of retreat. Shield gain is not a healing action; supported maps have no healer.

The last `filled` state is retained as the next-state snapshot; its unexecuted action is discarded. A dead enemy changing the nearest target cannot create a separation candidate. An episode ending before a follow-up horizon does not count as survival through that horizon. Enemy cooldown is not used. Global state and cooldown remain analysis-only inputs.

## Output

Default: `experiment_analysis/stage3_retreat_analysis/results/<mode>_<timestamp>/`. Set `--output-dir` to a **new** directory.

| File | Contents |
| --- | --- |
| `summary.md` | Stalker risk-context overview, all cooldowns, outcomes and combat phases separated |
| `context_summary.csv` | Both unit types; episode-weighted and pooled ratios with explicit denominators |
| `episodes.csv` | Outcome, source row, episode length and counts |
| `risk_contexts.csv.gz` | All risk-context actions, conditions and follow-up |
| `movement_steps.csv.gz` | All explicit move commands, including unclear/no displacement, target-death flags and follow-up |
| `separation_segments.csv` | Consecutive candidate steps grouped by unit and combat phase; follow-up anchored at segment start |
| `episode_context_counts.csv.gz` | Per-episode counts underlying the summary |
| `manifest.json` | Parameters, status, provenance, coordinate checks and data-selection scope |

Detailed CSVs are compressed and can be read directly with `pandas.read_csv()`. HP/shield losses are separately summed positive ratio decreases, not raw damage or net change. Follow-up includes subsequent policy actions. Segments are not independently validated strategies or successful retreats.

## Tests

```bash
python -m unittest experiment_analysis.stage3_retreat_analysis.test_retreat_analysis -v
```

H5 integration tests require h5py. Tests cover padding, source tail selection, H5/NPZ equivalence, cooldown/outcome coverage, enemy death and motion, no displacement, follow-up censoring, episode weighting and mixed-model rejection.
