# Stage 4: Policy Action Scores

Inspect the trained HiSSD policy on **expert H5 episode 809, `3s5z`, A2, steps 12–15**. No training, SC2 process, replay player, or GPU is required. Use the existing HiSSD Python environment; the original controller still imports PyTorch, h5py, SMAC/PySC2 and their dependencies.

Episode numbers start at **1 within the specified H5** (`809` means row `808`). Agent IDs and decision steps start at **0**. This is an expert-data episode, separate from the 100 policy episodes collected in Stage 2.

## Run on Ubuntu

From the repository root:

```bash
python -m experiment_analysis.stage4_policy_scores.probe_action_scores probe \
  --checkpoint-path '/home/user/JY/HiSSD/results/mto/sc2/stalker-zealot-expert/2s3z-expert+2s4z-expert+3s5z-expert/mto/mto_2026-10-31_13-19-37/models' \
  --load-step 30000 \
  --dataset-file dataset/3s5z/expert/qmix_2022-04-02_15-07-12/part_0.h5 \
  --map 3s5z --episode 809 --agents 2 --steps 12 13 14 15 \
  --expected-model-sha256 aff8288ac3f1d256b48430f408b01aed858ab06d7890b939ea2f5b9608354774 \
  --expected-dataset-sha256 f0b3f73a2e85fb14d6e59ccb026ecde097a7470cce0adb83840fc57bebcfc086 \
  --device cpu
```

The training run's `config.json` is required and found beside `models/`. Use `--training-config /path/to/config.json` if it was moved. A hash mismatch stops execution; it must be resolved before treating the output as a test of the same checkpoint/case. For another model or H5, supply its own verified hashes. `--load-step` requires the exact checkpoint.

Output: `experiment_analysis/stage4_policy_scores/results/scores_<timestamp>/`. Use `--output-dir /path/to/new_folder` to choose a new destination. Existing directories are never overwritten. Generated results are ignored by Git.

## What the code does

1. Verify the checkpoint and H5 hashes, actual episode length, action masks, stored one-hot actions, and `state[t+1]` previous-action alignment.
2. Load the **saved training configuration** and original HiSSD controller/agent. Create static map decomposers without constructing a game environment.
3. Initialize hidden states once at step 0. Process every recorded observation from step 0 through step 15, for every agent. At each step, only the current observation and previously recorded actions are available; future observations and the current action label are hidden.
4. Copy the model scores, then add the expert's recorded action to the history for the next step. **Predicted actions never replace the expert history.**
5. Compare available actions for A2 at steps 12–15. Global state is used for validation and HP/shield/cooldown annotations; it is absent from the batch passed to the actor. Weights remain fixed under `eval()` and `no_grad()`.

## Read the output

Start with `summary.md`, then `decisions.csv`. In this source case the recorded actions are **west, west, north, attack** at steps 12, 13, 14, 15 respectively. A different sequence means the source case must be checked.

| Field / file | Meaning |
| --- | --- |
| `recorded_label` | Expert action in the H5 |
| `greedy_label` | Highest-scoring available action, matching the original greedy selector's tie rule |
| `recorded_valid_rank` | Expert action's rank among available actions; ties share a rank |
| `move_minus_attack` | Best available move score minus best available attack score |
| `top1_margin` | Highest minus second-highest available score; zero for a top tie |
| `cooldown_normalized` | Global-state cooldown divided by SMAC's unit-specific maximum; zero means ready |
| `action_scores.csv` | All scores and masks for the requested agents/steps, including unavailable actions |
| `raw_scores.npz` | Scores for every agent at steps 0–15, plus masks and recorded actions |
| `metadata.json` | Source episode, source/model hashes, configuration hash, device, inference seed and completion status |
| `training_config.json`, `effective_config.json` | Original training settings and inference settings |

A positive `move_minus_attack` means a move outranks all available attacks, **but stop may still rank first**. Check `greedy_label`. Missing move/attack groups produce `N/A` (empty CSV cells), not a zero score. An action mask permits a command; it does not guarantee an immediate hit.

These are **Decoder action-classification scores**, not Bellman estimates of future return. Compare ranks and within-decision score gaps; do not read the gap as extra reward or compare its scale as calibrated confidence across checkpoints.

| Observation | Supported interpretation | Still untested |
| --- | --- | --- |
| The model chooses the recorded moves | It can select these moves given this expert history | Whether its own rollout reaches this history; whether the moves cause a win |
| Moves rank just below attack | The greedy choice differs by the recorded score gap | Whether changing the choice helps |
| Moves rank far below attack | This checkpoint does not reproduce these moves under this history | Whether the reason is the representation, loss, training coverage, or history mismatch |

The input is a **recorded-history probe**, not a new rollout. Even if the model preferred a different action at step 5, step 6 still uses the expert's actual action and observation. A disagreement does not identify which internal skill module caused it.

## Optional: inspect on Mac

The Mac needs the same `agent.th`, training configuration, and either the H5 or a small exported prefix. It does not need to run the SC2 4.10 engine. The Python environment must include compatible HiSSD dependencies and the patched SMAC map registry.

On Ubuntu, export only this episode's prefix:

```bash
python -m experiment_analysis.stage4_policy_scores.probe_action_scores export \
  --training-config '/home/user/JY/HiSSD/results/mto/sc2/stalker-zealot-expert/2s3z-expert+2s4z-expert+3s5z-expert/mto/mto_2026-10-31_13-19-37/config.json' \
  --dataset-file dataset/3s5z/expert/qmix_2022-04-02_15-07-12/part_0.h5 \
  --map 3s5z --episode 809 --through-step 15 \
  --expected-dataset-sha256 f0b3f73a2e85fb14d6e59ccb026ecde097a7470cce0adb83840fc57bebcfc086 \
  --output-dir experiment_analysis/stage4_policy_scores/results/expert_809_prefix
```

Transfer that folder and the checkpoint to Mac. Keep the checkpoint layout as `run/config.json` and `run/models/30000/agent.th`; optimizer and mixer files are unnecessary. Then run from the repository root:

```bash
python -m experiment_analysis.stage4_policy_scores.probe_action_scores probe \
  --checkpoint-path /path/to/run/models --load-step 30000 \
  --bundle-dir /path/to/expert_809_prefix \
  --map 3s5z --episode 809 --agents 2 --steps 12 13 14 15 \
  --expected-model-sha256 aff8288ac3f1d256b48430f408b01aed858ab06d7890b939ea2f5b9608354774 \
  --expected-dataset-sha256 f0b3f73a2e85fb14d6e59ccb026ecde097a7470cce0adb83840fc57bebcfc086 \
  --device cpu
```

## Next: test an action intervention in SC2

The score probe above is implemented. The following is the next experiment, **after** inspecting its output:

1. **Establish a reproducible encounter.** Expert H5 809 does not supply enough simulator state or original seed to guarantee exact reconstruction. The Stage 2 map seed is not that H5 episode's seed. If reconstruction fails, use new, reproducible policy encounters and label them as new cases.
2. Run a baseline and intervention from an identical pre-intervention history. Verify positions, HP, shields, cooldowns, alive units, observations, action masks and controller history before the branch. Reject pairs whose prefixes diverge.
3. At a predefined risk event, override only the chosen agent's legal action. Define retreat direction, duration and return-to-attack rule before looking at the result. Resume the policy using the **actually executed** actions as previous-action inputs. Do not reuse recorded next observations/rewards after changing an action.
4. Record team win/defeat/timeout, return, enemy/ally losses, focal-unit survival and re-engagement. Include an appropriate legal-movement control if claiming that a specific retreat strategy helps. Surviving longer alone is insufficient.
5. Repeat across encounters/seeds and report paired effects. One successful branch is a case study, not evidence that the latent space lacks strategy capacity or that retreat generally improves performance.

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 python -W ignore::DeprecationWarning -W ignore::FutureWarning \
  -m unittest experiment_analysis.stage4_policy_scores.test_policy_scores -v
```

Tests cover history reconstruction, current-label/future-observation isolation, masks/ties, H5 boundaries and checksums. With HiSSD dependencies installed, an original-controller integration test verifies equality with `EpisodeBatch`, unchanged weights, H5/bundle equivalence, and no SC2 environment construction. A randomly initialized test model is used; passing tests does not supply the trained checkpoint's actual scores.
