"""SMAC 에피소드의 성과 집계. SC2/PyTorch 없이 계산을 검증할 수 있다."""

import math
from statistics import NormalDist, fmean, median, pstdev


def wilson_interval(wins, episodes):
    """승률의 95% Wilson 신뢰구간. 0승·전승에서도 구간이 0으로 붕괴하지 않는다."""
    if episodes <= 0 or not 0 <= wins <= episodes:
        raise ValueError("0 <= wins <= episodes이고 episodes > 0이어야 합니다.")
    z = NormalDist().inv_cdf(0.975)
    p = wins / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def episode_record(map_name, split, episode, env_seed, stats, episode_return,
                   n_agents, n_enemies, episode_limit, timeouts_before,
                   timeouts_after, restarts_before, restarts_after):
    """한 판의 종료 정보로 원본 행을 만든다. 환경 오류를 패배로 섞지 않는다."""
    required = {"battle_won", "dead_allies", "dead_enemies", "ep_length", "n_episodes"}
    if not required.issubset(stats) or stats["n_episodes"] != 1:
        raise RuntimeError("정상적인 한 에피소드의 종료 정보가 없습니다. SC2 연결 로그를 확인하세요.")
    if restarts_after != restarts_before:
        raise RuntimeError("에피소드 도중 SC2가 강제 재시작됐습니다. 이 판은 성과에 포함하지 않습니다.")
    timeout_delta = timeouts_after - timeouts_before
    if timeout_delta not in (0, 1):
        raise RuntimeError("SMAC 시간 초과 카운터가 예상과 다르게 변경됐습니다.")
    won = bool(stats["battle_won"])
    # continuing_episode=False에서는 info['episode_limit']이 생략될 수 있다.
    # 길이만 비교하면 마지막 스텝의 정상 승패를 오분류하므로 SMAC 카운터를 사용한다.
    timeout = timeout_delta == 1
    if won and timeout:
        raise RuntimeError("승리와 시간 초과가 동시에 기록됐습니다.")
    dead_allies, dead_enemies = int(stats["dead_allies"]), int(stats["dead_enemies"])
    length = int(stats["ep_length"])
    if (n_agents <= 0 or n_enemies <= 0 or not 0 <= dead_allies <= n_agents
            or not 0 <= dead_enemies <= n_enemies or not 0 < length <= episode_limit):
        raise RuntimeError("유닛 수 또는 에피소드 길이가 환경 규격과 맞지 않습니다.")
    if not math.isfinite(float(episode_return)):
        raise RuntimeError("에피소드 보상이 유한한 수가 아닙니다.")
    enemies_alive = n_enemies - dead_enemies
    outcome = "win" if won else "timeout" if timeout else "draw" if (
        dead_allies == n_agents and enemies_alive == 0) else "defeat"
    return {
        "map": map_name, "split": split, "episode": episode, "env_seed": env_seed,
        "battle_won": won, "timeout": timeout, "outcome": outcome,
        "episode_length_steps": length, "episode_limit": episode_limit,
        "n_agents": n_agents, "initial_enemies": n_enemies,
        "dead_allies": dead_allies, "dead_enemies": dead_enemies,
        "enemies_alive": enemies_alive, "enemy_remaining_ratio": enemies_alive / n_enemies,
        "episode_return": float(episode_return),
    }


def summarize(records, map_name, split, expected_episodes):
    """조건부 평균의 분모를 분리한다. 승리/패배가 없으면 해당 값은 None이다."""
    if not records:
        raise ValueError("집계할 에피소드가 없습니다.")
    wins = [r for r in records if r["battle_won"]]
    nonwins = [r for r in records if not r["battle_won"]]
    lengths = [r["episode_length_steps"] for r in wins]
    ci_low, ci_high = wilson_interval(len(wins), len(records))
    return {
        "map": map_name, "split": split, "episodes": len(records),
        "expected_episodes": expected_episodes, "complete": len(records) == expected_episodes,
        "wins": len(wins), "non_wins": len(nonwins),
        "timeouts": sum(r["timeout"] for r in records),
        "win_rate": len(wins) / len(records), "win_ci95_low": ci_low,
        "win_ci95_high": ci_high,
        "timeout_rate": sum(r["timeout"] for r in records) / len(records),
        "nonwin_enemy_remaining_ratio_mean": (
            fmean(r["enemy_remaining_ratio"] for r in nonwins) if nonwins else None),
        "win_length_mean_steps": fmean(lengths) if lengths else None,
        "win_length_median_steps": median(lengths) if lengths else None,
        "return_mean": fmean(r["episode_return"] for r in records),
        "return_std": pstdev(r["episode_return"] for r in records),
        "dead_allies_mean": fmean(r["dead_allies"] for r in records),
        "dead_enemies_mean": fmean(r["dead_enemies"] for r in records),
    }


def markdown_report(summaries, checkpoint, status):
    """원본 CSV와 같은 집계 값으로 사람이 읽을 표를 만든다."""
    def percent(value):
        return "N/A" if value is None else f"{value * 100:.1f}%"

    def number(value):
        return "N/A" if value is None else f"{value:.1f}"

    lines = ["# Checkpoint Performance", "", f"Checkpoint: `{checkpoint}`", "",
             f"Status: `{status}`", "",
             "| 맵 | 학습 포함 | 평가 횟수 | 승률 (95% CI) | 시간 초과율 | 패배 시 적 잔존 비율 | 승리 길이 평균 / 중앙값 (step) |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in summaries:
        ci = f"{percent(row['win_ci95_low'])}–{percent(row['win_ci95_high'])}"
        count = f"{row['episodes']}/{row['expected_episodes']}"
        lines.append(
            f"| `{row['map']}` | {'포함' if row['split'] == 'seen' else '미포함'} | {count} | "
            f"{percent(row['win_rate'])} ({ci}) | {percent(row['timeout_rate'])} | "
            f"{percent(row['nonwin_enemy_remaining_ratio_mean'])} | "
            f"{number(row['win_length_mean_steps'])} / {number(row['win_length_median_steps'])} |")
    lines += ["", "| 구분 | 맵 수 | 평가 횟수 | 승리 / 비승리 | 승률 |",
              "| --- | ---: | ---: | --- | ---: |"]
    for split, label in (("seen", "학습에 사용한 맵"), ("unseen", "학습에 사용하지 않은 맵"), (None, "전체")):
        subset = [r for r in summaries if split is None or r["split"] == split]
        count = sum(r["episodes"] for r in subset)
        wins = sum(r["wins"] for r in subset)
        if count:
            lines.append(f"| {label} | {len(subset)} | {count} | {wins} / {count - wins} | {percent(wins / count)} |")
    lines += ["", "- 패배 조건부 지표의 분모는 비승리 에피소드입니다. 시간 초과와 동시 전멸도 포함합니다.",
              "- 적 잔존 비율은 각 비승리 에피소드의 생존 적 수 / 초기 적 수를 먼저 계산한 뒤 평균합니다.",
              "- 승리가 없으면 승리 길이, 비승리가 없으면 패배 시 적 잔존 비율은 N/A입니다.",
              "- 길이는 환경 의사결정 step이며 초 단위가 아닙니다.",
              "- 95% CI는 맵별 Wilson 구간입니다. 고정한 모델의 평가 불확실성이며 학습 seed 간 변동은 포함하지 않습니다.",
              "- 요약 승률은 완료된 에피소드 수로 가중합니다. 중단된 실행은 Status와 평가 횟수를 확인하세요.", ""]
    return "\n".join(lines)
