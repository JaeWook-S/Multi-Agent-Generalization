"""저장한 원본 궤적만 읽어 행동 진단을 다시 집계한다. SC2와 PyTorch는 필요 없다."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiment_analysis.stage1_map_selection.evaluate_performance import checkpoint_hash, write_json
from experiment_analysis.stage2_behavior_diagnostics.diagnostic_metrics import (
    OUTCOMES, analyze_episode, summarize_diagnostics)


def _csv_value(value):
    """중첩 마스크/유닛 목록은 Python repr 대신 JSON으로 저장해 다시 읽을 수 있게 한다."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value


def write_csv(path, records):
    """사건별 필드 차이를 모두 보존하고, 값이 없는 지표는 빈칸으로 남긴다."""
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def context_comparison(records, bootstrap_samples, bootstrap_seed):
    """유형·체력·실드·쿨다운·전투 구간을 함께 고정한 탐색용 비교표를 만든다.

    각 층 안에서도 먼저 에피소드별 비율을 구한 뒤 동일 가중 평균한다.
    연속된 agent-step을 독립 표본으로 bootstrap하지 않는다. 30%/70% 구간은
    분석용 고정 기준이며, 연속 원본 값도 별도 CSV에 그대로 남긴다.
    """
    def band(value):
        if value is None:
            return "unknown"
        return "0..0.3" if value <= 0.3 else "0.3..0.7" if value <= 0.7 else "0.7..1"

    grouped = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in records:
        relative = row["relative_step_to_first_ally_death"]
        phase = "no_ally_death" if relative is None else (
            "before_first_ally_death" if relative < 0 else "after_first_ally_death")
        cooldown = row["cooldown"]
        key = (row["outcome"], row["unit_type"], band(row["health_ratio"]),
               band(row["shield_ratio"]), "unknown" if cooldown is None else (
                   "ready" if cooldown == 0 else "waiting"), phase)
        counts = grouped[key][(row["map"], row["episode"])]
        counts[0] += int(row["is_nonattack"])
        counts[1] += 1
    result = []
    # 분석 전용 Generator는 모델 평가의 전역 RNG 상태를 변경하지 않는다.
    rng = np.random.default_rng(bootstrap_seed)
    names = ("outcome", "unit_type", "health_band", "shield_band", "cooldown_band", "phase")
    for key, by_episode in sorted(grouped.items()):
        values = np.asarray([n / d for n, d in by_episode.values()], dtype=float)
        interval = None
        if len(values) >= 2:
            means = np.empty(bootstrap_samples)
            for start in range(0, bootstrap_samples, 256):
                size = min(256, bootstrap_samples - start)
                means[start:start + size] = values[rng.integers(len(values), size=(size, len(values)))].mean(axis=1)
            interval = np.quantile(means, (0.025, 0.975)).tolist()
        result.append({**dict(zip(names, key)), "valid_episodes": len(values),
                       "opportunities": sum(d for n, d in by_episode.values()),
                       "nonattack_choices": sum(n for n, d in by_episode.values()),
                       "episode_ratio_mean": float(values.mean()),
                       "episode_ratio_median": float(np.median(values)),
                       "mean_ci95_low": interval[0] if interval else None,
                       "mean_ci95_high": interval[1] if interval else None})
    return result


def markdown_comparison(summary):
    """승리·전투 패배·시간 초과를 섞지 않고 유효 분모와 함께 주요 지표를 표시한다."""
    labels = {"win": "승리", "defeat": "전투 패배", "timeout": "시간 초과", "draw": "동시 전멸"}
    groups = summary["groups"]
    lines = ["# 2단계: 행동 진단", "", f"Checkpoint: `{summary.get('checkpoint')}`", "",
             f"수집 상태: `{summary['collection_status']}` · 분석 에피소드: {summary['analyzed_episodes']} · "
             f"사망 직전 관측 창: {summary['window']} step", "",
             "| 지표 | " + " | ".join(f"{labels[group]} (n={groups[group]['n']})" for group in OUTCOMES) + " |",
             "| --- | " + " | ".join("---" for _ in OUTCOMES) + " |"]

    def cell(group, metric):
        value = groups[group]["metrics"].get(metric)
        if not value or not value["valid_n"]:
            return "N/A (n=0)"
        ratio = "ratio" in metric
        scale, suffix = (100, "%") if ratio else (1, "")
        result = f"{value['mean'] * scale:.1f}{suffix} (n={value['valid_n']})"
        interval = value["mean_ci95"]
        if interval:
            result += f" [{interval[0] * scale:.1f}, {interval[1] * scale:.1f}]"
        return result

    for key, label in (("ally", "첫 사망: 아군"), ("enemy", "첫 사망: 적군"),
                       ("both", "첫 사망: 같은 전이 양쪽"), ("none", "사망 없음")):
        cells = []
        for group in OUTCOMES:
            count = groups[group]["first_death_side_counts"].get(key, 0)
            n = groups[group]["n"]
            cells.append(f"{count}/{n} ({100 * count / n:.1f}%)" if n else "N/A")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in (("Stalker", "첫 아군 사망: Stalker만"), ("Zealot", "첫 아군 사망: Zealot만"),
                       ("mixed", "첫 아군 사망: 두 종류 함께"), ("none", "아군 사망 없음")):
        cells = []
        for group in OUTCOMES:
            n = groups[group]["n"]
            count = groups[group]["first_ally_death_type_counts"].get(key, 0)
            cells.append(f"{count}/{n}" if n else "N/A")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for metric, label in (
            ("first_ally_death_snapshot", "첫 아군 사망 시점 (snapshot)"),
            ("enemy_kills_before_first_ally_death", "첫 아군 사망 전 적 처치 수"),
            ("enemy_kills_same_transition_as_first_ally_death", "같은 전이의 적 처치 수"),
            ("enemy_kills_through_first_ally_death", "첫 아군 사망 전이까지 누적 적 처치 수"),
            ("nonattack_ratio_stalker", "Stalker 비공격 선택 비율"),
            ("nonattack_ratio_zealot", "Zealot 비공격 선택 비율"),
            ("first_ally_death_invisibility_ratio", "첫 아군 사망 직전 비가시 비율")):
        lines.append(f"| {label} | " + " | ".join(cell(group, metric) for group in OUTCOMES) + " |")
    lines += ["", "- 수치 지표는 유효 에피소드별 값의 평균이며 n은 해당 지표의 유효 에피소드 수입니다.",
              "- 비율 지표의 대괄호는 에피소드 단위 bootstrap 95% 구간입니다. 학습 seed 간 변동은 포함하지 않습니다.",
              "- 원본에 기회/사건이 없으면 0이 아닌 N/A입니다. 중앙값은 comparison.json에 함께 저장됩니다.",
              "- 같은 전이 사망은 step_mul 범위에서 함께 발견됐다는 뜻이며 정확한 게임 내부 사망 순서는 알 수 없습니다.",
              "- 비공격 선택 비율은 공격 명령이 허용될 때 이동/정지를 고른 비율입니다. 실제 타격 가능성·DPS 손실을 뜻하지 않습니다.",
              "- 가시성은 실제 obs의 ally visible 열에서 읽습니다. 원시 쿨다운/global state는 분석용이며 actor 입력에 추가하지 않습니다.",
              "- context_comparison.csv는 유형/체력/실드/쿨다운/전투 구간별 비교입니다. 작은 valid_episodes 층은 신중하게 해석하세요.",
              "- 모든 차이는 관찰적 연관입니다. 정보 부족이나 행동 선택의 인과 효과를 확정하지 않습니다.", ""]
    return "\n".join(lines)


def analyze_run(input_dir, *, window=5, bootstrap_samples=1000, bootstrap_seed=0, output_dir=None):
    """원본 CSV에 완료로 등록된 판만 읽어 재집계한다. 고아 파일/다른 모델은 섞지 않는다."""
    if window < 1 or bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("window/bootstrap-samples는 양수, bootstrap-seed는 0 이상이어야 합니다.")
    input_dir = Path(input_dir).expanduser().resolve()
    run_meta = json.loads((input_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if run_meta.get("analysis_stage") != "stage2_behavior_diagnostics":
        raise ValueError("1단계 episodes.csv에는 스텝 로그가 없습니다. 2단계 수집 폴더를 지정하세요.")
    output = (Path(output_dir).expanduser().resolve() if output_dir else input_dir / "analysis" / f"window_{window}")
    # 원본 궤적/실행 기록을 재집계 출력으로 덮어쓰지 않도록 출력 위치를 제한한다.
    if output == input_dir or output == input_dir / "step_logs" or input_dir.is_relative_to(output):
        raise ValueError("분석 결과는 원본 수집 폴더와 분리된 하위/별도 폴더에 저장하세요.")
    if (input_dir / "step_logs").resolve() == output or (input_dir / "step_logs").resolve() in output.parents:
        raise ValueError("step_logs 아래에는 분석 결과를 쓰지 않습니다.")
    output.mkdir(parents=True, exist_ok=True)
    performance = []
    if (input_dir / "episodes.csv").is_file():
        with (input_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
            performance = list(csv.DictReader(stream))
    episode_rows, contexts, visibility, events, sources = [], [], [], [], []
    seen = set()
    for record in performance:
        task, episode = record["map"], int(record["episode"])
        identity = (task, episode)
        if identity in seen or task not in run_meta["maps"] or Path(task).name != task:
            raise ValueError("에피소드 목록에 중복 또는 허용되지 않은 맵 경로가 있습니다.")
        seen.add(identity)
        path = input_dir / "step_logs" / task / f"episode_{episode:06d}.npz"
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if (meta.get("agent_sha256") != run_meta["agent_sha256"] or meta.get("map") != task
                or meta.get("episode") != episode or meta.get("env_seed") != int(record["env_seed"])
                or meta.get("env_seed") != run_meta["map_seeds"][task]):
            raise ValueError(f"다른 모델/맵/에피소드/seed의 원본 로그가 섞였습니다: {path}")
        if meta.get("outcome") != record["outcome"]:
            raise ValueError("원본 궤적의 종료 유형이 성과 CSV와 다릅니다.")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if meta.get("transition_count") != int(record["episode_length_steps"]):
            raise ValueError("원본 전이 수와 성과 CSV 길이가 다릅니다.")
        derived = analyze_episode(arrays, meta, window=window)
        episode_rows.append(derived["episode_metrics"])
        contexts.extend(derived["nonattack_context"])
        visibility.extend(derived["visibility_windows"])
        events.extend(derived["events"])
        sources.append({"map": task, "episode": episode, "npz": str(path.relative_to(input_dir)),
                        "npz_sha256": checkpoint_hash(path), "metadata_sha256": checkpoint_hash(sidecar)})
    if len(performance) != run_meta["completed_episodes"]:
        raise ValueError("성과 CSV의 판 수와 run_metadata의 완료 판 수가 다릅니다.")
    summary = summarize_diagnostics(episode_rows, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed)
    summary.update({"checkpoint": run_meta.get("checkpoint"), "agent_sha256": run_meta.get("agent_sha256"),
                    "collection_status": run_meta["status"], "analyzed_episodes": len(episode_rows),
                    "window": window, "map_seeds": run_meta.get("map_seeds", {}),
                    "analysis_output": str(output), "context_bands": [0.3, 0.7]})
    for name, values in (("episode_diagnostics.csv", episode_rows), ("nonattack_context.csv", contexts),
                         ("visibility_windows.csv", visibility), ("events.csv", events)):
        write_csv(output / name, values)
    write_csv(output / "context_comparison.csv", context_comparison(contexts, bootstrap_samples, bootstrap_seed))
    write_json(output / "comparison.json", summary)
    write_json(output / "source_manifest.json", {"input_run": str(input_dir), "episodes": sources,
                                                "run_metadata_sha256": checkpoint_hash(input_dir / "run_metadata.json")})
    (output / "comparison.md").write_text(markdown_comparison(summary), encoding="utf-8")
    return summary


def main():
    """관측 창만 바꿔 재분석할 때도 원본 수집 자료와 모델은 바꾸지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    result = analyze_run(args.input_dir, window=args.window, bootstrap_samples=args.bootstrap_samples,
                         bootstrap_seed=args.bootstrap_seed, output_dir=args.output_dir)
    print(f"분석 완료: {result['analyzed_episodes']} episodes | {result['analysis_output']}/comparison.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
