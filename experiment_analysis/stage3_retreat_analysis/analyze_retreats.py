"""3단계: 학습 H5와 기존 정책 NPZ에서 위험 상황 및 거리 확보 후보를 조사한다."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment_analysis.stage3_retreat_analysis.retreat_metrics import (
    Parameters, contexts, count_records, episode_counts, separation_segments, summarize_counts,
)
from experiment_analysis.stage3_retreat_analysis.trajectory import (
    dataset_episodes, dataset_plan, policy_episodes, read_json, sha256,
)


class CsvSink:
    """상세 행을 하나씩 압축 저장하고, 결과가 비어 있어도 헤더를 남긴다."""

    def __init__(self, path, empty_fields=("map", "episode", "step", "unit_id", "separation_candidate")):
        self.path, self.empty_fields, self.writer = Path(path), empty_fields, None

    def __enter__(self):
        opener = gzip.open if self.path.suffix == ".gz" else open
        self.stream = opener(self.path, "wt", encoding="utf-8", newline="")
        return self

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                              else value for key, value in row.items()})

    def __exit__(self, *args):
        if self.writer is None:
            csv.DictWriter(self.stream, fieldnames=self.empty_fields).writeheader()
        self.stream.close()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def report(manifest, summary):
    """검증 범위와 분모를 함께 표시하고 관찰을 인과 결론으로 바꾸지 않는다."""
    lines = ["# 위험 상황과 거리 확보 후보", "",
             f"분석 상태: `{manifest['status']}` · 자료: `{manifest['mode']}` · "
             f"에피소드: {manifest['processed_episodes']}", "",
             "- 결과·쿨다운·첫 사망 시점·공격 명령 가능 여부로 에피소드/행동을 제외하지 않았습니다.",
             "- 위험 후보 상황: 최근 체력/실드 감소가 있거나 가까운 살아 있는 적이 있음. 실제 피격 가능성을 확정하지 않습니다.",
             "- 거리 확보 후보: 이동 명령 + 실제 변위 + 동일한 생존 적 집합에 대해 자신의 이동 성분과 관측 거리 모두 증가.",
             "- 좌표 단위는 시야 거리=1입니다. 쿨다운은 무기 최대 쿨다운 대비 비율이며 적 쿨다운은 사용하지 않습니다.",
             "- 이후 피해·생존은 후속 정책 행동도 포함한 관찰입니다. 이동의 효과나 의도를 뜻하지 않습니다.",
             "- horizon 이전에 게임이 끝나 살아 있으면 전체 horizon 생존은 결측으로 둡니다.", ""]
    if manifest["mode"] == "dataset":
        lines += [f"데이터 조사 범위: `{manifest['selection_scope']}`", "",
                  "H5 마지막 상태에서 양쪽이 살아 있으면 종료 유형은 unknown입니다. timeout/패배로 추측하지 않습니다.",
                  "학습 당시 데이터 목록이 없으므로 현재 loader 규칙을 복원했습니다. 과거 데이터와의 동일성은 당시 경로/파일 해시와 별도 대조해야 합니다.", ""]
    lines += ["아래는 쿨다운 전체, 위험 후보 상황, Stalker의 에피소드별 비율 평균입니다.", "",
              "| 맵 | 자료/종료 | 교전 구간 | 유효 에피소드 | 생존 agent-step | 거리 확보 후보 명령 | 후보가 있는 판 | 에피소드별 비율 평균 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        if row["cooldown_band"] != "all" or row["scope"] != "risk" or row["unit_type"] != "Stalker":
            continue
        if row["combat_phase"] == "all":
            continue
        lines.append(f"| {row['map']} | {row['source_kind']}/{row['outcome']} | {row['combat_phase']} | "
                     f"{row['episodes_with_context']} | {row['agent_steps']} | {row['separation_candidates']} | "
                     f"{row['episodes_with_candidate']} | {row['episode_candidate_ratio_mean']:.2%} |")
    lines += ["", "- contested: 적 2명 이상 / cleanup: 적 1명 이하. 단순 병력 수 구분이며 전투의 난이도 판정이 아닙니다.",
              "- 빈 층은 0%로 채우지 않습니다. 전체 유형/쿨다운/분모는 context_summary.csv를 확인하세요.",
              "- risk_contexts.csv.gz의 risk_band로 최근 피해·저체력·근접 접촉을 구분합니다.",
              "- movement_steps.csv.gz의 target_died_recently와 combat_phase로 표적 사망 뒤 재배치/마무리를 구분합니다.",
              "- separation_segments.csv는 연속 후보를 묶은 구간입니다. 독립 전략 사례 또는 후퇴 성공 수가 아닙니다.",
              "- shield_gain은 실드 증가 기록이며 동료의 치료로 해석하지 않습니다. 현재 지원 맵에는 힐러가 없습니다.", ""]
    return "\n".join(lines)


def run_analysis(args):
    """입력을 읽기 전용으로 분석하고 새 결과 폴더에만 저장한다."""
    parameters = Parameters(args.near_distance, args.low_health, args.recent_window,
                            args.min_move, args.min_separation, tuple(args.horizons)).validate()
    if args.limit_episodes is not None and args.limit_episodes < 1:
        raise ValueError("limit-episodes는 양수여야 합니다.")
    manifest = {"schema_version": 1, "mode": args.mode, "status": "running",
                "parameters": parameters.dictionary(), "processed_episodes": 0,
                "limit_episodes": args.limit_episodes, "episode_sources": [],
                "started_at": datetime.now().astimezone().isoformat(),
                "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                            ("analyze_retreats.py", "trajectory.py", "retreat_metrics.py")}}
    if args.mode == "dataset":
        config_path = args.training_config.expanduser().resolve()
        config = read_json(config_path)
        overrides = {}
        for item in args.source_dir or []:
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in overrides:
                raise ValueError("source-dir는 중복 없는 MAP/QUALITY=/path 형식이어야 합니다.")
            overrides[key] = value
        plan = dataset_plan(config, args.dataset_root, args.maps, args.audit_all, overrides)
        allowed = {f"{row['map']}/{row['quality']}" for row in plan}
        if not set(overrides).issubset(allowed):
            raise ValueError("source-dir에 선택하지 않은 맵/품질이 있습니다.")
        manifest.update(training_config=str(config_path), training_config_sha256=sha256(config_path),
                        dataset_root=str(args.dataset_root.expanduser().resolve()), selection_plan=plan,
                        selection_scope="all_source_episodes" if args.audit_all else "reconstructed_loader_tail",
                        historical_training_manifest_available=False, h5_sources=[])
        trajectories = dataset_episodes(plan, config, args.limit_episodes)
    else:
        runs = [path.expanduser().resolve() for path in args.run_dir]
        if len(set(runs)) != len(runs):
            raise ValueError("같은 run-dir를 중복 지정할 수 없습니다.")
        manifest["policy_runs"] = []
        for path in runs:
            meta = read_json(path / "run_metadata.json")
            manifest["policy_runs"].append({"path": str(path), "status": meta["status"],
                                             "agent_sha256": meta["agent_sha256"],
                                             "map_seeds": meta["map_seeds"],
                                             "run_metadata_sha256": sha256(path / "run_metadata.json"),
                                             "config_sha256": sha256(path / "effective_config.json"),
                                             "episodes_csv_sha256": sha256(path / "episodes.csv")})
        trajectories = (episode for path in runs for episode in policy_episodes(path, args.limit_episodes))
    output = args.output_dir
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = Path(__file__).resolve().parent / "results" / f"{args.mode}_{stamp}"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest["output_dir"] = str(output)
    write_json(output / "manifest.json", manifest)
    if args.mode == "dataset":
        write_json(output / "effective_config.json", config)
    outcome_counts = Counter()
    try:
        if args.mode == "dataset":
            for source in plan:
                print(f"H5 원본 확인: {source['path']}", flush=True)
                manifest["h5_sources"].append({"path": source["path"], "sha256": sha256(source["path"]),
                                                "bytes": Path(source["path"]).stat().st_size})
        with ExitStack() as stack:
            sinks = {name: stack.enter_context(CsvSink(output / filename)) for name, filename in (
                ("risk", "risk_contexts.csv.gz"), ("movement", "movement_steps.csv.gz"),
                ("segments", "separation_segments.csv"), ("counts", "episode_context_counts.csv.gz"),
                ("episodes", "episodes.csv"))}

            def process():
                """에피소드 단위로 읽고 상세 표는 즉시 저장한다. 원본 H5/NPZ는 변경하지 않는다."""
                for trajectory in trajectories:
                    rows = list(contexts(trajectory, parameters))
                    for row in rows:
                        if row["risk_context"]:
                            sinks["risk"].write(row)
                        if row["action_family"] == "move":
                            sinks["movement"].write(row)
                    segments = separation_segments(rows)
                    for segment in segments:
                        sinks["segments"].write(segment)
                    meta = trajectory.meta
                    outcome_counts[f"{meta['map']}/{meta['outcome']}"] += 1
                    sinks["episodes"].write({
                        **{key: meta.get(key) for key in ("source_kind", "source_id", "map", "episode", "outcome",
                                                        "source_file", "source_index", "quality", "env_seed", "agent_sha256")},
                        "outcome_source": meta["outcome_source"], "steps": trajectory.steps,
                        "episode_return": float(trajectory.rewards.sum()),
                        "alive_agent_steps": len(rows), "risk_context_steps": sum(row["risk_context"] for row in rows),
                        "movement_steps": sum(row["action_family"] == "move" for row in rows),
                        "separation_candidate_steps": sum(row["separation_candidate"] for row in rows),
                        "separation_segments": len(segments),
                    })
                    manifest["episode_sources"].append({
                        **{key: meta.get(key) for key in ("map", "episode", "source_file", "source_index", "source_sha256",
                                                        "metadata_sha256", "env_seed", "agent_sha256")},
                        "outcome": meta["outcome"], "calibration": trajectory.calibration})
                    manifest["processed_episodes"] += 1
                    count = manifest["processed_episodes"]
                    if count == 1 or count % 25 == 0:
                        print(f"분석 {count}판 완료 · {meta['map']} · episode {meta['episode']}", flush=True)
                    for record in count_records(episode_counts(rows), meta["episode"]):
                        sinks["counts"].write(record)
                        yield record

            summary = summarize_counts(process())
        if not manifest["processed_episodes"]:
            raise ValueError("분석할 에피소드가 없습니다.")
        with CsvSink(output / "context_summary.csv") as sink:
            for row in summary:
                sink.write(row)
        manifest.update(status="complete", outcome_counts=dict(outcome_counts),
                        finished_at=datetime.now().astimezone().isoformat())
        write_json(output / "manifest.json", manifest)
        (output / "summary.md").write_text(report(manifest, summary), encoding="utf-8")
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise
    print(f"완료: {output / 'summary.md'}", flush=True)
    return output


def parse_args(argv=None):
    """두 자료 경로에 같은 판정 기준을 적용하는 독립 분석 명령이다."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, description in (("dataset", "실제 학습 H5의 전략 사례 조사"), ("policy", "기존 2단계 NPZ 재분석")):
        command = sub.add_parser(mode, description=description)
        command.add_argument("--output-dir", type=Path, help="새 결과 폴더. 기존 폴더는 덮어쓰지 않음")
        command.add_argument("--limit-episodes", type=int, help="작동 확인용 일부만 분석. dataset은 맵별, policy는 run별 제한")
        command.add_argument("--near-distance", type=float, default=0.75, help="시야 거리=1 기준 가까운 적의 거리")
        command.add_argument("--low-health", type=float, default=0.30, help="저체력 구분만 변경하며 분석 대상을 제외하지 않음")
        command.add_argument("--recent-window", type=int, default=3, help="행동 직전 피해/표적 사망 확인 창")
        command.add_argument("--min-move", type=float, default=0.01, help="시야 거리 단위 최소 실제 변위")
        command.add_argument("--min-separation", type=float, default=0.01, help="시야 거리 단위 최소 거리 증가")
        command.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10], help="이후 경과 관측 길이; 의사결정 스텝 단위")
        if mode == "dataset":
            command.add_argument("--training-config", required=True, type=Path, help="학습 당시 config.json")
            command.add_argument("--dataset-root", type=Path, default=Path("dataset"))
            command.add_argument("--maps", nargs="+", help="생략하면 모든 train/pretrain 맵")
            command.add_argument("--source-dir", action="append", help="과거 소스 명시: MAP/QUALITY=/path/to/qmix_folder")
            command.add_argument("--audit-all", action="store_true", help="학습 subset이 아닌 소스 전체 조사. 무작위 subset 사용 시 필요")
        else:
            command.add_argument("--run-dir", type=Path, nargs="+", required=True, help="기존 2단계 결과 폴더들")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run_analysis(parse_args(argv))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"분석 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
