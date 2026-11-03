"""원본 NPZ의 위치·체력을 다시 그리는 2D 뷰어. SC2나 정책을 실행하지 않는다."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TEMPLATE_DIR = Path(__file__).resolve().parent


def load_episode(run_dir, map_name, episode, run_meta):
    """T개 행동과 T+1개 상태를 검증한다. step t의 행동 결과는 상태 t+1이다."""
    path = run_dir / "step_logs" / map_name / f"episode_{episode:06d}.npz"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta.get("status") != "valid" or meta.get("episode") != episode
            or meta.get("map") != map_name
            or meta.get("agent_sha256") != run_meta["agent_sha256"]
            or meta.get("env_seed") != run_meta["map_seeds"][map_name]):
        raise ValueError(f"다른 모델/맵/seed이거나 유효하지 않은 에피소드입니다: {path}")

    fields = ["actions", "avail_actions"]
    fields += [f"{side}_{name}" for side in ("ally", "enemy")
               for name in ("x", "y", "health", "health_max", "shield", "shield_max",
                            "cooldown", "type", "tag")]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in fields}
    steps, allies, enemies = meta["transition_count"], meta["n_agents"], meta["n_enemies"]
    if arrays["actions"].shape != (steps, allies):
        raise ValueError("행동 배열의 길이가 실제 전이 수와 다릅니다.")
    for side, count in (("ally", allies), ("enemy", enemies)):
        for name in ("x", "y", "health", "health_max", "shield", "shield_max", "cooldown"):
            value = arrays[f"{side}_{name}"]
            if value.shape != (steps + 1, count) or not np.isfinite(value).all():
                raise ValueError(f"상태 배열의 차원 또는 수치가 유효하지 않습니다: {side}_{name}")
        for name in ("type", "tag"):
            if arrays[f"{side}_{name}"].shape != (count,):
                raise ValueError(f"유닛 ID 매핑이 유효하지 않습니다: {side}_{name}")
    attack_offset = meta["n_actions_no_attack"]
    if arrays["avail_actions"].shape != (steps + 1, allies, attack_offset + enemies):
        raise ValueError("행동 마스크의 차원이 실제 유닛 수와 다릅니다.")

    # 첫 사망 확인 이전과 이후를 나눈 기존 분석의 기준을 그대로 사용한다.
    health = arrays["ally_health"]
    deaths = np.flatnonzero(((health[:-1] > 0) & (health[1:] <= 0)).any(axis=1))
    first_death = int(deaths[0] + 1) if len(deaths) else None
    events = []
    if first_death is not None:
        for t in range(first_death, steps):
            for unit in range(allies):
                max_hp = arrays["ally_health_max"][t, unit]
                max_shield = arrays["ally_shield_max"][t, unit]
                hp_ratio = health[t, unit] / max_hp if max_hp > 0 else 0
                shield_ratio = arrays["ally_shield"][t, unit] / max_shield if max_shield > 0 else 0
                action = int(arrays["actions"][t, unit])
                if (meta["outcome"] == "win" and arrays["ally_type"][unit] == "Stalker"
                        and health[t, unit] > 0 and hp_ratio <= .3 and shield_ratio <= .3
                        and arrays["ally_cooldown"][t, unit] == 0 and 1 <= action <= 5
                        and arrays["avail_actions"][t, unit, attack_offset:].any()):
                    events.append({"step": t, "unit": unit, "action": action})

    result = {"episode": episode, "map": map_name, "outcome": meta["outcome"],
              "steps": steps, "stepMul": meta["step_mul"], "seed": meta["env_seed"],
              "firstDeath": first_death, "events": events,
              "actions": arrays["actions"].astype(int).tolist(), "attackOffset": attack_offset,
              "source": path.name, "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for side, label in (("ally", "allies"), ("enemy", "enemies")):
        result[label] = {name: arrays[f"{side}_{name}"].tolist()
                         for name in ("x", "y", "health", "shield", "cooldown")}
        result[label].update({"maxHealth": arrays[f"{side}_health_max"].max(axis=0).tolist(),
                              "maxShield": arrays[f"{side}_shield_max"].max(axis=0).tolist(),
                              "types": arrays[f"{side}_type"].tolist(),
                              # SC2 태그를 JS 숫자로 바꾸면 큰 정수의 정밀도를 잃을 수 있다.
                              "tags": [str(int(tag)) for tag in arrays[f"{side}_tag"]]})
    return result


def build_viewer(run_dir, episodes=(1, 21, 44, 81, 89, 94), output=None):
    """여섯 사례를 한 화면에 담는다. 생성물만 쓰며 원본 NPZ/JSON은 수정하지 않는다."""
    run_dir = Path(run_dir).expanduser().resolve()
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("analysis_stage") != "stage2_behavior_diagnostics" or len(meta["maps"]) != 1:
        raise ValueError("한 맵을 수집한 2단계 결과 폴더를 지정하세요.")
    map_name = meta["maps"][0]
    if Path(map_name).name != map_name:
        raise ValueError("올바르지 않은 맵 이름입니다.")
    episodes = list(episodes)
    if not episodes or len(set(episodes)) != len(episodes) or any(e < 1 for e in episodes):
        raise ValueError("에피소드 번호는 중복 없는 양수여야 합니다.")
    cases = [load_episode(run_dir, map_name, e, meta) for e in episodes]
    data = {"map": map_name, "modelSha256": meta["agent_sha256"], "episodes": cases}
    # 데이터에 HTML 종료 태그가 포함되어도 실행 코드로 해석되지 않게 한다.
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    fragment = (TEMPLATE_DIR / "npz_case_viewer_template.html").read_text(encoding="utf-8")
    fragment = fragment.replace("__CASE_DATA__", encoded)
    output = Path(output).expanduser().resolve() if output else run_dir / "analysis/npz-case-viewer.html"
    if output.suffix != ".html" or (run_dir / "step_logs") in output.parents:
        raise ValueError("출력은 step_logs 바깥의 .html 파일이어야 합니다.")
    fragment_path = output.with_name(output.stem.replace("_", "-").lower() + "-inline.html")
    wrapper = (TEMPLATE_DIR / "npz_viewer_standalone.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(fragment, encoding="utf-8")
    output.write_text(wrapper.replace("__FRAGMENT__", fragment), encoding="utf-8")
    return output


def main():
    """결과 폴더만 있으면 실행 가능하다. SC2·PyTorch·pandas는 불러오지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[1, 21, 44, 81, 89, 94])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        output = build_viewer(args.run_dir, args.episodes, args.output)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f"브라우저에서 열 파일: {output}")


if __name__ == "__main__":
    main()
