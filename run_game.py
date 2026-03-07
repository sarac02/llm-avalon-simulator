from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from env import AvalonConfig, AvalonEnv, mission_rules_for_players
from avalon_role_agent import AvalonRoleAgent
from llm_caller import AvalonLLMCaller


ROOT = Path(__file__).resolve().parent
ROLES_DIR = ROOT / "roles"


def canonical_role_key(role_name: str) -> str:
    return role_name.strip().lower().replace(" ", "_")


def is_evil_role(role_name: str) -> bool:
    key = canonical_role_key(role_name)
    return any(tag in key for tag in ("assassin", "minion", "morgana", "mordred"))


def roles_from_folder() -> List[str]:
    names = []
    for p in sorted(ROLES_DIR.glob("*.ipynb")):
        key = p.stem.replace("_", " ").strip().title()
        if key == "Loyal Servant":
            names.append("Loyal Servant")
        elif key == "Minion Of Mordred":
            names.append("Minion of Mordred")
        else:
            names.append(key)
    return names


def load_role_briefs() -> Dict[str, str]:
    briefs: Dict[str, str] = {}
    for nb_path in sorted(ROLES_DIR.glob("*.ipynb")):
        key = nb_path.stem.lower()
        try:
            data = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text_chunks: List[str] = []
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            chunk = "".join(cell.get("source", [])).strip()
            if chunk:
                text_chunks.append(chunk)
            if len("\n\n".join(text_chunks)) > 1000:
                break
        briefs[key] = "\n\n".join(text_chunks)[:1200]
    return briefs


def prompt_num_players() -> int:
    while True:
        raw = input("Enter number of players (5-10): ").strip()
        try:
            n = int(raw)
        except ValueError:
            print("Please enter an integer from 5 to 10.")
            continue
        if 5 <= n <= 10:
            return n
        print("Invalid value. Number of players must be between 5 and 10.")


def evil_count_for_players(num_players: int) -> int:
    # Standard Avalon
    return {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}[num_players]


def build_role_list(num_players: int) -> List[str]:
    """
    Uses role pool from the roles folder and fills remaining good slots with Loyal Servant.
    Constraint requested: at least 2 Loyal Servants when num_players > 5.
    """
    available = set(roles_from_folder())
    evil_target = evil_count_for_players(num_players)
    good_target = num_players - evil_target

    good_specials = [r for r in ["Merlin", "Percival"] if r in available]
    evil_specials = [r for r in ["Assassin", "Morgana", "Mordred", "Minion of Mordred"] if r in available]

    if "Assassin" not in evil_specials:
        raise RuntimeError("roles folder must include Assassin role notebook.")
    if "Merlin" not in good_specials:
        raise RuntimeError("roles folder must include Merlin role notebook.")

    evil_roles = ["Assassin"]
    for role in evil_specials:
        if role == "Assassin":
            continue
        if len(evil_roles) >= evil_target:
            break
        evil_roles.append(role)
    good_roles = ["Merlin"]
    for role in good_specials:
        if role == "Merlin":
            continue
        if len(good_roles) >= good_target:
            break
        good_roles.append(role)
    loyal_count = good_target - len(good_roles)

    if num_players > 5 and loyal_count < 2:
        # Reduce optional good special first to satisfy requested constraint.
        while loyal_count < 2 and len(good_roles) > 1:
            good_roles.pop()
            loyal_count += 1

    good_roles += ["Loyal Servant"] * loyal_count
    roles = good_roles + evil_roles
    return roles


def main():
    print("Avalon rules in this simulator:")
    print("- Official Avalon rule: if the 5th proposal is rejected, Evil wins immediately.")
    print("- Good wins at 3 successful quests; Evil wins at 3 failed quests.")
    print("- If Good reaches 3 successes, Assassin gets one Merlin guess to steal win.")

    num_players = prompt_num_players()
    role_list = build_role_list(num_players)
    names = [f"P{i}" for i in range(num_players)]
    # Shuffle roles each game so assignments can change run-to-run.
    import random
    random.shuffle(role_list)
    lineup = list(zip(names, role_list))

    names = [n for n, _ in lineup]
    role_map = {n: r for n, r in lineup}
    role_briefs = load_role_briefs()
    llm_backend: Any = AvalonLLMCaller(timeout=60, retries=2, temperature=0.45)
    try:
        llm_backend.generate(
            system="Return valid JSON only.",
            user='{"ok": true}',
            max_tokens=16,
        )
    except Exception as exc:
        raise RuntimeError(
            "Real LLM backend is unavailable. Install/configure OpenAI-compatible client "
            "(e.g., pip install openai) and set OPENAI_API_KEY + AVALON_API_BASE before running the game."
        ) from exc

    agents: List[Any] = []
    for name in names:
        role_key = canonical_role_key(role_map[name]).replace(" ", "_")
        role_file_key = role_key
        if role_file_key == "servant":
            role_file_key = "loyal_servant"
        brief = role_briefs.get(role_file_key, "")
        agents.append(AvalonRoleAgent(name=name, role=role_map[name], llm=llm_backend, role_notes=brief))
    team_sizes, fails_required = mission_rules_for_players(len(agents))
    cfg = AvalonConfig(
        num_players=len(agents),
        team_sizes=team_sizes,
        fails_required=fails_required,
        seed=random.randint(0, 10_000_000),
        discussion_turns=len(agents),
        post_proposal_discussion_turns=len(agents),  # everyone speaks once after proposal for richer debate
        verbose=True,
        strict_agent_errors=True,
        log_output_path=os.environ.get("AVALON_LOG_OUTPUT"),  # e.g. AVALON_LOG_OUTPUT=avalon_outputs.jsonl
    )

    env = AvalonEnv(agents=agents, roles=role_map, config=cfg)
    env.reset(leader_idx=0)
    final = env.run_game()

    # Write transcript and accusations to logs/ (one timestamp per run)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"log_{ts}.txt"
    acc_path = logs_dir / f"accusations_{ts}.json"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(final.transcript))
        print(f"\nGame transcript written to {log_path}")
    except Exception:
        pass
    try:
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump({
                "winner": final.winner,
                "reason": final.reason,
                "score_good": final.score_good,
                "score_evil": final.score_evil,
                "round_idx": final.round_idx,
                "roles": role_map,
                "accusations": getattr(final, "accusations", []),
            }, f, indent=2, ensure_ascii=False)
        print(f"Accusation vectors written to {acc_path}")
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("=== GAME OVER ===")
    print("=" * 50)
    print("Winner:", final.winner)
    print("Reason:", final.reason)
    print("Final score: Good", final.score_good, ":", "Evil", final.score_evil)
    print("=" * 50)
    print("\nPlayers:", ", ".join(names))
    print("Roles:", role_map)
    print("--- Full Event Log ---")
    for e in final.public_events:
        print(e.type, e.payload)
    print("--- Full Conversation Transcript ---")
    for speaker, text in final.chat_log:
        print(f"{speaker}: {text}")
    print("--- Accusation vectors (for RL) ---")
    for a in getattr(final, "accusations", []):
        r = a.get("reasoning", "") or ""
        r_short = (r[:80] + "...") if len(r) > 80 else r
        print(f"  {a.get('speaker')}: evil={a.get('evil_suspects', [])}, good={a.get('good_suspects', [])}, reasoning: {r_short}")


if __name__ == "__main__":

    main()
