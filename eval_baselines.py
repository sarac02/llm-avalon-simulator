from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Any, Dict, List

from env import AvalonConfig, AvalonEnv, mission_rules_for_players
from avalon_role_agent import AvalonRoleAgent
from llm_caller import AvalonLLMCaller
from run_game import build_role_list, canonical_role_key, load_role_briefs, USE_RL, MERLIN_POLICY_PATH
from baseline_agents import RandomBaselineAgent, VanillaHeuristicBaselineAgent


def build_agents(
    names: List[str],
    role_map: Dict[str, str],
    baseline_mode: str,
) -> List[Any]:
    role_briefs = load_role_briefs()

    llm_backend = None
    merlin_policy = None

    if baseline_mode == "llm":
        llm_backend = AvalonLLMCaller(timeout=60, retries=2, temperature=0.45)
        llm_backend.generate(
            system="Return valid JSON only.",
            user='{"ok": true}',
            max_tokens=16,
        )

        if USE_RL:
            from rl.merlin_policy_inference import MerlinPolicy
            merlin_policy = MerlinPolicy(MERLIN_POLICY_PATH)

    agents: List[Any] = []
    for i, name in enumerate(names):
        if baseline_mode == "random":
            agents.append(
                RandomBaselineAgent(
                    name=name,
                    role=role_map[name],
                    seed=10_000 + i,
                )
            )
        elif baseline_mode == "heuristic":
            agents.append(
                VanillaHeuristicBaselineAgent(
                    name=name,
                    role=role_map[name],
                    seed=20_000 + i,
                )
            )
        elif baseline_mode == "llm":
            role_key = canonical_role_key(role_map[name]).replace(" ", "_")
            role_file_key = role_key
            if role_file_key == "servant":
                role_file_key = "loyal_servant"
            brief = role_briefs.get(role_file_key, "")
            agents.append(
                AvalonRoleAgent(
                    name=name,
                    role=role_map[name],
                    llm=llm_backend,
                    role_notes=brief,
                    merlin_policy=merlin_policy if name == "P0" else None,
                    use_rl=USE_RL if name == "P0" else False,
                )
            )
        else:
            raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    return agents


def play_one_game(num_players: int, baseline_mode: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)

    role_list = build_role_list(num_players)
    names = [f"P{i}" for i in range(num_players)]
    rng.shuffle(role_list)

    lineup = list(zip(names, role_list))
    names = [n for n, _ in lineup]
    role_map = {n: r for n, r in lineup}

    agents = build_agents(names, role_map, baseline_mode)

    team_sizes, fails_required = mission_rules_for_players(num_players)
    cfg = AvalonConfig(
        num_players=num_players,
        team_sizes=team_sizes,
        fails_required=fails_required,
        seed=seed,
        discussion_turns=len(agents),
        post_proposal_discussion_turns=len(agents),
        verbose=False,
        strict_agent_errors=True,
        log_output_path=None,
    )

    env = AvalonEnv(agents=agents, roles=role_map, config=cfg)
    env.reset(leader_idx=0)
    final = env.run_game()

    return {
        "winner": final.winner,
        "reason": final.reason,
        "score_good": final.score_good,
        "score_evil": final.score_evil,
        "roles": role_map,
    }


def evaluate(num_games: int, num_players: int, baseline_mode: str, seed: int) -> None:
    winners = Counter()
    reasons = Counter()
    good_scores = []
    evil_scores = []

    for i in range(num_games):
        game_seed = seed + i
        result = play_one_game(
            num_players=num_players,
            baseline_mode=baseline_mode,
            seed=game_seed,
        )
        winners[result["winner"]] += 1
        reasons[result["reason"]] += 1
        good_scores.append(result["score_good"])
        evil_scores.append(result["score_evil"])

        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(f"[{baseline_mode}] finished {i + 1}/{num_games} games")

    total = num_games
    good_wins = winners.get("good", 0)
    evil_wins = winners.get("evil", 0)

    print("\n" + "=" * 60)
    print(f"Baseline: {baseline_mode}")
    print(f"Players:  {num_players}")
    print(f"Games:    {num_games}")
    print("=" * 60)
    print(f"Good win rate: {good_wins}/{total} = {100.0 * good_wins / total:.2f}%")
    print(f"Evil win rate: {evil_wins}/{total} = {100.0 * evil_wins / total:.2f}%")
    print(f"Avg good score: {sum(good_scores) / total:.3f}")
    print(f"Avg evil score: {sum(evil_scores) / total:.3f}")
    print("\nWin reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count} ({100.0 * count / total:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--baseline", type=str, choices=["random", "heuristic", "llm"], default="random")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    evaluate(
        num_games=args.games,
        num_players=args.players,
        baseline_mode=args.baseline,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()