from __future__ import annotations

import random
from typing import Any, List


class RandomBaselineAgent:
    """
    Baseline 1: very simple random / rule-light bot.
    Good for a lower-bound sanity check.
    """

    def __init__(self, name: str, role: str, seed: int | None = None):
        self.name = name
        self.role = role
        self.rng = random.Random(seed)

    def _known_evil_players(self, state) -> set[str]:
        known = state.private_knowledge.get("evil_players", [])
        if isinstance(known, list):
            return {p for p in known if isinstance(p, str)}
        return set()

    def speak(self, state) -> str:
        if state.current_team:
            lean = self.rng.choice(["APPROVE_LEAN", "REJECT_LEAN"])
            return f"I am not fully sure yet, but I have a read on {state.current_proposer}. {lean}"
        return f"Early read: I want to watch how {state.current_proposer} builds this team."

    def propose_team(self, state, team_size: int) -> List[str]:
        players = list(state.players)
        return self.rng.sample(players, team_size)

    def vote_on_team(self, state, team: List[str]) -> bool:
        # Slightly less dumb than pure random:
        # approve own team more often, otherwise coin flip
        if self.name in team:
            return True
        return self.rng.random() < 0.5

    def mission_action(self, state) -> bool:
        # Env already forces good players to PASS, but we keep logic explicit.
        if state.my_alignment == "good":
            return True
        # Evil fails often, but not always.
        return self.rng.random() < 0.3  # True=PASS, False=FAIL

    def assassinate(self, state) -> str:
        candidates = [p for p in state.players if p != self.name]
        return self.rng.choice(candidates)


class VanillaHeuristicBaselineAgent:
    """
    Baseline 2: simple non-LLM heuristic bot.
    Uses only obvious public signals + known evil info if available.
    This is much stronger than random, but still simple and believable.
    """

    def __init__(self, name: str, role: str, seed: int | None = None):
        self.name = name
        self.role = role
        self.rng = random.Random(seed)

    def _known_evil_players(self, state) -> set[str]:
        known = state.private_knowledge.get("evil_players", [])
        if isinstance(known, list):
            return {p for p in known if isinstance(p, str)}
        return set()

    def _suspicion_scores(self, state) -> dict[str, float]:
        """
        Very lightweight suspicion tracker:
        - players on failed missions become suspicious
        - players on clean missions become a bit trusted
        """
        scores = {p: 0.0 for p in state.players}

        for mission in getattr(state, "missions", []):
            team = list(getattr(mission, "team", []) or [])
            fails = int(getattr(mission, "fail_count", 0))

            if fails > 0:
                for p in team:
                    scores[p] += 2.0 * fails
            else:
                for p in team:
                    scores[p] -= 0.75

        # If I'm good, trust myself strongly.
        if state.my_alignment == "good":
            scores[self.name] -= 3.0

        # If I know evil players, mark them very suspicious.
        for p in self._known_evil_players(state):
            scores[p] += 100.0

        # If I am evil, trust fellow evil for team-building purposes.
        if state.my_alignment == "evil":
            for p in self._known_evil_players(state):
                scores[p] -= 5.0
            scores[self.name] -= 5.0

        return scores

    def speak(self, state) -> str:
        scores = self._suspicion_scores(state)
        ordered = sorted(state.players, key=lambda p: scores[p])
        trusted = [p for p in ordered if p != self.name][:2]
        suspicious = [p for p in reversed(ordered) if p != self.name][:2]

        if state.current_team:
            good_team_signal = sum(scores[p] for p in state.current_team)
            lean = "APPROVE_LEAN" if good_team_signal <= 1.5 * len(state.current_team) else "REJECT_LEAN"
            return (
                f"I currently trust {trusted[0]} more than {suspicious[0]}. "
                f"This lineup gives me a clearer signal. {lean}"
            )

        return (
            f"My current read is that {trusted[0]} looks safer, while {suspicious[0]} is more suspicious "
            f"from the mission history."
        )

    def propose_team(self, state, team_size: int) -> List[str]:
        scores = self._suspicion_scores(state)
        players_sorted = sorted(state.players, key=lambda p: (scores[p], p))

        if state.my_alignment == "good":
            # Pick least suspicious players
            team = players_sorted[:team_size]
            if self.name not in team:
                team[-1] = self.name
            return team

        # Evil strategy: include self + one known evil if possible, then fill with least suspicious goods
        evil_pool = sorted(
            list(self._known_evil_players(state) | {self.name}),
            key=lambda p: (scores[p], p)
        )
        team: List[str] = []

        if self.name not in team:
            team.append(self.name)

        for p in evil_pool:
            if p not in team and len(team) < min(2, team_size):
                team.append(p)

        for p in players_sorted:
            if p not in team and len(team) < team_size:
                team.append(p)

        return team[:team_size]

    def vote_on_team(self, state, team: List[str]) -> bool:
        scores = self._suspicion_scores(state)

        if state.my_alignment == "good":
            # Reject if known evil on team
            if any(p in self._known_evil_players(state) for p in team):
                return False

            team_risk = sum(scores[p] for p in team)

            # On the 5th proposal, approve to avoid auto-loss if unsure
            if state.proposal_idx >= 4:
                return True

            return team_risk <= 1.5 * len(team)

        # Evil: usually approve teams containing any evil
        evil_set = self._known_evil_players(state) | {self.name}
        if any(p in evil_set for p in team):
            return True

        # Otherwise sometimes reject good-looking teams
        return False

    def mission_action(self, state) -> bool:
        if state.my_alignment == "good":
            return True

        # Evil heuristic:
        # - early game, sometimes pass to avoid exposure
        # - if evil already has 2 points, try to fail for win
        # - if good has 2 points, definitely fail
        evil_score = getattr(state, "num_fails", 0)
        good_score = getattr(state, "num_successes", 0)

        if good_score >= 2:
            return False  # FAIL
        if evil_score >= 2:
            return False  # FAIL for the win
        if state.round_idx == 1 and len(state.current_team) >= 3:
            return True   # PASS early sometimes to look safe
        return False      # FAIL otherwise

    def assassinate(self, state) -> str:
        # If Percival/Merlin candidates exist, prefer those.
        mk = state.private_knowledge.get("merlin_candidates", [])
        if isinstance(mk, list) and mk:
            candidates = [p for p in mk if p != self.name]
            if candidates:
                return self.rng.choice(candidates)

        # Otherwise kill the most trusted good-looking player.
        scores = self._suspicion_scores(state)
        candidates = [p for p in state.players if p != self.name]
        candidates.sort(key=lambda p: (scores[p], p))
        return candidates[0]