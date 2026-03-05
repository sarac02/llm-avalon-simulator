from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import random
import re


def mission_rules_for_players(num_players: int) -> Tuple[List[int], List[int]]:
    """Standard Avalon: 5 rounds, team sizes and fail thresholds per player count. Round 4 needs 2 fails for 7–10 players."""
    preset: Dict[int, Tuple[List[int], List[int]]] = {
        5: ([2, 3, 2, 3, 3], [1, 1, 1, 1, 1]),
        6: ([2, 3, 4, 3, 4], [1, 1, 1, 1, 1]),
        7: ([2, 3, 3, 4, 4], [1, 1, 1, 2, 1]),
        8: ([3, 4, 4, 5, 5], [1, 1, 1, 2, 1]),
        9: ([3, 4, 4, 5, 5], [1, 1, 1, 2, 1]),
        10: ([3, 4, 4, 5, 5], [1, 1, 1, 2, 1]),
    }
    if num_players in preset:
        return preset[num_players]
    sizes = [max(2, min(num_players - 1, 2 + i // 2)) for i in range(5)]
    fails = [1, 1, 1, 1, 1]
    return sizes, fails


def assert_avalon_config(num_players: int, team_sizes: List[int], fails_required: List[int]) -> None:
    """Crash loudly if config does not match Avalon rules (exact team sizes, 5 rounds, correct fail thresholds)."""
    if num_players < 5 or num_players > 10:
        raise ValueError(f"Avalon requires 5–10 players, got {num_players}")
    expected_sizes, expected_fails = mission_rules_for_players(num_players)
    if len(team_sizes) != 5 or team_sizes != expected_sizes:
        raise ValueError(f"team_sizes must be exactly 5 rounds for Avalon: expected {expected_sizes}, got {team_sizes}")
    if len(fails_required) != 5 or fails_required != expected_fails:
        raise ValueError(f"fails_required must match Avalon (round 4 = 2 fails for 7–10 players): expected {expected_fails}, got {fails_required}")


class Phase(str, Enum):
    DISCUSSION = "discussion"
    PROPOSE = "propose"
    TEAM_DISCUSSION = "team_discussion"
    TEAM_VOTE = "team_vote"
    QUEST = "quest"
    ASSASSINATE = "assassinate"
    GAME_OVER = "game_over"


@dataclass
class AvalonConfig:
    seed: int = 0
    num_players: int = 0
    team_sizes: List[int] = field(default_factory=list)
    fails_required: List[int] = field(default_factory=list)
    max_rejects: int = 5
    discussion_turns: Optional[int] = None  # None -> everyone once
    post_proposal_discussion_turns: Optional[int] = None  # None -> everyone once
    verbose: bool = True
    strict_agent_errors: bool = True


@dataclass
class PlayerInfo:
    name: str
    role: str
    alignment: str
    private_knowledge: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublicEvent:
    type: str
    payload: Dict[str, Any]


@dataclass
class ProposalRecord:
    round_idx: int
    proposal_idx: int
    proposer: str
    team: List[str]
    votes: Dict[str, bool] = field(default_factory=dict)
    approved: Optional[bool] = None


@dataclass
class MissionRecord:
    round_idx: int
    team: List[str]
    fail_count: int
    outcome: str


@dataclass
class GameState:
    players: List[str]
    leader: str
    round_idx: int = 1
    proposal_idx: int = 1
    score_good: int = 0
    score_evil: int = 0
    phase: Phase = Phase.DISCUSSION
    current_team: List[str] = field(default_factory=list)
    last_team_votes: Dict[str, bool] = field(default_factory=dict)
    last_quest_fails: int = 0
    chat_log: List[Tuple[str, str]] = field(default_factory=list)
    public_events: List[PublicEvent] = field(default_factory=list)
    proposals: List[ProposalRecord] = field(default_factory=list)
    missions: List[MissionRecord] = field(default_factory=list)
    winner: Optional[str] = None
    reason: Optional[str] = None
    transcript: List[str] = field(default_factory=list)
    rejected_rounds_total: int = 0


class AvalonEnv:
    def __init__(self, agents: List[Any], roles: Dict[str, str], config: Optional[AvalonConfig] = None):
        self.cfg = config or AvalonConfig(num_players=len(agents))
        if self.cfg.num_players <= 0:
            self.cfg.num_players = len(agents)
        if not self.cfg.team_sizes or not self.cfg.fails_required:
            team_sizes, fails_required = mission_rules_for_players(self.cfg.num_players)
            self.cfg.team_sizes = team_sizes
            self.cfg.fails_required = fails_required
        assert_avalon_config(self.cfg.num_players, self.cfg.team_sizes, self.cfg.fails_required)

        random.seed(self.cfg.seed)
        self.agents: Dict[str, Any] = {a.name: a for a in agents}
        self.players: List[str] = [a.name for a in agents]
        if len(self.players) != self.cfg.num_players:
            raise ValueError("cfg.num_players must match number of agents")

        self.player_info: Dict[str, PlayerInfo] = {}
        for name in self.players:
            role = roles.get(name, "Loyal Servant")
            role_l = role.lower()
            is_evil = any(x in role_l for x in ["assassin", "minion", "morgana", "mordred"])
            self.player_info[name] = PlayerInfo(name=name, role=role, alignment=("evil" if is_evil else "good"))

        evil_players = [p for p in self.players if self.player_info[p].alignment == "evil"]
        merlin_player = next((p for p in self.players if "merlin" in self.player_info[p].role.lower()), None)
        morgana_player = next((p for p in self.players if "morgana" in self.player_info[p].role.lower()), None)

        for name in self.players:
            role_l = self.player_info[name].role.lower()
            info = self.player_info[name].private_knowledge
            info["known_evil_players"] = []

            # Merlin sees evil players except Mordred.
            if "merlin" in role_l:
                info["known_evil_players"] = [
                    p for p in evil_players
                    if "mordred" not in self.player_info[p].role.lower() and p != name
                ]
                # Backward-compatible key used by current agents.
                info["evil_players"] = list(info["known_evil_players"])

            # Percival sees Merlin/Morgana but cannot distinguish.
            if "percival" in role_l:
                candidates = [p for p in [merlin_player, morgana_player] if p is not None and p != name]
                random.shuffle(candidates)
                info["merlin_candidates"] = candidates

            if self.player_info[name].alignment == "evil":
                info["known_evil_players"] = [p for p in evil_players if p != name]
                # Backward-compatible key used by current agents.
                info["evil_players"] = list(info["known_evil_players"])

        self._leader_idx = 0
        self.state = GameState(players=self.players[:], leader=self.players[0])
        self._log("=== GAME INIT ===")
        self._log(f"Players ({len(self.players)}): {', '.join(self.players)}")
        role_view = ", ".join([f"{p}:{self.player_info[p].role}" for p in self.players])
        self._log(f"Assigned roles: {role_view}")

    def reset(self, leader_idx: int = 0) -> GameState:
        self._leader_idx = leader_idx % len(self.players)
        self.state = GameState(players=self.players[:], leader=self.players[self._leader_idx])
        self._log("=== GAME START ===")
        self._log(f"Starting leader: {self.state.leader}")
        return self.state

    def run_game(self) -> GameState:
        while self.state.phase != Phase.GAME_OVER:
            self.step()
        return self.state

    def _agent_state(self, me: str) -> Any:
        s = self.state
        return SimpleNamespace(
            players=s.players[:],
            me=me,
            my_role=self.player_info[me].role,
            my_alignment=self.player_info[me].alignment,
            private_knowledge=dict(self.player_info[me].private_knowledge),
            round_idx=s.round_idx,
            proposal_idx=s.proposal_idx,
            current_proposer=s.leader,
            current_team=s.current_team[:],
            chat_log=list(s.chat_log),
            missions=list(s.missions),
            required_team_size=self.cfg.team_sizes[s.round_idx - 1],
            extra={"last_team_votes": dict(s.last_team_votes), "last_quest_fails": s.last_quest_fails},
        )

    def _call_agent(self, name: str, fn_name: str, *args) -> Any:
        agent = self.agents[name]
        try:
            if hasattr(agent, fn_name):
                return getattr(agent, fn_name)(*args)
            if self.cfg.strict_agent_errors:
                raise RuntimeError(f"Agent {name} missing method {fn_name}")
        except Exception as exc:
            if self.cfg.strict_agent_errors:
                raise RuntimeError(f"Agent call failed for {name}.{fn_name}: {exc}") from exc
        return self._default_action(name, fn_name, *args)

    @staticmethod
    def _sanitize_public_message(msg: str) -> str:
        text = (msg or "").strip()
        # Normalize to one printable line.
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        # Strip approve/reject tags; discussion is pitch-only, no required lean.
        text = re.sub(r"\s*(APPROVE_LEAN|REJECT_LEAN)\s*", " ", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            return ""
        return text

    def _finalize_discussion_message(self, speaker: str, msg: str) -> str:
        s = self.state
        cleaned = self._sanitize_public_message(msg)
        recent_lines = [m for _, m in s.chat_log[-4:]]
        if (not cleaned) or cleaned in recent_lines:
            retry = self._call_agent(speaker, "speak", self._agent_state(speaker))
            retry_msg = str(retry.get("message", "")) if isinstance(retry, dict) else str(retry)
            retry_clean = self._sanitize_public_message(retry_msg)
            if retry_clean and retry_clean not in recent_lines:
                cleaned = retry_clean
        if not cleaned:
            cleaned = "I'll support the group's choice."
        return self._sanitize_public_message(cleaned)

    def _default_action(self, name: str, fn_name: str, *args) -> Any:
        if fn_name == "speak":
            return {"message": "I'll support the group's choice.", "reasoning": "", "proposed_team": []}
        if fn_name == "propose_team":
            state, team_size = args
            return self._validate_team([state.me], int(team_size), state.current_proposer)
        if fn_name == "vote_on_team":
            return True
        if fn_name == "mission_action":
            # Good pass, evil strategic fail.
            return self.player_info[name].alignment != "evil" or self.state.score_good < 2
        if fn_name == "assassinate":
            for p in self.players:
                if p != name:
                    return p
            return name
        return None

    @staticmethod
    def _as_bool_vote(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            value = value.get("approve", value.get("decision", value.get("vote", False)))
        value = getattr(value, "value", value)
        if isinstance(value, str):
            t = value.strip().lower()
            if t in {"approve", "yes", "y", "true", "support"}:
                return True
            if t in {"reject", "no", "n", "false", "deny"}:
                return False
        return bool(value)

    @staticmethod
    def _as_bool_mission(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            value = value.get("success", value.get("action", value.get("vote", True)))
        value = getattr(value, "value", value)
        if isinstance(value, str):
            t = value.strip().lower()
            if t in {"pass", "success", "succeed", "true", "yes"}:
                return True
            if t in {"fail", "failure", "false", "no"}:
                return False
        return bool(value)

    def _validate_team(self, team: List[str], size: int, leader: str) -> List[str]:
        seen = set()
        cleaned: List[str] = []
        for p in team or []:
            if p in self.players and p not in seen:
                cleaned.append(p)
                seen.add(p)
        if leader in self.players and leader not in seen and len(cleaned) < size:
            cleaned.append(leader)
            seen.add(leader)
        for p in self.players:
            if len(cleaned) >= size:
                break
            if p not in seen:
                cleaned.append(p)
                seen.add(p)
        return cleaned[:size]

    def _advance_leader(self) -> None:
        self._leader_idx = (self._leader_idx + 1) % len(self.players)
        self.state.leader = self.players[self._leader_idx]

    def _find_holder(self, role_substring: str) -> Optional[str]:
        key = role_substring.lower()
        for n, info in self.player_info.items():
            if key in info.role.lower():
                return n
        return None

    def _end_game(self, winner: str, reason: str) -> None:
        self.state.winner = winner
        self.state.reason = reason
        self.state.phase = Phase.GAME_OVER
        self.state.public_events.append(PublicEvent("game_over", {"winner": winner, "reason": reason}))
        self._log(f"GAME OVER -> winner={winner}, reason={reason}")

    def _log(self, line: str) -> None:
        self.state.transcript.append(line)
        if self.cfg.verbose:
            print(line)

    @staticmethod
    def _normalize_line_for_dedupe(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.sub(r"(approve_lean|reject_lean)", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _near_duplicate(self, a: str, b: str) -> bool:
        na = self._normalize_line_for_dedupe(a)
        nb = self._normalize_line_for_dedupe(b)
        if not na or not nb:
            return False
        return na == nb or na.startswith(nb[: max(1, min(len(nb), 80))]) or nb.startswith(na[: max(1, min(len(na), 80))])

    @staticmethod
    def _clean_reasoning_for_log(text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r"[\r\n\t]+", " ", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if len(t) > 220:
            t = t[:220].rstrip(" .") + "..."
        return t

    def step(self) -> None:
        s = self.state
        if s.phase == Phase.GAME_OVER:
            return

        if s.phase == Phase.DISCUSSION:
            self._log(f"\n[ROUND {s.round_idx}] Discussion phase")
            turn_names = self.players[:]
            if self.cfg.discussion_turns is not None:
                turn_names = turn_names[: self.cfg.discussion_turns]
            for name in turn_names:
                raw = self._call_agent(name, "speak", self._agent_state(name))
                if isinstance(raw, dict):
                    msg = self._finalize_discussion_message(name, str(raw.get("message", "")))
                    reasoning = self._clean_reasoning_for_log(str(raw.get("reasoning", "")))
                    if reasoning:
                        self._log(f"- [{name} reasoning] {reasoning}")
                else:
                    msg = self._finalize_discussion_message(name, str(raw))
                if s.chat_log and s.chat_log[-1][0] == name and self._near_duplicate(s.chat_log[-1][1], msg):
                    prev = s.chat_log[-1][1]
                    s.chat_log[-1] = (name, msg if len(msg) >= len(prev) else prev)
                else:
                    s.chat_log.append((name, msg))
                self._log(f"- {name} says: {msg}")
            s.public_events.append(PublicEvent("discussion_done", {"round_idx": s.round_idx}))
            s.phase = Phase.PROPOSE
            return

        if s.phase == Phase.PROPOSE:
            leader = s.leader
            team_size = self.cfg.team_sizes[s.round_idx - 1]
            self._log(f"[ROUND {s.round_idx}] Proposal phase (leader={leader}, team_size={team_size})")
            raw_team = self._call_agent(leader, "propose_team", self._agent_state(leader), team_size)
            if isinstance(raw_team, dict):
                raw_team = raw_team.get("team", [])
            team = self._validate_team(list(raw_team or []), team_size, leader)
            assert len(team) == team_size, f"Avalon requires exact team size {team_size} for this round, got {len(team)}"
            s.current_team = team
            s.proposals.append(
                ProposalRecord(
                    round_idx=s.round_idx,
                    proposal_idx=s.proposal_idx,
                    proposer=leader,
                    team=team[:],
                )
            )
            s.public_events.append(PublicEvent("team_proposed", {"leader": leader, "team": team}))
            self._log(f"- Team proposed: {team}")
            # If post-proposal discussion turns are zero, skip directly to vote.
            if self.cfg.post_proposal_discussion_turns == 0:
                s.phase = Phase.TEAM_VOTE
            else:
                s.phase = Phase.TEAM_DISCUSSION
            return

        if s.phase == Phase.TEAM_DISCUSSION:
            self._log(f"[ROUND {s.round_idx}] Post-proposal discussion (team={s.current_team})")
            # Start from leader, then cycle. This gives rebuttal/defense before vote.
            turn_names = self.players[self._leader_idx :] + self.players[: self._leader_idx]
            if self.cfg.post_proposal_discussion_turns is not None:
                turn_names = turn_names[: self.cfg.post_proposal_discussion_turns]
            for name in turn_names:
                raw = self._call_agent(name, "speak", self._agent_state(name))
                if isinstance(raw, dict):
                    msg = self._finalize_discussion_message(name, str(raw.get("message", "")))
                    reasoning = self._clean_reasoning_for_log(str(raw.get("reasoning", "")))
                    if reasoning:
                        self._log(f"- [{name} reasoning] {reasoning}")
                else:
                    msg = self._finalize_discussion_message(name, str(raw))
                if s.chat_log and s.chat_log[-1][0] == name and self._near_duplicate(s.chat_log[-1][1], msg):
                    prev = s.chat_log[-1][1]
                    s.chat_log[-1] = (name, msg if len(msg) >= len(prev) else prev)
                else:
                    s.chat_log.append((name, msg))
                self._log(f"- {name} (post-proposal) says: {msg}")
            s.public_events.append(
                PublicEvent(
                    "post_proposal_discussion_done",
                    {"round_idx": s.round_idx, "proposal_idx": s.proposal_idx, "team": s.current_team[:]},
                )
            )
            s.phase = Phase.TEAM_VOTE
            return

        if s.phase == Phase.TEAM_VOTE:
            self._log(f"[ROUND {s.round_idx}] Team vote phase")

            votes: Dict[str, bool] = {}
            for name in self.players:
                raw_vote = self._call_agent(name, "vote_on_team", self._agent_state(name), s.current_team[:])
                votes[name] = self._as_bool_vote(raw_vote)
                self._log(f"- {name} votes: {'APPROVE' if votes[name] else 'REJECT'}")
            assert len(votes) == len(self.players), f"Must have exactly one vote per player, got {len(votes)}"
            approved = sum(1 for v in votes.values() if v) > (len(self.players) // 2)
            s.last_team_votes = votes
            if s.proposals:
                s.proposals[-1].votes = dict(votes)
                s.proposals[-1].approved = approved
            s.public_events.append(PublicEvent("team_vote", {"team": s.current_team[:], "votes": votes, "approved": approved}))
            self._log(f"- Vote result: {'APPROVED' if approved else 'REJECTED'}")

            if approved:
                s.phase = Phase.QUEST
                return

            s.proposal_idx += 1
            if s.proposal_idx > self.cfg.max_rejects:
                # Official Avalon rule: if 5th proposal is rejected, Evil wins immediately.
                s.rejected_rounds_total += 1
                s.public_events.append(
                    PublicEvent(
                        "game_over_by_rejections",
                        {
                            "round_idx": s.round_idx,
                            "proposal_idx": self.cfg.max_rejects,
                            "reason": "fifth_rejected_proposal",
                        },
                    )
                )
                self._end_game("evil", "five_failed_votes")
                return
            self._advance_leader()
            self._log(f"- Next leader: {self.state.leader}")
            s.phase = Phase.DISCUSSION
            return

        if s.phase == Phase.QUEST:
            team = s.current_team[:]
            self._log(f"[ROUND {s.round_idx}] Quest phase (team={team})")
            successes = 0
            for name in team:
                raw_action = self._call_agent(name, "mission_action", self._agent_state(name))
                is_success = self._as_bool_mission(raw_action)
                if self.player_info[name].alignment == "good":
                    is_success = True  # Only evil can choose to fail; Merlin/Servant cannot fail
                assert self.player_info[name].alignment != "good" or is_success, f"Good player {name} cannot fail the quest"
                if is_success:
                    successes += 1
                self._log(f"- {name} mission action: {'PASS' if is_success else 'FAIL'}")
            fails = len(team) - successes
            fail_threshold = self.cfg.fails_required[s.round_idx - 1]
            assert 1 <= s.round_idx <= len(self.cfg.fails_required), f"Round index {s.round_idx} out of range for fail threshold"
            mission_failed = fails >= fail_threshold
            s.last_quest_fails = fails
            s.missions.append(
                MissionRecord(
                    round_idx=s.round_idx,
                    team=team,
                    fail_count=fails,
                    outcome=("fail" if mission_failed else "success"),
                )
            )
            s.public_events.append(
                PublicEvent(
                    "quest_result",
                    {"round_idx": s.round_idx, "team": team, "fails": fails, "success": (not mission_failed)},
                )
            )
            self._log(f"- Quest result: {'FAILED' if mission_failed else 'SUCCESS'} (fails={fails}, threshold={fail_threshold})")

            if mission_failed:
                s.score_evil += 1
            else:
                s.score_good += 1
            self._log(f"- Score now -> Good {s.score_good} : Evil {s.score_evil}")

            if s.score_evil >= 3:
                self._end_game("evil", "three_failed_quests")
                return
            if s.score_good >= 3:
                self._log("- Good reached 3 successes. Moving to assassination phase.")
                s.phase = Phase.ASSASSINATE
                return

            s.round_idx += 1
            if s.round_idx > len(self.cfg.team_sizes):
                # Safety: if this ever happens, decide by current score.
                winner = "good" if s.score_good >= s.score_evil else "evil"
                self._end_game(winner, "max_rounds")
                return
            s.proposal_idx = 1
            s.current_team = []
            self._advance_leader()
            self._log(f"- Next round leader: {self.state.leader}")
            s.phase = Phase.DISCUSSION
            return

        if s.phase == Phase.ASSASSINATE:
            self._log("[ENDGAME] Assassination phase")
            assassin = self._find_holder("assassin")
            merlin = self._find_holder("merlin")
            if assassin is None or merlin is None:
                self._end_game("good", "no_assassin_or_merlin")
                return

            target_raw = self._call_agent(assassin, "assassinate", self._agent_state(assassin))
            if isinstance(target_raw, dict):
                target_raw = target_raw.get("target", target_raw.get("guess", ""))
            if isinstance(target_raw, int):
                target = self.players[target_raw] if 0 <= target_raw < len(self.players) else ""
            else:
                target = str(target_raw)
            if target == assassin or target not in self.players:
                target = next((p for p in self.players if p != assassin), assassin)

            hit = target == merlin
            s.public_events.append(
                PublicEvent("assassination", {"assassin": assassin, "target": target, "hit_merlin": hit})
            )
            self._log(f"- Assassin {assassin} guess: {target}; correct (hit Merlin): {hit}")
            s.transcript.append(f"[ASSASSINATION] guess={target}, correct={hit}")
            self._end_game("evil" if hit else "good", ("assassination_hit_merlin" if hit else "assassination_missed"))
            return

        raise RuntimeError(f"Unknown phase: {s.phase}")