# -*- coding: utf-8 -*-
"""avalon_env.ipynb

Design:
- Agent-centric: env takes a list of agent objects (each with .name), not player indices.
- Phase flow: discussion -> proposal -> voting -> mission -> repeat; after 3 good wins, assassination.
- Two agent interfaces supported:
  1) Role API (preferred): speak(state), propose_team(state, team_size), vote_on_team(state, team),
     mission_action(state), assassinate(state). State is an object with .players, .me, .round_idx,
     .num_successes, .num_fails, .current_proposer, .current_team, .chat_log (matches roles/ notebooks).
  2) Fallback: act(system_prompt, user_prompt) -> str. Env sends JSON-shaped prompts and parses response.

Compatibility:
- roles/: Implement MerlinAgent/AssassinAgent etc. with the role API; state is SimpleNamespace with
  the fields above. No __main__ or notebook globals required.
- LLM_caller.AvalonAgent: Add act(system_prompt, user_prompt) that calls your LLM and returns the
  response; env will parse JSON for vote/proposal/mission/assassination. For discussion, either
  implement speak(state) (e.g. by mapping state to your history and calling get_consolidated_action)
  or the env will use act() with a speech prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import random
import json
import re
import asyncio
import inspect

"""Core enums / dataclasses"""

class TeamVote(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"

class MissionAction(str, Enum):
    PASS = "pass"
    FAIL = "fail"

class MissionOutcome(str, Enum):
    SUCCESS = "success"
    FAIL = "fail"

class Phase(str, Enum):
    DISCUSSION = "discussion"
    PROPOSAL = "proposal"
    VOTING = "voting"
    MISSION = "mission"
    ASSASSINATION = "assassination"
    GAME_OVER = "game_over"

@dataclass
class Proposal:
    round_idx: int
    proposal_idx: int
    proposer: str
    team: List[str]
    votes: Dict[str, TeamVote] = field(default_factory=dict)
    approved: Optional[bool] = None

@dataclass
class Mission:
    round_idx: int
    team: List[str]
    outcome: Optional[MissionOutcome] = None
    fail_count: Optional[int] = None

@dataclass
class AvalonConfig:
    # Standard Avalon 5-player mission sizes: 2,3,2,3,3
    mission_team_sizes: Dict[int, int] = field(default_factory=lambda: {1: 2, 2: 3, 3: 2, 4: 3, 5: 3})
    max_rounds: int = 5

    # If a proposal is rejected 5 times in a round, Evil auto-wins (standard rule).
    max_proposals_per_round: int = 5

    # Discussion: number of public statements per phase (simple cap)
    discussion_turn_cap: int = 8

    # Random seed for reproducibility
    seed: int = 0

@dataclass
class PlayerInfo:
    name: str
    role: str
    alignment: str  # "good" or "evil"
    # For routing private info. Merlin needs to know evil list (except maybe Mordred variant).
    private_knowledge: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PublicLogEvent:
    type: str
    payload: Dict[str, Any]

@dataclass
class GameState:
    # Public state
    players: List[str]
    round_idx: int
    proposal_idx: int
    leader: str
    score_good: int
    score_evil: int
    phase: Phase
    proposed_team: List[str] = field(default_factory=list)

    # Public logs (for analysis + to show to agents)
    chat_log: List[Tuple[str, str]] = field(default_factory=list)
    proposals: List[Proposal] = field(default_factory=list)
    missions: List[Mission] = field(default_factory=list)
    public_events: List[PublicLogEvent] = field(default_factory=list)

    # Optional extras for research
    extra: Dict[str, Any] = field(default_factory=dict)

"""JOSN parsing helpers (robust)"""

def _extract_json_obj(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    # direct parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # find first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

"""Environment"""

class AvalonEnv:
    """
    Owns: phases, legality checks, hidden info routing, and scoring.

    Philosophy:
    - Give agents a clean, explicit "state block"
    - Ask for a single JSON action depending on phase
    - Validate; if invalid -> fallback heuristic (so sims never crash)
    """

    def __init__(self, agents: List[Any], roles: Dict[str, str], config: Optional[AvalonConfig] = None):
        self.cfg = config or AvalonConfig()
        random.seed(self.cfg.seed)

        self.agents: Dict[str, Any] = {a.name: a for a in agents}
        self.players: List[str] = [a.name for a in agents]

        # Build player infos (alignment derived from role name; robust to casing/spacing)
        self.player_info: Dict[str, PlayerInfo] = {}
        for name in self.players:
            role = roles[name]
            rl = role.lower()
            alignment = "evil" if (
                "assassin" in rl or "minion" in rl or "morgana" in rl or "oberon" in rl or "mordred" in rl
            ) else "good"
            self.player_info[name] = PlayerInfo(name=name, role=role, alignment=alignment)

        # Private knowledge routing (robust to role string variants)
        evil_players = [n for n in self.players if self.player_info[n].alignment == "evil"]
        for name in self.players:
            r = self.player_info[name].role.lower()
            if "merlin" in r:
                self.player_info[name].private_knowledge["evil_players"] = [p for p in evil_players if p != name]
            if self.player_info[name].alignment == "evil":
                self.player_info[name].private_knowledge["evil_players"] = [p for p in evil_players if p != name]

        # Game state init
        self.state = GameState(
            players=list(self.players),
            round_idx=1,
            proposal_idx=1,
            leader=self.players[0],
            score_good=0,
            score_evil=0,
            phase=Phase.DISCUSSION,
        )

        self._leader_idx = 0
        self._game_over_reason: Optional[str] = None
    # ---------- Public API ----------

    async def initialize(self) -> None:
        for a in self.agents.values():
            if hasattr(a, "initialize_agent"):
                await self._call_maybe_async(a.initialize_agent)

    async def run_game(self) -> Dict[str, Any]:
        """
        Runs until GAME_OVER. Returns a structured summary dict.
        """
        while self.state.phase != Phase.GAME_OVER:
            await self.step()

        reason = self._game_over_reason
        if reason == "assassination_success":
            winner = "evil"
        elif self.state.score_good >= 3:
            winner = "good"
        elif self.state.score_evil >= 3 or reason == "too_many_rejections":
            winner = "evil"
        else:
            winner = "good" if self.state.score_good > self.state.score_evil else "evil"
        return {
            "winner": winner,
            "score_good": self.state.score_good,
            "score_evil": self.state.score_evil,
            "reason": reason,
            "chat_log": list(self.state.chat_log),
            "proposals": [p.__dict__ for p in self.state.proposals],
            "missions": [m.__dict__ for m in self.state.missions],
            "public_events": [{"type": e.type, "payload": e.payload} for e in self.state.public_events],
        }

    async def step(self) -> None:
        phase = self.state.phase
        if phase == Phase.DISCUSSION:
            await self._phase_discussion()
            self.state.phase = Phase.PROPOSAL
            return

        if phase == Phase.PROPOSAL:
            await self._phase_proposal()
            self.state.phase = Phase.VOTING
            return

        if phase == Phase.VOTING:
            approved = await self._phase_voting()
            if approved:
                self.state.phase = Phase.MISSION
            else:
                # rejected proposal -> rotate leader, maybe evil auto-win on 5 rejects
                if self.state.proposal_idx >= self.cfg.max_proposals_per_round:
                    self._end_game("too_many_rejections")
                else:
                    self._advance_leader()
                    self.state.proposal_idx += 1
                    self.state.phase = Phase.DISCUSSION
            return

        if phase == Phase.MISSION:
            await self._phase_mission()
            # Check win conditions
            if self.state.score_evil >= 3:
                self._end_game("evil_three_fails")
                return
            if self.state.score_good >= 3:
                # assassination stage
                self.state.phase = Phase.ASSASSINATION
                return

            # next round
            self.state.round_idx += 1
            if self.state.round_idx > self.cfg.max_rounds:
                # should rarely happen if scoring logic is correct
                self._end_game("max_rounds_reached")
                return
            self.state.proposal_idx = 1
            self._advance_leader()
            self.state.phase = Phase.DISCUSSION
            return

        if phase == Phase.ASSASSINATION:
            await self._phase_assassination()
            return

    # ---------- Phase implementations ----------

    async def _phase_discussion(self) -> None:
        """
        Simple: leader + a few others speak.
        You can later replace this with your fancy "mention-based next speaker" logic from LLM_caller.py :contentReference[oaicite:4]{index=4}.
        """
        turn_cap = self.cfg.discussion_turn_cap
        speakers = self._discussion_order(turn_cap)

        for name in speakers:
            msg = await self._ask_speech(name)
            self.state.chat_log.append((name, msg))
            self.state.public_events.append(PublicLogEvent(type="speech", payload={"speaker": name, "message": msg}))

    async def _phase_proposal(self) -> None:
        leader = self.state.leader
        team_size = self.cfg.mission_team_sizes[self.state.round_idx]

        team = await self._ask_proposal(leader, team_size=team_size)
        # validate + fallback
        team = self._validate_team(team, team_size, leader)

        self.state.proposed_team = team
        prop = Proposal(
            round_idx=self.state.round_idx,
            proposal_idx=self.state.proposal_idx,
            proposer=leader,
            team=team,
        )
        self.state.proposals.append(prop)
        self.state.public_events.append(PublicLogEvent(type="proposal", payload={"leader": leader, "team": team}))

    async def _phase_voting(self) -> bool:
        team = list(self.state.proposed_team)
        votes: Dict[str, TeamVote] = {}
        for name in self.players:
            v = await self._ask_vote(name, team)
            votes[name] = v

        approved = sum(1 for v in votes.values() if v == TeamVote.APPROVE) > (len(self.players) // 2)

        # write back into current proposal
        prop = self.state.proposals[-1]
        prop.votes = votes
        prop.approved = approved

        self.state.public_events.append(PublicLogEvent(
            type="vote_result",
            payload={"team": team, "votes": {k: v.value for k, v in votes.items()}, "approved": approved}
        ))

        return approved

    async def _phase_mission(self) -> None:
        team = list(self.state.proposed_team)

        # only team members submit mission actions; good must PASS; evil can choose
        actions: Dict[str, MissionAction] = {}
        for name in team:
            action = await self._ask_mission_action(name, team)
            actions[name] = action

        fail_count = sum(1 for a in actions.values() if a == MissionAction.FAIL)
        outcome = MissionOutcome.FAIL if fail_count > 0 else MissionOutcome.SUCCESS

        self.state.missions.append(Mission(round_idx=self.state.round_idx, team=team, outcome=outcome, fail_count=fail_count))
        self.state.public_events.append(PublicLogEvent(
            type="mission_result",
            payload={"round": self.state.round_idx, "team": team, "fail_count": fail_count, "outcome": outcome.value}
        ))

        if outcome == MissionOutcome.SUCCESS:
            self.state.score_good += 1
        else:
            self.state.score_evil += 1

        # clear proposal team
        self.state.proposed_team = []

    async def _phase_assassination(self) -> None:
        assassin = self._find_role("assassin")
        if assassin is None:
            # if no assassin role exists in this run, good just wins
            self._end_game("no_assassin_present")
            return

        target = await self._ask_assassinate(assassin)
        merlin = self._find_role("merlin")

        if merlin is not None and target == merlin:
            self._end_game("assassination_success")  # evil steals win
        else:
            self._end_game("assassination_fail")     # good keeps win

    # ---------- Prompts / action requests ----------

    def _public_state_block(self) -> str:
        return (
            f"Players: {', '.join(self.players)}\n"
            f"Round: {self.state.round_idx} | Proposal: {self.state.proposal_idx}\n"
            f"Leader: {self.state.leader}\n"
            f"Score: Good={self.state.score_good}, Evil={self.state.score_evil}\n"
            f"Current proposed team: {self.state.proposed_team}\n"
        )

    def _private_block(self, me: str) -> str:
        info = self.player_info[me]
        role = info.role
        pk = info.private_knowledge
        return (
            f"YOU ARE: {me}\n"
            f"SECRET ROLE: {role}\n"
            f"PRIVATE KNOWLEDGE: {json.dumps(pk)}\n"
        )

    def _recent_chat(self, k: int = 12) -> str:
        recent = self.state.chat_log[-k:]
        if not recent:
            return "(no chat yet)"
        return "\n".join([f"{s}: {m}" for s, m in recent])

    async def _ask_speech(self, me):
        agent = self.agents[me]

        if self._has_role_api(agent) and hasattr(agent, "speak"):
            role_state = self._build_role_state(me)
            msg = await self._call_maybe_async(agent.speak, role_state)
            return msg if msg else "Let’s evaluate more carefully."

        # fallback: use Protocol act() with JSON prompt
        return await self._ask_act_json(me, "speech", {"message": "your discussion message (1-3 sentences)"})

    async def _ask_proposal(self, me, team_size):
        agent = self.agents[me]

        if self._has_role_api(agent) and hasattr(agent, "propose_team"):
            role_state = self._build_role_state(me)
            team = await self._call_maybe_async(agent.propose_team, role_state, team_size)
            return list(team) if isinstance(team, list) else []

        # fallback: use Protocol act() with JSON prompt; always validate before return
        team = await self._ask_act_json(me, "propose_team", {"team": f"list of exactly {team_size} player names"})
        team = list(team)[:team_size] if isinstance(team, list) else []
        return self._validate_team(team, team_size, me)

    async def _ask_vote(self, me, team):
        agent = self.agents[me]

        if self._has_role_api(agent) and hasattr(agent, "vote_on_team"):
            role_state = self._build_role_state(me)
            decision = await self._call_maybe_async(agent.vote_on_team, role_state, team)
            val = getattr(decision, "value", decision)
            val = str(val).lower()
            if val == "approve":
                return TeamVote.APPROVE
            return TeamVote.REJECT

        raw = await self._ask_act_json(me, "vote", {"decision": "approve or reject"}, extra=f"Team: {team}")
        val = (str(raw).strip().lower() if raw else "reject")
        return TeamVote.APPROVE if val == "approve" else TeamVote.REJECT

    async def _ask_mission_action(self, me, team):
        # Avalon: good can only PASS; only evil may choose FAIL
        alignment = self.player_info[me].alignment

        agent = self.agents[me]
        if self._has_role_api(agent) and hasattr(agent, "mission_action"):
            role_state = self._build_role_state(me)
            action = await self._call_maybe_async(agent.mission_action, role_state)
            val = getattr(action, "value", action)
            val = str(val).lower()
            if alignment == "good":
                return MissionAction.PASS
            return MissionAction.FAIL if val == "fail" else MissionAction.PASS

        if alignment == "good":
            return MissionAction.PASS
        raw = await self._ask_act_json(me, "mission_action", {"action": "pass or fail (evil only)"})
        val = (str(raw).strip().lower() if raw else "pass")
        return MissionAction.FAIL if val == "fail" else MissionAction.PASS

    async def _ask_assassinate(self, assassin):
        agent = self.agents[assassin]

        if self._has_role_api(agent) and hasattr(agent, "assassinate"):
            role_state = self._build_role_state(assassin)
            target = await self._call_maybe_async(agent.assassinate, role_state)
            if target in self.players and target != assassin:
                return target
            for p in self.players:
                if p != assassin:
                    return p
            return assassin

        target = await self._ask_act_json(assassin, "assassinate", {"target": "player name you guess is Merlin"})
        if target in self.players and target != assassin:
            return target
        for p in self.players:
            if p != assassin:
                return p
        return assassin

    async def _ask_act_json(self, me: str, action_type: str, json_shape: Dict[str, str], extra: str = "") -> Any:
        """Fallback: call agent.act() with system/user prompts and parse JSON response."""
        agent = self.agents[me]
        if not hasattr(agent, "act"):
            return None
        system_prompt = (
            self._public_state_block() + "\n" + self._private_block(me)
            + "\n\nRespond with a single JSON object. No other text."
        )
        user_prompt = (
            f"Action type: {action_type}. Required JSON shape: {json.dumps(json_shape)}\n"
            f"Recent chat:\n{self._recent_chat()}\n"
        )
        if action_type == "vote":
            user_prompt += 'Valid values for "decision": exactly "approve" or "reject".\n'
        elif action_type == "mission_action":
            user_prompt += 'Valid values for "action": exactly "pass" or "fail".\n'
        elif action_type == "propose_team":
            user_prompt += (
                f'"team" must be a JSON array of valid player names from {self.players}. '
                "Length must match the required mission team size for this round. No extra keys. No explanation.\n"
            )
        if extra:
            user_prompt += f"\n{extra}\n"
        user_prompt += "Output only the JSON object."
        raw = await agent.act(system_prompt=system_prompt, user_prompt=user_prompt)
        obj = _extract_json_obj(raw or "")
        if not obj:
            return None
        if action_type == "speech":
            return (obj.get("message") or "").strip() or "Let's evaluate more carefully."
        if action_type == "vote":
            return obj.get("decision")
        if action_type == "mission_action":
            return obj.get("action")
        if action_type == "assassinate":
            return obj.get("target")
        if action_type == "propose_team":
            t = obj.get("team")
            return list(t) if isinstance(t, list) else None
        return None

    def _has_role_api(self, agent):
        return all(hasattr(agent, m) for m in [
            "speak", "propose_team", "vote_on_team", "mission_action", "assassinate"
        ])

    async def _call_maybe_async(self, fn, *args):
        if inspect.iscoroutinefunction(fn):
            return await fn(*args)
        return await asyncio.to_thread(fn, *args)

    def _build_role_state(self, me):
        payload = dict(
            players=list(self.players),
            me=me,
            round_idx=self.state.round_idx,
            proposal_idx=self.state.proposal_idx,
            num_successes=self.state.score_good,
            num_fails=self.state.score_evil,
            current_proposer=self.state.leader,
            current_team=list(self.state.proposed_team),
            chat_log=list(self.state.chat_log),
            proposals=list(self.state.proposals),
            missions=list(self.state.missions),
        )
        return SimpleNamespace(**payload)

    # ---------- Utilities ----------

    def _discussion_order(self, cap: int) -> List[str]:
        # simple: start with leader, then cycle
        order = []
        idx = self._leader_idx
        for _ in range(min(cap, len(self.players) * 2)):
            order.append(self.players[idx % len(self.players)])
            idx += 1
        return order

    def _advance_leader(self) -> None:
        self._leader_idx = (self._leader_idx + 1) % len(self.players)
        self.state.leader = self.players[self._leader_idx]

    def _validate_team(self, team: List[str], team_size: int, leader: str) -> List[str]:
        # Keep only valid unique names in order
        seen = set()
        cleaned = []
        for p in team:
            if p in self.players and p not in seen:
                cleaned.append(p)
                seen.add(p)
        # pad if too small
        for p in self.players:
            if len(cleaned) >= team_size:
                break
            if p not in seen:
                cleaned.append(p)
                seen.add(p)
        # truncate if too large
        cleaned = cleaned[:team_size]
        # Ensure non-empty
        if len(cleaned) != team_size:
            # hard fallback: leader + next players
            cleaned = [leader]
            for p in self.players:
                if len(cleaned) >= team_size:
                    break
                if p != leader:
                    cleaned.append(p)
        return cleaned

    def _find_role(self, role_name: str) -> Optional[str]:
        role_name = role_name.lower()
        for n, info in self.player_info.items():
            if role_name in info.role.lower():
                return n
        return None

    def _end_game(self, reason: str) -> None:
        self._game_over_reason = reason
        self.state.public_events.append(PublicLogEvent(type="game_over", payload={"reason": reason}))
        self.state.phase = Phase.GAME_OVER