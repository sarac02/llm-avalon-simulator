from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm_caller import AvalonLLMCaller, _extract_json_obj


class AvalonRoleAgent:
    def __init__(
        self,
        name,
        role,
        llm,
        role_notes="",
        merlin_policy=None,
        use_rl=False,
    ):
        self.name = name
        self.role = role
        self.llm = llm
        self.role_notes = role_notes.strip()
        self.last_reasoning: str = ""
        self.merlin_policy = merlin_policy
        self.use_rl = use_rl

    def _system(self, state) -> str:
        notes = f"\nRole notes:\n{self.role_notes}\n" if self.role_notes else ""
        alignment = state.my_alignment
        tactical = (
            "As good: use mission results and voting patterns to find evil; express trust in players who've been on successful quests or voted consistently with good outcomes. "
            "As evil: you are on a team with the Assassin and other evil roles; your goal is to help evil win. Always act in evil's interest: (1) Support your allies—put them in good_suspects, suggest they go on the quest, approve lineups that include you or an ally so you can sabotage or build trust. (2) Deflect suspicion onto good players (e.g. after a failed mission, publicly blame a good player who was on the team, not your ally). (3) Never contradict public facts: if your ally was on a failed mission, do not say they 'have shown no trouble'—instead deflect (e.g. 'I suspect P1'). Blend in—never mention sabotage or allies in your public message. "
            "CRITICAL (evil only): Your evil_suspects list must NEVER include your allies or yourself (see private_knowledge known_evil_players). Put allies in good_suspects and advocate for them to be on the team. When an ally was on a failed quest, deflect suspicion onto a good player in your public message; in private reasoning you know your ally sabotaged. "
            "Your reasoning is private (logged, not spoken): if you are evil, your reasoning MUST state your evil strategy—who you are deflecting onto, who you are protecting, supporting your ally so they get on the team—so you do not sound like a clueless good player; only the public message must not reveal your role. "
            "Sound like real players: reference specific events (e.g. 'last round P0 and P1 went and the mission failed, so one of them is likely evil'), react to what others said (agree, disagree, add your take), and build your argument from mission history and votes—not generic filler."
        )
        return (
            f"You are playing Avalon. You are {self.name}. Your true role and alignment are SECRET; the table only sees your public statements.\n\n"
            "## What you MUST do\n"
            "- Speak in character: base your words on public information only (mission outcomes, who proposed/voted how, what others said).\n"
            "- Express clear reads: who you trust (good_suspects) and who you find suspicious (evil_suspects), with brief reasoning.\n"
            "- Be specific: name players by id (P0, P1, …) and tie your view to events (e.g. 'P2 was on the failed mission').\n\n"
            "## What you must NEVER do in your public message\n"
            "- Reveal your role (Merlin, Assassin, Minion, etc.) or say you are good/evil.\n"
            "- Refer to yourself in the third person: you are " + self.name + ", so do not say '" + self.name + " thinks' or '" + self.name + " is trustworthy'—say 'I think', 'I'm', 'I trust'.\n"
            "- Claim to 'know' who is good or evil; only say who you suspect or trust based on public evidence.\n"
            "- Mention sabotage, failing on purpose, or coordinating with teammates in a way that implies you are evil.\n"
            "- Use private knowledge (evil_players, merlin_candidates, etc.) in your wording—use it only in your internal reasoning.\n\n"
            f"## Your situation\n"
            f"Your identity: {self.name}. Secret role: {state.my_role}. Alignment: {alignment}.\n"
            f"{tactical}\n\n"
            f"Private knowledge (for your reasoning only; never leak in your message): {json.dumps(state.private_knowledge)}.{notes}"
        )

    def _game_context_summary(self, state) -> str:
        """Build a concise summary of missions and recent proposals for discussion context."""
        parts: List[str] = []
        if state.missions:
            parts.append("Missions so far (use these to form and state your reads—e.g. who was on a failed quest):")
            for m in state.missions[-5:]:  # last 5 missions
                team = list(getattr(m, "team", []) or [])
                fails = int(getattr(m, "fail_count", 0))
                outcome = "SUCCESS" if fails == 0 else f"FAILED ({fails} fail(s))"
                parts.append(f"  Round {getattr(m, 'round_idx', '?')}: team {team} -> {outcome}.")
            last = state.missions[-1]
            lt = list(getattr(last, "team", []) or [])
            lf = int(getattr(last, "fail_count", 0))
            parts.append(f"  → Last mission: {lt} -> {'FAILED' if lf > 0 else 'SUCCESS'}. Reference this in your message when relevant.")
        if state.proposals:
            for p in state.proposals[-3:]:  # last 3 proposals
                team = list(getattr(p, "team", []) or [])
                proposer = getattr(p, "proposer", "?")
                approved = getattr(p, "approved", None)
                if approved is not None:
                    result = "approved" if approved else "rejected"
                    votes = getattr(p, "votes", {}) or {}
                    yay = sum(1 for v in votes.values() if v)
                    parts.append(f"  Proposal by {proposer}: {team} -> {result} (votes: {yay}/{len(votes)}).")
                else:
                    parts.append(f"  Proposal by {proposer}: {team} (vote pending).")
        return "\n".join(parts) if parts else "No missions or proposals yet."

    def _chat(self, state, k: int = 32) -> str:
        """Recent chat for context; no cap so agents can reference full discussion."""
        recent = state.chat_log[-k:]
        if not recent:
            return "(no chat yet)"
        return "\n".join([f"{s}: {m}" for s, m in recent])

    @staticmethod
    def _clean_text(text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r"[\r\n\t]+", " ", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        m = re.search(r"(APPROVE_LEAN|REJECT_LEAN)", t)
        if m:
            t = t[: m.end()]
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
        dedup: List[str] = []
        for p in parts:
            if p not in dedup:
                dedup.append(p)
        return " ".join(dedup).strip()

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        low = (text or "").strip().lower()
        markers = [
            "rewrite this",
            "include at least one explicit player id",
            "output is a single valid json object",
            "important:",
            "return strict json",
        ]
        return any(m in low for m in markers)

    @staticmethod
    def _mentions_player(text: str) -> bool:
        return bool(re.search(r"\bP\d+\b", text or ""))

    @staticmethod
    def _infer_suspects_from_message(msg: str, players_set: set) -> tuple:
        """Infer good_suspects and evil_suspects from message text when the model returned empty arrays."""
        text = (msg or "").lower()
        good_words = ("reliable", "solid", "trust", "agree", "comfortable", "good", "support", "send", "want", "trustworthy", "rely")
        evil_words = ("uneasy", "wary", "suspicious", "watch", "eye on", "worried", "doubt", "quiet", "bit off")
        # Match P# in original message to preserve case; fallback to lowercase match on text
        raw_found = re.findall(r"\b(P\d+)\b", msg or "", re.IGNORECASE)
        # Normalize to canonical id from players_set (e.g. P0, P1) so we don't drop due to case
        def canonical(pid: str):
            return next((x for x in players_set if x.upper() == (pid or "").upper()), None)
        found = [canonical(p) for p in raw_found if canonical(p)]
        seen: set = set()
        good: List[str] = []
        evil: List[str] = []
        for pid in found:
            if pid in seen:
                continue
            seen.add(pid)
            idx = text.find(pid.lower())
            if idx < 0:
                continue
            start = max(0, idx - 60)
            end = min(len(text), idx + 60)
            window = text[start:end]
            if any(w in window for w in evil_words):
                evil.append(pid)
            elif any(w in window for w in good_words):
                good.append(pid)
            else:
                good.append(pid)  # default to good if unclear
        if not good and not evil and found:
            good.append(found[0])
        return (list(dict.fromkeys(good)), list(dict.fromkeys(evil)))

    @staticmethod
    def _evil_pass_reasoning_ok(reasoning: str) -> bool:
        """True if reasoning acknowledges being evil and passing to fool (required when evil chooses pass)."""
        r = (reasoning or "").strip().lower()
        if len(r) < 20:
            return False
        # Must suggest evil is aware of passing for strategic reasons (fool, blend, avoid suspicion, etc.)
        markers = (
            "evil", "fool", "suspicion", "blend", "pretend", "avoid suspicion",
            "look good", "seem good", "build trust", "they trust", "passing to",
            "so they think", "not reveal", "hide", "undercover",
        )
        return any(m in r for m in markers)

    @staticmethod
    def _is_generic_text(text: str) -> bool:
        low = (text or "").lower()
        return any(
            x in low
            for x in [
                "vote consistency",
                "mission outcomes",
                "moving the game state forward",
                "controlled test",
                "we should evaluate this team",
                "i list good",
                "based on current reads",
            ]
        )

    @staticmethod
    def _message_suggests_approve(msg: str) -> bool:
        """True if message content clearly argues for approving (so lean should be approve)."""
        t = (msg or "").lower()
        approve_phrases = ("i trust", "i approve", "solid team", "good chance", "we should go", "comfortable with", "support this", "in favor")
        reject_phrases = ("don't trust", "reject", "suspicious", "against", "would not send", "uneasy", "block this")
        if any(r in t for r in reject_phrases):
            return False
        return any(a in t for a in approve_phrases)

    @staticmethod
    def _message_suggests_reject(msg: str) -> bool:
        """True if message content clearly argues for rejecting."""
        t = (msg or "").lower()
        reject_phrases = ("don't trust", "reject", "suspicious", "against", "would not send", "uneasy", "block this", "not comfortable")
        approve_phrases = ("i trust", "i approve", "solid team", "good chance", "we should go", "comfortable with")
        if any(a in t for a in approve_phrases):
            return False
        return any(r in t for r in reject_phrases)

    def _force_json(self, *, system: str, user: str, schema: str, max_tokens: int) -> Dict[str, Any]:
        
        # PRINT PROMPTS OUT
        print("\n" + "="*80)
        print(f"PROMPT FOR {self.name} ({self.role})")
        print("="*80)
        print("SYSTEM PROMPT:")
        print(system)
        print("\nUSER PROMPT:")
        print(user)
        print("="*80 + "\n")

        raw = self.llm.generate(system=system, user=user, max_tokens=max_tokens)
        obj = _extract_json_obj(raw)
        if obj:
            return obj
        retry = self.llm.generate(
            system=system,
            user=user + "\nIMPORTANT: output one valid JSON object only.",
            max_tokens=max_tokens + 80,
        )
        obj2 = _extract_json_obj(retry)
        if obj2:
            return obj2
        repair = self.llm.generate(
            system="You are a JSON formatter.",
            user=f"Convert this into strict JSON schema {schema}: {retry or raw}",
            max_tokens=220,
        )
        return _extract_json_obj(repair)

    def _known_evil_players(self, state) -> set[str]:
        known = state.private_knowledge.get("evil_players", [])
        return {p for p in known if isinstance(p, str)} if isinstance(known, list) else set()

    @staticmethod
    def _team_key(team: List[str]) -> tuple:
        return tuple(sorted(team))

    def _seen_clean_team(self, state, team: List[str]) -> bool:
        key = self._team_key(team)
        for m in reversed(state.missions):
            mt = list(getattr(m, "team", []) or [])
            if self._team_key(mt) == key and int(getattr(m, "fail_count", 1)) == 0:
                return True
        return False

    def _seen_failed_team(self, state, team: List[str]) -> bool:
        key = self._team_key(team)
        for m in reversed(state.missions):
            mt = list(getattr(m, "team", []) or [])
            if self._team_key(mt) == key and int(getattr(m, "fail_count", 0)) > 0:
                return True
        return False

    def _known_evil_in_team(self, state, team: List[str]) -> bool:
        return any(p in self._known_evil_players(state) for p in team)

    def _default_lean(self, state, team_override: List[str] | None = None) -> str:
        team = list(team_override if team_override is not None else (getattr(state, "current_team", []) or []))
        if not team:
            return "approve" if ((sum(ord(c) for c in self.name) + state.round_idx + state.proposal_idx) % 2 == 0) else "reject"
        if state.my_alignment == "good":
            if self._known_evil_in_team(state, team):
                return "reject"
            if self._seen_clean_team(state, team):
                return "approve"
            if self._seen_failed_team(state, team) and state.proposal_idx < 4:
                return "reject"
            if state.proposal_idx >= 4:
                return "approve"
        else:
            known = self._known_evil_players(state) | {self.name}
            if any(p in known for p in team):
                return "approve"
        return "reject"

    def _human_context_summary(self, state) -> str:
        if not state.missions:
            return "Early game and little hard evidence."
        m = state.missions[-1]
        fails = int(getattr(m, "fail_count", 0))
        team = list(getattr(m, "team", []) or [])
        if fails == 0:
            return f"The last mission with {team} succeeded cleanly."
        return f"The last mission with {team} failed, so someone there is suspicious."

    def speak(self, state):
        team_size = getattr(state, "team_size", 2)
        if state.current_team:
            team_context = (
                f"This round requires exactly {team_size} players. "
                f"The proposed team has exactly {len(state.current_team)} members: {state.current_team}. "
                "Do not refer to a different number of players or invent team members."
            )
        else:
            team_context = (
                f"This round the team will have exactly {team_size} members. "
                "CRITICAL: There is NO proposed team right now. The leader will propose a team only AFTER this discussion. "
                "Do NOT mention 'the proposed team', 'this team', 'the lineup', or any specific roster—none exists yet. Only discuss who you think should go or your general stance toward the leader."
            )
        game_summary = self._game_context_summary(state)
        recent_chat = self._chat(state)
        merlin_rl_hint = self._format_merlin_rl_hint(state)

        if not state.current_team:
            mission_nudge = ""
            if state.missions:
                last_m = state.missions[-1]
                lteam = list(getattr(last_m, "team", []) or [])
                lf = int(getattr(last_m, "fail_count", 0))
                mission_nudge = (
                    f" Last round we sent {lteam} and the mission {'FAILED' if lf > 0 else 'succeeded'}. "
                    f"You MUST reference this like real players: e.g. 'Last time {lteam} went and it failed—one of them is likely evil'; "
                    "'I don't trust P1 after that'; 'P2 said P1 is suspicious and I agree.' React to what others just said (agree, disagree, add your take). "
                    "Engaging conversation—not one-line generic replies.\n"
                )
            evil_encourage = ""
            if state.my_alignment == "evil":
                evil_encourage = " As evil: support your allies—suggest including yourself or allies in who should go; put allies in good_suspects. If the last mission failed with your ally on it, deflect publicly onto a good player who was on that team (e.g. say you suspect P1), not your ally; in reasoning you can state you are protecting your ally. Act in evil's interest.\n"
            phase_guidance = (
                "PHASE: Discussion (before any team is proposed). The leader will propose a team only after everyone speaks.\n"
                "PURPOSE: Open table talk. (1) Say who you're suspicious of or trust and why (tie to mission history if any). (2) Suggest who should go on the quest (e.g. 'I'd send P1 and P3'). (3) React to what others said—agree, disagree, or build on their point.\n"
                "EFFECT: Your words shape what team the leader will propose—so name specific players (P#) and reference past missions/votes when relevant. Do NOT end with APPROVE_LEAN or REJECT_LEAN.\n"
                + mission_nudge
                + evil_encourage
                + "Do NOT refer to a 'proposed team' or lineup; none exists yet. Just talk about who you trust, who you're wary of, and who you'd send."
            )
        else:
            evil_on_team_nudge = ""
            if state.my_alignment == "evil" and state.current_team:
                on_team = self.name in state.current_team
                ally_on_team = any(p in state.current_team for p in (state.private_knowledge.get("evil_players") or state.private_knowledge.get("known_evil_players") or []))
                if on_team:
                    evil_on_team_nudge = (
                        " You are ON this proposed team. Your reasoning MUST state your evil plan in private, e.g.: "
                        "'I encouraged them to put me on the team so I can sabotage and help evil win' or "
                        "'They put me on—I can fail this mission' or 'I'll pass to build trust so they keep sending me.' "
                        "Support this team (approve) so you get to go. Do not sound like a good player just happy to be picked.\n"
                    )
                elif ally_on_team:
                    evil_on_team_nudge = " One of your allies is ON this proposed team. Support this team (approve) so they get to go and can sabotage or build trust. Your reasoning can state this in private. In your message argue for approving without revealing anything.\n"
            phase_guidance = (
                "PHASE: Post-proposal discussion. A specific team is on the table; the vote happens right after this.\n"
                "PURPOSE: Table talk about this exact proposal. (1) Say clearly why you're for or against this lineup—reference mission history if relevant (e.g. 'P1 was on the failed quest last time'). (2) Try to sway others' votes. (3) Defend or question the proposer's choices.\n"
                "EFFECT: Your words shape whether this team gets approved or rejected. Take a clear position and end your message with APPROVE_LEAN or REJECT_LEAN.\n"
                + evil_on_team_nudge
                + "Reference the actual lineup (" + str(state.current_team) + ") and mission history. End your message with exactly APPROVE_LEAN or REJECT_LEAN."
            )

        extra = getattr(state, "extra", None) or {}
        retry_hint = extra.get("retry_hint", "").strip() if isinstance(extra, dict) else ""
        retry_block = f"\n\nImportant: {retry_hint}\n" if retry_hint else ""

        players_list = ", ".join(state.players)
        has_lean = bool(state.current_team)  # Post-proposal only: message ends with APPROVE_LEAN/REJECT_LEAN and we have lean
        user = (
            "Output a single JSON object with these exact keys: reasoning, message"
            + (', lean' if has_lean else '')
            + ", evil_suspects, good_suspects.\n"
            "REQUIRED: Do NOT return both evil_suspects and good_suspects as empty []. You must put at least one player id (e.g. P0, P1) in good_suspects and/or evil_suspects.\n\n"
            "--- CONTEXT ---\n"
            + (
                f"Round {state.round_idx}, proposal {state.proposal_idx}. Leader (will propose a team AFTER this discussion): {state.current_proposer}. No team exists yet.\n"
                if not state.current_team
                else f"Round {state.round_idx}, proposal {state.proposal_idx}. Leader proposed: {state.current_proposer}. A team is on the table.\n"
            )
            + f"{team_context}\n\n"
            "Game history:\n"
            f"{game_summary}\n\n"
            f"{merlin_rl_hint}"
            "What others just said (respond to specific points when relevant):\n"
            f"{recent_chat}\n\n"
            + (
                "Last round a mission was run—reference who was on it and whether it failed. In your message and reasoning: accuse or defend like real people (e.g. 'Last time P0 and P1 went and it failed, so I suspect one of them'; 'I agree with P2 that P1 is suspicious'). React to what others just said. No length cap—write as much as you need for a real conversation.\n\n"
                if state.missions else ""
            )
            + "--- YOUR TASK ---\n"
            f"{phase_guidance}\n\n"
            f"{retry_block}"
            "--- OUTPUT FORMAT ---\n"
            "• reasoning (string): 2–4 sentences of actual thinking: WHY you trust/suspect whom (reference mission history, voting, behavior). Do NOT just restate the lists. If you are evil and on the team, state e.g. 'I got myself on the team so I can sabotage' or 'I'll pass to build trust.' No length cap. Required.\n"
            + (
                "• message (string): What you say at the table. 2–4 sentences. Use 'I' for yourself (do not refer to " + self.name + " in third person). Persuade others; name other players by id (P0, P1). Do NOT end with APPROVE_LEAN or REJECT_LEAN. No length cap. Required.\n"
                if not state.current_team
                else "• message (string): What you say at the table. 2–4 sentences. Use 'I' for yourself (do not refer to " + self.name + " in third person). Make a case for or against this lineup; name other players by id. End with APPROVE_LEAN or REJECT_LEAN. No length cap. Required.\n"
                "• lean (string): \"approve\" or \"reject\". Must match the tag at the end of your message.\n"
            )
            + (
                "• good_suspects (array): Player ids you'd want on the quest or trust, e.g. [\"P0\", \"P3\"]. Required: list at least one player (who you'd send or trust).\n"
                "• evil_suspects (array): Player ids you're wary of or would not send, e.g. [\"P2\"]. Can be [] if you have no suspicion yet. You must list at least one player in good_suspects.\n"
                if not state.current_team
                else "• good_suspects (array): Player ids you trust (on or off this team), e.g. [\"P0\", \"P1\"]. Required: at least one.\n"
                "• evil_suspects (array): Player ids you suspect as evil, e.g. [\"P2\"]. Can be []. You must list at least one player in good_suspects or evil_suspects—do not return both empty.\n"
            )
            + f"\nValid player ids: [{players_list}]. Use only these exact ids. Do not return both arrays empty.\n"
            + ("If you are evil: do NOT put your allies or yourself (see private_knowledge known_evil_players) in evil_suspects; put allies in good_suspects to build cover. Never list yourself in evil_suspects.\n" if state.my_alignment == "evil" else "")
            + (
                f"Post-proposal: the team being voted on is {state.current_team}. List in good_suspects who you trust (e.g. from this team) and/or in evil_suspects who you suspect.\n"
                if state.current_team else ""
            )
            + "Give a distinct view; do not copy another player's phrasing."
        )
        schema = (
            '{"reasoning": string, "message": string, "lean":"approve|reject", "evil_suspects":[string], "good_suspects":[string]}'
            if has_lean
            else '{"reasoning": string, "message": string, "evil_suspects":[string], "good_suspects":[string]}'
        )
        obj = self._force_json(
            system=self._system(state),
            user=user,
            schema=schema,
            max_tokens=600,
        )
        lean = str(obj.get("lean", "")).strip().lower() if has_lean else ""
        if has_lean and lean not in {"approve", "reject"}:
            lean = self._default_lean(state)

        reasoning = self._clean_text(str(obj.get("reasoning", "") or "").strip())
        original_reasoning = reasoning
        if not reasoning or len(reasoning) < 18 or self._is_generic_text(reasoning):
            evil_hint = " If you are evil, your reasoning MUST state your evil strategy (e.g. who you're deflecting onto, who you're protecting, building trust with whom)—do not sound like a clueless good player." if state.my_alignment == "evil" else ""
            if state.my_alignment == "evil" and state.current_team and self.name in state.current_team:
                evil_hint = " You are ON the proposed team. Your reasoning MUST state your evil plan: e.g. 'I encouraged them to put me on the team so I can sabotage and help evil win' or 'They put me on—I can fail this mission' or 'I'll pass to build trust so they keep sending me.'"
            repair = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"reasoning": string}\n\n'
                    f"Round {state.round_idx}, proposal {state.proposal_idx}. Players: {state.players}. "
                    + (
                        "No team proposed yet. In 1–2 sentences state your read: which other players you think are good and which you suspect as evil, and why (e.g. behavior, voting, who was on the last failed mission). Use only public info. No role reveal."
                        if not state.current_team
                        else f"Proposed team: {state.current_team}. In 1–2 sentences state your read on the table (who you think is good vs evil) and why you lean approve or reject this lineup. No role reveal."
                    )
                    + evil_hint
                ),
                schema='{"reasoning": string}',
                max_tokens=280,
            )
            repair_reasoning = self._clean_text(str(repair.get("reasoning", "") or ""))
            # Keep repair only if it's better; otherwise keep original so we don't drop to empty
            if len(repair_reasoning) >= 18 and not self._is_generic_text(repair_reasoning):
                reasoning = repair_reasoning
            elif original_reasoning:
                reasoning = original_reasoning
            else:
                reasoning = repair_reasoning
        if not reasoning or len(reasoning) < 10:
            # One more try: get a single sentence of reasoning (include evil strategy if evil)
            q = (
                "which players you think are good vs evil and why, or your stance on the leader's upcoming proposal."
                if not state.current_team
                else "why you lean approve or reject this team and your read on who is good vs evil."
            )
            evil_q = " If you are evil, state your evil strategy (deflecting, protecting ally, building trust)." if state.my_alignment == "evil" else ""
            fallback = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"reasoning": string}\n\n'
                    f"Round {state.round_idx}. One sentence only: {q} Use only public information. No role reveal.{evil_q}"
                ),
                schema='{"reasoning": string}',
                max_tokens=500,
            )
            reasoning = self._clean_text(str(fallback.get("reasoning", "") or ""))
        self.last_reasoning = reasoning or ""

        msg = self._clean_text(str(obj.get("message", "")))
        original_msg = msg
        bad_msg = (
            (not msg)
            or self._is_generic_text(msg)
            or self._looks_like_prompt_leak(msg)
            or ((not state.current_team) and ("empty team" in msg.lower()))
        )
        if bad_msg:
            repair_prompt = (
                "Write a short table-talk message (1–2 sentences). Say who you trust or are wary of, suggest who should go. Mention at least one player by id (P0, P1). Do NOT end with APPROVE_LEAN or REJECT_LEAN."
                if not state.current_team
                else "Write a short table-talk message (1–2 sentences). Say why you're for or against this lineup. Mention at least one player by id. End with exactly APPROVE_LEAN or REJECT_LEAN."
            )
            repair = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"message": string}\n\n'
                    f"Round {state.round_idx}. Proposer: {state.current_proposer}. {self._human_context_summary(state)}\n\n"
                    f"Task: {phase_guidance}\n\n"
                    "Recent chat: " + (recent_chat) + "\n\n"
                    + repair_prompt
                ),
                schema='{"message": string}',
                max_tokens=500,
            )
            repair_msg = self._clean_text(str(repair.get("message", "")))
            # Keep repair only if it's real content; otherwise keep original so we don't drop to empty
            if repair_msg and not self._looks_like_prompt_leak(repair_msg):
                msg = repair_msg
            elif original_msg:
                msg = original_msg
            else:
                msg = repair_msg
            if not msg:
                free_prompt = (
                    "Reply in one short sentence. Say who you trust or are wary of. Name at least one player (e.g. P0). Do NOT end with APPROVE_LEAN or REJECT_LEAN."
                    if not state.current_team
                    else "Reply in one short sentence. Say why you're for or against this team. Name at least one player. End your reply with exactly APPROVE_LEAN or REJECT_LEAN."
                )
                free = self.llm.generate(
                    system=self._system(state),
                    user=f"Round {state.round_idx}. Leader: {state.current_proposer}. {phase_guidance}\n\n{free_prompt}",
                    max_tokens=500,
                )
                msg = self._clean_text(free)
                if not msg and has_lean:
                    msg = ("APPROVE_LEAN" if lean == "approve" else "REJECT_LEAN")
        if not self._mentions_player(msg):
            msg = f"{msg.rstrip('. ')} {state.current_proposer} is my current reference point."
        # Align lean with message content so vote matches what they said (e.g. "I trust P1" must not end with REJECT_LEAN)
        if has_lean:
            msg_content = re.sub(r"\s*(APPROVE_LEAN|REJECT_LEAN)\s*$", "", msg).strip()
            if self._message_suggests_approve(msg_content) and lean == "reject":
                lean = "approve"
            elif self._message_suggests_reject(msg_content) and lean == "approve":
                lean = "reject"
        if has_lean and not msg.endswith(("APPROVE_LEAN", "REJECT_LEAN")):
            msg = msg.rstrip(". ")
            msg += " APPROVE_LEAN" if lean == "approve" else " REJECT_LEAN"
        # Discussion phase: no lean—strip APPROVE_LEAN/REJECT_LEAN if the model added them
        if not has_lean and msg:
            msg = re.sub(r"\s*(APPROVE_LEAN|REJECT_LEAN)\s*$", "", msg).strip() or msg

        # Structured accusations for RL: only include valid player ids
        players_set = set(state.players)
        raw_evil = obj.get("evil_suspects", [])
        raw_good = obj.get("good_suspects", [])
        evil_suspects = [p for p in (raw_evil if isinstance(raw_evil, list) else []) if isinstance(p, str) and p in players_set]
        good_suspects = [p for p in (raw_good if isinstance(raw_good, list) else []) if isinstance(p, str) and p in players_set]

        # If model left both accusation lists empty, one focused repair for accusations only
        if not evil_suspects and not good_suspects and len(players_set) > 1:
            acc_phase = "No team proposed yet" if not state.current_team else "A team has been proposed"
            acc_task = (
                "Suggest who should go: put in good_suspects at least one player you'd want on the quest (e.g. the leader or someone you trust), and optionally in evil_suspects anyone you're wary of."
                if not state.current_team
                else "React to the lineup: put in good_suspects at least one player you trust, and optionally in evil_suspects anyone you suspect. You must list at least one player total."
            )
            # Include the agent's own message so the model can extract who they trust/suspect from what they said
            my_msg = f' You just said: "{msg[:400]}". Use that to fill good_suspects (players you said you trust/rely on) and evil_suspects (players you said you are wary of or watching).' if msg else ""
            acc_repair = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"evil_suspects": [string], "good_suspects": [string]}\n\n'
                    "Your previous response had empty evil_suspects and good_suspects. That is invalid. You MUST list at least one player id in one or both arrays.\n\n"
                    f"Phase: {acc_phase}. Players: {list(state.players)}. Leader: {state.current_proposer}.{my_msg}\n\n"
                    f"{acc_task} Example: {{\"good_suspects\": [\"P0\", \"P3\"], \"evil_suspects\": [\"P2\"]}}. Use only these player ids. Do not return [] for both."
                ),
                schema='{"evil_suspects":[string],"good_suspects":[string]}',
                max_tokens=500,
            )
            re_evil = acc_repair.get("evil_suspects", []) or []
            rg = acc_repair.get("good_suspects", []) or []
            evil_suspects = [p for p in (re_evil if isinstance(re_evil, list) else []) if isinstance(p, str) and p in players_set]
            good_suspects = [p for p in (rg if isinstance(rg, list) else []) if isinstance(p, str) and p in players_set]

        # Last resort: infer from message text if both arrays still empty (e.g. "P1 seems reliable" -> good_suspects, "uneasy about P2" -> evil_suspects)
        if not evil_suspects and not good_suspects and msg and len(players_set) > 1:
            good_suspects, evil_suspects = self._infer_suspects_from_message(msg, players_set)

        # Evil must never list allies or themselves in evil_suspects (they know who is evil)
        if state.my_alignment == "evil":
            allies = self._known_evil_players(state)
            evil_suspects = [p for p in evil_suspects if p not in allies and p != self.name]
            # If filtering left both lists empty (e.g. model only accused allies/self), ensure at least one in good_suspects
            if not evil_suspects and not good_suspects and players_set:
                non_ally = next((p for p in state.players if p not in allies and p != self.name), None)
                if non_ally:
                    good_suspects = [non_ally]

        # Reasoning must be actual thinking—repair if empty or generic; accept any non-empty repair if we had nothing
        if (not self.last_reasoning or self._is_generic_text(self.last_reasoning)) and (evil_suspects or good_suspects):
            repair_reason = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"reasoning": string}\n\n'
                    f"Round {state.round_idx}. You have good_suspects={good_suspects} and evil_suspects={evil_suspects}. "
                    "In 2–4 sentences give your actual reasoning: WHY do you trust the good list and WHY do you suspect the evil list? "
                    "Reference evidence (mission outcome, voting, behavior). Do NOT just say 'I list good X and evil Y'—explain your thinking. No length cap."
                ),
                schema='{"reasoning": string}',
                max_tokens=500,
            )
            repaired = self._clean_text(str(repair_reason.get("reasoning", "") or ""))
            if repaired:
                if not self._is_generic_text(repaired):
                    self.last_reasoning = repaired
                elif not self.last_reasoning:
                    self.last_reasoning = repaired  # use anyway when we had nothing
        # Guarantee non-empty reasoning: retry until we get something from the LLM
        for _ in range(2):
            if self.last_reasoning:
                break
            final_reason = self._force_json(
                system=self._system(state),
                user=(
                    'Output JSON: {"reasoning": string}\n\n'
                    f"Round {state.round_idx}. {self.name}. In one sentence, why do you trust or suspect the players you mentioned? "
                    "Or why are you for or against the current proposal? Use only public info. No role reveal. Required: output a non-empty reasoning string. No length cap."
                ),
                schema='{"reasoning": string}',
                max_tokens=200,
            )
            self.last_reasoning = self._clean_text(str(final_reason.get("reasoning", "") or ""))
        # Last resort: use what they said at the table (so every agent has reasoning)
        if not self.last_reasoning and msg:
            self.last_reasoning = ("Stated: " + msg[:280].strip()).rstrip()

        return {
            "message": self._clean_text(msg),
            "reasoning": self.last_reasoning,
            "evil_suspects": evil_suspects,
            "good_suspects": good_suspects,
        }

    def propose_team(self, state, team_size: int):
        merlin_rl_hint = self._format_merlin_rl_hint(state)
        game_summary = self._game_context_summary(state)
        user = (
            'Output JSON: {"team": [string], "reasoning": string}\n\n'
            f"You are the leader this round. You must propose a team of exactly {team_size} players.\n\n"
            f"Players: {state.players}. Round: {state.round_idx}.\n\n"
            "Game history:\n"
            f"{game_summary}\n\n"
            f"{merlin_rl_hint}"
            "Choose exactly " + str(team_size) + " player ids for your team. "
            "team: array of exactly " + str(team_size) + " strings (e.g. [\"P0\", \"P2\"]). "
            "reasoning: one sentence explaining why you chose this lineup (use public info: mission results, who you trust or suspect). No role reveal."
        )
        obj = self._force_json(
            system=self._system(state),
            user=user,
            schema='{"team": [string], "reasoning": string}',
            max_tokens=180,
        )
        self.last_reasoning = self._clean_text(str(obj.get("reasoning", "")))
        raw_team = obj.get("team", [])
        chosen: List[str] = []
        if isinstance(raw_team, list):
            for p in raw_team:
                if p in state.players and p not in chosen:
                    chosen.append(p)
        if state.my_alignment == "good":
            known_evil = self._known_evil_players(state)
            ranked = sorted(
                state.players,
                key=lambda p: (
                    p in known_evil,
                    sum(1 for m in state.missions if p in list(getattr(m, "team", []) or []) and int(getattr(m, "fail_count", 0)) > 0),
                ),
            )
        else:
            ranked = list(state.players)
        for p in ranked:
            if len(chosen) >= int(team_size):
                break
            if p not in chosen:
                chosen.append(p)
        return chosen[: int(team_size)]

    def vote_on_team(self, state, team: List[str]):
        """Vote must align with the discussion: use LLM so the vote matches what the agent said."""
        game_summary = self._game_context_summary(state)
        merlin_rl_hint = self._format_merlin_rl_hint(state)
        recent_chat = self._chat(state, k=20)
        user = (
            'Output JSON: {"vote": "approve" or "reject", "reasoning": string}\n\n'
            f"Proposed team to vote on: {team}. Round {state.round_idx}, proposal {state.proposal_idx}. Proposer: {state.current_proposer}.\n\n"
            "Game history:\n"
            f"{game_summary}\n\n"
            f"{merlin_rl_hint}"
            "What you and others just said (your vote should be consistent with your stated position):\n"
            f"{recent_chat}\n\n"
            "Decide whether you approve or reject this team. Your vote must align with what you said in the discussion (e.g. if you said you're comfortable approving, vote approve; if you said you're cautious or want to reject, vote reject). "
            "If you are good: approve when the team seems acceptable for the information we have; reject when you have strong reasons (e.g. known evil in team, or you need more information). "
            "If you are evil: approve or reject in a way that helps your side and is consistent with your public statements. "
            "Reply with vote (exactly \"approve\" or \"reject\") and one sentence of reasoning."
        )
        obj = self._force_json(
            system=self._system(state),
            user=user,
            schema='{"vote":"approve|reject","reasoning":string}',
            max_tokens=120,
        )
        self.last_reasoning = self._clean_text(str(obj.get("reasoning", "") or ""))
        raw = str(obj.get("vote", "")).strip().lower()
        if raw in ("approve", "reject"):
            return raw
        # Fallback: use the lean from this agent's last message in chat (vote must align with what they said)
        chat_log = getattr(state, "chat_log", []) or []
        for speaker, text in reversed(chat_log):
            if speaker == self.name and text:
                if "APPROVE_LEAN" in text:
                    self.last_reasoning = "Vote aligned with stated position (approve)."
                    return "approve"
                if "REJECT_LEAN" in text:
                    self.last_reasoning = "Vote aligned with stated position (reject)."
                    return "reject"
                break
        decision = self._default_lean(state, team_override=team)
        self.last_reasoning = "Vote aligned with game state (fallback)."
        return decision

    def mission_action(self, state):
        user = (
            'Output JSON: {"action": "pass" or "fail", "reasoning": string}\n\n'
            f"Round {state.round_idx}. You are on the quest team: {state.current_team}.\n\n"
            "Good players always choose \"pass\" (they always want the mission to succeed).\n"
            "Evil players may choose \"fail\" to sabotage, or \"pass\" to fool others—but if you are evil and choose \"pass\", your reasoning MUST state that you are evil and passing to fool/build trust/avoid suspicion. "
            "If an evil player chooses \"pass\" without that acknowledgment in reasoning, the action is treated as \"fail\" (sabotage).\n\n"
            "If you are good: action = \"pass\"; reasoning = why you support this mission (one sentence).\n"
            "If you are evil and sabotage: action = \"fail\"; reasoning = why you are failing (one sentence).\n"
            "If you are evil and pretend: action = \"pass\"; reasoning must explicitly show you are evil and passing to fool (e.g. \"I am evil but passing to avoid suspicion\" or \"Passing so they trust me\").\n\n"
            "Reply with action (exactly \"pass\" or \"fail\") and reasoning. This reasoning is private; you may reference your role when explaining an evil pass."
        )
        obj = self._force_json(
            system=self._system(state),
            user=user,
            schema='{"action":"pass|fail","reasoning":string}',
            max_tokens=120,
        )
        reasoning = self._clean_text(str(obj.get("reasoning", "")))
        self.last_reasoning = reasoning
        action = str(obj.get("action", "")).strip().lower()
        if action not in {"pass", "fail"}:
            action = "pass"
        # Evil may only pass if reasoning acknowledges they are evil and passing to fool; otherwise force fail
        if state.my_alignment == "evil" and action == "pass" and not self._evil_pass_reasoning_ok(reasoning):
            action = "fail"
            self.last_reasoning = (reasoning or "").strip() or "Sabotaging; did not justify passing as evil."
        return action

    def assassinate(self, state):
        game_summary = self._game_context_summary(state)
        merlin_rl_hint = self._format_merlin_rl_hint(state)
        user = (
            'Output JSON: {"target": string, "reasoning": string}\n\n'
            "Good has won 3 quests. You (Assassin) get one guess to kill Merlin. If you kill Merlin, Evil wins.\n\n"
            f"Players: {state.players}. You are {self.name} (do not target yourself).\n\n"
            "Game history:\n"
            f"{game_summary}\n\n"
            f"{merlin_rl_hint}"
            "Recent discussion:\n"
            f"{self._chat(state)}\n\n"
            "Choose one player id as target (the player you believe is Merlin). "
            "Merlin often: gave information that helped good, was on successful teams, or was trusted by others. "
            "reasoning: one sentence explaining why you suspect this player. Use only public information."
        )
        obj = self._force_json(
            system=self._system(state),
            user=user,
            schema='{"target": string, "reasoning": string}',
            max_tokens=140,
        )
        self.last_reasoning = self._clean_text(str(obj.get("reasoning", "")))
        target = str(obj.get("target", "")).strip()
        if target in state.players and target != self.name:
            return target
        for p in state.players:
            if p != self.name:
                return p
        return self.name
    
    def _get_merlin_scores(self, state) -> Dict[str, float] | None:
        if not self.use_rl or self.merlin_policy is None:
            return None

        try:
            current_state = self._build_merlin_state(state)
            return self.merlin_policy.score_candidates(current_state, viewer=self.name)
        except Exception:
            return None


    def _format_merlin_rl_hint(self, state) -> str:
        scores = self._get_merlin_scores(state)
        if not scores:
            return ""

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        lines = ["--- LEARNED MERLIN-LIKELIHOOD SIGNAL ---"]
        for p, s in ranked:
            lines.append(f"{p}: {s:.3f}")


        lines.append(
            "Use this as private guidance for who is most likely Merlin."
            "Merlin knows who is evil."
            "Do not mention signals or Merlin."
        )

        return "\n".join(lines) + "\n\n"


    def _ranked_merlin_candidates(self, state) -> List[str]:
        scores = self._get_merlin_scores(state)
        if not scores:
            return []
        return [p for p, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    
    def _build_merlin_state(self, state):
        return {
            "players": state.players,
            "roles": state.extra.get("roles", {}),
            "alignments": state.extra.get("alignments", {}),
            "round_idx": state.round_idx,
            "proposal_idx_in_round": state.proposal_idx,
            "score_good": state.num_successes,
            "score_evil": state.num_fails,
            "consecutive_rejections_before": state.proposal_idx - 1,
            "current_team": state.current_team,
            "vote_rows_so_far": state.extra.get("vote_rows_so_far", []),
        }
