from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROLE_LINE_RE = re.compile(r"^(P\d+):\s*(.+)$")
ROUND_HEADER_RE = re.compile(r"^\[ROUND (\d+)\]\s+(.+)$")
PROPOSAL_PHASE_RE = re.compile(r"^\[ROUND (\d+)\] Proposal phase \(leader=(P\d+), team_size=(\d+)\)$")
POST_PROPOSAL_RE = re.compile(r"^\[ROUND (\d+)\] Post-proposal discussion \(team=(.+)\)$")
TEAM_PROPOSED_RE = re.compile(r"^- Team proposed:\s*(.+)$")
TEAM_VOTE_PHASE_RE = re.compile(r"^\[ROUND (\d+)\] Team vote phase$")
QUEST_PHASE_RE = re.compile(r"^\[ROUND (\d+)\] Quest phase \(team=(.+)\)$")
VOTE_LINE_RE = re.compile(r"^- (P\d+) votes:\s*(APPROVE|REJECT)$")
VOTE_RESULT_RE = re.compile(r"^- Vote result:\s*(APPROVED|REJECTED)$")
MISSION_ACTION_RE = re.compile(r"^- (P\d+) mission action:\s*(PASS|FAIL)$")
QUEST_RESULT_RE = re.compile(r"^- Quest result:\s*(SUCCESS|FAILED)\s+\(fails=(\d+), threshold=(\d+)\)$")
SCORE_RE = re.compile(r"^- Score now -> Good (\d+)\s*:\s*Evil (\d+)$")
NEXT_LEADER_RE = re.compile(r"^- Next (?:round )?leader:\s*(P\d+)$")
ASSASSINATION_RE = re.compile(r"^- Assassin (P\d+) targets (P\d+); hit_merlin=(True|False)$")
GAME_OVER_INLINE_RE = re.compile(r"^GAME OVER -> winner=(good|evil), reason=([a-zA-Z0-9_]+)$")
FINAL_SCORE_RE = re.compile(r"^Final score:\s*Good\s+(\d+)\s*:\s*Evil\s+(\d+)$")
POST_PROPOSAL_STANCE_RE = re.compile(r"^- (P\d+) \(post-proposal\) says: .* (APPROVE_LEAN|REJECT_LEAN)$")


def is_evil_role(role_name: str) -> bool:
    key = role_name.strip().lower()
    evil_tags = ("assassin", "minion", "morgana", "mordred", "oberon")
    return any(tag in key for tag in evil_tags)


def to_alignment(role_name: str) -> str:
    return "evil" if is_evil_role(role_name) else "good"


def safe_list(text: str) -> List[str]:
    try:
        v = ast.literal_eval(text.strip())
    except Exception:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def extract_timestamp_from_name(path: Path, prefix: str) -> str:
    stem = path.stem
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


@dataclass
class ProposalRecord:
    game_id: str
    round_idx: int
    proposal_idx_in_round: int
    proposal_id: int
    leader: str
    team_size: int
    team: List[str] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)
    vote_result: Optional[str] = None
    approved: Optional[bool] = None
    approve_count: int = 0
    reject_count: int = 0
    post_proposal_stances: Dict[str, str] = field(default_factory=dict)
    quest_result: Optional[str] = None
    quest_fails: Optional[int] = None
    quest_fail_threshold: Optional[int] = None
    mission_actions: Dict[str, str] = field(default_factory=dict)
    score_good_before: int = 0
    score_evil_before: int = 0
    score_good_after: int = 0
    score_evil_after: int = 0
    next_leader: Optional[str] = None


def parse_single_log(log_path: Path, accusation_path: Optional[Path]) -> Dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    game_id = extract_timestamp_from_name(log_path, "log_")
    roles: Dict[str, str] = {}

    proposals: List[ProposalRecord] = []
    missions: List[Dict[str, Any]] = []

    current_round = 0
    proposal_counter = 0
    round_proposal_counter: Dict[int, int] = {}
    score_good = 0
    score_evil = 0

    current_proposal: Optional[ProposalRecord] = None
    pending_approved_proposal: Optional[ProposalRecord] = None

    winner = ""
    reason = ""
    assassin = None
    assassination_target = None
    assassination_hit_merlin = None

    in_role_assignments = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == "=== ROLE ASSIGNMENTS ===":
            in_role_assignments = True
            continue
        if line == "=== GAME TRANSCRIPT ===":
            in_role_assignments = False
            continue

        if in_role_assignments:
            m = ROLE_LINE_RE.match(line)
            if m:
                roles[m.group(1)] = m.group(2).strip()
            continue

        m_round = ROUND_HEADER_RE.match(line)
        if m_round:
            current_round = int(m_round.group(1))

        m = PROPOSAL_PHASE_RE.match(line)
        if m:
            rnd = int(m.group(1))
            leader = m.group(2)
            team_size = int(m.group(3))
            proposal_counter += 1
            round_proposal_counter[rnd] = round_proposal_counter.get(rnd, 0) + 1
            current_proposal = ProposalRecord(
                game_id=game_id,
                round_idx=rnd,
                proposal_idx_in_round=round_proposal_counter[rnd],
                proposal_id=proposal_counter,
                leader=leader,
                team_size=team_size,
                score_good_before=score_good,
                score_evil_before=score_evil,
            )
            proposals.append(current_proposal)
            continue

        m = TEAM_PROPOSED_RE.match(line)
        if m and current_proposal:
            current_proposal.team = safe_list(m.group(1))
            continue

        m = POST_PROPOSAL_STANCE_RE.match(line)
        if m and current_proposal:
            current_proposal.post_proposal_stances[m.group(1)] = m.group(2)
            continue

        m = TEAM_VOTE_PHASE_RE.match(line)
        if m:
            continue

        m = VOTE_LINE_RE.match(line)
        if m and current_proposal:
            voter, vote = m.group(1), m.group(2)
            current_proposal.votes[voter] = vote
            continue

        m = VOTE_RESULT_RE.match(line)
        if m and current_proposal:
            current_proposal.vote_result = m.group(1)
            current_proposal.approved = m.group(1) == "APPROVED"
            current_proposal.approve_count = sum(1 for v in current_proposal.votes.values() if v == "APPROVE")
            current_proposal.reject_count = sum(1 for v in current_proposal.votes.values() if v == "REJECT")
            if current_proposal.approved:
                pending_approved_proposal = current_proposal
            else:
                current_proposal.score_good_after = score_good
                current_proposal.score_evil_after = score_evil
            continue

        m = QUEST_PHASE_RE.match(line)
        if m:
            continue

        m = MISSION_ACTION_RE.match(line)
        if m and pending_approved_proposal:
            pending_approved_proposal.mission_actions[m.group(1)] = m.group(2)
            continue

        m = QUEST_RESULT_RE.match(line)
        if m and pending_approved_proposal:
            pending_approved_proposal.quest_result = m.group(1)
            pending_approved_proposal.quest_fails = int(m.group(2))
            pending_approved_proposal.quest_fail_threshold = int(m.group(3))
            continue

        m = SCORE_RE.match(line)
        if m:
            score_good = int(m.group(1))
            score_evil = int(m.group(2))
            if pending_approved_proposal:
                pending_approved_proposal.score_good_after = score_good
                pending_approved_proposal.score_evil_after = score_evil
                missions.append(
                    {
                        "game_id": game_id,
                        "round_idx": pending_approved_proposal.round_idx,
                        "proposal_id": pending_approved_proposal.proposal_id,
                        "team": pending_approved_proposal.team,
                        "quest_result": pending_approved_proposal.quest_result,
                        "quest_fails": pending_approved_proposal.quest_fails,
                        "quest_fail_threshold": pending_approved_proposal.quest_fail_threshold,
                        "mission_actions": dict(pending_approved_proposal.mission_actions),
                        "score_good_before": pending_approved_proposal.score_good_before,
                        "score_evil_before": pending_approved_proposal.score_evil_before,
                        "score_good_after": score_good,
                        "score_evil_after": score_evil,
                    }
                )
            continue

        m = NEXT_LEADER_RE.match(line)
        if m:
            next_leader = m.group(1)
            if pending_approved_proposal:
                pending_approved_proposal.next_leader = next_leader
                pending_approved_proposal = None
            elif current_proposal:
                current_proposal.next_leader = next_leader
            continue

        m = ASSASSINATION_RE.match(line)
        if m:
            assassin = m.group(1)
            assassination_target = m.group(2)
            assassination_hit_merlin = (m.group(3) == "True")
            continue

        m = GAME_OVER_INLINE_RE.match(line)
        if m:
            winner = m.group(1)
            reason = m.group(2)
            continue

        if line.startswith("Winner:"):
            winner = line.split(":", 1)[1].strip().lower()
            continue
        if line.startswith("Reason:"):
            reason = line.split(":", 1)[1].strip()
            continue
        m = FINAL_SCORE_RE.match(line)
        if m:
            score_good = int(m.group(1))
            score_evil = int(m.group(2))

    accusations_payload = {"accusations": [], "roles": {}}
    if accusation_path and accusation_path.exists():
        try:
            accusations_payload = json.loads(accusation_path.read_text(encoding="utf-8"))
        except Exception:
            accusations_payload = {"accusations": [], "roles": {}}

    if not roles and accusations_payload.get("roles"):
        roles = {k: str(v) for k, v in accusations_payload.get("roles", {}).items()}

    acc_index: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for a in accusations_payload.get("accusations", []):
        key = (int(a.get("round_idx", 0)), str(a.get("phase", "")), str(a.get("speaker", "")))
        acc_index[key] = a

    players = sorted(roles.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    role_alignment = {p: to_alignment(r) for p, r in roles.items()}

    votes_rows: List[Dict[str, Any]] = []
    running_vote_stats = {p: {"approve": 0, "reject": 0} for p in players}
    running_team_stats = {p: {"teams": 0, "quest_success": 0, "quest_fail": 0} for p in players}

    for prop in proposals:
        team_set = set(prop.team)
        evil_on_team = sum(1 for p in prop.team if role_alignment.get(p) == "evil")
        team_has_merlin = any("merlin" in roles.get(p, "").lower() for p in prop.team)

        for voter, vote in prop.votes.items():
            acc = acc_index.get((prop.round_idx, "team_discussion", voter)) or acc_index.get(
                (prop.round_idx, "discussion", voter),
                {},
            )
            evil_suspects = [str(x) for x in acc.get("evil_suspects", [])]
            good_suspects = [str(x) for x in acc.get("good_suspects", [])]
            stance = prop.post_proposal_stances.get(voter)
            accuses_team_count = sum(1 for p in prop.team if p in evil_suspects)
            trusts_team_count = sum(1 for p in prop.team if p in good_suspects)

            votes_rows.append(
                {
                    "game_id": game_id,
                    "round_idx": prop.round_idx,
                    "proposal_id": prop.proposal_id,
                    "proposal_idx_in_round": prop.proposal_idx_in_round,
                    "voter": voter,
                    "voter_role": roles.get(voter),
                    "voter_alignment": role_alignment.get(voter),
                    "leader": prop.leader,
                    "leader_role": roles.get(prop.leader),
                    "leader_alignment": role_alignment.get(prop.leader),
                    "team": prop.team,
                    "team_size": prop.team_size,
                    "voter_on_team": voter in team_set,
                    "vote": vote,
                    "vote_binary": 1 if vote == "APPROVE" else 0,
                    "proposal_approved": prop.approved,
                    "quest_result_if_approved": prop.quest_result,
                    "score_good_before": prop.score_good_before,
                    "score_evil_before": prop.score_evil_before,
                    "consecutive_rejections_before": prop.proposal_idx_in_round - 1,
                    "evil_on_team_count": evil_on_team,
                    "team_has_merlin": team_has_merlin,
                    "player_approve_count_before": running_vote_stats.get(voter, {}).get("approve", 0),
                    "player_reject_count_before": running_vote_stats.get(voter, {}).get("reject", 0),
                    "player_team_count_before": running_team_stats.get(voter, {}).get("teams", 0),
                    "player_quest_success_before": running_team_stats.get(voter, {}).get("quest_success", 0),
                    "player_quest_fail_before": running_team_stats.get(voter, {}).get("quest_fail", 0),
                    "latest_evil_suspects": evil_suspects,
                    "latest_good_suspects": good_suspects,
                    "latest_reasoning_len": len(str(acc.get("reasoning", ""))),
                    "accuses_leader": prop.leader in evil_suspects,
                    "trusts_leader": prop.leader in good_suspects,
                    "accuses_team_member_count": accuses_team_count,
                    "trusts_team_member_count": trusts_team_count,
                    "post_proposal_stance": stance,
                    "stance_matches_vote": (
                        (stance == "APPROVE_LEAN" and vote == "APPROVE")
                        or (stance == "REJECT_LEAN" and vote == "REJECT")
                        if stance
                        else None
                    ),
                }
            )

            if vote == "APPROVE":
                running_vote_stats[voter]["approve"] += 1
            else:
                running_vote_stats[voter]["reject"] += 1

        if prop.approved:
            for member in prop.team:
                running_team_stats[member]["teams"] += 1
                if prop.quest_result == "SUCCESS":
                    running_team_stats[member]["quest_success"] += 1
                elif prop.quest_result == "FAILED":
                    running_team_stats[member]["quest_fail"] += 1

    proposal_rows: List[Dict[str, Any]] = []
    for prop in proposals:
        team_evil_roles = [roles.get(p) for p in prop.team if role_alignment.get(p) == "evil"]
        proposal_rows.append(
            {
                "game_id": game_id,
                "round_idx": prop.round_idx,
                "proposal_id": prop.proposal_id,
                "proposal_idx_in_round": prop.proposal_idx_in_round,
                "leader": prop.leader,
                "leader_role": roles.get(prop.leader),
                "leader_alignment": role_alignment.get(prop.leader),
                "team": prop.team,
                "team_size": prop.team_size,
                "team_roles": [roles.get(p) for p in prop.team],
                "score_good_before": prop.score_good_before,
                "score_evil_before": prop.score_evil_before,
                "score_good_after": prop.score_good_after,
                "score_evil_after": prop.score_evil_after,
                "consecutive_rejections_before": prop.proposal_idx_in_round - 1,
                "approve_count": prop.approve_count,
                "reject_count": prop.reject_count,
                "proposal_approved": prop.approved,
                "quest_result": prop.quest_result,
                "quest_fails": prop.quest_fails,
                "quest_fail_threshold": prop.quest_fail_threshold,
                "mission_actions": prop.mission_actions,
                "team_evil_count": len(team_evil_roles),
                "team_evil_roles": team_evil_roles,
                "team_has_merlin": any("merlin" in (roles.get(p, "").lower()) for p in prop.team),
                "next_leader": prop.next_leader,
            }
        )

    mission_rows: List[Dict[str, Any]] = []
    for m in missions:
        team = m["team"]
        mission_rows.append(
            {
                **m,
                "team_roles": [roles.get(p) for p in team],
                "team_alignments": [role_alignment.get(p) for p in team],
                "team_evil_count": sum(1 for p in team if role_alignment.get(p) == "evil"),
                "team_has_merlin": any("merlin" in (roles.get(p, "").lower()) for p in team),
            }
        )

    total_approved = sum(1 for p in proposals if p.approved)
    total_rejected = len(proposals) - total_approved

    game_row = {
        "game_id": game_id,
        "log_file": str(log_path),
        "accusations_file": str(accusation_path) if accusation_path else None,
        "num_players": len(players),
        "players": players,
        "roles": roles,
        "alignments": role_alignment,
        "winner": winner,
        "reason": reason,
        "score_good": score_good,
        "score_evil": score_evil,
        "rounds_reached": max((p.round_idx for p in proposals), default=0),
        "total_proposals": len(proposals),
        "approved_proposals": total_approved,
        "rejected_proposals": total_rejected,
        "total_quests": len(missions),
        "quests_success": sum(1 for m in missions if m.get("quest_result") == "SUCCESS"),
        "quests_failed": sum(1 for m in missions if m.get("quest_result") == "FAILED"),
        "assassin": assassin,
        "assassination_target": assassination_target,
        "assassination_hit_merlin": assassination_hit_merlin,
    }

    return {
        "game": game_row,
        "proposals": proposal_rows,
        "votes": votes_rows,
        "missions": mission_rows,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Avalon logs into RL-ready features.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log_*.txt and accusations_*.json")
    parser.add_argument("--out-dir", default="logs/parsed", help="Output directory for parsed JSONL files")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_dir.glob("log_*.txt"))
    game_rows: List[Dict[str, Any]] = []
    proposal_rows: List[Dict[str, Any]] = []
    vote_rows: List[Dict[str, Any]] = []
    mission_rows: List[Dict[str, Any]] = []

    for log_file in log_files:
        ts = extract_timestamp_from_name(log_file, "log_")
        acc_file = logs_dir / f"accusations_{ts}.json"
        parsed = parse_single_log(log_file, acc_file if acc_file.exists() else None)
        game_rows.append(parsed["game"])
        proposal_rows.extend(parsed["proposals"])
        vote_rows.extend(parsed["votes"])
        mission_rows.extend(parsed["missions"])

    write_jsonl(out_dir / "games.jsonl", game_rows)
    write_jsonl(out_dir / "proposals.jsonl", proposal_rows)
    write_jsonl(out_dir / "votes.jsonl", vote_rows)
    write_jsonl(out_dir / "missions.jsonl", mission_rows)

    summary = {
        "num_games": len(game_rows),
        "num_proposals": len(proposal_rows),
        "num_votes": len(vote_rows),
        "num_missions": len(mission_rows),
        "output_dir": str(out_dir),
        "files": {
            "games": str(out_dir / "games.jsonl"),
            "proposals": str(out_dir / "proposals.jsonl"),
            "votes": str(out_dir / "votes.jsonl"),
            "missions": str(out_dir / "missions.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
