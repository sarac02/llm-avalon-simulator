from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np


def aggregate_candidate_features_runtime(vote_rows_so_far, mission_rows_so_far, candidate, viewer):
    accuse_count = trust_count = approve_count = reject_count = 0
    on_team_count = on_team_approve_count = on_team_reject_count = 0
    proposed_by_evil_count = proposed_by_good_count = 0
    accused_leader_count = trusted_leader_count = 0
    accuses_team_member_total = trusts_team_member_total = 0
    latest_reasoning_len_sum = 0
    visible_rounds = set()
    team_sizes_with_candidate = []
    joint_team_with_viewer = 0

    for row in vote_rows_so_far:
        team = row.get("team", [])
        evil_suspects = row.get("latest_evil_suspects", [])
        good_suspects = row.get("latest_good_suspects", [])
        if candidate in evil_suspects:
            accuse_count += 1
        if candidate in good_suspects:
            trust_count += 1
        if row.get("voter") == candidate:
            if row.get("vote") == "APPROVE":
                approve_count += 1
            else:
                reject_count += 1
            if row.get("accuses_leader"):
                accused_leader_count += 1
            if row.get("trusts_leader"):
                trusted_leader_count += 1
            accuses_team_member_total += int(row.get("accuses_team_member_count", 0))
            trusts_team_member_total += int(row.get("trusts_team_member_count", 0))
            latest_reasoning_len_sum += int(row.get("latest_reasoning_len", 0))
        if candidate in team:
            on_team_count += 1
            team_sizes_with_candidate.append(int(row.get("team_size", len(team))))
            if row.get("vote") == "APPROVE":
                on_team_approve_count += 1
            else:
                on_team_reject_count += 1
            if row.get("leader_alignment") == "evil":
                proposed_by_evil_count += 1
            elif row.get("leader_alignment") == "good":
                proposed_by_good_count += 1
            if viewer in team:
                joint_team_with_viewer += 1
        visible_rounds.add(int(row.get("round_idx", 0)))

    mission_seen = success_on_team = fail_on_team = 0
    fail_rate_weighted_team_evil = 0.0
    for mission in mission_rows_so_far:
        team = mission.get("team", [])
        if candidate in team:
            mission_seen += 1
            if mission.get("quest_result") == "FAILED":
                fail_on_team += 1
                fail_rate_weighted_team_evil += float(mission.get("team_evil_count", 0))
            elif mission.get("quest_result") == "SUCCESS":
                success_on_team += 1

    rounds_observed = max(len(visible_rounds), 1)
    total_votes = approve_count + reject_count
    total_mentions = accuse_count + trust_count
    return {
        "candidate_accuse_count": float(accuse_count),
        "candidate_trust_count": float(trust_count),
        "candidate_net_suspicion": float(accuse_count - trust_count),
        "candidate_accuse_rate": float(accuse_count / max(total_mentions, 1)),
        "candidate_trust_rate": float(trust_count / max(total_mentions, 1)),
        "candidate_approve_count": float(approve_count),
        "candidate_reject_count": float(reject_count),
        "candidate_vote_approve_rate": float(approve_count / max(total_votes, 1)),
        "candidate_vote_reject_rate": float(reject_count / max(total_votes, 1)),
        "candidate_on_team_count": float(on_team_count),
        "candidate_on_team_approve_count": float(on_team_approve_count),
        "candidate_on_team_reject_count": float(on_team_reject_count),
        "candidate_avg_team_size": float(sum(team_sizes_with_candidate) / len(team_sizes_with_candidate)) if team_sizes_with_candidate else 0.0,
        "candidate_proposed_by_evil_count": float(proposed_by_evil_count),
        "candidate_proposed_by_good_count": float(proposed_by_good_count),
        "candidate_joint_team_with_viewer": float(joint_team_with_viewer),
        "candidate_accused_leader_count": float(accused_leader_count),
        "candidate_trusted_leader_count": float(trusted_leader_count),
        "candidate_avg_accused_teammates": float(accuses_team_member_total / max(total_votes, 1)),
        "candidate_avg_trusted_teammates": float(trusts_team_member_total / max(total_votes, 1)),
        "candidate_avg_reasoning_len": float(latest_reasoning_len_sum / max(total_votes, 1)),
        "candidate_visibility": float((total_mentions + on_team_count) / rounds_observed),
        "candidate_missions_seen": float(mission_seen),
        "candidate_success_on_team": float(success_on_team),
        "candidate_fail_on_team": float(fail_on_team),
        "candidate_fail_rate_on_team": float(fail_on_team / max(mission_seen, 1)),
        "candidate_fail_weighted_team_evil": float(fail_rate_weighted_team_evil),
    }


def build_features_for_candidate_runtime(current_state, viewer, candidate):
    roles = current_state["roles"]
    alignments = current_state["alignments"]
    current_team = current_state.get("current_team", [])
    vote_rows_so_far = current_state.get("vote_rows_so_far", [])
    mission_rows_so_far = current_state.get("mission_rows_so_far", [])
    candidate_feats = aggregate_candidate_features_runtime(vote_rows_so_far, mission_rows_so_far, candidate, viewer)
    return {
        "round_idx": float(current_state["round_idx"]),
        "proposal_idx_in_round": float(current_state["proposal_idx_in_round"]),
        "score_good_before": float(current_state["score_good"]),
        "score_evil_before": float(current_state["score_evil"]),
        "consecutive_rejections_before": float(current_state["consecutive_rejections_before"]),
        "viewer_is_evil": float(alignments.get(viewer) == "evil"),
        "viewer_is_assassin": float("assassin" in str(roles.get(viewer, "")).lower()),
        "candidate_is_viewer": float(candidate == viewer),
        "candidate_on_current_team": float(candidate in current_team),
        "viewer_on_current_team": float(viewer in current_team),
        "leader_is_candidate": float(candidate == current_state.get("leader")),
        "leader_is_viewer": float(viewer == current_state.get("leader")),
        "leader_alignment_evil": float(current_state.get("leader_alignment") == "evil"),
        "current_team_size": float(len(current_team)),
        "current_team_evil_count": float(sum(1 for p in current_team if alignments.get(p) == "evil")),
        "current_team_has_candidate_and_viewer": float(candidate in current_team and viewer in current_team),
        **candidate_feats,
    }


class SuspicionPolicy:
    def __init__(self, model_path="suspicion_policy.pkl"):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def score_candidates(self, current_state, viewer):
        scores = {}
        for candidate in current_state["players"]:
            feat = build_features_for_candidate_runtime(current_state, viewer, candidate)
            x = np.array([[feat[k] for k in self.feature_names]], dtype=float)
            scores[candidate] = float(self.model.predict_proba(x)[0, 1])
        return scores

    def most_suspicious_player(self, current_state, viewer, include_self=False):
        scores = self.score_candidates(current_state, viewer)
        if not include_self and viewer in scores:
            scores = {p: s for p, s in scores.items() if p != viewer}
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def suspicion_ranking(self, current_state, viewer, include_self=False):
        scores = self.score_candidates(current_state, viewer)
        if not include_self and viewer in scores:
            scores = {p: s for p, s in scores.items() if p != viewer}
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
