from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np


def aggregate_state_features_for_candidate_runtime(
    roles: Dict[str, str],
    alignments: Dict[str, str],
    players: List[str],
    vote_rows_so_far: List[Dict[str, Any]],
    candidate: str,
) -> Dict[str, float]:
    accuse_count = 0
    trust_count = 0
    on_team_count = 0
    on_team_approve_count = 0
    approve_count = 0
    reject_count = 0
    team_sizes_with_candidate = []
    round_seen = set()

    for row in vote_rows_so_far:
        team = row.get("team", [])
        evil_suspects = row.get("latest_evil_suspects", [])
        good_suspects = row.get("latest_good_suspects", [])

        if candidate in evil_suspects:
            accuse_count += 1
        if candidate in good_suspects:
            trust_count += 1
        if candidate in team:
            on_team_count += 1
            team_sizes_with_candidate.append(int(row.get("team_size", len(team))))
            if row.get("vote") == "APPROVE":
                on_team_approve_count += 1

        if row.get("voter") == candidate:
            if row.get("vote") == "APPROVE":
                approve_count += 1
            else:
                reject_count += 1

        round_seen.add(int(row.get("round_idx", 0)))

    rounds_observed = max(len(round_seen), 1)

    return {
        "candidate_is_self_evil": float(alignments.get(candidate) == "evil"),
        "candidate_accuse_count": float(accuse_count),
        "candidate_trust_count": float(trust_count),
        "candidate_net_trust": float(trust_count - accuse_count),
        "candidate_on_team_count": float(on_team_count),
        "candidate_on_team_approve_count": float(on_team_approve_count),
        "candidate_avg_team_size": float(sum(team_sizes_with_candidate) / len(team_sizes_with_candidate))
        if team_sizes_with_candidate else 0.0,
        "candidate_approve_count": float(approve_count),
        "candidate_reject_count": float(reject_count),
        "candidate_vote_approve_rate": float(approve_count / max(approve_count + reject_count, 1)),
        "candidate_visibility": float((trust_count + accuse_count + on_team_count) / rounds_observed),
    }


def build_features_for_candidate_runtime(
    current_state: Dict[str, Any],
    viewer: str,
    candidate: str,
) -> Dict[str, float]:
    roles = current_state["roles"]
    alignments = current_state["alignments"]
    players = current_state["players"]
    vote_rows_so_far = current_state.get("vote_rows_so_far", [])
    current_team = current_state.get("current_team", [])

    candidate_feats = aggregate_state_features_for_candidate_runtime(
        roles=roles,
        alignments=alignments,
        players=players,
        vote_rows_so_far=vote_rows_so_far,
        candidate=candidate,
    )

    evil_on_team_count = sum(1 for p in current_team if alignments.get(p) == "evil")

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
        "current_team_size": float(len(current_team)),
        "current_team_evil_count": float(evil_on_team_count),
        **candidate_feats,
    }


class MerlinPolicy:
    def __init__(self, model_path: str = "merlin_policy.pkl"):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def score_candidates(self, current_state: Dict[str, Any], viewer: str) -> Dict[str, float]:
        players = current_state["players"]
        scores: Dict[str, float] = {}

        for candidate in players:
            feat = build_features_for_candidate_runtime(current_state, viewer, candidate)
            x = np.array([[feat[k] for k in self.feature_names]], dtype=float)
            prob = float(self.model.predict_proba(x)[0, 1])
            scores[candidate] = prob

        return scores

    def most_likely_merlin(self, current_state: Dict[str, Any], viewer: str) -> str:
        scores = self.score_candidates(current_state, viewer)
        return max(scores.items(), key=lambda kv: kv[1])[0]