from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return []


def aggregate_viewer_history_features_runtime(
    vote_rows_so_far: List[Dict[str, Any]],
    viewer: str,
    team: List[str],
) -> Dict[str, float]:
    approve_count = 0
    reject_count = 0
    on_team_count = 0
    on_team_approve_count = 0
    viewer_as_leader_count = 0
    accused_by_others = 0
    trusted_by_others = 0
    team_member_votes = 0
    team_member_approves = 0

    viewer_last_known_evil_suspects: List[str] = []
    viewer_last_known_good_suspects: List[str] = []
    team_set = set(team)

    # Proposal-level dedupe for team repetition features
    seen_prop = {}

    for row in vote_rows_so_far:
        row_team = safe_list(row.get("team", []))
        row_vote = row.get("vote")
        key = (int(row.get("round_idx", 0)), int(row.get("proposal_idx_in_round", 0)))
        if key not in seen_prop:
            seen_prop[key] = {
                "team": row_team,
                "proposal_approved": row.get("proposal_approved"),
                "quest_result_if_approved": row.get("quest_result_if_approved"),
            }

        if row.get("voter") == viewer:
            if row_vote == "APPROVE":
                approve_count += 1
            elif row_vote == "REJECT":
                reject_count += 1

            if viewer in row_team:
                on_team_count += 1
                if row_vote == "APPROVE":
                    on_team_approve_count += 1

            viewer_last_known_evil_suspects = safe_list(row.get("latest_evil_suspects", []))
            viewer_last_known_good_suspects = safe_list(row.get("latest_good_suspects", []))

        if row.get("leader") == viewer:
            viewer_as_leader_count += 1

        evil_suspects = safe_list(row.get("latest_evil_suspects", []))
        good_suspects = safe_list(row.get("latest_good_suspects", []))
        if viewer in evil_suspects:
            accused_by_others += 1
        if viewer in good_suspects:
            trusted_by_others += 1

        if row.get("voter") in team_set:
            team_member_votes += 1
            if row_vote == "APPROVE":
                team_member_approves += 1

    exact_team_seen = 0
    exact_team_approved = 0
    exact_team_success = 0
    exact_team_failed = 0
    proposal_approved_count = 0

    team_key = tuple(sorted(team))
    for p in seen_prop.values():
        p_team = safe_list(p.get("team", []))
        if p.get("proposal_approved") is True:
            proposal_approved_count += 1
        if tuple(sorted(p_team)) == team_key:
            exact_team_seen += 1
            if p.get("proposal_approved") is True:
                exact_team_approved += 1
            q = p.get("quest_result_if_approved")
            if q == "SUCCESS":
                exact_team_success += 1
            elif q == "FAILED":
                exact_team_failed += 1

    viewer_accuses_team_count = sum(1 for p in team if p in set(viewer_last_known_evil_suspects))
    viewer_trusts_team_count = sum(1 for p in team if p in set(viewer_last_known_good_suspects))

    total_votes = approve_count + reject_count
    proposal_total = max(len(seen_prop), 1)

    return {
        "viewer_prior_votes": float(total_votes),
        "viewer_prior_approve_count": float(approve_count),
        "viewer_prior_reject_count": float(reject_count),
        "viewer_prior_approve_rate": float(approve_count / max(total_votes, 1)),
        "viewer_prior_on_team_count": float(on_team_count),
        "viewer_prior_on_team_approve_rate": float(on_team_approve_count / max(on_team_count, 1)),
        "viewer_prior_leader_count": float(viewer_as_leader_count),
        "viewer_prior_accused_by_others": float(accused_by_others),
        "viewer_prior_trusted_by_others": float(trusted_by_others),
        "viewer_prior_net_trust": float(trusted_by_others - accused_by_others),
        "team_member_prior_vote_approve_rate": float(team_member_approves / max(team_member_votes, 1)),
        "prior_proposals_seen": float(len(seen_prop)),
        "prior_proposals_approved": float(proposal_approved_count),
        "prior_proposal_approval_rate": float(proposal_approved_count / proposal_total),
        "exact_team_seen_count": float(exact_team_seen),
        "exact_team_approved_count": float(exact_team_approved),
        "exact_team_success_count": float(exact_team_success),
        "exact_team_failed_count": float(exact_team_failed),
        "viewer_accuses_team_count": float(viewer_accuses_team_count),
        "viewer_trusts_team_count": float(viewer_trusts_team_count),
    }


def build_features_for_vote_runtime(
    current_state: Dict[str, Any],
    viewer: str,
    team: List[str] | None = None,
) -> Dict[str, float]:
    roles = current_state.get("roles", {})
    alignments = current_state.get("alignments", {})
    vote_rows_so_far = current_state.get("vote_rows_so_far", [])

    active_team = safe_list(team if team is not None else current_state.get("current_team", []))
    leader = str(current_state.get("current_leader") or current_state.get("current_proposer") or "")

    viewer_role = roles.get(viewer, "")
    viewer_alignment = alignments.get(viewer, "")

    hist = aggregate_viewer_history_features_runtime(
        vote_rows_so_far=vote_rows_so_far,
        viewer=viewer,
        team=active_team,
    )

    score_good = float(current_state.get("score_good", current_state.get("num_successes", 0)))
    score_evil = float(current_state.get("score_evil", current_state.get("num_fails", 0)))

    return {
        "round_idx": float(current_state.get("round_idx", 0)),
        "proposal_idx_in_round": float(current_state.get("proposal_idx_in_round", current_state.get("proposal_idx", 0))),
        "score_good_before": score_good,
        "score_evil_before": score_evil,
        "consecutive_rejections_before": float(current_state.get("consecutive_rejections_before", max(int(current_state.get("proposal_idx", 1)) - 1, 0))),
        "team_size": float(len(active_team)),
        "viewer_on_team": float(viewer in set(active_team)),
        "leader_is_viewer": float(leader == viewer),
        "viewer_is_evil": float(viewer_alignment == "evil"),
        "viewer_is_assassin": float("assassin" in str(viewer_role).lower()),
        "team_has_viewer_trusted_player": float(hist["viewer_trusts_team_count"] > 0),
        "team_has_viewer_accused_player": float(hist["viewer_accuses_team_count"] > 0),
        "score_margin_good_minus_evil": float(score_good - score_evil),
        **hist,
    }


class TeamApprovalPolicy:
    def __init__(self, model_path: str = "team_approval_policy.pkl"):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def approve_probability(
        self,
        current_state: Dict[str, Any],
        viewer: str,
        team: List[str] | None = None,
    ) -> float:
        feat = build_features_for_vote_runtime(
            current_state=current_state,
            viewer=viewer,
            team=team,
        )
        x = np.array([[feat[k] for k in self.feature_names]], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

    def decide_vote(
        self,
        current_state: Dict[str, Any],
        viewer: str,
        team: List[str] | None = None,
        threshold: float = 0.5,
    ) -> str:
        p = self.approve_probability(current_state=current_state, viewer=viewer, team=team)
        return "approve" if p >= threshold else "reject"
