from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return []


def is_evil_role(role_name: str | None) -> bool:
    if role_name is None:
        return False
    key = role_name.lower()
    return any(tag in key for tag in ["assassin", "minion", "morgana", "mordred", "oberon"])


def build_game_maps(
    games: List[Dict[str, Any]],
    votes: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    game_map = {g["game_id"]: g for g in games}
    votes_by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for v in votes:
        votes_by_game[v["game_id"]].append(v)

    for gid in votes_by_game:
        votes_by_game[gid] = sorted(
            votes_by_game[gid],
            key=lambda r: (int(r["round_idx"]), int(r["proposal_idx_in_round"]), str(r["voter"])),
        )
    return game_map, votes_by_game


def proposal_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row.get("round_idx", 0)), int(row.get("proposal_idx_in_round", 0))


def proposal_snapshots(vote_rows_so_far: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate vote-level rows into proposal-level snapshots.
    """
    seen: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in vote_rows_so_far:
        key = proposal_key(row)
        if key not in seen:
            seen[key] = {
                "round_idx": key[0],
                "proposal_idx_in_round": key[1],
                "team": safe_list(row.get("team", [])),
                "team_size": int(row.get("team_size", len(safe_list(row.get("team", []))) or 0)),
                "leader": row.get("leader"),
                "proposal_approved": row.get("proposal_approved"),
                "quest_result_if_approved": row.get("quest_result_if_approved"),
            }
    return [seen[k] for k in sorted(seen.keys())]


def aggregate_viewer_history_features(
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

    for row in vote_rows_so_far:
        row_team = safe_list(row.get("team", []))
        row_vote = row.get("vote")

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

        if row.get("voter") in set(team):
            team_member_votes += 1
            if row_vote == "APPROVE":
                team_member_approves += 1

    proposals_so_far = proposal_snapshots(vote_rows_so_far)
    exact_team_seen = 0
    exact_team_approved = 0
    exact_team_success = 0
    exact_team_failed = 0
    proposal_approved_count = 0

    team_key = tuple(sorted(team))
    for p in proposals_so_far:
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
        "prior_proposals_seen": float(len(proposals_so_far)),
        "prior_proposals_approved": float(proposal_approved_count),
        "prior_proposal_approval_rate": float(proposal_approved_count / max(len(proposals_so_far), 1)),
        "exact_team_seen_count": float(exact_team_seen),
        "exact_team_approved_count": float(exact_team_approved),
        "exact_team_success_count": float(exact_team_success),
        "exact_team_failed_count": float(exact_team_failed),
        "viewer_accuses_team_count": float(viewer_accuses_team_count),
        "viewer_trusts_team_count": float(viewer_trusts_team_count),
    }


def build_vote_features(
    game_row: Dict[str, Any],
    vote_rows_so_far: List[Dict[str, Any]],
    current_row: Dict[str, Any],
) -> Dict[str, float]:
    roles = game_row.get("roles", {})
    alignments = game_row.get("alignments", {})

    viewer = str(current_row["voter"])
    team = safe_list(current_row.get("team", []))
    leader = str(current_row.get("leader", ""))

    viewer_role = current_row.get("voter_role") or roles.get(viewer)
    viewer_alignment = current_row.get("voter_alignment") or alignments.get(viewer)

    hist = aggregate_viewer_history_features(
        vote_rows_so_far=vote_rows_so_far,
        viewer=viewer,
        team=team,
    )

    return {
        "round_idx": float(current_row.get("round_idx", 0)),
        "proposal_idx_in_round": float(current_row.get("proposal_idx_in_round", 0)),
        "score_good_before": float(current_row.get("score_good_before", 0)),
        "score_evil_before": float(current_row.get("score_evil_before", 0)),
        "consecutive_rejections_before": float(current_row.get("consecutive_rejections_before", 0)),
        "team_size": float(current_row.get("team_size", len(team))),
        "viewer_on_team": float(viewer in set(team)),
        "leader_is_viewer": float(leader == viewer),
        "viewer_is_evil": float(viewer_alignment == "evil" or is_evil_role(viewer_role)),
        "viewer_is_assassin": float("assassin" in str(viewer_role or "").lower()),
        "team_has_viewer_trusted_player": float(hist["viewer_trusts_team_count"] > 0),
        "team_has_viewer_accused_player": float(hist["viewer_accuses_team_count"] > 0),
        "score_margin_good_minus_evil": float(current_row.get("score_good_before", 0) - current_row.get("score_evil_before", 0)),
        **hist,
    }


def build_training_examples(
    games: List[Dict[str, Any]],
    votes: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    game_map, votes_by_game = build_game_maps(games, votes)

    all_feature_dicts: List[Dict[str, float]] = []
    all_labels: List[int] = []
    all_weights: List[float] = []

    for game_id, rows in votes_by_game.items():
        game_row = game_map.get(game_id)
        if not game_row:
            continue

        # Group by proposal so each vote sees only history from prior proposals.
        proposal_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            proposal_groups[proposal_key(r)].append(r)
        ordered_keys = sorted(proposal_groups.keys())

        vote_rows_so_far: List[Dict[str, Any]] = []
        winner = game_row.get("winner")

        for key in ordered_keys:
            proposal_rows = proposal_groups[key]

            for row in proposal_rows:
                feat = build_vote_features(
                    game_row=game_row,
                    vote_rows_so_far=vote_rows_so_far,
                    current_row=row,
                )
                label = int(row.get("vote_binary", 1 if row.get("vote") == "APPROVE" else 0))

                viewer_alignment = row.get("voter_alignment") or game_row.get("alignments", {}).get(row.get("voter"))
                side_won = (
                    (winner == "good" and viewer_alignment == "good")
                    or (winner == "evil" and viewer_alignment == "evil")
                )

                weight = 1.0
                if side_won:
                    weight += 1.0

                # Additional reward shaping from realized quest outcome (if approved)
                approved = row.get("proposal_approved")
                q = row.get("quest_result_if_approved")
                if approved is True and q in {"SUCCESS", "FAILED"}:
                    if viewer_alignment == "good":
                        if (q == "SUCCESS" and label == 1) or (q == "FAILED" and label == 0):
                            weight += 0.75
                    elif viewer_alignment == "evil":
                        if (q == "FAILED" and label == 1) or (q == "SUCCESS" and label == 0):
                            weight += 0.75

                all_feature_dicts.append(feat)
                all_labels.append(label)
                all_weights.append(weight)

            vote_rows_so_far.extend(proposal_rows)

    if not all_feature_dicts:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), []

    feature_names = sorted(all_feature_dicts[0].keys())
    X = np.array([[fd[k] for k in feature_names] for fd in all_feature_dicts], dtype=float)
    y = np.array(all_labels, dtype=int)
    w = np.array(all_weights, dtype=float)
    return X, y, w, feature_names


def train_team_approval_policy(
    games_path: str | Path,
    votes_path: str | Path,
    model_out_path: str | Path = "team_approval_policy.pkl",
) -> None:
    games = load_jsonl(games_path)
    votes = load_jsonl(votes_path)

    X, y, w, feature_names = build_training_examples(games, votes)
    if len(X) == 0:
        raise RuntimeError("No training examples were built.")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=w)

    pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred) if len(set(y.tolist())) > 1 else float("nan")
    acc = accuracy_score(y, (pred >= 0.5).astype(int))

    artifact = {
        "model": model,
        "feature_names": feature_names,
    }
    with open(model_out_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved team-approval policy model to {model_out_path}")
    print(f"Training examples: {len(X)}")
    print(f"Train Accuracy: {acc:.4f}")
    if auc == auc:
        print(f"Train AUC: {auc:.4f}")


if __name__ == "__main__":
    train_team_approval_policy(
        games_path="logs/parsed/games.jsonl",
        votes_path="logs/parsed/votes.jsonl",
        model_out_path="team_approval_policy.pkl",
    )
