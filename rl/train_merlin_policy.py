from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_merlin_role(role_name: str | None) -> bool:
    return role_name is not None and "merlin" in role_name.lower()


def is_evil_role(role_name: str | None) -> bool:
    if role_name is None:
        return False
    key = role_name.lower()
    return any(tag in key for tag in ["assassin", "minion", "morgana", "mordred", "oberon"])


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return []


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
            key=lambda r: (r["round_idx"], r["proposal_idx_in_round"], r["voter"]),
        )
    return game_map, votes_by_game


def aggregate_state_features_for_candidate(
    game_row: Dict[str, Any],
    vote_rows_up_to_now: List[Dict[str, Any]],
    candidate: str,
) -> Dict[str, float]:
    """
    Build features describing how Merlin-like `candidate` looks at the current state.
    """
    roles = game_row["roles"]
    alignments = game_row["alignments"]
    players = game_row["players"]

    accuse_count = 0
    trust_count = 0
    on_team_count = 0
    on_team_approve_count = 0
    approve_count = 0
    reject_count = 0

    team_sizes_with_candidate = []
    round_seen = set()

    for row in vote_rows_up_to_now:
        team = safe_list(row["team"])
        evil_suspects = safe_list(row.get("latest_evil_suspects"))
        good_suspects = safe_list(row.get("latest_good_suspects"))

        if candidate in evil_suspects:
            accuse_count += 1
        if candidate in good_suspects:
            trust_count += 1
        if candidate in team:
            on_team_count += 1
            team_sizes_with_candidate.append(int(row["team_size"]))
            if row["vote"] == "APPROVE":
                on_team_approve_count += 1

        if row["voter"] == candidate:
            if row["vote"] == "APPROVE":
                approve_count += 1
            else:
                reject_count += 1

        round_seen.add(int(row["round_idx"]))

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


def build_example_features(
    game_row: Dict[str, Any],
    vote_rows_up_to_now: List[Dict[str, Any]],
    viewer: str,
    candidate: str,
    current_row: Dict[str, Any],
) -> Dict[str, float]:
    """
    Features for 'how likely is candidate Merlin from viewer's perspective at this point?'
    """
    roles = game_row["roles"]
    alignments = game_row["alignments"]
    players = game_row["players"]

    candidate_feats = aggregate_state_features_for_candidate(game_row, vote_rows_up_to_now, candidate)

    current_team = safe_list(current_row["team"])
    evil_on_team_count = int(current_row["evil_on_team_count"])

    return {
        "round_idx": float(current_row["round_idx"]),
        "proposal_idx_in_round": float(current_row["proposal_idx_in_round"]),
        "score_good_before": float(current_row["score_good_before"]),
        "score_evil_before": float(current_row["score_evil_before"]),
        "consecutive_rejections_before": float(current_row["consecutive_rejections_before"]),
        "viewer_is_evil": float(alignments.get(viewer) == "evil"),
        "viewer_is_assassin": float("assassin" in str(roles.get(viewer, "")).lower()),
        "candidate_is_viewer": float(candidate == viewer),
        "candidate_on_current_team": float(candidate in current_team),
        "current_team_size": float(len(current_team)),
        "current_team_evil_count": float(evil_on_team_count),
        **candidate_feats,
    }


def build_merlin_training_examples(
    games: List[Dict[str, Any]],
    votes: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    game_map, votes_by_game = build_game_maps(games, votes)

    all_feature_dicts: List[Dict[str, float]] = []
    all_labels: List[int] = []
    all_weights: List[float] = []

    for game_id, rows in votes_by_game.items():
        game_row = game_map[game_id]
        roles = game_row["roles"]
        alignments = game_row["alignments"]
        players = game_row["players"]
        winner = game_row["winner"]

        merlin_candidates = [p for p, r in roles.items() if is_merlin_role(r)]
        if len(merlin_candidates) != 1:
            continue
        true_merlin = merlin_candidates[0]

        # Build examples only from evil viewers; they care about Merlin identification
        evil_viewers = [p for p in players if alignments.get(p) == "evil"]
        if not evil_viewers:
            continue

        vote_rows_so_far: List[Dict[str, Any]] = []

        for row in rows:
            for viewer in evil_viewers:
                for candidate in players:
                    feat = build_example_features(
                        game_row=game_row,
                        vote_rows_up_to_now=vote_rows_so_far,
                        viewer=viewer,
                        candidate=candidate,
                        current_row=row,
                    )

                    label = 1 if candidate == true_merlin else 0

                    # Reward-like weight:
                    # winning evil games matter more, and assassination-hit evil wins matter even more
                    terminal_reward = 1.0 if winner == "evil" else -1.0
                    weight = 1.0
                    if winner == "evil":
                        weight += 1.0
                        if game_row.get("assassination_hit_merlin") is True:
                            weight += 1.0
                    else:
                        weight += 0.0

                    all_feature_dicts.append(feat)
                    all_labels.append(label)
                    all_weights.append(weight)

            vote_rows_so_far.append(row)

    feature_names = sorted(all_feature_dicts[0].keys()) if all_feature_dicts else []
    X = np.array([[fd[k] for k in feature_names] for fd in all_feature_dicts], dtype=float)
    y = np.array(all_labels, dtype=int)
    w = np.array(all_weights, dtype=float)

    return X, y, w, feature_names


def train_merlin_policy(
    games_path: str | Path,
    votes_path: str | Path,
    model_out_path: str | Path = "merlin_policy.pkl",
) -> None:
    games = load_jsonl(games_path)
    votes = load_jsonl(votes_path)

    X, y, w, feature_names = build_merlin_training_examples(games, votes)

    if len(X) == 0:
        raise RuntimeError("No training examples were built.")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=w)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    artifact = {
        "model": model,
        "feature_names": feature_names,
    }

    with open(model_out_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved Merlin policy model to {model_out_path}")
    print(f"Training examples: {len(X)}")
    print(f"Train AUC: {auc:.4f}")


if __name__ == "__main__":
    train_merlin_policy(
        games_path="logs/parsed/games.jsonl",
        votes_path="logs/parsed/votes.jsonl",
        model_out_path="merlin_policy.pkl",
    )