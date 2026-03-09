from __future__ import annotations

import json
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


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return []


def aggregate_state_features_for_candidate(
    game_row: Dict[str, Any],
    vote_rows_up_to_now: List[Dict[str, Any]],
    candidate: str,
) -> Dict[str, float]:
    alignments = game_row["alignments"]
    accuse_count = 0
    trust_count = 0
    on_team_count = 0
    on_team_approve_count = 0
    approve_count = 0
    reject_count = 0
    team_sizes_with_candidate = []
    round_seen = set()

    for row in vote_rows_up_to_now:
        team = safe_list(row.get("team"))
        evil_suspects = safe_list(row.get("latest_evil_suspects"))
        good_suspects = safe_list(row.get("latest_good_suspects"))

        if candidate in evil_suspects: accuse_count += 1
        if candidate in good_suspects: trust_count += 1
        if candidate in team:
            on_team_count += 1
            team_sizes_with_candidate.append(int(row.get("team_size", len(team))))
            if row.get("vote") == "APPROVE": on_team_approve_count += 1

        if row.get("voter") == candidate:
            if row.get("vote") == "APPROVE": approve_count += 1
            else: reject_count += 1
        round_seen.add(int(row.get("round_idx", 0)))

    rounds_observed = max(len(round_seen), 1)

    return {
        "candidate_is_self_evil": float(alignments.get(candidate) == "evil"),
        "candidate_accuse_count": float(accuse_count),
        "candidate_trust_count": float(trust_count),
        "candidate_net_trust": float(trust_count - accuse_count),
        "candidate_on_team_count": float(on_team_count),
        "candidate_on_team_approve_count": float(on_team_approve_count),
        "candidate_avg_team_size": float(sum(team_sizes_with_candidate) / len(team_sizes_with_candidate)) if team_sizes_with_candidate else 0.0,
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
    current_prop: Dict[str, Any],
) -> Dict[str, float]:
    alignments = game_row["alignments"]
    candidate_feats = aggregate_state_features_for_candidate(game_row, vote_rows_up_to_now, candidate)

    return {
        "round_idx": float(current_prop["round_idx"]),
        "proposal_idx_in_round": float(current_prop["proposal_idx_in_round"]),
        "score_good_before": float(current_prop["score_good_before"]),
        "score_evil_before": float(current_prop["score_evil_before"]),
        "consecutive_rejections_before": float(current_prop["consecutive_rejections_before"]),
        "viewer_is_evil": float(alignments.get(viewer) == "evil"),
        "candidate_is_viewer": float(candidate == viewer),
        **candidate_feats,
    }


def build_team_gen_training_examples(
    games: List[Dict[str, Any]],
    votes: List[Dict[str, Any]],
    proposals: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    game_map = {g["game_id"]: g for g in games}
    votes_by_game = defaultdict(list)
    for v in votes:
        votes_by_game[v["game_id"]].append(v)

    all_feature_dicts: List[Dict[str, float]] = []
    all_labels: List[int] = []

    for prop in proposals:
        if not prop.get("proposal_approved"): continue
        
        game_id = prop["game_id"]
        game_row = game_map[game_id]
        leader = prop["leader"]
        alignments = game_row["alignments"]

        if alignments.get(leader) == "good":
            if prop.get("quest_result") != "SUCCESS": continue
        else:
            if prop.get("quest_result") != "FAILED" or prop.get("team_evil_count", 0) > 1: continue

        vote_rows_so_far = [
            v for v in votes_by_game[game_id] 
            if v["round_idx"] < prop["round_idx"] or 
            (v["round_idx"] == prop["round_idx"] and v["proposal_idx_in_round"] < prop["proposal_idx_in_round"])
        ]

        for candidate in game_row["players"]:
            feat = build_example_features(game_row, vote_rows_so_far, leader, candidate, prop)
            label = 1 if candidate in safe_list(prop.get("team")) else 0
            
            all_feature_dicts.append(feat)
            all_labels.append(label)

    feature_names = sorted(all_feature_dicts[0].keys()) if all_feature_dicts else []
    X = np.array([[fd[k] for k in feature_names] for fd in all_feature_dicts], dtype=float)
    y = np.array(all_labels, dtype=int)

    return X, y, feature_names


def train_team_generation_policy(
    games_path: str | Path = "logs/parsed/games.jsonl",
    votes_path: str | Path = "logs/parsed/votes.jsonl",
    proposals_path: str | Path = "logs/parsed/proposals.jsonl",
    model_out_path: str | Path = "team_generation_policy.pkl",
) -> None:
    games = load_jsonl(games_path)
    votes = load_jsonl(votes_path)
    proposals = load_jsonl(proposals_path)

    X, y, feature_names = build_team_gen_training_examples(games, votes, proposals)

    if len(X) == 0:
        raise RuntimeError("No optimal proposals found to train on.")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    with open(model_out_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)

    print(f"Saved Team Gen policy model to {model_out_path}")
    print(f"Training examples: {len(X)}")
    print(f"Train AUC: {auc:.4f}")

if __name__ == "__main__":
    train_team_generation_policy()