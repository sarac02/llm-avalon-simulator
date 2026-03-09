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


def build_game_maps(games, votes, missions):
    game_map = {g["game_id"]: g for g in games}
    votes_by_game = defaultdict(list)
    for v in votes:
        votes_by_game[v["game_id"]].append(v)
    for gid in votes_by_game:
        votes_by_game[gid] = sorted(votes_by_game[gid], key=lambda r: (r["round_idx"], r["proposal_idx_in_round"], r["voter"]))
    missions_by_game_and_key = defaultdict(dict)
    for m in missions:
        missions_by_game_and_key[m["game_id"]][(int(m["round_idx"]), int(m["proposal_id"]))] = m
    return game_map, votes_by_game, missions_by_game_and_key


def aggregate_candidate_features(vote_rows_up_to_now, mission_rows_up_to_now, candidate, viewer):
    accuse_count = trust_count = approve_count = reject_count = 0
    on_team_count = on_team_approve_count = on_team_reject_count = 0
    proposed_by_evil_count = proposed_by_good_count = 0
    accused_leader_count = trusted_leader_count = 0
    accuses_team_member_total = trusts_team_member_total = 0
    latest_reasoning_len_sum = 0
    visible_rounds = set()
    team_sizes_with_candidate = []
    joint_team_with_viewer = 0

    for row in vote_rows_up_to_now:
        team = safe_list(row.get("team"))
        evil_suspects = safe_list(row.get("latest_evil_suspects"))
        good_suspects = safe_list(row.get("latest_good_suspects"))

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
    for mission in mission_rows_up_to_now:
        team = safe_list(mission.get("team"))
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


def build_example_features(game_row, vote_rows_up_to_now, mission_rows_up_to_now, viewer, candidate, current_row):
    roles = game_row["roles"]
    alignments = game_row["alignments"]
    current_team = safe_list(current_row.get("team"))
    candidate_feats = aggregate_candidate_features(vote_rows_up_to_now, mission_rows_up_to_now, candidate, viewer)
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
        "viewer_on_current_team": float(viewer in current_team),
        "leader_is_candidate": float(candidate == current_row.get("leader")),
        "leader_is_viewer": float(viewer == current_row.get("leader")),
        "leader_alignment_evil": float(current_row.get("leader_alignment") == "evil"),
        "current_team_size": float(len(current_team)),
        "current_team_evil_count": float(current_row.get("evil_on_team_count", 0)),
        "current_team_has_candidate_and_viewer": float(candidate in current_team and viewer in current_team),
        **candidate_feats,
    }


def build_suspicion_training_examples(games, votes, missions):
    game_map, votes_by_game, missions_by_game_and_key = build_game_maps(games, votes, missions)
    all_feature_dicts = []
    all_labels = []
    all_weights = []

    for game_id, rows in votes_by_game.items():
        game_row = game_map[game_id]
        alignments = game_row["alignments"]
        players = game_row["players"]
        winner = game_row.get("winner")
        vote_rows_so_far = []
        mission_rows_so_far = []
        mission_lookup = missions_by_game_and_key.get(game_id, {})
        added_mission_keys = set()

        for row in rows:
            current_key = (int(row["round_idx"]), int(row["proposal_id"]))
            team_now = safe_list(row.get("team"))
            for viewer in players:
                for candidate in players:
                    feat = build_example_features(game_row, vote_rows_so_far, mission_rows_so_far, viewer, candidate, row)
                    label = 1 if alignments.get(candidate) == "evil" else 0
                    weight = 1.0
                    if candidate == viewer:
                        weight += 0.25
                    if winner == "good" and alignments.get(candidate) == "evil":
                        weight += 0.5
                    if row.get("proposal_approved") and row.get("quest_result_if_approved") == "FAILED" and candidate in team_now:
                        weight += 0.75
                    if row.get("proposal_approved") and row.get("quest_result_if_approved") == "SUCCESS" and candidate in team_now:
                        weight += 0.15
                    if candidate == row.get("leader"):
                        weight += 0.2
                    all_feature_dicts.append(feat)
                    all_labels.append(label)
                    all_weights.append(weight)
            vote_rows_so_far.append(row)
            if row.get("proposal_approved") and current_key in mission_lookup and current_key not in added_mission_keys:
                mission_rows_so_far.append(mission_lookup[current_key])
                added_mission_keys.add(current_key)

    if not all_feature_dicts:
        raise RuntimeError("No training examples were built.")
    feature_names = sorted(all_feature_dicts[0].keys())
    X = np.array([[fd[k] for k in feature_names] for fd in all_feature_dicts], dtype=float)
    y = np.array(all_labels, dtype=int)
    w = np.array(all_weights, dtype=float)
    return X, y, w, feature_names


def train_suspicion_policy(games_path, votes_path, missions_path, model_out_path="suspicion_policy.pkl"):
    games = load_jsonl(games_path)
    votes = load_jsonl(votes_path)
    missions = load_jsonl(missions_path)
    X, y, w, feature_names = build_suspicion_training_examples(games, votes, missions)
    model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(X, y, sample_weight=w)
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    with open(model_out_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"Saved suspicion policy model to {model_out_path}")
    print(f"Training examples: {len(X)}")
    print(f"Positive rate (evil candidates): {float(np.mean(y)):.4f}")
    print(f"Train AUC: {auc:.4f}")


if __name__ == "__main__":
    train_suspicion_policy("logs/parsed/games.jsonl", "logs/parsed/votes.jsonl", "logs/parsed/missions.jsonl", "suspicion_policy.pkl")
