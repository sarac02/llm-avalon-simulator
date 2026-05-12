# llm-avalon-simulator
# LLM Avalon Simulator

A multi-agent LLM simulator for **The Resistance: Avalon**, designed to study how language-model agents reason under hidden information, deception, partial observability, team formation, voting, and role-based objectives.

The project simulates a full Avalon game loop with LLM-driven players. Each agent receives a private role, observes the public game state, participates in discussion, proposes teams, votes on teams, chooses quest actions, and performs endgame assassination when applicable. The environment enforces Avalon rules while the agents make strategic decisions through role-conditioned prompts and structured response parsing.

---

## Overview

Avalon is a hidden-role social deduction game where players are divided into two teams:

- **Good players** try to complete three successful quests.
- **Evil players** try to sabotage quests or identify Merlin at the end.
- **Merlin** knows the evil players but must avoid revealing themselves.
- **Assassin** can steal the win for evil by correctly identifying Merlin after three successful quests.
- **Percival**, **Morgana**, loyal servants, and other roles introduce uncertainty and deception.

This simulator uses LLM agents to model these behaviors and evaluate whether language models can reason strategically in a multi-agent, partially observable environment.

---

## Key Features

- Full Avalon game loop:
  - Discussion
  - Team proposal
  - Team vote
  - Quest execution
  - Win/loss tracking
  - Merlin assassination phase

- Role-conditioned LLM agents:
  - Merlin, Assassin, Percival, Morgana, Loyal Servant, and other supported roles
  - Each agent receives only the information allowed by its role
  - Agents generate natural language reasoning and structured game actions

- Rule-enforced environment:
  - Validates team sizes
  - Tracks proposal attempts
  - Enforces majority vote approval
  - Handles failed proposals
  - Computes quest success/failure
  - Triggers assassination after three good quest wins

- Modular architecture:
  - Game environment and rules are separated from agent logic
  - LLM API calls are isolated in a caller module
  - Role behavior, response parsing, and validation live in the agent module
  - RL-style policies can be plugged in for team generation, suspicion, and approval behavior

- Logging and evaluation support:
  - Step-by-step game traces
  - Proposal, vote, quest, and outcome logging
  - Log parsing utilities for downstream analysis and RL-style datasets

---


Files: 

env.py — Avalon game state and rules: phases (discussion → propose → vote → quest → assassination), team sizes, fail thresholds, win/loss, and step-by-step logging.

avalon_role_agent.py — One player’s “brain”: builds prompts from role/state, calls the LLM for speak / propose_team / vote_on_team / mission_action / assassinate, and parses/validates responses (no role leak, no contradiction with facts). This was included in llm and env before but to make things easier I think putting the agent logic into its own file is better.. so the LLM module only handles API calls, and everything about how a player decides (prompts, parsing, checks) lives in one place instead of being mixed into the caller or the env.

llm_caller.py — sends chat requests to the configured model, handles JSON extraction from model text, and retries/timeouts.

run_game.py — Entry point: loads role briefs from the roles/ folder, prompts for player count, builds role list and agents, creates AvalonEnv, runs the game, and prints the outcome.


<img width="500" height="313" alt="Screenshot 2026-03-05 at 3 04 05 AM" src="https://github.com/user-attachments/assets/d93f221b-a0f0-4555-958c-37c8f2ae9185" />


<img width="502" height="313" alt="Screenshot 2026-03-05 at 3 04 18 AM" src="https://github.com/user-attachments/assets/1762ef7f-6574-493e-8b41-a09a2bc26285" />


<img width="500" height="313" alt="Screenshot 2026-03-05 at 3 04 47 AM" src="https://github.com/user-attachments/assets/dda0d24f-04ee-4c34-b3d0-b3be0f6d50d9" />
