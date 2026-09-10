# llm-avalon-simulator

A full simulation of *The Resistance: Avalon*, the hidden-role social deduction game,
played entirely by LLM agents. Every player is a separate agent with a secret role,
private knowledge, and no visibility into any other agent's internal reasoning, they
only see what a human player at the table would see: chat, proposed teams, votes, and
quest outcomes.

## Why this is hard

Avalon is a bad fit for LLMs by default. Left alone, a model will happily reveal its
role, invent quest outcomes that didn't happen, or reason in vague generalities instead
of naming names. Most of this project's code exists to close those gaps:

- **Fact-grounded reasoning.** Before every discussion turn, each agent is handed the
  literal quest history and vote history as structured facts and explicitly told not to
  contradict them. Responses get checked against those facts (`_message_contradicts_facts`)
  and rejected if they invent an outcome that didn't happen.
- **Leak detection.** Every message is scanned for role leaks (`_looks_like_role_leak`)
  and prompt leaks (`_looks_like_prompt_leak`) before it reaches the game log, with a
  clean fallback if a response leaks.
- **Anti-vagueness retry.** A first pass that produces a generic message ("a trusted
  player", "someone aligned with me") gets one retry with a stricter prompt demanding
  specific player IDs and a concrete reason.
- **Structured accusation output.** Beyond free-form chat, each turn also asks the agent
  for a structured `ACCUSATION` JSON object (`accused_evil`, `accused_good`, plus
  reasoning for each), logged separately from the transcript so per-player suspicion can
  be analyzed or turned into features for downstream RL work.

## Rules engine

`env.py` implements the actual Avalon rulebook as a state machine (`Phase.DISCUSSION` ->
`PROPOSE` -> `TEAM_DISCUSSION` -> `TEAM_VOTE` -> `QUEST` -> ... -> `ASSASSINATE`), not an
approximation of it:

- Correct team size and fail-count-required tables for every player count from 5 to 10,
  including the round-4-needs-2-fails rule that only applies at 7+ players.
- Five proposals per round; a fifth rejected proposal ends the game for Evil immediately.
- Good wins at 3 successful quests, but only provisionally, the Assassin then gets one
  guess at Merlin's identity to steal the win back.
- `assert_avalon_config` crashes loudly on startup if the round config doesn't exactly
  match the real rulebook for the given player count, so a bug can't silently produce an
  illegal game.

## Roles

Each role (Merlin, Percival, Loyal Servant, Assassin, Morgana, Mordred, Minion of
Mordred) has its own brief stored as a notebook under `roles/`, loaded and handed to
that role's agent as private context, alongside role-appropriate private knowledge
(Merlin sees who's evil, Percival sees Merlin and Morgana but not which is which, and so
on). Evil-role counts per lobby size follow the standard Avalon table (2 evil at 5
players, up to 4 at 10).

## Running it

Requires an OpenAI-compatible LLM backend (`pip install openai`, then set
`OPENAI_API_KEY` and optionally `AVALON_API_BASE`):

```bash
pip install -r requirements.txt
python run_game.py
```

You'll be prompted for a player count (5-10); the simulator builds a legal role list,
assigns agents, and runs the full game to completion, printing discussion, proposals,
votes, quest results, and (if it gets there) the Assassin's endgame guess.

## Structure

| File | Role |
|---|---|
| `env.py` | Game state machine and rules engine |
| `avalon_role_agent.py` | Per-agent reasoning: discussion, proposals, votes, accusations, all the anti-hallucination guardrails |
| `llm_caller.py` | Thin OpenAI-compatible LLM client wrapper |
| `roles/*.ipynb` | Per-role briefs and private-knowledge notes |
| `run_game.py` | CLI entry point: builds a legal lineup and runs one game |
