# llm-avalon-simulator

Files: 

env.py — Avalon game state and rules: phases (discussion → propose → vote → quest → assassination), team sizes, fail thresholds, win/loss, and step-by-step logging.

avalon_role_agent.py — One player’s “brain”: builds prompts from role/state, calls the LLM for speak / propose_team / vote_on_team / mission_action / assassinate, and parses/validates responses (no role leak, no contradiction with facts). This was included in llm and env before but to make things easier I think putting the agent logic into its own file is better.. so the LLM module only handles API calls, and everything about how a player decides (prompts, parsing, checks) lives in one place instead of being mixed into the caller or the env.

llm_caller.py — sends chat requests to the configured model, handles JSON extraction from model text, and retries/timeouts.

run_game.py — Entry point: loads role briefs from the roles/ folder, prompts for player count, builds role list and agents, creates AvalonEnv, runs the game, and prints the outcome.
