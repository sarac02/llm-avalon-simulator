# llm-avalon-simulator

Files: 

env.py — Avalon game state and rules: phases (discussion → propose → vote → quest → assassination), team sizes, fail thresholds, win/loss, and step-by-step logging.

avalon_role_agent.py — One player’s “brain”: builds prompts from role/state, calls the LLM for speak / propose_team / vote_on_team / mission_action / assassinate, and parses/validates responses (no role leak, no contradiction with facts). This was included in llm and env before but to make things easier I think putting the agent logic into its own file is better.. so the LLM module only handles API calls, and everything about how a player decides (prompts, parsing, checks) lives in one place instead of being mixed into the caller or the env.

llm_caller.py — sends chat requests to the configured model, handles JSON extraction from model text, and retries/timeouts.

run_game.py — Entry point: loads role briefs from the roles/ folder, prompts for player count, builds role list and agents, creates AvalonEnv, runs the game, and prints the outcome.


<img width="500" height="313" alt="Screenshot 2026-03-05 at 3 04 05 AM" src="https://github.com/user-attachments/assets/d93f221b-a0f0-4555-958c-37c8f2ae9185" />


<img width="502" height="313" alt="Screenshot 2026-03-05 at 3 04 18 AM" src="https://github.com/user-attachments/assets/1762ef7f-6574-493e-8b41-a09a2bc26285" />


<img width="500" height="313" alt="Screenshot 2026-03-05 at 3 04 47 AM" src="https://github.com/user-attachments/assets/dda0d24f-04ee-4c34-b3d0-b3be0f6d50d9" />
