import asyncio
import json
import litellm
import re
import random

MODEL = "ollama/deepseek-r1:8b" 
OLLAMA_URL = "http://localhost:11434"

GAME_RULES = """
### AVALON STRATEGIC PROTOCOL ###
1. IDENTITY & COVER: NEVER reveal your role name. All claim to be 'Loyal Servant'.
2. WIN CONDITIONS: Good wins if 3 missions pass. Evil wins if 3 fail OR they find Merlin.
3. MERLIN: You know the Spies. Guide Good without being caught.
4. ASSASSIN: Find Merlin. Watch for players who 'know too much'.
"""

class AvalonAgent:
    def __init__(self, name, role, goal, secret_info=""):
        self.name = name
        self.role = role
        self.goal = goal
        self.secret_info = secret_info
        self.mental_map = "Initial state: Awaiting briefing."

    async def _call_llm(self, system_prompt, user_prompt):
        """Standardized helper for all LLM interactions."""
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=OLLAMA_URL,
                max_tokens=2048, 
                temperature=0.6,
                timeout=300 
            )
            content = response.choices[0].message.content
            return content.split("</think>")[-1].strip() if "</think>" in content else content
        except Exception as e:
            return f"Error: {e}"

    async def initialize_agent(self):
        """
        The agent processes their role and knowledge before the game starts.
        """
        system_prompt = (
            f"{GAME_RULES}\n"
            f"INITIALIZATION: You are {self.name}, the {self.role}.\n"
            f"SECRET KNOWLEDGE: {self.secret_info}\n"
            "TASK: Based on your role and knowledge, what is your initial strategy? "
            "Write your PRIVATE MENTAL MAP. Do NOT speak to the table yet."
        )
        self.mental_map = await self._call_llm(system_prompt, "Formulate your plan.")
        print(f"🧠 {self.name} ({self.role}) has formulated a secret strategy.")

    async def get_consolidated_action(self, state, history):
        system_prompt = (
            f"{GAME_RULES}\n"
            f"YOU ARE: {self.name}. PUBLIC PERSONA: Loyal Servant.\n"
            f"SECRET ROLE: {self.role}. {self.secret_info}\n"
            f"LAST MENTAL MAP: {self.mental_map}\n"
            "INSTRUCTION: Update your map, then provide a public statement. "
            "You MUST end your response with: STATEMENT: [your public message]"
        )
        usr_prompt = f"History: {json.dumps(history)}\n\nWhat is your next move?"
        
        clean_content = await self._call_llm(system_prompt, usr_prompt)
        
        if "STATEMENT:" in clean_content:
            parts = clean_content.split("STATEMENT:")
            self.mental_map = parts[0].replace("NEW_MAP:", "").strip()
            return parts[1].strip()
        
        return clean_content[:200]

class AvalonEnvironment:
    def __init__(self, players):
        self.players = {p.name: p for p in players}
        self.player_names = list(self.players.keys())
        self.history = []
        self.current_speaker = "Alice"
        self.turn_count = 0
        self.max_turns = 10
        self.consecutive_turns = {name: 0 for name in self.player_names}

    def get_next_speaker(self, last_message):
        mentioned = [n for n in self.player_names if re.search(rf'\b{n}\b', last_message, re.IGNORECASE) and n != self.current_speaker]
        if mentioned and self.consecutive_turns.get(mentioned[0], 0) < 2:
            return mentioned[0]
        counts = {name: sum(1 for h in self.history if h['player'] == name) for name in self.player_names}
        candidates = [n for n in self.player_names if n != self.current_speaker]
        return min(candidates, key=lambda n: counts[n])

    async def run_turn(self):
        p = self.players[self.current_speaker]
        for name in self.player_names:
            self.consecutive_turns[name] = self.consecutive_turns[name] + 1 if name == p.name else 0
        state = {"turn": self.turn_count, "active": p.name}
        statement = await p.get_consolidated_action(state, self.history)
        self.history.append({"player": p.name, "message": statement})
        print(f"💬 {p.name}: {statement}")
        self.current_speaker = self.get_next_speaker(statement)
        self.turn_count += 1

async def main():
    agents = [
        AvalonAgent("Alice", "Merlin", "Guide Good.", "Secret: Bob and Eve are Evil."),
        AvalonAgent("Bob", "Assassin", "Find Merlin.", "Secret: Eve is your Evil teammate."),
        AvalonAgent("Charlie", "Loyal Servant", "Pass missions.", "Secret: You know nothing yet."),
        AvalonAgent("Dave", "Loyal Servant", "Find spies.", "Secret: You know nothing yet."),
        AvalonAgent("Eve", "Minion of Mordred", "Sabotage missions.", "Secret: Bob is your Evil teammate.")
    ]
    
    print("🛡️ AVALON SIMULATION: INITIALIZING ROLES 🛡️")
    print("-" * 60)
    init_tasks = [a.initialize_agent() for a in agents]
    await asyncio.gather(*init_tasks)
    
    print("\n🚀 STARTING GAME DISCUSSION 🚀")
    print("-" * 60)
    env = AvalonEnvironment(agents)
    while env.turn_count < env.max_turns:
        await env.run_turn()
        await asyncio.sleep(0.5)

    print("-" * 60)
    print(f"🔍 MERLIN'S FINAL STRATEGY LOG:\n{agents[0].mental_map}")

if __name__ == "__main__":
    asyncio.run(main())