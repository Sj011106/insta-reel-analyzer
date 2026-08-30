from agents import run_agent_crew


def analyze_reel(caption: str, api_key: str) -> str:
    print("  → Launching CrewAI multi-agent system...")
    result = run_agent_crew(caption)
    return result