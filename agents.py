from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from knowledge_base import search_similar_projects
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── LangGraph State ───────────────────────────────────────────────
# This is what gets passed between agents
# Each agent reads from state and writes back to it
class AgentState(TypedDict):
    reel_content: str           # original reel content
    github_context: str         # RAG retrieved projects
    identified_projects: str    # Agent 1 output
    github_research: str        # Agent 2 output
    roadmap: str                # Agent 3 output
    final_roadmap: str          # Agent 4 output


# ── LLM Setup ────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.8-27b",
    max_tokens=4096
)


# ── Node 1: RAG Search ────────────────────────────────────────────
def rag_search_node(state: AgentState) -> AgentState:
    """
    First node — searches FAISS knowledge base for similar projects.
    This is the Retrieval step of RAG.
    """
    print("\n  📚 Node 1: RAG Search")
    print("  → Searching FAISS knowledge base...")

    similar_projects = search_similar_projects(state["reel_content"], n_results=5)

    if similar_projects:
        github_context = "RETRIEVED FROM FAISS KNOWLEDGE BASE:\n"
        for proj in similar_projects:
            github_context += f"""
- {proj['name']} ({proj['similarity']}% similar)
  URL: {proj['url']}
  Language: {proj['language']}
  Stars: ⭐ {proj['stars']}
  Description: {proj['description']}
"""
        print(f"  ✅ Found {len(similar_projects)} similar projects")
    else:
        github_context = "No similar projects found in knowledge base."
        print("  ⚠ No similar projects found")

    return {**state, "github_context": github_context}


# ── Node 2: Project Identifier Agent ─────────────────────────────
def project_identifier_node(state: AgentState) -> AgentState:
    """
    Agent 1 — identifies what CS projects the reel is about.
    """
    print("\n  🤖 Node 2: Project Identifier Agent")
    print("  → Identifying projects...")

    messages = [
        SystemMessage(content="""You are an expert CS educator who identifies CS projects 
        from raw text extracted from Instagram reels. The text may be messy since 
        it comes from OCR and audio transcription. You extract project names, 
        what they do, and what technologies are involved. Be concise and clear."""),
        HumanMessage(content=f"""Analyze this content from an Instagram reel and identify 
ALL CS projects being shown or described.

REEL CONTENT:
{state['reel_content']}

For each project:
- Project name
- What it does in 1-2 sentences  
- Main technologies mentioned or implied

If multiple projects are shown, identify ALL of them.""")
    ]

    response = llm.invoke(messages)
    print("  ✅ Project Identifier done")
    return {**state, "identified_projects": response.content}


# ── Node 3: GitHub Research Agent ────────────────────────────────
def github_research_node(state: AgentState) -> AgentState:
    """
    Agent 2 — matches identified projects with real GitHub repos.
    """
    print("\n  🤖 Node 3: GitHub Research Agent")
    print("  → Matching with GitHub repos...")

    messages = [
        SystemMessage(content="""You are a GitHub research specialist who matches CS projects 
        with real open-source examples. You always provide actual GitHub URLs so 
        students can explore real code and learn from working examples."""),
        HumanMessage(content=f"""These CS projects were identified from an Instagram reel:

{state['identified_projects']}

These are real GitHub projects retrieved from our knowledge base:
{state['github_context']}

For each identified project:
- Match it with the most relevant GitHub repos from above
- Explain why each repo is relevant  
- Note the tech stack used in the real repos
- Always include the actual GitHub URLs""")
    ]

    response = llm.invoke(messages)
    print("  ✅ GitHub Research Agent done")
    return {**state, "github_research": response.content}


# ── Node 4: Roadmap Builder Agent ────────────────────────────────
def roadmap_builder_node(state: AgentState) -> AgentState:
    """
    Agent 3 — builds a detailed beginner roadmap.
    """
    print("\n  🤖 Node 4: Roadmap Builder Agent")
    print("  → Building roadmap...")

    messages = [
        SystemMessage(content="""You are a CS mentor who builds clear beginner-friendly 
        learning roadmaps. You break complex projects into weekly steps that a 
        first or second year CS student can follow. You always ground roadmaps 
        in real technologies and actual GitHub projects."""),
        HumanMessage(content=f"""Using the identified projects and GitHub research below,
build a comprehensive beginner roadmap.

IDENTIFIED PROJECTS:
{state['identified_projects']}

GITHUB RESEARCH:
{state['github_research']}

For EACH project provide:

1. PROJECT NAME

2. WHAT IT DOES
   2-3 simple sentences for a complete beginner.

3. TECH STACK
   Each technology and exactly why it is needed.

4. BEGINNER ROADMAP
   Week by week (3-4 weeks). Each week needs:
   - A clear goal
   - 2-3 specific tasks

5. REAL GITHUB PROJECTS TO STUDY
   List the GitHub repos with their actual URLs.

6. FREE RESOURCES
   2 specific free resources to get started.

Separate each project with: ─────────────────""")
    ]

    response = llm.invoke(messages)
    print("  ✅ Roadmap Builder done")
    return {**state, "roadmap": response.content}


# ── Node 5: Quality Reviewer Agent ───────────────────────────────
def quality_reviewer_node(state: AgentState) -> AgentState:
    """
    Agent 4 — reviews and improves the roadmap.
    """
    print("\n  🤖 Node 5: Quality Reviewer Agent")
    print("  → Reviewing and polishing...")

    messages = [
        SystemMessage(content="""You are a senior CS educator who reviews learning roadmaps 
        for quality. You ensure roadmaps are complete, accurate, beginner-friendly,
        and have clear actionable steps. You improve vague sections and make sure
        all GitHub URLs are preserved."""),
        HumanMessage(content=f"""Review and improve this CS project roadmap:

{state['roadmap']}

Check for:
- Is each week clear and achievable for a beginner?
- Are tech stack explanations simple enough?
- Are the GitHub URLs included and preserved?
- Is difficulty progression logical?
- Are there any vague steps that need more detail?

Fix any issues and return the final polished roadmap.
Keep the same format. Do NOT remove GitHub URLs.
IMPORTANT: Do NOT include any meta-commentary like "Key Improvements Made", "Changes I made", or "What I fixed". Only output the roadmap itself — nothing before or after it.""")
    ]

    response = llm.invoke(messages)
    print("  ✅ Quality Reviewer done")
    return {**state, "final_roadmap": response.content}


# ── Build the LangGraph ───────────────────────────────────────────
def build_graph():
    """
    Builds the LangGraph agent workflow.
    Nodes are agents, edges define the flow between them.
    """
    graph = StateGraph(AgentState)

    # add all nodes (agents)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("project_identifier", project_identifier_node)
    graph.add_node("github_research", github_research_node)
    graph.add_node("roadmap_builder", roadmap_builder_node)
    graph.add_node("quality_reviewer", quality_reviewer_node)

    # define the flow between nodes
    # this is the graph's edges
    graph.set_entry_point("rag_search")
    graph.add_edge("rag_search", "project_identifier")
    graph.add_edge("project_identifier", "github_research")
    graph.add_edge("github_research", "roadmap_builder")
    graph.add_edge("roadmap_builder", "quality_reviewer")
    graph.add_edge("quality_reviewer", END)

    return graph.compile()


# ── Main function called by analyzer.py ──────────────────────────
def run_agent_crew(reel_content: str) -> str:
    """
    Runs the full LangGraph multi-agent pipeline.
    """
    print("\n🕸️  Starting LangGraph multi-agent pipeline...\n")

    # build the graph
    app = build_graph()

    # initial state
    initial_state = AgentState(
        reel_content=reel_content,
        github_context="",
        identified_projects="",
        github_research="",
        roadmap="",
        final_roadmap=""
    )

    # run the graph
    final_state = app.invoke(initial_state)

    print("\n✅ LangGraph pipeline complete!\n")
    return final_state["final_roadmap"]