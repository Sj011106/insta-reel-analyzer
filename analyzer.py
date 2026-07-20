from groq import Groq


def analyze_reel(caption: str, api_key: str) -> str:

    client = Groq(api_key=api_key)

    prompt = f"""You are a helpful CS mentor assistant. A computer science student found an Instagram reel
that shows or describes a CS project, but they don't know how to build it.

Here is the caption from the reel:
---
{caption}
---

Based on this caption, please provide the following in a clear, beginner-friendly format:

1. PROJECT NAME
   What is this project called? Give it a clear name.

2. WHAT IT DOES
   Explain what this project does in 2-3 simple sentences.

3. TECH STACK
   List every technology needed. For each one, write one sentence explaining WHY it is used.

4. BEGINNER ROADMAP
   Break it down week by week (4-6 weeks). Each week should have:
   - A clear goal
   - 2-3 specific tasks

5. WHAT TO SEARCH ON GITHUB
   Give 3-5 search terms to find similar open-source projects.

6. RESOURCES TO GET STARTED
   List 2-3 free resources for the most important technology.
"""

    print("  → Sending caption to Groq...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
    )

    return response.choices[0].message.content