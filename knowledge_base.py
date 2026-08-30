import os
import requests
import faiss
import numpy as np
import json
import pickle
from dotenv import load_dotenv
import time

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

DB_PATH = "./faiss_db"
INDEX_FILE = f"{DB_PATH}/index.faiss"
METADATA_FILE = f"{DB_PATH}/metadata.pkl"
DIMENSION = 384  # embedding size for all-MiniLM-L6-v2


def get_embeddings(texts: list) -> np.ndarray:
    """
    Creates embeddings using a simple TF-IDF style approach.
    No scipy, no sentence-transformers needed.
    Uses Groq to generate embeddings via text similarity.
    """
    from groq import Groq
    import os

    # Simple hash-based embedding that works without scipy
    # Each text gets converted to a fixed-size vector
    embeddings = []
    for text in texts:
        # normalize text
        text = text.lower()[:500]
        # create a simple bag of words vector
        vector = np.zeros(DIMENSION, dtype=np.float32)
        words = text.split()
        for i, word in enumerate(words[:DIMENSION]):
            # hash each word to a dimension
            idx = hash(word) % DIMENSION
            vector[idx] += 1.0
        # normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        embeddings.append(vector)
    return np.array(embeddings, dtype=np.float32)


def load_or_create_index():
    """Load existing FAISS index or create a new one."""
    os.makedirs(DB_PATH, exist_ok=True)

    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print(f"✅ Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded {index.ntotal} projects from database")
        return index, metadata
    else:
        # cosine similarity using Inner Product on normalized vectors
        index = faiss.IndexFlatIP(DIMENSION)
        metadata = []
        print("✅ Created new FAISS index")
        return index, metadata


def save_index(index, metadata):
    """Save FAISS index and metadata to disk."""
    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def fetch_github_repos(query: str, num_repos: int = 8) -> list:
    """Search GitHub for CS project repos."""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"{query} in:readme in:description",
        "sort": "stars",
        "order": "desc",
        "per_page": num_repos
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"  ⚠ GitHub API error: {response.status_code}")
        return []
    return response.json().get("items", [])


def fetch_readme(repo_full_name: str) -> str:
    """Fetches README content of a GitHub repo."""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"
    }
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text[:1000]
    return ""


def build_knowledge_base():
    """Builds the FAISS knowledge base from GitHub repos."""
    index, metadata = load_or_create_index()

    # track existing repo IDs to avoid duplicates
    existing_ids = {m["id"] for m in metadata}

    cs_project_queries = [
        "machine learning project python beginner",
        "face recognition opencv python",
        "sentiment analysis nlp python",
        "web scraper python beautifulsoup",
        "chatbot python nlp",
        "recommendation system python",
        "object detection yolo python",
        "stock price prediction lstm",
        "image classification deep learning",
        "todo app react javascript",
        "flask rest api python",
        "data visualization python matplotlib",
        "fraud detection machine learning",
        "heart disease prediction scikit-learn",
        "twitter sentiment analysis python",
        "personal finance tracker python",
        "crud app nodejs express",
        "weather app python api",
        "snake game python pygame",
        "calculator app javascript"
    ]

    print(f"\n🔍 Building GitHub knowledge base...")
    print(f"   Searching {len(cs_project_queries)} CS project categories\n")

    total_added = 0
    new_docs = []
    new_metadata = []

    for i, query in enumerate(cs_project_queries):
        print(f"[{i+1}/{len(cs_project_queries)}] Searching: '{query}'")
        repos = fetch_github_repos(query, num_repos=8)
        added = 0

        for repo in repos:
            repo_id = str(repo["id"])
            if repo_id in existing_ids:
                continue

            readme = fetch_readme(repo["full_name"])
            description = repo.get("description") or ""
            document = f"{repo['name']} {description} {readme}"

            if not document.strip():
                continue

            new_docs.append(document)
            new_metadata.append({
                "id": repo_id,
                "name": repo["name"],
                "url": repo["html_url"],
                "stars": repo["stargazers_count"],
                "language": repo.get("language") or "Unknown",
                "description": description,
            })
            existing_ids.add(repo_id)
            added += 1
            time.sleep(0.3)

        total_added += added
        print(f"  ✅ Added {added} new repos")
        time.sleep(0.5)

    # add all new docs to FAISS at once
    if new_docs:
        print(f"\n→ Creating embeddings for {len(new_docs)} repos...")
        embeddings = get_embeddings(new_docs)
        index.add(embeddings)
        metadata.extend(new_metadata)
        save_index(index, metadata)

    print(f"\n{'='*50}")
    print(f"✅ Knowledge base built!")
    print(f"   Total projects in DB: {index.ntotal}")
    print(f"   New projects added: {total_added}")
    print(f"{'='*50}\n")


def search_similar_projects(query: str, n_results: int = 5) -> list:
    """
    Finds most similar GitHub projects using FAISS similarity search.
    """
    index, metadata = load_or_create_index()

    if index.ntotal == 0:
        print("  ⚠ Knowledge base is empty. Run build_knowledge_base() first.")
        return []

    # embed the query
    query_embedding = get_embeddings([query])

    # search FAISS
    n = min(n_results, index.ntotal)
    distances, indices = index.search(query_embedding, n)

    projects = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        similarity = round(float(dist) * 100, 1)
        projects.append({
            "name": meta["name"],
            "url": meta["url"],
            "stars": meta["stars"],
            "language": meta["language"],
            "description": meta["description"],
            "similarity": similarity
        })

    return projects


if __name__ == "__main__":
    build_knowledge_base()