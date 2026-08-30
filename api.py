from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from scraper import scrape_reel
from analyzer import analyze_reel

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# allows frontend to talk to backend without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# defines what data frontend sends us
class ReelRequest(BaseModel):
    url: str

# serves index.html when someone visits localhost:8000
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# main endpoint — frontend sends URL here, we return roadmap
@app.post("/analyze")
def analyze(request: ReelRequest):
    try:
        print(f"\n→ Received URL: {request.url}")

        # Step 1 — scrape
        reel_data = scrape_reel(request.url)

        # Step 2 — analyze
        roadmap = analyze_reel(reel_data["caption"], GROQ_API_KEY)

        return {
            "success": True,
            "roadmap": roadmap,
            "caption": reel_data["caption"]
        }

    except Exception as e:
        print(f"→ Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }