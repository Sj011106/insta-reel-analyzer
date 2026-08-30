from scraper import scrape_reel
from analyzer import analyze_reel

# ─────────────────────────────────────────────
#  STEP 1: Paste your free Gemini API key here
#  Get it from: https://aistudio.google.com
#  Takes 2 minutes, no credit card needed
# ─────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def main():
    print("\n" + "=" * 55)
    print("   Instagram Reel → CS Project Analyzer")
    print("=" * 55)

    # Get URL from user
    url = input("\nPaste the Instagram Reel URL: ").strip()

    if not url:
        print("No URL entered. Exiting.")
        return

    # ── Stage 1: Scrape ──────────────────────────────────────
    print("\n[1/2] Scraping reel caption from Instagram...")
    try:
        reel_data = scrape_reel(url)
    except ValueError as e:
        print(f"\n❌ URL Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ Instagram scraping failed: {e}")
        print("   This can happen if the post is private or Instagram blocked the request.")
        print("   Try again in a few minutes, or try a different public reel.")
        return

    print(f"\n  ✅ Caption extracted! ({len(reel_data['caption'])} characters)")

    # Show the raw caption so you can see what Gemini is working with
    print("\n" + "-" * 40)
    print("RAW CAPTION:")
    print("-" * 40)
    print(reel_data["caption"][:500])  # show first 500 chars
    if len(reel_data["caption"]) > 500:
        print("... (caption truncated for display)")

    # ── Stage 2: Analyze with Gemini ─────────────────────────
    print("\n[2/2] Analyzing with Gemini AI...")
    try:
        roadmap = analyze_reel(reel_data["caption"], GROQ_API_KEY)
    except Exception as e:
        print(f"\n❌ Gemini API error: {e}")
        print("   Double-check your API key in main.py")
        return

    # ── Output ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   PROJECT ROADMAP")
    print("=" * 55 + "\n")
    print(roadmap)
    print("\n" + "=" * 55)

    # Save to a file so you don't lose it
    output_filename = f"roadmap_{reel_data['shortcode']}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {reel_data['url']}\n\n")
        f.write(roadmap)

    print(f"✅ Roadmap saved to: {output_filename}")


if __name__ == "__main__":
    main()
