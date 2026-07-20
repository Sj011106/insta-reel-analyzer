import yt_dlp
import re


def get_shortcode_from_url(url: str) -> str:
    match = re.search(r"/(?:reel|p)/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError(
            "Could not find a reel/post shortcode in the URL.\n"
            "Make sure the URL looks like: https://www.instagram.com/reel/ABC123/"
        )
    return match.group(1)


def scrape_reel(url: str) -> dict:
    """
    Uses yt-dlp to extract caption/description from an Instagram reel.
    Much more reliable than instaloader for public reels.
    """
    shortcode = get_shortcode_from_url(url)

    print(f"  → Fetching reel with shortcode: {shortcode}")

    ydl_opts = {
        "quiet": True,           # don't print yt-dlp's own output
        "skip_download": True,   # we only want metadata, not the video
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # yt-dlp stores the caption in "description"
    caption = info.get("description") or ""
    title = info.get("title") or ""

    # combine title + description for more context
    full_text = f"{title}\n{caption}".strip()

    if not full_text:
        print("  ⚠  Warning: No caption found in this reel.")
        print("     Gemini will still try, but results may be vague.")

    return {
        "caption": full_text,
        "shortcode": shortcode,
        "url": url,
    }