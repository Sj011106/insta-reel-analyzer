import yt_dlp
import whisper
import easyocr
import cv2
import re
import os
from PIL import Image
import numpy as np


def get_shortcode_from_url(url: str) -> str:
    match = re.search(r"/(?:reel|p)/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError(
            "Could not find a shortcode in the URL.\n"
            "Make sure the URL looks like: https://www.instagram.com/p/ABC123/"
        )
    return match.group(1)


def get_caption(url: str) -> str:
    print("  → Fetching caption...")
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "cookiefile": "cookies.txt",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            caption = info.get("description") or ""
            title = info.get("title") or ""
            return f"{title}\n{caption}".strip()
    except Exception as e:
        print(f"  ⚠  Could not fetch caption: {e}")
        return ""


def download_media(url: str, shortcode: str) -> dict:
    """
    Tries yt-dlp first for reels/videos.
    Falls back to instaloader for carousel image posts.
    """
    print(f"  → Downloading media: {shortcode}")

    # First try yt-dlp (works for reels/videos)
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "cookiefile": "cookies.txt",
            "outtmpl": f"reel_{shortcode}_%(autonumber)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        files = sorted([
            f for f in os.listdir(".")
            if f.startswith(f"reel_{shortcode}") and
            not f.endswith(".part") and
            not f.endswith(".ytdl")
        ])

        if files:
            video_extensions = {".mp4", ".webm", ".mkv", ".mov"}
            has_video = any(
                os.path.splitext(f)[1].lower() in video_extensions
                for f in files
            )
            post_type = "video" if has_video else "carousel"
            print(f"  ✅ Downloaded via yt-dlp: {len(files)} file(s)")
            return {"type": post_type, "files": files}

    except Exception as e:
        print(f"  ℹ️  yt-dlp failed ({e}), trying instaloader for carousel...")

    # Fallback — instaloader for carousel image posts
    try:
        import instaloader
        L = instaloader.Instaloader(
            quiet=True,
            download_video_thumbnails=False,
            save_metadata=False,
            post_metadata_txt_pattern="",
        )

        # load cookies from cookies.txt into instaloader session
        import http.cookiejar
        cookie_jar = http.cookiejar.MozillaCookieJar()
        cookie_jar.load("cookies.txt", ignore_discard=True, ignore_expires=True)
        for cookie in cookie_jar:
            L.context._session.cookies.set(cookie.name, cookie.value)

        post = instaloader.Post.from_shortcode(L.context, shortcode)

        # download all images from the carousel
        files = []
        for i, node in enumerate(post.get_sidecar_nodes()):
            filename = f"reel_{shortcode}_{i+1}.jpg"
            L.download_pic(filename, node.display_url, node.date_utc)
            if os.path.exists(filename):
                files.append(filename)

        if not files:
            # single image post
            filename = f"reel_{shortcode}_1.jpg"
            L.download_pic(filename, post.url, post.date_utc)
            if os.path.exists(filename):
                files.append(filename)

        print(f"  ✅ Downloaded via instaloader: {len(files)} image(s)")
        return {"type": "carousel", "files": files}

    except Exception as e:
        raise Exception(f"Could not download media: {e}")


def extract_audio_transcript(video_file: str) -> str:
    print("  → Transcribing audio with Whisper...")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_file)
        transcript = result["text"].strip()
        if transcript:
            print(f"  ✅ Audio transcript: {len(transcript)} characters")
        else:
            print("  ℹ️  No speech detected")
        return transcript
    except Exception as e:
        print(f"  ⚠  Audio transcription failed: {e}")
        return ""


def ocr_image(reader, image_path: str) -> list:
    """Run OCR on a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    results = reader.readtext(img)
    return [text.strip() for (_, text, conf) in results if conf > 0.4 and text.strip()]


def ocr_video_frames(reader, video_file: str) -> str:
    """Run OCR on video frames every 0.5 seconds."""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)

    all_text = []
    seen_text = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            results = reader.readtext(frame)
            for (_, text, conf) in results:
                cleaned = text.strip()
                if conf > 0.4 and cleaned and cleaned not in seen_text:
                    seen_text.add(cleaned)
                    all_text.append(cleaned)
        frame_count += 1

    cap.release()
    return " | ".join(all_text)


def extract_screen_text(media: dict) -> str:
    print("  → Reading text from media...")
    try:
        reader = easyocr.Reader(["en"], verbose=False)

        if media["type"] == "carousel":
            # read each image separately, label by slide number
            all_slides = []
            image_files = [f for f in media["files"] if not f.endswith(".mp4")]

            for i, img_file in enumerate(image_files):
                texts = ocr_image(reader, img_file)
                if texts:
                    slide_text = " ".join(texts)
                    all_slides.append(f"[SLIDE {i+1}]\n{slide_text}")
                    print(f"  ✅ Slide {i+1}: {len(texts)} text blocks found")

            return "\n\n".join(all_slides)

        else:
            # it's a video reel
            video_file = media["files"][0] if media["files"] else None
            if not video_file:
                return ""
            result = ocr_video_frames(reader, video_file)
            if result:
                print(f"  ✅ Screen text found in video")
            return result

    except Exception as e:
        print(f"  ⚠  Screen text extraction failed: {e}")
        return ""


def scrape_reel(url: str) -> dict:
    shortcode = get_shortcode_from_url(url)

    # Step 1 — download media (video or carousel images)
    media = download_media(url, shortcode)
    print(f"  ✅ Downloaded {len(media['files'])} file(s) — type: {media['type']}")

    # Step 2 — fetch caption
    caption = get_caption(url)
    if caption:
        print(f"  ✅ Caption fetched: {len(caption)} characters")

    # Step 3 — audio transcript (only for video reels)
    audio_transcript = ""
    if media["type"] == "video" and media["files"]:
        audio_transcript = extract_audio_transcript(media["files"][0])

    # Step 4 — OCR screen text
    screen_text = extract_screen_text(media)

    # Step 5 — combine everything
    combined = ""
    if caption:
        combined += f"CAPTION:\n{caption}\n\n"
    if audio_transcript:
        combined += f"SPOKEN AUDIO:\n{audio_transcript}\n\n"
    if screen_text:
        combined += f"TEXT VISIBLE ON SCREEN:\n{screen_text}"

    if not combined.strip():
        combined = "No content could be extracted."

    print(f"\n  ✅ Total content extracted: {len(combined)} characters")

    # Step 6 — clean up downloaded files
    for f in media["files"]:
        if os.path.exists(f):
            os.remove(f)

    return {
        "caption": combined,
        "shortcode": shortcode,
        "url": url,
    }