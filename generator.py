import os
import subprocess
import sys
import re
import time
import random
import shutil
from pathlib import Path

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
print("Installing Dependencies...")
try:
    libs = ["chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", "requests", "beautifulsoup4", "pydub"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs + ["--quiet"])
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg libsndfile1", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
import requests
from chatterbox.tts import ChatterboxTTS

# ==========================================
# 2. CONFIG & INPUTS (Injected by GitHub Action)
# ==========================================
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_URL = """{{VOICE_URL_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Secrets
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

# Setup Directories
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. UNIVERSAL VISUAL MAP (Expanded)
# ==========================================
VISUAL_MAP = {
    "tech": ["futuristic technology", "circuit board", "data center", "hologram", "coding"],
    "business": ["office meeting", "handshake", "skyscraper time lapse", "financial chart"],
    "finance": ["money falling", "stock market", "bitcoin", "gold coins", "wallet"],
    "nature": ["forest aerial", "mountain landscape", "ocean waves", "sunset"],
    "travel": ["airplane taking off", "busy airport", "tropical beach", "city street"],
    "history": ["ancient ruins", "old library", "vintage map", "museum artifacts", "castle"],
    "health": ["doctor", "dna spiral", "microscope", "healthy food", "yoga"],
    "happy": ["people laughing", "friends celebrating", "smiling face", "party"],
    "sad": ["rainy window", "lonely person", "dark clouds", "tears"],
    "abstract": ["ink in water", "light leaks", "particles", "geometric shapes"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(terms[:2])
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: queries.append(random.choice(words) + " cinematic")
    if not queries: queries = VISUAL_MAP["abstract"]
    return list(set(queries))[:3]

# ==========================================
# 4. CORE FUNCTIONS
# ==========================================

# --- YOUR REQUESTED AUDIO DOWNLOAD LOGIC ---
def download_voice_sample(url, path):
    print(f"Downloading from: {url}")
    try:
        token = os.environ.get('GH_PAT', '')
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
        headers['Accept'] = 'application/vnd.github.v3.raw'
        
        r = requests.get(url, headers=headers, allow_redirects=True)
        
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print("Download successful.")
            return True
        else:
            print(f"Download failed: {r.status_code}")
            print(r.text[:200])
            return False
    except Exception as e:
        print(f"Error during download: {e}")
        return False

def generate_script(topic, minutes):
    words = int(minutes * 130)
    print(f"Generating {words} word script for: {topic}")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Write a YouTube script about '{topic}'. Length: {words} words. Plain text only. No scene directions. No timestamps. Engaging tone. Do not use asterisks."
    try:
        return model.generate_content(prompt).text.strip().replace("*", "")
    except Exception as e:
        print(f"Script Error: {e}")
        return f"Welcome to our video about {topic}. Please enjoy the visuals."

# --- ROBUST CLONING (Prevents GPU Crash) ---
def clone_voice_robust(text, ref_audio_path, out_path):
    print("Cloning Voice (Robust Mode)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        ref_audio, sr = torchaudio.load(ref_audio_path)

        # CRITICAL: Split text to avoid CUDA OOM
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_chunks = []
        
        for i, sentence in enumerate(sentences):
            # Simple cleaning
            clean_sent = sentence.replace('"', '').replace("*", "")
            print(f"  Synthesizing chunk {i+1}/{len(sentences)}...")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=clean_sent, 
                        audio_prompt_path=ref_audio_path, 
                        exaggeration=0.5, 
                        cfg_weight=0.5
                    )
                    all_chunks.append(wav.cpu())
            except Exception as e:
                print(f"  ⚠️ Chunk failed: {e}")
                continue

        if not all_chunks:
            print("❌ All chunks failed.")
            return False

        print("  Stitching audio...")
        final_wav = torch.cat(all_chunks, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return False

def get_subtitles(audio_path):
    print("Transcribing...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    srt_path = TEMP_DIR / "subs.srt"
    with open(srt_path, "w") as f: f.write(t.export_subtitles_srt())
    sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
    return sentences, srt_path

def search_and_download_clips(sentences):
    clips = []
    used_ids = set()
    for i, sent in enumerate(sentences):
        duration = max(3.0, sent['end'] - sent['start'])
        queries = get_visual_queries(sent['text'])
        print(f"Seg {i}: '{queries[0]}' ({duration:.1f}s)")
        
        found_url = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=5)
                for v in r.json().get('videos', []):
                    if v['id'] not in used_ids:
                        found_url = v['video_files'][0]['link']; used_ids.add(v['id']); break
            except: pass
            
        if not found_url and PIXABAY_KEYS:
            try:
                k = random.choice(PIXABAY_KEYS)
                r = requests.get(f"https://pixabay.com/api/videos/?key={k}&q={queries[0]}&video_type=film", timeout=5)
                for h in r.json().get('hits', []):
                    if h['id'] not in used_ids:
                        found_url = h['videos']['medium']['url']; used_ids.add(h['id']); break
            except: pass

        if found_url:
            p = TEMP_DIR / f"clip_{i}.mp4"
            with open(p, "wb") as f: f.write(requests.get(found_url).content)
            out_p = TEMP_DIR / f"seg_{i}.mp4"
            subprocess.run(f"ffmpeg -y -i {p} -t {duration} -vf scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2 -c:v libx264 -preset fast -an {out_p}", shell=True)
            clips.append(str(out_p))
        else:
            p = TEMP_DIR / f"seg_{i}.mp4"
            subprocess.run(f"ffmpeg -y -f lavfi -i color=c=black:s=1920x1080:d={duration} -an {p}", shell=True)
            clips.append(str(p))
    return clips

def render_video(clips, audio, srt, output):
    print("Rendering Final Video...")
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4", shell=True)
    srt_esc = str(srt).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio),
        "-vf", f"subtitles={srt_esc}:force_style='FontName=Arial,FontSize=24'",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(output)
    ]
    subprocess.run(cmd)

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
print("--- STARTING VIDEO PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Failed to download voice sample")
    exit(1)

final_text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_robust(final_text, ref_voice, audio_out):
    sentences, srt_file = get_subtitles(audio_out)
    if sentences:
        print("Fetching Visuals...")
        clips = search_and_download_clips(sentences)
        
        # Unique Output Name based on Job ID
        final_video_name = f"final_video_{JOB_ID}.mp4"
        final_path = OUTPUT_DIR / final_video_name
        
        render_video(clips, audio_out, srt_file, final_path)
        print(f"--- DONE: {final_video_name} ---")
    else: print("Subtitles failed")
else: print("Audio failed")
