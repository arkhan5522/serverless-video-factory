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
    # Chatterbox and Audio
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", "requests", "beautifulsoup4", "pydub", "--quiet"])
    # FFmpeg
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
import requests
from chatterbox.tts import ChatterboxTTS

# ==========================================
# 2. CONFIGURATION
# ==========================================
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_URL = """{{VOICE_URL_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. VISUAL DICTIONARY
# ==========================================
VISUAL_MAP = {
    "tech": ["technology abstract", "server room", "coding screen", "robot", "circuit board"],
    "business": ["corporate meeting", "handshake", "office skyscraper", "financial chart"],
    "nature": ["forest aerial", "ocean waves", "mountain sunset", "flowers blooming"],
    "history": ["ancient ruins", "museum", "old map", "vintage library", "castle"],
    "abstract": ["ink in water", "light leaks", "particles", "geometric shapes", "smoke"],
    "happy": ["people laughing", "party", "smiling face", "celebration"],
    "sad": ["rainy window", "lonely person", "dark clouds"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(terms[:2])
    
    # Extract nouns for variety
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: queries.append(random.choice(words) + " cinematic")
    
    if not queries: queries = VISUAL_MAP["abstract"]
    return list(set(queries))[:3]

# ==========================================
# 4. CORE FUNCTIONS
# ==========================================

def download_voice_sample(url, path):
    print(f"Downloading voice: {url}")
    try:
        token = os.environ.get('GH_PAT', '')
        headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3.raw'} if token else {}
        r = requests.get(url, headers=headers, allow_redirects=True)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def generate_script(topic, minutes):
    words = int(minutes * 130)
    print(f"Generating {words} word script for: {topic}")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Write a YouTube script about '{topic}'. Length: {words} words. Plain text only. No scene directions. No timestamps. Engaging tone. Do not use asterisks."
    try:
        return model.generate_content(prompt).text.strip().replace("*", "").replace("#", "")
    except: return f"Welcome to our video about {topic}. Enjoy the visuals."

# --- FIXED AUDIO LOGIC (Your Method) ---
def clone_voice_chunked(text, ref_audio_path, out_path):
    print("Cloning Voice (Chunked)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Sentences split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            # Clean chunk
            chunk = chunk.replace('"', '').replace("'", "")
            print(f"  Synthesizing chunk {i+1}/{len(sentences)}...")
            
            try:
                with torch.no_grad():
                    # YOUR EXACT LOGIC HERE: Passing path string, not tensor
                    wav = model.generate(
                        text=chunk,
                        audio_prompt_path=str(ref_audio_path), # <--- KEY FIX
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )
                    all_wavs.append(wav.cpu())
            except Exception as e:
                print(f"  ⚠️ Chunk skipped: {e}")

        if not all_wavs: return False

        print("  Stitching audio...")
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True

    except Exception as e:
        print(f"❌ Audio Error: {e}")
        return False

def get_subtitles(audio_path):
    print("Transcribing...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error: return [], None
    
    srt_path = TEMP_DIR / "subs.srt"
    with open(srt_path, "w") as f: f.write(t.export_subtitles_srt())
    
    sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
    return sentences, srt_path

# --- FIXED VIDEO LOGIC (No Shell Syntax Errors) ---
def search_and_download_clips(sentences):
    clips = []
    used_ids = set()
    
    for i, sent in enumerate(sentences):
        duration = max(3.0, sent['end'] - sent['start'])
        queries = get_visual_queries(sent['text'])
        print(f"Seg {i}: '{queries[0]}' ({duration:.1f}s)")
        
        found_url = None
        # Pexels
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=5)
                for v in r.json().get('videos', []):
                    if v['id'] not in used_ids:
                        found_url = v['video_files'][0]['link']; used_ids.add(v['id']); break
            except: pass
            
        # Pixabay
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
            
            # SAFE COMMAND (No shell=True)
            cmd = [
                "ffmpeg", "-y", "-i", str(p),
                "-t", str(duration),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "fast",
                "-an", str(out_p)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append(str(out_p))
        else:
            # Fallback
            p = TEMP_DIR / f"seg_{i}.mp4"
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-an", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append(str(p))
            
    return clips

def render_video(clips, audio, srt, output):
    print("Rendering Final Video...")
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Escape for filter
    srt_esc = str(srt).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio),
        "-vf", f"subtitles={srt_esc}:force_style='FontName=Arial,FontSize=24'",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 5. EXECUTION
# ==========================================
print("--- STARTING VIDEO PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Failed to download voice")
    exit(1)

text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_chunked(text, ref_voice, audio_out):
    sentences, srt_file = get_subtitles(audio_out)
    if sentences:
        print("Fetching Visuals...")
        clips = search_and_download_clips(sentences)
        
        # Unique Name
        final_video_name = f"final_video_{JOB_ID}.mp4"
        final_path = OUTPUT_DIR / final_video_name
        
        render_video(clips, audio_out, srt_file, final_path)
        print(f"--- DONE: {final_video_name} ---")
    else: print("Subtitles failed")
else: print("Audio failed")
