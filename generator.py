import os
import subprocess
import sys
import re
import time
import random
import shutil
import json
import concurrent.futures
from pathlib import Path

# ==========================================
# 1. AUTO-INSTALLER
# ==========================================
print("Installing Dependencies...")
try:
    libs = ["chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", "requests", "beautifulsoup4", "pydub", "--quiet"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
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
    "tech": ["futuristic technology", "server room", "coding screen", "robot arm", "circuit board", "hologram"],
    "business": ["corporate meeting", "handshake slow motion", "office skyscraper", "man in suit walking"],
    "finance": ["stock market graph", "money falling", "gold coins", "bitcoin", "wallet"],
    "nature": ["forest aerial", "ocean waves", "mountain sunset", "flowers blooming", "waterfall"],
    "history": ["ancient ruins", "museum statue", "old map", "vintage library", "pyramids"],
    "abstract": ["ink in water", "light leaks", "particles dust", "geometric shapes", "smoke swirling", "bokeh lights"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    for cat, terms in VISUAL_MAP.items():
        if cat in text:
            queries.extend(random.sample(terms, min(2, len(terms))))
    
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: 
        queries.append(random.choice(words) + " cinematic")
    
    if not queries: 
        queries = random.sample(VISUAL_MAP["abstract"], 3)
    
    random.shuffle(queries)
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
    words = int(minutes * 150)
    print(f"Generating {words} word script for: {topic} ({minutes} mins)")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # For long videos, we ask for structure
    prompt = f"""
    Write a {words}-word YouTube script about '{topic}'.
    Format: Plain spoken text ONLY. No headers. No *Asterisks*.
    """
    try:
        text = model.generate_content(prompt).text
        text = text.replace("*", "").replace("#", "")
        return text.strip()
    except: return f"Welcome to our video about {topic}."

def clone_voice_chunked(text, ref_audio_path, out_path):
    print("Cloning Voice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        all_wavs = []
        for i, chunk in enumerate(sentences):
            chunk = chunk.replace('"', '').replace("'", "")
            try:
                with torch.no_grad():
                    wav = model.generate(text=chunk, audio_prompt_path=str(ref_audio_path), exaggeration=0.5, cfg_weight=0.5)
                    all_wavs.append(wav.cpu())
            except: pass
        if not all_wavs: return False
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def generate_styled_subtitles(audio_path):
    print("Generating Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error: return [], None
    
    ass_path = TEMP_DIR / "style.ass"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w") as f:
        f.write(header)
        for s in t.get_sentences():
            start = format_time_ass(s.start)
            end = format_time_ass(s.end)
            clean_text = s.text.replace("'", "").replace('"', '')
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{clean_text}\n")
            
    sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
    return sentences, ass_path

def format_time_ass(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms//10:02d}"

# --- VIDEO PROCESSING ---
def sanitize_clip(input_path, output_path, duration):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-t", str(duration),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-an", str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # AGGRESSIVE CLEANUP: Delete raw file immediately
    if os.path.exists(input_path):
        os.remove(input_path)
    return output_path

def download_and_process_clip(data):
    i, sent, used_ids = data
    duration = max(3.5, sent['end'] - sent['start'])
    queries = get_visual_queries(sent['text'])
    out_p = TEMP_DIR / f"seg_{i}.mp4"
    
    found_url = None
    # Pexels
    if PEXELS_KEYS:
        try:
            h = {"Authorization": random.choice(PEXELS_KEYS)}
            r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=5)
            videos = r.json().get('videos', [])
            for v in videos:
                if v['id'] not in used_ids:
                    found_url = v['video_files'][0]['link']; used_ids.add(v['id']); break
            if not found_url and videos: found_url = videos[0]['video_files'][0]['link']
        except: pass
        
    # Pixabay
    if not found_url and PIXABAY_KEYS:
        try:
            k = random.choice(PIXABAY_KEYS)
            r = requests.get(f"https://pixabay.com/api/videos/?key={k}&q={queries[0]}&video_type=film", timeout=5)
            hits = r.json().get('hits', [])
            if hits: found_url = hits[0]['videos']['medium']['url']
        except: pass

    if found_url:
        raw_p = TEMP_DIR / f"raw_{i}.mp4"
        try:
            with open(raw_p, "wb") as f: f.write(requests.get(found_url).content)
            sanitize_clip(raw_p, out_p, duration)
            print(f"  ✓ Clip {i} Ready")
            return str(out_p)
        except: pass
    
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-t", str(duration), "-vf", "fps=30,pix_fmt=yuv420p", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_p)

# --- BATCH PROCESSOR ---
def process_batches(sentences, audio_path, ass_file, final_output):
    """Handles long videos by processing in 50-segment batches"""
    BATCH_SIZE = 50
    total_segments = len(sentences)
    batch_files = []
    
    used_ids = set()
    
    for start_idx in range(0, total_segments, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_segments)
        print(f"\n--- Processing Batch {start_idx}-{end_idx} / {total_segments} ---")
        
        batch_sentences = sentences[start_idx:end_idx]
        
        # Parallel Download for this batch
        tasks = []
        for i, sent in enumerate(batch_sentences):
            # Maintain global index for filenames
            global_idx = start_idx + i
            tasks.append((global_idx, sent, used_ids))
            
        processed_clips = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            processed_clips = list(executor.map(download_and_process_clip, tasks))
            
        # Concat this batch into a temp video part
        batch_list = TEMP_DIR / f"batch_{start_idx}.txt"
        batch_video = TEMP_DIR / f"video_part_{start_idx}.mp4"
        
        with open(batch_list, "w") as f:
            for c in processed_clips: f.write(f"file '{c}'\n")
            
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(batch_list), "-c", "copy", str(batch_video)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # CLEANUP: Delete the small clips immediately
        print("  🧹 Cleaning up batch clips...")
        for c in processed_clips:
            if os.path.exists(c): os.remove(c)
            
        batch_files.append(str(batch_video))
        
    # Final Merge of Batches
    print("\n--- Merging All Batches ---")
    final_list = TEMP_DIR / "final_list.txt"
    visual_only = TEMP_DIR / "visual_full.mp4"
    
    with open(final_list, "w") as f:
        for b in batch_files: f.write(f"file '{b}'\n")
        
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(final_list), "-c", "copy", str(visual_only)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Final Burn
    ass_abs = Path(ass_file).resolve()
    ass_str = str(ass_abs).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y", "-i", str(visual_only), "-i", str(audio_path),
        "-vf", f"ass='{ass_str}'",
        "-c:v", "libx264", "-preset", "medium", "-b:v", "6000k",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(final_output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 5. EXECUTION
# ==========================================
print("--- STARTING MARATHON VIDEO PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Failed to download voice")
    exit(1)

text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_chunked(text, ref_voice, audio_out):
    sentences, ass_file = generate_styled_subtitles(audio_out)
    if sentences:
        final_video_name = f"final_video_{JOB_ID}.mp4"
        final_path = OUTPUT_DIR / final_video_name
        
        # Use Batch Processor
        process_batches(sentences, audio_out, ass_file, final_path)
        
        print(f"--- DONE: {final_video_name} ---")
    else: print("Subtitles failed")
else: print("Audio failed")
