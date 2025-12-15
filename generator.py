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
# 3. MASSIVE VISUAL DICTIONARY (Broad & Deep)
# ==========================================
VISUAL_MAP = {
    # TECH & FUTURE
    "tech": ["futuristic technology", "server room", "coding screen", "robot arm", "circuit board", "hologram", "microchip"],
    "ai": ["artificial intelligence abstract", "brain neural network", "cyborg eye", "digital mind", "robot face"],
    "cyber": ["hacker hoodie", "matrix code", "security padlock", "digital shield", "cyber city", "fingerprint scan"],
    "innovation": ["light bulb turning on", "rocket launch", "drone flying", "3d printer printing", "vr headset"],

    # BUSINESS
    "business": ["corporate meeting", "handshake slow motion", "office skyscraper", "man in suit walking", "team brainstorming"],
    "finance": ["stock market graph", "money falling", "gold coins", "bitcoin", "wallet", "credit card", "bank vault"],
    "work": ["typing on keyboard", "writing in notebook", "whiteboard strategy", "laptop coffee", "architect drawing"],

    # NATURE
    "nature": ["forest aerial", "ocean waves", "mountain sunset", "flowers blooming", "waterfall", "desert sand"],
    "space": ["galaxy stars", "planet earth rotating", "astronaut", "moon surface", "nebula", "mars surface"],
    "elements": ["fire flames", "water splash slow motion", "smoke swirling", "lightning storm", "clouds timelapse"],

    # HISTORY
    "history": ["ancient ruins", "museum statue", "old map", "vintage library", "pyramids", "colosseum rome", "medieval castle"],
    "war": ["soldiers marching", "tank mud", "flag waving", "military aircraft", "cannon fire"],
    "art": ["painting canvas", "sculpture museum", "artist drawing", "color palette", "gallery walk"],

    # LIFESTYLE
    "happy": ["people laughing", "party confetti", "smiling face", "friends jumping", "toast cheers"],
    "sad": ["rainy window", "lonely person sitting", "dark clouds", "tear eye", "empty chair"],
    "health": ["doctor stethoscope", "yoga sunset", "healthy food", "running track", "gym workout"],
    "city": ["city timelapse night", "busy street walking", "traffic lights", "subway train", "skyscraper view"],

    # CONCEPTS (Abstract mappings)
    "time": ["clock ticking", "hourglass sand", "time lapse city", "calendar flip", "sunrise sunset"],
    "freedom": ["bird flying", "running field", "open road", "mountain top"],
    "mystery": ["foggy forest", "dark alley", "shadow figure", "candle light", "old book dust"],
    "communication": ["talking phone", "writing letter", "satellite dish", "network nodes"],
    
    # SAFETY NET
    "abstract": ["ink in water", "light leaks", "particles dust", "geometric shapes", "bokeh lights", "fractal animation"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    
    # 1. Direct Map
    for cat, terms in VISUAL_MAP.items():
        if cat in text:
            queries.extend(random.sample(terms, min(2, len(terms))))
            
    # 2. Concept Mapping
    if "idea" in text: queries.append("light bulb")
    if "global" in text: queries.append("earth rotating")
    if "speed" in text: queries.append("fast car")
    if "love" in text: queries.append("couple holding hands")
    if "ancient" in text: queries.append("ruins")
    if "future" in text: queries.append("futuristic city")
    
    # 3. Noun Extraction (Broad)
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: 
        queries.append(random.choice(words) + " cinematic")
    
    # 4. Fallback
    if not queries: 
        queries = random.sample(VISUAL_MAP["abstract"], 3)
    
    random.shuffle(queries)
    return list(set(queries))[:4]

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
    # Precise Timing: 150 words per minute
    words = int(minutes * 150)
    print(f"Generating {words} word script for: {topic} ({minutes} mins)")
    
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Write a YouTube video script about '{topic}'.
    
    STRICT CONSTRAINTS:
    - Total Word Count: Approximately {words} words (CRITICAL for timing).
    - Format: Plain spoken text ONLY. No [Scene Directions]. No (Timecodes). No *Asterisks*.
    - Structure: Engaging Hook -> Intro -> detailed Body -> Conclusion.
    - Style: Professional, storytelling, engaging.
    """
    try:
        text = model.generate_content(prompt).text
        # Clean specific artifacts
        text = text.replace("*", "").replace("#", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return text.strip()
    except: return f"Welcome to our video about {topic}. Please enjoy the visuals."

# --- AUDIO GENERATION ---
def clone_voice_chunked(text, ref_audio_path, out_path):
    print("Cloning Voice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Audio Device: {device}")
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            chunk = chunk.replace('"', '').replace("'", "")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk,
                        audio_prompt_path=str(ref_audio_path),
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )
                    all_wavs.append(wav.cpu())
            except: pass

        if not all_wavs: return False
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# --- SUBTITLES ---
def generate_styled_subtitles(audio_path):
    print("Generating Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error: return [], None
    
    ass_path = TEMP_DIR / "style.ass"
    # Yellow, Bold, Shadow
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,70,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,70,1
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

# --- OPTIMIZED VIDEO PROCESSING ---
def sanitize_clip(input_path, output_path, duration):
    # Preset: ultrafast for sanitization (we just need format match)
    # This is much faster than 'slow' and safe for intermediate clips
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-t", str(duration),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-an",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def download_and_process_clip(data):
    """Helper for parallel execution"""
    i, sent, used_ids = data
    duration = max(3.5, sent['end'] - sent['start'])
    queries = get_visual_queries(sent['text'])
    out_p = TEMP_DIR / f"seg_{i}.mp4"
    
    found_url = None
    
    # Try Pexels
    if PEXELS_KEYS:
        try:
            h = {"Authorization": random.choice(PEXELS_KEYS)}
            r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=5)
            videos = r.json().get('videos', [])
            for v in videos:
                # Basic loop avoidance
                if v['id'] not in used_ids:
                    found_url = v['video_files'][0]['link']
                    used_ids.add(v['id']) # Note: This isn't thread-safe strictly but good enough for randomness
                    break
            if not found_url and videos: found_url = videos[0]['video_files'][0]['link']
        except: pass
        
    # Try Pixabay
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
            print(f"  ✓ Clip {i} Ready ({queries[0]})")
            return str(out_p)
        except: pass
    
    # Fallback Black Clip
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-t", str(duration), "-vf", "fps=30,pix_fmt=yuv420p", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  ⚠️ Clip {i} Fallback")
    return str(out_p)

def search_and_download_clips_parallel(sentences):
    print(f"Fetching {len(sentences)} clips in PARALLEL (4 Threads)...")
    clips = [None] * len(sentences)
    used_ids = set()
    
    # Prepare data for threads
    tasks = []
    for i, sent in enumerate(sentences):
        tasks.append((i, sent, used_ids))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(download_and_process_clip, tasks)
        
    return list(results)

def render_video(clips, audio, ass_file, output):
    print("Rendering Final Cinema-Quality Video...")
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    # Fast Concat
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    ass_abs = Path(ass_file).resolve()
    ass_str = str(ass_abs).replace("\\", "/").replace(":", "\\:")
    
    # Preset: Medium (Best balance of Speed vs Quality for 1080p)
    # Slow preset is diminishing returns for this use case
    cmd = [
        "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio),
        "-vf", f"ass='{ass_str}'",
        "-c:v", "libx264", "-preset", "medium", "-b:v", "6000k",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 5. EXECUTION
# ==========================================
print("--- STARTING TURBO HIGH-FIDELITY PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Failed to download voice")
    exit(1)

text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_chunked(text, ref_voice, audio_out):
    sentences, ass_file = generate_styled_subtitles(audio_out)
    if sentences:
        clips = search_and_download_clips_parallel(sentences)
        final_video_name = f"final_video_{JOB_ID}.mp4"
        final_path = OUTPUT_DIR / final_video_name
        render_video(clips, audio_out, ass_file, final_path)
        print(f"--- DONE: {final_video_name} ---")
    else: print("Subtitles failed")
else: print("Audio failed")
