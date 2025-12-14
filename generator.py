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
# 3. MASSIVE VISUAL DICTIONARY
# ==========================================
VISUAL_MAP = {
    # Tech & Future
    "tech": ["futuristic technology", "server room", "coding screen", "robot arm", "circuit board", "hologram"],
    "ai": ["artificial intelligence abstract", "brain neural network", "cyborg eye", "digital mind"],
    "cyber": ["hacker hoodie", "matrix code", "security padlock", "digital shield", "cyber city"],
    
    # Business & Money
    "business": ["corporate meeting", "handshake slow motion", "office skyscraper", "man in suit walking"],
    "finance": ["stock market graph", "money falling", "gold coins", "bitcoin", "wallet"],
    "work": ["typing on keyboard", "writing in notebook", "team brainstorming", "whiteboard strategy"],

    # Nature & World
    "nature": ["forest aerial", "ocean waves", "mountain sunset", "flowers blooming", "waterfall"],
    "space": ["galaxy stars", "planet earth rotating", "astronaut", "rocket launch", "moon surface"],
    "city": ["city timelapse night", "busy street walking", "traffic lights", "subway train"],
    
    # History & Mystery
    "history": ["ancient ruins", "museum statue", "old map", "vintage library", "pyramids"],
    "mystery": ["foggy forest", "dark alley", "shadow figure", "candle light", "old book dust"],
    "war": ["soldiers marching", "tank mud", "flag waving", "military aircraft"],

    # Lifestyle & Emotions
    "happy": ["people laughing", "party confetti", "smiling face", "friends jumping", "toast cheers"],
    "sad": ["rainy window", "lonely person sitting", "dark clouds", "tear eye", "empty chair"],
    "health": ["doctor stethoscope", "dna spiral", "microscope", "yoga sunset", "healthy food"],
    "love": ["couple holding hands", "wedding ring", "heart shape", "rose flower"],

    # Abstract (The Ultimate Fallback)
    "abstract": ["ink in water", "light leaks", "particles dust", "geometric shapes", "smoke swirling", "bokeh lights"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    
    # Check massive dictionary
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(terms[:3])
    
    # Extract long nouns
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: queries.append(random.choice(words) + " cinematic")
    
    # Always ensure fallback
    if not queries: queries = VISUAL_MAP["abstract"]
    
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
    words = int(minutes * 140) # Slightly faster pace
    print(f"Generating {words} word script for: {topic}")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Write a YouTube script about '{topic}'. Length: {words} words. Plain text only. No scene directions. No timestamps. Engaging, storytelling tone. Do not use asterisks or hashtags."
    try:
        return model.generate_content(prompt).text.strip().replace("*", "").replace("#", "")
    except: return f"Welcome to our video about {topic}. Please enjoy the visuals."

# --- ROBUST AUDIO (Chunked + Path Fix) ---
def clone_voice_chunked(text, ref_audio_path, out_path):
    print("Cloning Voice (Robust Chunking)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            chunk = chunk.replace('"', '').replace("'", "")
            print(f"  Synthesizing chunk {i+1}/{len(sentences)}...")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk,
                        audio_prompt_path=str(ref_audio_path),
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )
                    all_wavs.append(wav.cpu())
            except Exception as e:
                print(f"  ⚠️ Chunk skipped: {e}")

        if not all_wavs: return False
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True
    except Exception as e:
        print(f"❌ Audio Error: {e}")
        return False

# --- STYLED SUBTITLES (.ASS Format) ---
def generate_styled_subtitles(audio_path):
    print("Transcribing & Styling Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    
    if t.status == aai.TranscriptStatus.error: return [], None
    
    ass_path = TEMP_DIR / "style.ass"
    
    # ASS Header
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(ass_path, "w") as f:
        f.write(header)
        for s in t.get_sentences():
            start = format_time_ass(s.start)
            end = format_time_ass(s.end)
            # Remove scary chars
            clean_text = s.text.replace("'", "").replace('"', '')
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{clean_text}\n")
            
    sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
    return sentences, ass_path

def format_time_ass(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms//10:02d}"

# --- VIDEO SANITIZER (The Anti-Freeze Fix) ---
def sanitize_clip(input_path, output_path, duration):
    """Re-encodes clip to strict standard to prevent concat freeze"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-t", str(duration),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", 
        "-an", # Remove audio
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

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

        out_p = TEMP_DIR / f"seg_{i}.mp4"
        
        if found_url:
            raw_p = TEMP_DIR / f"raw_{i}.mp4"
            with open(raw_p, "wb") as f: f.write(requests.get(found_url).content)
            sanitize_clip(raw_p, out_p, duration)
        else:
            # Black fallback
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-t", str(duration), "-vf", "fps=30", "-pix_fmt", "yuv420p", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        clips.append(str(out_p))
            
    return clips

def render_video(clips, audio, ass_file, output):
    print("Rendering Final Video...")
    
    # 1. Concat Visuals
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Burn Subtitles & Audio
    # Note: Using absolute path for ASS file is safer
    ass_abs = Path(ass_file).resolve()
    ass_str = str(ass_abs).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio),
        "-vf", f"ass='{ass_str}'",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-shortest", 
        str(output)
    ]
    subprocess.run(cmd)

# ==========================================
# 5. EXECUTION
# ==========================================
print("--- STARTING T4 VIDEO PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Failed to download voice")
    exit(1)

text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_chunked(text, ref_voice, audio_out):
    sentences, ass_file = generate_styled_subtitles(audio_out)
    if sentences:
        print("Fetching Visuals...")
        clips = search_and_download_clips(sentences)
        
        final_video_name = f"final_video_{JOB_ID}.mp4"
        final_path = OUTPUT_DIR / final_video_name
        
        render_video(clips, audio_out, ass_file, final_path)
        print(f"--- DONE: {final_video_name} ---")
    else: print("Subtitles failed")
else: print("Audio failed")
