import os
import subprocess
import sys
import re
import time
import random
import shutil
import json
import concurrent.futures
import requests
from pathlib import Path

# ==========================================
# 1. AUTO-INSTALLER
# ==========================================
print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", 
        "requests", "beautifulsoup4", "pydub", "nltk", "numpy", "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
import nltk
from pydub import AudioSegment
from chatterbox.tts import ChatterboxTTS

# Ensure NLTK data
nltk.download('punkt', quiet=True)

# ==========================================
# 2. CONFIGURATION & KEYS
# ==========================================
# Template Placeholders
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}""" # e.g. uploads/voice_123.mp3
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""   # e.g. uploads/logo_123.png
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# API Keys
# Handle Single or Multiple Gemini Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
if not GEMINI_KEYS: print("⚠️ WARNING: No Gemini Keys Found!")

ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. STATUS & COMMUNICATION SYSTEM
# ==========================================
def update_status(progress, message, status="processing", file_url=None):
    """
    Updates the status JSON on GitHub for the frontend progress bar.
    """
    print(f"--- STATUS: {progress}% | {message} ---")
    
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    
    if not repo or not token: return

    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    
    # Payload
    status_data = {
        "progress": progress,
        "message": message,
        "status": status,
        "job_id": JOB_ID,
        "timestamp": time.time()
    }
    if file_url: status_data["file_io_url"] = file_url

    import base64
    content_b64 = base64.b64encode(json.dumps(status_data).encode()).decode()

    # Get SHA if exists
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        get_res = requests.get(url, headers=headers)
        sha = get_res.json().get("sha") if get_res.status_code == 200 else None
        
        data = {
            "message": f"Status Update {progress}%",
            "content": content_b64,
            "branch": "main" 
        }
        if sha: data["sha"] = sha
        
        requests.put(url, headers=headers, json=data)
    except Exception as e:
        print(f"Status Update Failed: {e}")

def upload_to_fileio(file_path):
    """Uploads final video to File.io for instant sharing"""
    print("Uploading to File.io...")
    try:
        with open(file_path, 'rb') as f:
            # Expires in 14 days or 1 download (default free tier)
            response = requests.post('https://file.io', files={'file': f})
        
        if response.status_code == 200:
            link = response.json().get('link')
            print(f"File.io Link: {link}")
            return link
    except Exception as e:
        print(f"File.io Error: {e}")
    return None

def download_asset(repo_path, local_path):
    """Downloads Voice or Logo from the Repo"""
    print(f"Downloading asset: {repo_path}")
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 4. ROBUST SCRIPT GENERATION (ROTATING KEYS)
# ==========================================
def get_gemini_response(prompt, model_name='gemini-1.5-flash'):
    """Tries keys one by one until success"""
    # Shuffle keys to load balance
    random.shuffle(GEMINI_KEYS)
    
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"   ⚠️ Key {key[:5]}... exhausted. Switching...")
                continue # Try next key
            else:
                print(f"   ⚠️ Gemini Error: {e}")
    
    print("❌ All Gemini Keys Failed!")
    return None

def generate_script(topic, minutes):
    words = int(minutes * 145)
    print(f"--- 📝 Generating Script (~{words} words) ---")
    
    # 15+ Mins = Chunked
    if minutes > 15:
        chunks = 5
        words_per_chunk = int(words / chunks)
        full_script = []
        
        for i in range(chunks):
            update_status(5 + (i*2), f"Writing Script Part {i+1}/{chunks}...")
            
            context = ""
            if full_script: context = f"Previous context: ...{full_script[-1][-300:]}"
            
            prompt = f"""
            Write Part {i+1} of 5 for a documentary about '{topic}'.
            {context}
            Target Length: ~{words_per_chunk} words.
            Style: Engaging narration. NO headers. NO intros like 'Welcome back'.
            """
            text = get_gemini_response(prompt)
            if text: full_script.append(text.replace("**","").replace("##",""))
            
        return " ".join(full_script)

    # < 15 Mins = Single Shot
    else:
        prompt = f"""
        Write a YouTube script about '{topic}'.
        Length: {words} words.
        Format: Spoken narration only. No visual cues.
        """
        return get_gemini_response(prompt) or f"Script generation failed for {topic}."

# ==========================================
# 5. VISUAL & AUDIO ENGINES
# ==========================================
def get_visual_queries(text):
    text = text.lower()
    queries = []
    
    # Mapping
    map_ = {
        "tech": ["technology", "AI", "computer", "future"],
        "business": ["office", "meeting", "handshake", "city"],
        "nature": ["forest", "river", "sky", "mountains"],
        "history": ["ancient", "museum", "ruins", "map"],
    }
    
    for k, v in map_.items():
        if k in text: queries.extend(v)
            
    words = [w for w in re.findall(r'\w+', text) if len(w) > 6]
    if words: queries.append(random.choice(words))
    
    if not queries: queries = ["abstract background", "cinematic light"]
    random.shuffle(queries)
    return list(set(queries))[:3]

def clone_voice_robust(text, ref_audio, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        sentences = nltk.sent_tokenize(text)
        
        tensors = []
        for i, sent in enumerate(sentences):
            if len(sent) < 2: continue
            
            # Progress update for long audio
            if i % 5 == 0:
                pct = 20 + int((i / len(sentences)) * 30) # 20% to 50%
                update_status(pct, f"Synthesizing Voice ({i}/{len(sentences)})")
            
            try:
                with torch.no_grad():
                    wav = model.generate(sent, str(ref_audio), exaggeration=0.5)
                    tensors.append(wav.cpu())
                    tensors.append(torch.zeros(1, int(24000 * 0.25)))
            except: pass
            
        if not tensors: return False
        final = torch.cat(tensors, dim=1)
        torchaudio.save(out_path, final, 24000)
        return True
    except Exception as e:
        print(f"TTS Error: {e}"); return False

def process_visuals_with_logo(sentences, audio_path, ass_file, logo_path, final_out):
    print(f"--- 🎬 Processing Visuals ---")
    BATCH_SIZE = 50
    used_ids = set()
    parts = []
    
    # 1. Download Clips
    def download_clip(args):
        i, sent, used = args
        dur = max(3.5, sent['end'] - sent['start'])
        q = get_visual_queries(sent['text'])[0]
        out = TEMP_DIR / f"seg_{i}.mp4"
        
        # Search Pexels/Pixabay
        found_url = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={q}&orientation=landscape&size=medium", headers=h, timeout=4)
                vids = r.json().get('videos', [])
                for v in vids:
                    if v['id'] not in used:
                        found_url = v['video_files'][0]['link']; used.add(v['id']); break
            except: pass
        
        # FFMPEG Sanitization
        if found_url:
            try:
                raw = TEMP_DIR / f"raw_{i}.mp4"
                with open(raw, "wb") as f: f.write(requests.get(found_url).content)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(raw), "-t", str(dur),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if raw.exists(): os.remove(raw)
                return str(out)
            except: pass
            
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}", "-t", str(dur), "-vf", "fps=30", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    # Batch Processing
    for start in range(0, len(sentences), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(sentences))
        pct = 50 + int((start / len(sentences)) * 40) # 50% to 90%
        update_status(pct, f"Rendering Visual Batch {start}-{end}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            clips = list(ex.map(download_clip, [(k, sentences[k], used_ids) for k in range(start, end)]))
            
        batch_txt = TEMP_DIR / f"list_{start}.txt"
        batch_vid = TEMP_DIR / f"part_{start}.mp4"
        with open(batch_txt, "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(batch_txt), "-c", "copy", str(batch_vid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for c in clips: 
            if os.path.exists(c): os.remove(c)
        parts.append(str(batch_vid))

    # 2. Final Merge & Logo Overlay
    update_status(95, "Finalizing & Adding Logo...")
    full_vis = TEMP_DIR / "full_visual.mp4"
    list_txt = TEMP_DIR / "full_list.txt"
    with open(list_txt, "w") as f:
        for p in parts: f.write(f"file '{p}'\n")
    
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(full_vis)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    ass_fix = str(Path(ass_file).resolve()).replace("\\", "/").replace(":", "\\:")
    
    # Build Complex Filter for Logo + Subtitles
    # Resize logo to 150px wide, put at 30,30 padding
    filter_complex = f"[1:v]scale=150:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass='{ass_fix}'[v2]"
    
    cmd = [
        "ffmpeg", "-y", 
        "-i", str(full_vis),     # Input 0: Video
        "-i", str(logo_path),    # Input 1: Logo
        "-i", str(audio_path),   # Input 2: Audio
        "-filter_complex", filter_complex,
        "-map", "[v2]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "medium", "-b:v", "4500k",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(final_out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 6. MAIN EXECUTION FLOW
# ==========================================
print("--- 🚀 STARTING JOB ---")
update_status(1, "Initializing Cloud Environment...")

# 1. Download Assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png" # Will convert any img to this path

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Failed to download Voice", "failed"); exit(1)
    
if not download_asset(LOGO_PATH, ref_logo):
    print("Logo download failed or not provided, skipping logo overlay.")
    # Create dummy transparent logo to prevent ffmpeg crash
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black@0.0:s=150x150", "-frames:v", "1", str(ref_logo)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 2. Script
update_status(10, "Generating AI Script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

# 3. Audio
update_status(20, "Starting Voice Synthesis...")
audio_out = TEMP_DIR / "out.wav"
if clone_voice_robust(text, ref_voice, audio_out):
    
    # 4. Subtitles
    update_status(50, "Generating Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_out))
    
    if t.status != aai.TranscriptStatus.error:
        # Generate ASS file
        ass_path = TEMP_DIR / "style.ass"
        with open(ass_path, "w") as f:
            f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
            for s in t.get_sentences():
                start = f"{int(s.start/3600000)}:{int(s.start/60000)%60:02d}:{int(s.start/1000)%60:02d}.{int(s.start%1000/10):02d}"
                end = f"{int(s.end/3600000)}:{int(s.end/60000)%60:02d}:{int(s.end/1000)%60:02d}.{int(s.end%1000/10):02d}"
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{s.text}\n")
        
        sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
        
        # 5. Visuals & Logo
        final_video = OUTPUT_DIR / f"final_video_{JOB_ID}.mp4"
        process_visuals_with_logo(sentences, audio_out, ass_path, ref_logo, final_video)
        
        # 6. Upload & Finish
        update_status(99, "Uploading to Cloud...")
        file_link = upload_to_fileio(final_video)
        
        if file_link:
            update_status(100, "Completed Successfully!", "completed", file_link)
        else:
            update_status(100, "Video Done (Upload Failed)", "completed")
            
    else:
        update_status(0, "Subtitle Generation Failed", "failed")
else:
    update_status(0, "Audio Generation Failed", "failed")
