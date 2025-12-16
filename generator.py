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
        "requests", "beautifulsoup4", "pydub", "numpy", "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai

# ==========================================
# 2. CONFIGURATION & KEYS
# ==========================================
# Template Placeholders
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()] # Multi-Key Support

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
# 3. STATUS & UTILS
# ==========================================
def update_status(progress, message, status="processing", file_url=None):
    """Updates frontend progress bar via GitHub API"""
    print(f"--- STATUS: {progress}% | {message} ---")
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return

    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    
    data = {
        "progress": progress,
        "message": message,
        "status": status,
        "timestamp": time.time()
    }
    if file_url: data["file_io_url"] = file_url
    
    import base64
    content_b64 = base64.b64encode(json.dumps(data).encode()).decode()
    
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        # Get SHA to update existing file
        get_res = requests.get(url, headers=headers)
        sha = get_res.json().get("sha") if get_res.status_code == 200 else None
        
        payload = {"message": "status update", "content": content_b64, "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def upload_to_fileio(file_path):
    """Uploads to File.io"""
    try:
        print("Uploading to File.io...")
        with open(file_path, 'rb') as f:
            r = requests.post('https://file.io', files={'file': f})
        if r.status_code == 200: return r.json().get('link')
    except Exception as e: print(f"File.io Error: {e}")
    return None

def download_asset(repo_path, local_path):
    """Downloads Voice/Logo"""
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 4. SCRIPT GENERATION (ROTATING KEYS)
# ==========================================
def generate_script(topic, minutes):
    words = int(minutes * 150)
    print(f"Generating Script (~{words} words)...")
    
    random.shuffle(GEMINI_KEYS) # Rotate keys
    
    prompt = f"""
    Write a YouTube video script about '{topic}'.
    STRICT CONSTRAINTS:
    - Total Word Count: Approximately {words} words.
    - Format: Plain spoken text ONLY. No [Scene Directions]. No *Asterisks*.
    - Structure: Engaging Hook -> Intro -> detailed Body -> Conclusion.
    """
    
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash') # Or 1.5-flash
            text = model.generate_content(prompt).text
            return text.replace("*", "").replace("#", "").replace("[", "").replace("]", "").strip()
        except Exception as e:
            print(f"Key failed, switching... Error: {e}")
            continue
            
    return f"Welcome to our video about {topic}. Please enjoy the visuals."

# ==========================================
# 5. AUDIO ENGINE (STABLE CHATTERBOX)
# ==========================================
def clone_voice_chunked(text, ref_audio_path, out_path):
    print("Cloning Voice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Audio Device: {device}")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Use Regex Split (From your working script)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            # Update Progress Bar
            if i % 5 == 0:
                pct = 20 + int((i / len(sentences)) * 30) # 20% -> 50%
                update_status(pct, f"Synthesizing Voice ({i}/{len(sentences)})")

            chunk = chunk.replace('"', '').replace("'", "")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk,
                        audio_prompt_path=str(ref_audio_path),
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )
                    all_wavs.append(wav.cpu()) # Move to CPU immediately
                    
                    # Memory Cleanup
                    if device == "cuda": torch.cuda.empty_cache()
            except Exception as e:
                print(f"Skipped chunk {i}: {e}")

        if not all_wavs: return False
        
        print("Merging Audio...")
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, model.sr)
        return True
    except Exception as e:
        print(f"TTS Critical Error: {e}")
        return False

# ==========================================
# 6. VISUALS & LOGO ENGINE
# ==========================================
VISUAL_MAP = {
    "tech": ["futuristic technology", "robot", "coding", "circuit"],
    "business": ["meeting", "office", "handshake", "skyscraper"],
    "nature": ["forest", "ocean", "mountain", "sunset"],
    "history": ["ruins", "museum", "old map", "castle"],
    "abstract": ["abstract background", "particles", "light leaks"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(random.sample(terms, min(2, len(terms))))
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: queries.append(random.choice(words) + " cinematic")
    if not queries: queries = ["abstract background"]
    random.shuffle(queries)
    return list(set(queries))[:3]

def download_and_process_clip(data):
    i, sent, used_ids = data
    duration = max(3.5, sent['end'] - sent['start'])
    queries = get_visual_queries(sent['text'])
    out_p = TEMP_DIR / f"seg_{i}.mp4"
    
    found_url = None
    if PEXELS_KEYS:
        try:
            h = {"Authorization": random.choice(PEXELS_KEYS)}
            r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=5)
            videos = r.json().get('videos', [])
            for v in videos:
                if v['id'] not in used_ids:
                    found_url = v['video_files'][0]['link']; used_ids.add(v['id']); break
        except: pass

    if found_url:
        try:
            raw_p = TEMP_DIR / f"raw_{i}.mp4"
            with open(raw_p, "wb") as f: f.write(requests.get(found_url).content)
            # Fast Sanitize
            subprocess.run([
                "ffmpeg", "-y", "-i", str(raw_p), "-t", str(duration),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out_p)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(out_p)
        except: pass
    
    # Fallback
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-t", str(duration), "-vf", "fps=30", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_p)

# ==========================================
# 7. EXECUTION & LOGO RENDER
# ==========================================
print("--- 🚀 STARTING JOB ---")
update_status(1, "Initializing...")

# 1. Assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice Download Failed", "failed"); exit(1)

download_asset(LOGO_PATH, ref_logo) # Optional download

# 2. Script
update_status(10, "Generating Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT

# 3. Audio
update_status(20, "Synthesizing Audio...")
audio_out = TEMP_DIR / "tts_out.wav"

if clone_voice_chunked(text, ref_voice, audio_out):
    
    # 4. Subtitles
    update_status(50, "Generating Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_out))
    
    if t.status != aai.TranscriptStatus.error:
        # Create ASS File
        ass_path = TEMP_DIR / "style.ass"
        with open(ass_path, "w") as f:
            f.write("""[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n""")
            for s in t.get_sentences():
                start = f"{int(s.start/3600000)}:{int(s.start/60000)%60:02d}:{int(s.start/1000)%60:02d}.{int(s.start%1000/10):02d}"
                end = f"{int(s.end/3600000)}:{int(s.end/60000)%60:02d}:{int(s.end/1000)%60:02d}.{int(s.end%1000/10):02d}"
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{s.text}\n")
        
        sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
        
        # 5. Visuals
        update_status(60, "Downloading Visuals...")
        used_ids = set()
        tasks = [(i, sent, used_ids) for i, sent in enumerate(sentences)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            clips = list(ex.map(download_and_process_clip, tasks))
            
        # 6. Render
        update_status(85, "Final Rendering (Logo + Subs)...")
        
        # Concat Visuals First
        with open("list.txt", "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        final_out = OUTPUT_DIR / f"final_video_{JOB_ID}.mp4"
        ass_fix = str(Path(ass_path).resolve()).replace("\\", "/").replace(":", "\\:")
        
        # LOGO LOGIC
        if os.path.exists(ref_logo):
            # Complex Filter: Scale Logo -> Overlay Top Left -> Burn Subtitles
            filter_complex = f"[1:v]scale=150:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass='{ass_fix}'[v2]"
            inputs = ["-i", "visual.mp4", "-i", str(ref_logo), "-i", str(audio_out)]
            mapping = ["-map", "[v2]", "-map", "2:a"]
        else:
            # No Logo: Just Subtitles
            filter_complex = f"ass='{ass_fix}'"
            inputs = ["-i", "visual.mp4", "-i", str(audio_out)]
            mapping = ["-c:v", "libx264", "-c:a", "aac"]
            
        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex] + mapping + ["-c:v", "libx264", "-preset", "medium", "-b:v", "4500k", "-c:a", "aac", "-shortest", str(final_out)]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 7. Upload
        update_status(99, "Uploading to Cloud...")
        link = upload_to_fileio(final_out)
        if link:
            update_status(100, "Done!", "completed", link)
        else:
            update_status(100, "Done (Upload Error)", "completed")
            
    else: update_status(0, "Subtitles Failed", "failed")
else: update_status(0, "Audio Failed", "failed")
