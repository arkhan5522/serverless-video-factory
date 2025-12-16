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
import gc
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
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Multi-Key Handling
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]

ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. UTILS: STATUS & UPLOAD
# ==========================================
def update_status(progress, message, status="processing", file_url=None):
    """Sends real-time updates to HTML"""
    print(f"--- {progress}% | {message} ---")
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return

    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    data = {"progress": progress, "message": message, "status": status, "timestamp": time.time()}
    if file_url: data["file_io_url"] = file_url
    
    import base64
    content = base64.b64encode(json.dumps(data).encode()).decode()
    
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        sha = requests.get(url, headers=headers).json().get("sha")
        payload = {"message": "update", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def upload_to_fileio(file_path):
    print("Uploading to File.io...")
    for i in range(3): # 3 Retries
        try:
            with open(file_path, 'rb') as f:
                r = requests.post('https://file.io', files={'file': f})
            if r.status_code == 200: 
                return r.json().get('link')
            else:
                print(f"File.io Error {r.status_code}: {r.text}")
        except Exception as e: 
            print(f"Upload Retry {i}: {e}")
            time.sleep(2)
    return None

def download_asset(repo_path, local_path):
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
# 4. SMART SCRIPT ENGINE (CHUNKING)
# ==========================================
def generate_script(topic, minutes):
    words = int(minutes * 150)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    # CHUNKING FOR LONG VIDEOS (>15 Mins)
    if minutes > 15:
        chunks = int(minutes / 5) # 1 chunk per 5 mins
        full_script = []
        print(f"Mode: Marathon ({chunks} Chapters)")
        
        for i in range(chunks):
            update_status(5 + i, f"Writing Chapter {i+1}/{chunks}...")
            context = full_script[-1][-300:] if full_script else "Start of documentary"
            
            prompt = f"""
            Write Part {i+1} of a documentary script about '{topic}'.
            Previous Context: "...{context}"
            Target Length: ~750 words.
            Style: Engaging, professional narration.
            Constraint: Write ONLY the spoken text. NO headers (Part 1), NO [Music cues].
            """
            text = call_gemini(prompt)
            if text: full_script.append(text)
            
        return " ".join(full_script)
    
    # SINGLE SHOT (<15 Mins)
    else:
        prompt = f"""
        Write a complete YouTube script about '{topic}'.
        Target Word Count: {words}.
        Format: Pure spoken narration. NO visual instructions. NO [Bracketed text].
        Style: High-retention storytelling.
        """
        return call_gemini(prompt)

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except: continue
    return ""

# ==========================================
# 5. ROBUST AUDIO ENGINE (MEMORY SAFE)
# ==========================================
def clone_voice_robust(text, ref_audio, out_path):
    print("Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Clean & Split
        clean_text = re.sub(r'\[.*?\]', '', text) # Remove [Music] etc
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        all_wavs = []
        print(f"Total Sentences: {len(sentences)}")
        
        for i, chunk in enumerate(sentences):
            # Progress Update
            if i % 10 == 0: 
                pct = 20 + int((i/len(sentences))*30)
                update_status(pct, f"Synthesizing Audio ({i}/{len(sentences)})")
            
            try:
                # Generate
                chunk = chunk.replace('"', '').replace("'", "")
                with torch.no_grad():
                    wav = model.generate(chunk, str(ref_audio), exaggeration=0.5)
                    all_wavs.append(wav.cpu()) # Move to CPU immediately to save VRAM
                
                # Force Garbage Collection (Fixes the 90% crash)
                if i % 20 == 0:
                    if device == "cuda": torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Skipped chunk {i}: {e}")

        if not all_wavs: 
            update_status(0, "TTS Output Empty", "failed")
            return False
        
        print("Merging Audio Streams...")
        final_wav = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final_wav, 24000)
        return True

    except Exception as e:
        update_status(0, f"Audio Crash: {str(e)}", "failed")
        return False

# ==========================================
# 6. VISUALS & STYLES
# ==========================================
VISUAL_MAP = {
    "tech": ["futuristic data center", "ai robot", "hologram city", "cyberpunk", "microchip"],
    "money": ["bitcoin gold", "stock market", "money falling", "luxury car", "mansion"],
    "nature": ["forest drone", "ocean waves", "mountain cinematic", "waterfall", "galaxy stars"],
    "horror": ["spooky forest", "abandoned house", "shadow figure", "foggy street", "skull"],
    "history": ["ancient rome", "egypt pyramids", "vintage map", "medieval castle", "war tank"],
    "travel": ["paris eiffel", "tropical beach", "airplane wing", "nyc times square", "road trip"],
    "abstract": ["ink in water", "light leaks", "particles dust", "geometric loop", "smoke"]
}

def get_visual_query(text):
    text = text.lower()
    queries = []
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(random.sample(terms, min(2, len(terms))))
    
    words = [w for w in re.findall(r'\w+', text) if len(w) > 6]
    if words: queries.append(random.choice(words) + " cinematic 4k")
    if not queries: queries = ["abstract background 4k"]
    
    random.shuffle(queries)
    return queries[0]

def get_subtitle_style():
    styles = [
        # 1. Netflix (White/Shadow)
        "Fontname=Arial,Fontsize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,Bold=0,Outline=1,Shadow=1,Alignment=2,MarginV=30",
        # 2. YouTube (Yellow/Black Stroke)
        "Fontname=Impact,Fontsize=24,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BackColour=&H80000000,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=40",
        # 3. Cinematic (Minimal)
        "Fontname=Roboto,Fontsize=18,PrimaryColour=&H00E0E0E0,OutlineColour=&H00000000,BackColour=&H00000000,Bold=1,Outline=1,Shadow=0,Alignment=2,MarginV=25,Spacing=1",
        # 4. Boxed (White on Black Box)
        "Fontname=Verdana,Fontsize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H60000000,Bold=1,BorderStyle=3,Outline=0,Shadow=0,Alignment=2,MarginV=35",
        # 5. Neon (Cyan Glow)
        "Fontname=Arial,Fontsize=22,PrimaryColour=&H00FFFF00,OutlineColour=&H00000000,BackColour=&H00000000,Bold=1,Outline=1,Shadow=2,Alignment=2,MarginV=35"
    ]
    return random.choice(styles)

# ==========================================
# 7. MAIN PIPELINE
# ==========================================
print("--- 🚀 STARTING ---")
update_status(1, "Initializing Cloud Engine...")

# 1. Assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"
if not download_asset(VOICE_PATH, ref_voice): 
    update_status(0, "Missing Voice File", "failed"); exit(1)
download_asset(LOGO_PATH, ref_logo)

# 2. Script
update_status(10, "Creating Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
if len(text) < 100: 
    update_status(0, "Script Generation Failed", "failed"); exit(1)

# 3. Audio
update_status(20, "Synthesizing Voice...")
audio_out = TEMP_DIR / "out.wav"
if not clone_voice_robust(text, ref_voice, audio_out): exit(1)

# 4. Subtitles
update_status(50, "Generating Subtitles...")
aai.settings.api_key = ASSEMBLY_KEY
t = aai.Transcriber().transcribe(str(audio_out))

if t.status == aai.TranscriptStatus.error:
    update_status(0, "Subtitle Error", "failed"); exit(1)

# Generate ASS
ass_file = TEMP_DIR / "style.ass"
style = get_subtitle_style()
with open(ass_file, "w") as f:
    f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, BorderStyle, Spacing\nStyle: Default,{style}\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for s in t.get_sentences():
        start = f"{int(s.start/3600000)}:{int(s.start/60000)%60:02d}:{int(s.start/1000)%60:02d}.{int(s.start%1000/10):02d}"
        end = f"{int(s.end/3600000)}:{int(s.end/60000)%60:02d}:{int(s.end/1000)%60:02d}.{int(s.end%1000/10):02d}"
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{s.text}\n")

# 5. Visuals & Render
sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
update_status(60, "Downloading Visuals...")

used_ids = set()
def download_clip(args):
    i, sent = args
    dur = max(3.5, sent['end'] - sent['start'])
    query = get_visual_query(sent['text'])
    out = TEMP_DIR / f"s_{i}.mp4"
    
    found = None
    if PEXELS_KEYS:
        try:
            h = {"Authorization": random.choice(PEXELS_KEYS)}
            r = requests.get(f"https://api.pexels.com/videos/search?query={query}&size=medium&orientation=landscape", headers=h, timeout=5)
            vids = r.json().get('videos', [])
            for v in vids:
                if v['id'] not in used_ids: found = v['video_files'][0]['link']; used_ids.add(v['id']); break
        except: pass
        
    if found:
        try:
            raw = TEMP_DIR / f"r_{i}.mp4"
            with open(raw, "wb") as f: f.write(requests.get(found).content)
            subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(dur), "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30", "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(out)
        except: pass
    
    # Fallback
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}", "-t", str(dur), "-vf", "fps=30", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    clips = list(ex.map(download_clip, [(i, s) for i, s in enumerate(sentences)]))

with open("list.txt", "w") as f:
    for c in clips: f.write(f"file '{c}'\n")
    
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 6. Final Render
update_status(85, "Final Rendering...")
final_out = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
ass_path = str(Path(ass_file).resolve()).replace("\\", "/").replace(":", "\\:")

if os.path.exists(ref_logo):
    # Logo 300px + Subtitles
    filter = f"[1:v]scale=300:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass='{ass_path}'[v2]"
    inputs = ["-i", "visual.mp4", "-i", str(ref_logo), "-i", str(audio_out)]
    maps = ["-map", "[v2]", "-map", "2:a"]
else:
    filter = f"ass='{ass_path}'"
    inputs = ["-i", "visual.mp4", "-i", str(audio_out)]
    maps = ["-c:v", "libx264", "-c:a", "aac"]

cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter] + maps + ["-preset", "medium", "-b:v", "5000k", "-shortest", str(final_out)]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 7. Upload
update_status(99, "Uploading Video...")
link = upload_to_fileio(final_out)

if link:
    update_status(100, "Done!", "completed", link)
else:
    update_status(100, "Done (Upload Failed - Check Logs)", "completed")
