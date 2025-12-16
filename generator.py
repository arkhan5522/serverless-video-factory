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
# 1. INSTALLATION
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
# 2. CONFIGURATION
# ==========================================
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]

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
# 3. UTILS
# ==========================================
def update_status(progress, message, status="processing", file_url=None):
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
        payload = {"message": "upd", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def robust_upload(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None

    # 1. Try File.io
    print("Attempting File.io...")
    try:
        with open(file_path, 'rb') as f:
            r = requests.post('https://file.io', files={'file': f}, timeout=60)
        if r.status_code == 200:
            link = r.json().get('link')
            return link
    except Exception as e: print(f"File.io Error: {e}")

    # 2. Fallback to Transfer.sh
    print("Attempting Transfer.sh Fallback...")
    try:
        cmd = ["curl", "--upload-file", str(file_path), f"https://transfer.sh/{os.path.basename(file_path)}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        link = result.stdout.strip()
        if "transfer.sh" in link: return link
    except Exception as e: print(f"Transfer.sh Error: {e}")
    return None

def download_asset(path, local):
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 4. SCRIPTING
# ==========================================
def generate_script(topic, minutes):
    words = int(minutes * 150)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            prompt = f"Write Part {i+1}/{chunks} of a documentary about '{topic}'. Context: {full_script[-1][-200:] if full_script else 'Start'}. Length: 700 words. Spoken Text ONLY."
            full_script.append(call_gemini(prompt))
        return " ".join(full_script)
    else:
        prompt = f"Write a YouTube script about '{topic}'. {words} words. Spoken Text ONLY. No [Music] tags."
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
# 5. AUDIO
# ==========================================
def clone_voice_robust(text, ref_audio, out_path):
    print("Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        all_wavs = []
        print(f"Sentences to process: {len(sentences)}")
        
        for i, chunk in enumerate(sentences):
            if i%10==0: update_status(20 + int((i/len(sentences))*30), f"Voice Gen {i}/{len(sentences)}")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk.replace('"',''), 
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                
                if i%20==0 and device=="cuda": 
                    torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"Skipping chunk {i}: {e}")
            
        if not all_wavs: return False
        torchaudio.save(out_path, torch.cat(all_wavs, dim=1), 24000)
        return True
    except Exception as e:
        print(f"Critical Audio Fail: {e}")
        return False

# ==========================================
# 6. VISUALS (FIXED SHELL ERROR)
# ==========================================
def get_subtitle_style():
    # 'Sans' font for compatibility
    styles = [
        "Fontname=Sans,Fontsize=24,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,Bold=1,Outline=1,Shadow=1,MarginV=35", # Standard
        "Fontname=Sans,Fontsize=26,PrimaryColour=&H0000FFFF,BackColour=&H80000000,Bold=1,Outline=2,Shadow=0,MarginV=45", # Yellow
        "Fontname=Sans,Fontsize=22,PrimaryColour=&H00FFFFFF,BackColour=&H60000000,Bold=1,BorderStyle=3,MarginV=40",      # Box
        "Fontname=Sans,Fontsize=24,PrimaryColour=&H00FFFF00,BackColour=&H00000000,Bold=1,Outline=1,Shadow=2,MarginV=40"  # Neon
    ]
    return random.choice(styles)

def process_visuals(sentences, audio_path, ass_file, logo_path, final_out):
    print("Visuals & Render...")
    
    # 1. Download Visuals
    def get_clip(args):
        i, sent = args
        dur = max(3.5, sent['end'] - sent['start'])
        words = [w for w in re.findall(r'\w+', sent['text'].lower()) if len(w)>5]
        query = (random.choice(words) + " cinematic") if words else "abstract background"
        out = TEMP_DIR / f"s_{i}.mp4"
        
        found = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={query}&size=medium&orientation=landscape", headers=h, timeout=5)
                found = r.json()['videos'][0]['video_files'][0]['link']
            except: pass
            
        if found:
            try:
                raw = TEMP_DIR / f"r_{i}.mp4"
                with open(raw, "wb") as f: f.write(requests.get(found).content)
                
                # FIX: Use LIST for command to avoid Shell Syntax Error with Parentheses
                cmd = [
                    "ffmpeg", "-y", "-i", str(raw), "-t", str(dur),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(out)
            except: pass
        
        # Fallback Black Clip (Also safe list)
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}",
            "-t", str(dur), "-vf", "fps=30", str(out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        clips = list(ex.map(get_clip, [(i, s) for i, s in enumerate(sentences)]))

    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    # Concat
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Final Render
    if os.path.exists(logo_path):
        filter_complex = f"[1:v]scale=230:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass=temp/style.ass[v2]"
        cmd = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_complex, "-map", "[v2]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium", "-b:v", "5000k", "-c:a", "aac", "-shortest", str(final_out)
        ]
    else:
        filter_complex = "ass=temp/style.ass"
        cmd = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "medium", "-b:v", "5000k", "-c:a", "aac", "-shortest", str(final_out)
        ]
        
    print("Running Final Render...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 7. EXECUTION
# ==========================================
print("--- 🚀 START ---")
update_status(1, "Initializing...")

ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"
if not download_asset(VOICE_PATH, ref_voice): update_status(0, "Voice Fail", "failed"); exit(1)
download_asset(LOGO_PATH, ref_logo)

update_status(10, "Scripting...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT

update_status(20, "Audio...")
audio_out = TEMP_DIR / "out.wav"
if clone_voice_robust(text, ref_voice, audio_out):
    update_status(50, "Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_out))
    
    if t.status != aai.TranscriptStatus.error:
        # Generate ASS (Use Generic Font)
        ass_file = TEMP_DIR / "style.ass"
        style = get_subtitle_style()
        with open(ass_file, "w") as f:
            f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Outline, Shadow, MarginL, MarginR, MarginV, BorderStyle, Spacing\nStyle: Default,{style},0,0,10,10,0\n[Events]\nFormat: Layer, Start, End, Style, Text\n")
            for s in t.get_sentences():
                start_sec = s.start / 1000
                end_sec = s.end / 1000
                def fmt(s):
                    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s % 1) * 100)
                    return f"{h}:{m:02d}:{sec:02d}.{ms:02d}"
                f.write(f"Dialogue: 0,{fmt(start_sec)},{fmt(end_sec)},Default,{s.text}\n")
        
        # Render
        sents = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
        update_status(60, "Visuals...")
        final = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
        process_visuals(sents, audio_out, ass_file, ref_logo, final)
        
        # Upload (Robust)
        update_status(99, "Uploading...")
        link = robust_upload(final)
        
        if link: update_status(100, "Done!", "completed", link)
        else: update_status(100, "Upload Failed (Check Logs)", "failed")
        
    else: update_status(0, "Subs Failed", "failed")
else: update_status(0, "Audio Failed", "failed")
