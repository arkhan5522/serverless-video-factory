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
# 3. UTILS: STATUS & UPLOAD
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
    filename = os.path.basename(file_path)
    
    # 1. Try Transfer.sh (Best for CLI)
    print("Attempting Transfer.sh...")
    try:
        url = f"https://transfer.sh/{filename}"
        with open(file_path, 'rb') as f:
            r = requests.put(url, data=f)
        if r.status_code == 200:
            link = r.text.strip()
            return link
    except Exception as e: print(f"Transfer.sh Error: {e}")

    # 2. Fallback File.io
    print("Attempting File.io...")
    try:
        with open(file_path, 'rb') as f:
            r = requests.post('https://file.io', files={'file': f})
        if r.status_code == 200: return r.json().get('link')
    except: pass
    
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
# 4. MASSIVE VISUAL DICTIONARY
# ==========================================
VISUAL_MAP = {
    # TECH & FUTURE
    "tech": ["server room", "circuit board", "hologram", "robot face", "coding screen"],
    "ai": ["artificial intelligence", "neural network", "cyborg", "digital brain"],
    "future": ["futuristic city", "flying car", "spaceship", "neon street"],
    "crypto": ["bitcoin", "blockchain", "ethereum", "digital wallet", "stock graph"],
    "hacker": ["hoodie hacker", "matrix code", "security lock", "glitch effect"],
    
    # BUSINESS & MONEY
    "business": ["business meeting", "handshake", "office skyscrapers", "suit walking"],
    "money": ["counting money", "gold bars", "falling coins", "luxury car", "mansion"],
    "success": ["man on mountain", "cheering crowd", "trophy", "private jet"],
    "growth": ["plant growing", "chart going up", "sunrise time lapse", "building construction"],
    
    # EMOTION & HUMAN
    "happy": ["friends laughing", "party confetti", "smiling woman", "dancing crowd"],
    "sad": ["rainy window", "lonely man", "crying eye", "dark tunnel"],
    "fear": ["shadowy figure", "scary forest", "scream", "nightmare"],
    "love": ["couple holding hands", "wedding", "heart shape", "sunset kiss"],
    
    # NATURE & WORLD
    "nature": ["waterfall drone", "forest mist", "blooming flower", "ocean waves"],
    "space": ["galaxy stars", "planet earth", "astronaut", "black hole"],
    "city": ["nyc time lapse", "tokyo night", "traffic lights", "subway train"],
    "travel": ["airplane wing", "passport", "beach palm trees", "eiffel tower"],
    
    # CONCEPTS
    "time": ["clock ticking", "hourglass", "calendar", "time lapse clouds"],
    "idea": ["light bulb", "brainstorming", "sketching", "spark"],
    "network": ["mesh network", "fiber optic", "globe connections", "satellite"],
    "war": ["tank", "explosion", "soldier", "military jet"],
    "history": ["ancient ruins", "pyramids", "old map", "museum statue"],
    "abstract": ["ink in water", "smoke swirls", "light leaks", "geometric loop"]
}

def get_visual_query(text):
    text = text.lower()
    # Check dictionary first
    for cat, terms in VISUAL_MAP.items():
        if cat in text: 
            return random.choice(terms)
    
    # Fallback to noun extraction
    words = [w for w in re.findall(r'\w+', text) if len(w) > 5]
    if words: 
        return random.choice(words) + " cinematic"
    return "abstract background 4k"

# ==========================================
# 5. SCRIPT & AUDIO
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
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"Write Part {i+1}/{chunks} of a documentary about '{topic}'. Context: {context}. Length: 700 words. Spoken Text ONLY."
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
    return "Script generation failed."

def clone_voice_robust(text, ref_audio, out_path):
    print("Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            if i%10==0: update_status(20 + int((i/len(sentences))*30), f"Voice Gen {i}/{len(sentences)}")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk.replace('"',''), 
                        audio_prompt_path=str(ref_audio), # Named Arg Fix
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                if i%20==0 and device=="cuda": torch.cuda.empty_cache(); gc.collect()
            except: pass
            
        if not all_wavs: return False
        torchaudio.save(out_path, torch.cat(all_wavs, dim=1), 24000)
        return True
    except: return False

# ==========================================
# 6. VISUALS & RENDER
# ==========================================
def process_visuals(sentences, audio_path, ass_file, logo_path, final_out):
    print("Visuals & Render...")
    used_links = set() # Track used videos to prevent repetition
    
    def get_clip(args):
        i, sent = args
        dur = max(3.5, sent['end'] - sent['start'])
        query = get_visual_query(sent['text'])
        out = TEMP_DIR / f"s_{i}.mp4"
        
        found_link = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                # Fetch 15 results to find a unique one
                r = requests.get(f"https://api.pexels.com/videos/search?query={query}&size=medium&orientation=landscape&per_page=15", headers=h, timeout=5)
                videos = r.json().get('videos', [])
                random.shuffle(videos) # Shuffle results
                
                for v in videos:
                    link = v['video_files'][0]['link']
                    # Simple check to avoid duplicates in this run
                    # Note: imperfect in threads but good enough for variety
                    if link not in used_links:
                        found_link = link
                        used_links.add(link)
                        break
            except: pass
            
        if found_link:
            try:
                raw = TEMP_DIR / f"r_{i}.mp4"
                with open(raw, "wb") as f: f.write(requests.get(found_link).content)
                # Use list command to avoid shell syntax errors
                cmd = [
                    "ffmpeg", "-y", "-i", str(raw), "-t", str(dur),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(out)
            except: pass
        
        # Fallback Black Clip
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}",
            "-t", str(dur), "-vf", "fps=30", str(out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    # Parallel Download
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        clips = list(ex.map(get_clip, [(i, s) for i, s in enumerate(sentences)]))

    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4", shell=True)

    # Final Render with Subtitles & Logo
    # Fontsize 30, BorderStyle=3 (Opaque Box) for max visibility
    style = "Fontname=Sans,Fontsize=30,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,Bold=1,BorderStyle=3,MarginV=50"
    
    # Create valid ASS file
    with open(ass_file, "w") as f:
        f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, BorderStyle, MarginV\nStyle: Default,Sans,30,&H00FFFFFF,&H60000000,-1,3,50\n[Events]\nFormat: Layer, Start, End, Style, Text\n")
        for s in sentences:
            start = s['start']; end = s['end']
            def fmt(t):
                h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t%1)*100)
                return f"{h}:{m:02d}:{s:02d}.{ms:02d}"
            f.write(f"Dialogue: 0,{fmt(start)},{fmt(end)},Default,{s['text']}\n")

    # Construct Filter Complex
    # [0:v] is video, [1:v] is logo
    # Scale logo to 230px, overlay it
    # Then burn subtitles using relative path "temp/style.ass"
    if os.path.exists(logo_path):
        filter_complex = "[1:v]scale=230:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass=temp/style.ass[v2]"
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
        ass_file = TEMP_DIR / "style.ass"
        sents = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
        
        update_status(60, "Visuals...")
        final = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
        process_visuals(sents, audio_out, ass_file, ref_logo, final)
        
        update_status(99, "Uploading...")
        link = robust_upload(final)
        
        if link: update_status(100, "Done!", "completed", link)
        else: update_status(100, "Upload Failed (Check Logs)", "failed")
        
    else: update_status(0, "Subs Failed", "failed")
else: update_status(0, "Audio Failed", "failed")
