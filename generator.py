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
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Multi-Key Rotation
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
# 3. MASSIVE VISUAL DICTIONARY
# ==========================================
VISUAL_MAP = {
    # TECH / SCI-FI
    "tech": ["futuristic data center", "ai robot face", "hologram city", "cyberpunk street", "microchip macro"],
    "ai": ["artificial intelligence brain", "neural network", "glowing nodes", "cyborg eye"],
    "crypto": ["bitcoin gold coin", "stock market graph", "blockchain network", "digital wallet"],
    
    # NATURE / TRAVEL
    "nature": ["drone forest aerial", "mountain cinematic", "ocean waves slow motion", "waterfall 4k"],
    "space": ["galaxy stars deep space", "planet earth rotating", "mars surface", "astronaut floating"],
    "travel": ["airplane wing window", "paris eiffel tower", "tropical beach drone", "road trip highway"],
    
    # EMOTION / HUMAN
    "happy": ["friends laughing party", "diverse people smiling", "concert crowd cheering", "success celebration"],
    "sad": ["rainy window moody", "lonely person silhouette", "dark storm clouds", "candle blowing out"],
    "horror": ["spooky forest fog", "abandoned house", "shadow figure", "creepy doll"],
    
    # BUSINESS / HISTORY
    "business": ["corporate handshake", "office meeting time lapse", "man in suit walking", "money falling"],
    "history": ["ancient rome ruins", "egypt pyramids", "vintage map old", "medieval castle"],
    "war": ["military tank", "soldiers marching", "explosion fire", "jet fighter"],
    
    # ABSTRACT FALLBACKS
    "abstract": ["ink in water colorful", "light leaks cinematic", "particles dust floating", "geometric shapes loop"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    
    # 1. Dictionary Match
    for cat, terms in VISUAL_MAP.items():
        if cat in text: queries.extend(random.sample(terms, min(2, len(terms))))
    
    # 2. Extract Key Nouns (Simple Regex)
    words = [w for w in re.findall(r'\w+', text) if len(w) > 6]
    if words: queries.append(random.choice(words) + " cinematic 4k")
    
    # 3. Fallback
    if not queries: queries = random.sample(VISUAL_MAP["abstract"], 3)
    
    random.shuffle(queries)
    return list(set(queries))[:3]

# ==========================================
# 4. STATUS & FILE.IO
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
        payload = {"message": "update", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def upload_to_fileio(file_path):
    print("Uploading to File.io...")
    for i in range(3): # Retry logic
        try:
            with open(file_path, 'rb') as f:
                r = requests.post('https://file.io', files={'file': f})
            if r.status_code == 200: return r.json().get('link')
        except: time.sleep(2)
    return None

def download_asset(path_key, local_path):
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path_key}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 5. SMART SCRIPT GENERATOR
# ==========================================
def generate_script(topic, minutes):
    words = int(minutes * 150)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    # CHUNK LOGIC FOR LONG VIDEOS (>15 mins)
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5 + i, f"Writing Script Part {i+1}/{chunks}...")
            prompt = f"""
            Write Part {i+1} of {chunks} for a documentary about '{topic}'.
            Context: {full_script[-1][-200:] if full_script else 'Start of video'}
            Format: Spoken Narration ONLY. NO music cues. NO '[Intro]'.
            Length: ~700 words.
            """
            text = call_gemini(prompt)
            if text: full_script.append(text)
        return " ".join(full_script)
    
    # SHORT VIDEO
    else:
        prompt = f"""
        Write a YouTube script about '{topic}'.
        Length: {words} words.
        Format: Spoken Narration ONLY. Do NOT write 'Scene 1', 'Music Fades', or '[Host]'. Just the text.
        """
        return call_gemini(prompt)

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except: continue
    return "Script generation error. Proceeding with backup text."

# ==========================================
# 6. ROBUST AUDIO ENGINE
# ==========================================
def clone_voice_robust(text, ref_audio, out_path):
    print("Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Regex Split (Best for stability)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            if i % 5 == 0: update_status(20 + int((i/len(sentences))*30), f"Voice Gen ({i}/{len(sentences)})")
            
            try:
                chunk = chunk.replace('"', '').replace("'", "")
                wav = model.generate(chunk, str(ref_audio), exaggeration=0.5)
                all_wavs.append(wav.cpu())
                if device == "cuda": torch.cuda.empty_cache() # CRITICAL FIX
            except: pass

        if not all_wavs: return False
        final = torch.cat(all_wavs, dim=1)
        torchaudio.save(out_path, final, 24000)
        return True
    except: return False

# ==========================================
# 7. PROFESSIONAL SUBTITLE STYLES
# ==========================================
def get_random_style():
    styles = [
        # 1. Netflix Standard (White with Shadow)
        "Fontname=Arial,Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,Bold=0,Italic=0,Outline=1,Shadow=1,Alignment=2,MarginV=50",
        
        # 2. YouTuber Bold (Yellow with Thick Outline)
        "Fontname=Impact,Fontsize=28,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BackColour=&H80000000,Bold=1,Italic=0,Outline=2,Shadow=0,Alignment=2,MarginV=60",
        
        # 3. Modern Box (White text on Black Box)
        "Fontname=Roboto,Fontsize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H60000000,Bold=1,Italic=0,BorderStyle=3,Outline=0,Shadow=0,Alignment=2,MarginV=40",
        
        # 4. Cinematic Minimal (Small, Spaced, Semi-transparent)
        "Fontname=Arial,Fontsize=18,PrimaryColour=&H00E0E0E0,OutlineColour=&H00000000,BackColour=&H00000000,Bold=1,Italic=0,Outline=1,Shadow=0,Alignment=2,MarginV=30,Spacing=2",
        
        # 5. Neon Glow (Cyan with Glow effect simulated by shadow)
        "Fontname=Verdana,Fontsize=24,PrimaryColour=&H00FFFF00,OutlineColour=&H00000000,BackColour=&H00000000,Bold=1,Italic=0,Outline=1,Shadow=2,Alignment=2,MarginV=50"
    ]
    return random.choice(styles)

# ==========================================
# 8. VISUAL PROCESSING
# ==========================================
def process_visuals(sentences, audio_path, ass_file, logo_path, final_out):
    print("Processing Visuals...")
    used_ids = set()
    
    def download_clip(args):
        i, sent = args
        dur = max(3.5, sent['end'] - sent['start'])
        query = get_visual_queries(sent['text'])[0]
        out = TEMP_DIR / f"s_{i}.mp4"
        
        # Pexels Search
        found = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={query}&size=medium&orientation=landscape", headers=h, timeout=5)
                vids = r.json().get('videos', [])
                for v in vids:
                    if v['id'] not in used_ids:
                        found = v['video_files'][0]['link']; used_ids.add(v['id']); break
            except: pass
            
        if found:
            try:
                raw = TEMP_DIR / f"r_{i}.mp4"
                with open(raw, "wb") as f: f.write(requests.get(found).content)
                subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(dur), "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30", "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(out)
            except: pass
            
        # Fallback Black
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}", "-t", str(dur), "-vf", "fps=30", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    # Parallel DL
    tasks = [(i, s) for i, s in enumerate(sentences)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        clips = list(ex.map(download_clip, tasks))
        
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
        
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "visual.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Final Render (Logo 300px + Subs)
    ass_path = str(Path(ass_file).resolve()).replace("\\", "/").replace(":", "\\:")
    
    if os.path.exists(logo_path):
        filter = f"[1:v]scale=300:-1[logo];[0:v][logo]overlay=30:30[v1];[v1]ass='{ass_path}'[v2]"
        inputs = ["-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path)]
        maps = ["-map", "[v2]", "-map", "2:a"]
    else:
        filter = f"ass='{ass_path}'"
        inputs = ["-i", "visual.mp4", "-i", str(audio_path)]
        maps = ["-c:v", "libx264", "-c:a", "aac"]
        
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter] + maps + ["-preset", "medium", "-shortest", str(final_out)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 9. EXECUTION FLOW
# ==========================================
print("--- 🚀 START ---")
update_status(1, "Booting...")

# Assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"
if not download_asset(VOICE_PATH, ref_voice): update_status(0, "Voice Fail", "failed"); exit(1)
download_asset(LOGO_PATH, ref_logo)

# Script
update_status(10, "Writing Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT

# Audio
update_status(20, "Audio Gen...")
audio_out = TEMP_DIR / "out.wav"
if clone_voice_robust(text, ref_voice, audio_out):
    
    # Subtitles
    update_status(50, "Generating Subs...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_out))
    
    if t.status != aai.TranscriptStatus.error:
        # Generate Styled ASS
        ass_file = TEMP_DIR / "style.ass"
        chosen_style = get_random_style()
        print(f"Chosen Subtitle Style: {chosen_style}")
        
        with open(ass_file, "w") as f:
            f.write(f"""[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, BorderStyle, Spacing\nStyle: Default,{chosen_style}\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n""")
            for s in t.get_sentences():
                # Split long sentences into 2-word chunks for faster pacing if needed, 
                # but standard sentence-level is safer for sync. We stick to sentence level for now.
                start = f"{int(s.start/3600000)}:{int(s.start/60000)%60:02d}:{int(s.start/1000)%60:02d}.{int(s.start%1000/10):02d}"
                end = f"{int(s.end/3600000)}:{int(s.end/60000)%60:02d}:{int(s.end/1000)%60:02d}.{int(s.end%1000/10):02d}"
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{s.text}\n")
        
        sents = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
        
        # Visuals & Render
        update_status(60, "Visual Engine...")
        final = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
        process_visuals(sents, audio_out, ass_file, ref_logo, final)
        
        # Upload
        update_status(95, "Uploading...")
        link = upload_to_fileio(final)
        if link: update_status(100, "Done!", "completed", link)
        else: update_status(100, "Done (Upload Failed)", "completed")
        
    else: update_status(0, "Subs Failed", "failed")
else: update_status(0, "Audio Failed", "failed")
