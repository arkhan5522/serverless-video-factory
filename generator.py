"""
VIDEO FACTORY V3 - PRODUCTION ENGINE
======================================
Submagic-level quality video generation:
- Groq GPT-OSS-120B for intelligent query generation
- Chatterbox Multilingual TTS (English/Spanish)
- Resemble Enhance for studio-grade audio
- Sentence-synced visuals (each clip matches its narration)
- Ken Burns cinematic effects (zoom/pan on clips)
- Multiple subtitle style presets (movie, modern, bold)
- Color-graded output for professional look
- Optimized for Kaggle P100 16GB GPU
"""

import os, subprocess, sys, re, time, random, shutil, json
import concurrent.futures, requests, gc
from pathlib import Path

# ==========================================
# 1. INSTALLATION
# ==========================================
print("--- Installing Dependencies ---")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "groq", "chatterbox-tts", "assemblyai", "google-generativeai",
    "transformers", "sentencepiece", "requests", "pydub", "numpy", "pillow"
], capture_output=True)

# Resemble Enhance - install from git to avoid dep conflicts
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
    "resemble-enhance"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "librosa", "scipy", "df_conformer"], capture_output=True)

subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True,
    capture_output=True)

import torch, torchaudio
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
LANGUAGE = """{{LANGUAGE_PLACEHOLDER}}"""

raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

IS_SPANISH = LANGUAGE.lower().strip() in ["spanish", "es", "espanol"]
USED_URLS = set()
AI_QUERIES = []



# ==========================================
# 3. GROQ QUERY ENGINE
# ==========================================
def generate_queries(script_text, num_queries):
    """Generate precise English queries via Groq GPT-OSS-120B"""
    if not GROQ_KEY:
        print("  No Groq key, using fallback")
        return [random.choice(FALLBACK) for _ in range(num_queries)]
    
    print(f"  Generating {num_queries} queries via Groq...")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        all_q = []
        
        words = script_text.split()
        # 2 batches for efficiency
        segments = [' '.join(words[:len(words)//2])[:2500],
                    ' '.join(words[len(words)//2:])[:2500]]
        
        for seg in segments:
            needed = min(num_queries - len(all_q), num_queries // 2 + 5)
            if needed <= 0: break
            
            r = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You generate short English stock footage search queries. Each query is 3-5 words describing a visual scene a camera can capture. Never include people, faces, religion, violence, or NSFW content."},
                    {"role": "user", "content": f"From this script, generate {needed} video search queries. Focus on the specific visuals described - objects, places, nature, technology, architecture. One per line:\n\n{seg}"}
                ],
                model="openai/gpt-oss-120b",
                max_tokens=600,
                temperature=0.7
            )
            
            text = r.choices[0].message.content
            for line in text.strip().split('\n'):
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                if 3 < len(cleaned) < 50 and _safe(cleaned):
                    all_q.append(cleaned)
            time.sleep(0.5)
        
        while len(all_q) < num_queries:
            all_q.append(random.choice(FALLBACK))
        
        print(f"  Got {len(all_q)} queries")
        return all_q[:num_queries]
    except Exception as e:
        print(f"  Groq error: {e}")
        return [random.choice(FALLBACK) for _ in range(num_queries)]

FALLBACK = [
    "forest trees cinematic 4k", "mountain landscape aerial",
    "waterfall nature slow motion", "ocean waves sunset golden",
    "city skyline night timelapse", "clouds rolling mountains",
    "river flowing rocks forest", "northern lights sky",
    "desert sand dunes aerial", "coral reef underwater 4k",
    "volcano eruption dramatic", "snowfall pine trees winter",
    "lightning storm clouds", "sunrise over valley fog",
    "autumn leaves falling wind", "space earth orbit view"
]

def _safe(q):
    bad = ['woman','women','girl','female','bikini','nude','naked','sexy',
           'jesus','christ','church','mosque','temple','bible','buddha',
           'gun','weapon','war','blood','violence','kill','alcohol',
           'drug','gambling','pork','lgbtq','person','people','crowd']
    ql = q.lower()
    return not any(t in ql for t in bad)



# ==========================================
# 4. SUBTITLE SYSTEM (Multiple Styles)
# ==========================================
SUBTITLE_PRESETS = {
    "cinema": {
        "name": "Cinema (Netflix Style)",
        "font": "Arial", "size": 54, "bold": -1,
        "primary": "&H00FFFFFF", "outline_c": "&H00000000",
        "back": "&H80000000", "border": 3, "outline": 0,
        "shadow": 0, "margin": 40, "spacing": 0.5
    },
    "modern_white": {
        "name": "Modern White Bold",
        "font": "Arial Black", "size": 58, "bold": -1,
        "primary": "&H00FFFFFF", "outline_c": "&H00111111",
        "back": "&H00000000", "border": 1, "outline": 4,
        "shadow": 2, "margin": 45, "spacing": 1
    },
    "neon_yellow": {
        "name": "Neon Yellow Pop",
        "font": "Arial Black", "size": 60, "bold": -1,
        "primary": "&H0000FFFF", "outline_c": "&H00000000",
        "back": "&H00000000", "border": 1, "outline": 5,
        "shadow": 3, "margin": 50, "spacing": 1.5
    },
    "soft_shadow": {
        "name": "Soft Shadow Elegant",
        "font": "Arial", "size": 52, "bold": -1,
        "primary": "&H00FFFFFF", "outline_c": "&H00333333",
        "back": "&H00000000", "border": 1, "outline": 3,
        "shadow": 4, "margin": 42, "spacing": 0.8
    },
    "highlight_green": {
        "name": "Highlight Green",
        "font": "Arial Black", "size": 56, "bold": -1,
        "primary": "&H0000FF88", "outline_c": "&H00003311",
        "back": "&H00000000", "border": 1, "outline": 4,
        "shadow": 2, "margin": 48, "spacing": 1.2
    },
}

def create_subtitles(sentences, ass_path, style_name=None):
    """Create clean, movie-grade subtitles - bottom, max 2 lines, readable"""
    if not style_name:
        style_name = random.choice(list(SUBTITLE_PRESETS.keys()))
    s = SUBTITLE_PRESETS[style_name]
    print(f"  Subtitle style: {s['name']}")
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\nScriptType: v4.00+\n")
        f.write("PlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{s['font']},{s['size']},{s['primary']},&H00FFFFFF,"
                f"{s['outline_c']},{s['back']},{s['bold']},0,0,0,100,100,"
                f"{s['spacing']},0,{s['border']},{s['outline']},{s['shadow']},"
                f"2,50,50,{s['margin']},1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for sent in sentences:
            t1 = _fmt(sent['start'])
            t2 = _fmt(sent['end'])
            text = sent['text'].strip().rstrip('.,;:')
            
            # Smart 2-line wrap at natural break point
            if len(text) > 40:
                words = text.split()
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                text = f"{line1}\\N{line2}"
            
            f.write(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{text}\n")

def _fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"



# ==========================================
# 5. AUDIO ENGINE (Chatterbox + Enhance)
# ==========================================
def generate_audio(text, ref_audio, out_path):
    """High-quality voice: Chatterbox TTS + Resemble Enhance mastering"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_path = TEMP_DIR / "raw_tts.wav"
    
    # --- TTS ---
    print(f"  TTS: {'Multilingual (ES)' if IS_SPANISH else 'English'} on {device}")
    try:
        if IS_SPANISH:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        
        sr = model.sr
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Group 2-3 sentences per chunk for natural flow
        chunks = []
        buf, blen = [], 0
        for s in sents:
            if blen + len(s) > 160 and buf:
                chunks.append(' '.join(buf)); buf, blen = [s], len(s)
            else:
                buf.append(s); blen += len(s) + 1
        if buf: chunks.append(' '.join(buf))
        
        print(f"  {len(sents)} sentences -> {len(chunks)} chunks")
        wavs = []
        for i, c in enumerate(chunks):
            if i % 5 == 0:
                update_status(18 + int((i/len(chunks))*27), f"TTS {i+1}/{len(chunks)}")
            try:
                with torch.no_grad():
                    if IS_SPANISH:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio), language_id="es")
                    else:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio), exaggeration=0.6, cfg_weight=0.4)
                    wavs.append(w.cpu())
                if i % 8 == 0: torch.cuda.empty_cache()
            except: continue
        
        if not wavs: return False
        
        # Concat with natural pauses
        full = wavs[0]
        for w in wavs[1:]:
            pause = torch.zeros((full.shape[0], int(random.uniform(0.12, 0.22) * sr)))
            full = torch.cat([full, pause, w], dim=1)
        full = torch.cat([full, torch.zeros((full.shape[0], int(1.5 * sr)))], dim=1)
        torchaudio.save(str(raw_path), full, sr)
        print(f"  TTS done: {full.shape[1]/sr:.1f}s at {sr}Hz")
        
        del model, wavs, full; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"  TTS error: {e}"); return False
    
    # --- ENHANCE ---
    print("  Enhancing audio (denoise + upscale)...")
    try:
        from resemble_enhance.enhancer.inference import enhance as re_enhance
        dwav, osr = torchaudio.load(str(raw_path))
        
        # Process in 25s chunks
        chunk_s = 25 * osr
        parts = []
        for i in range(0, dwav.shape[1], chunk_s):
            chunk = dwav[:, i:i+chunk_s]
            try:
                hw, esr = re_enhance(dwav=chunk, sr=osr, device=device, lambd=0.6)
                parts.append(hw.cpu())
            except:
                parts.append(torchaudio.transforms.Resample(osr, 44100)(chunk).cpu())
                esr = 44100
            torch.cuda.empty_cache()
        
        final = torch.cat(parts, dim=1)
        torchaudio.save(str(out_path), final, esr)
        print(f"  Enhanced: {esr}Hz, {final.shape[1]/esr:.1f}s")
        del parts, final, dwav; torch.cuda.empty_cache(); gc.collect()
        return True
    except Exception as e:
        print(f"  Enhance unavailable ({str(e)[:40]}), using raw")
        shutil.copy2(str(raw_path), str(out_path))
        return True



# ==========================================
# 6. VIDEO ENGINE (Cinematic Quality)
# ==========================================
def search_and_download(query, idx, duration):
    """Search Pexels/Pixabay and download best HD clip"""
    urls = []
    page = random.randint(1, 3)
    
    # Pexels
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 15, "page": page, "orientation": "landscape"},
                timeout=12)
            if r.status_code == 200:
                for v in r.json().get('videos', []):
                    files = v.get('video_files', [])
                    hd = [f for f in files if f.get('quality') == 'hd' and f.get('width', 0) >= 1280]
                    if not hd: hd = [f for f in files if f.get('quality') in ['hd', 'large']]
                    if hd:
                        url = random.choice(hd)['link']
                        if url not in USED_URLS: urls.append(url)
        except: pass
    
    # Pixabay
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key": key, "q": query, "per_page": 15, "page": page},
                timeout=12)
            if r.status_code == 200:
                for v in r.json().get('hits', []):
                    vd = v.get('videos', {})
                    url = vd.get('large', vd.get('medium', {})).get('url')
                    if url and url not in USED_URLS: urls.append(url)
        except: pass
    
    # Download first available
    for url in urls[:3]:
        try:
            raw = TEMP_DIR / f"raw_{idx}.mp4"
            out = TEMP_DIR / f"clip_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            
            if os.path.getsize(raw) < 5000: continue
            
            # Ken Burns effect: slow zoom or pan for cinematic feel
            kb_effects = [
                "scale=2048:1152,zoompan=z='min(zoom+0.0008,1.15)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30",
                "scale=2048:1152,zoompan=z='1.15':d=1:x='if(eq(on,1),0,x+1)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30",
                "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30"
            ]
            # Use Ken Burns 60% of the time, static crop 40%
            effect = random.choice(kb_effects[:2]) if random.random() < 0.6 else kb_effects[2]
            
            subprocess.run([
                "ffmpeg", "-y", "-i", str(raw), "-t", str(duration),
                "-vf", effect,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", str(out)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            
            os.remove(raw)
            if os.path.exists(out) and os.path.getsize(out) > 2000:
                USED_URLS.add(url)
                return str(out)
        except: continue
    return None

def process_clip(args):
    """Process single clip with retry"""
    i, sent, total = args
    dur = max(3.5, sent['end'] - sent['start'])
    query = AI_QUERIES[i] if i < len(AI_QUERIES) else random.choice(FALLBACK)
    
    for attempt in range(3):
        clip = search_and_download(query, i, dur)
        if clip: return (i, clip)
        query = random.choice(FALLBACK)
        time.sleep(0.3)
    return (i, None)



# ==========================================
# 7. RENDER ENGINE
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_nosub, out_sub):
    """Full render pipeline: download clips, concat, apply logo+subs"""
    n = len(sentences)
    print(f"\n  Rendering {n} clips (5 workers)...")
    
    # Parallel download
    clips = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(process_clip, (i, s, n)): i for i, s in enumerate(sentences)}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            try:
                idx, path = f.result()
                if path: clips[idx] = path; done += 1
                update_status(55 + int((done/n)*22), f"Clips {done}/{n}")
            except: pass
    
    # Fill gaps (loop nearest valid clip)
    valid = [i for i, c in enumerate(clips) if c and os.path.exists(c)]
    if not valid: return False
    
    for i in range(n):
        if clips[i] and os.path.exists(clips[i]): continue
        nearest = min(valid, key=lambda x: abs(x-i))
        dur = max(3.5, sentences[i]['end'] - sentences[i]['start'])
        gap = TEMP_DIR / f"gap_{i}.mp4"
        subprocess.run(["ffmpeg","-y","-stream_loop","-1","-i",clips[nearest],
            "-t",str(dur),"-c:v","libx264","-preset","ultrafast","-crf","18","-an",str(gap)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips[i] = str(gap) if os.path.exists(gap) else clips[nearest]
    
    # Concat (stream copy - instant, no quality loss)
    print("  Concatenating...")
    with open("list.txt","w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists("visual.mp4"):
        subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c:v libx264 -preset ultrafast -crf 18 visual.mp4",
            shell=True, capture_output=True)
    if not os.path.exists("visual.mp4"): return False
    
    # === V1: No subtitles (900p) ===
    update_status(80, "Rendering V1 (900p)...")
    if logo_path and os.path.exists(logo_path):
        filt = "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=160:-1[l];[bg][l]overlay=20:20[v]"
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    else:
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(audio_path),
            "-vf","scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if not os.path.exists(out_nosub): return False
    print(f"  V1: {os.path.getsize(out_nosub)/(1024**2):.0f}MB")
    
    # === V2: With subtitles (1080p) ===
    update_status(88, "Rendering V2 (1080p + subs)...")
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];[wl]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    else:
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","1:a",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if os.path.exists(out_sub):
        print(f"  V2: {os.path.getsize(out_sub)/(1024**2):.0f}MB")
    return True



# ==========================================
# 8. UTILITIES
# ==========================================
LOG_BUF = []
def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")
    LOG_BUF.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(LOG_BUF) > 30: LOG_BUF.pop(0)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return
    import base64
    data = {"progress": progress, "message": message, "status": status,
            "logs": "\n".join(LOG_BUF), "timestamp": time.time()}
    if file_url: data["file_io_url"] = file_url
    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        gr = requests.get(url, headers=headers)
        sha = gr.json().get("sha") if gr.status_code == 200 else None
        payload = {"message": f"s{progress}", "content": base64.b64encode(json.dumps(data).encode()).decode(), "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"})
        if r.status_code == 200:
            with open(local, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def upload_drive(file_path):
    if not os.path.exists(file_path): return None
    print(f"  Uploading {os.path.basename(file_path)}...")
    cid = os.environ.get("OAUTH_CLIENT_ID")
    cs = os.environ.get("OAUTH_CLIENT_SECRET")
    rt = os.environ.get("OAUTH_REFRESH_TOKEN")
    fid = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not all([cid,cs,rt]): return None
    try:
        tok = requests.post("https://oauth2.googleapis.com/token",
            data={"client_id":cid,"client_secret":cs,"refresh_token":rt,"grant_type":"refresh_token"}).json()['access_token']
    except: return None
    
    sz = os.path.getsize(file_path)
    meta = {"name": os.path.basename(file_path), "mimeType": "video/mp4"}
    if fid: meta["parents"] = [fid]
    
    resp = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json",
                 "X-Upload-Content-Type":"video/mp4","X-Upload-Content-Length":str(sz)}, json=meta)
    if resp.status_code != 200: return None
    
    with open(file_path,"rb") as f:
        ur = requests.put(resp.headers["Location"], headers={"Content-Length":str(sz)}, data=f)
    if ur.status_code in [200,201]:
        fid2 = ur.json().get('id')
        requests.post(f"https://www.googleapis.com/drive/v3/files/{fid2}/permissions",
            headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},
            json={'role':'reader','type':'anyone'})
        link = f"https://drive.google.com/file/d/{fid2}/view?usp=sharing"
        print(f"  -> {link}")
        return link
    return None

def generate_script(topic, mins):
    words = int(mins * 180)
    lang = "Write in Spanish." if IS_SPANISH else "Write in English."
    prompt = f"""Write a documentary narration about '{topic}'. {words} words.
{lang}
Rules: Only narration text. No brackets, no directions. Islamic guidelines. Family-friendly."""
    random.shuffle(GEMINI_KEYS)
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            r = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt)
            return re.sub(r'\[.*?\]', '', r.text.replace("*","").replace("#","").strip())
        except: continue
    return ""



# ==========================================
# 9. MAIN
# ==========================================
print(f"\n{'='*50}")
print(f"  VIDEO FACTORY V3 - PRODUCTION ENGINE")
print(f"  Lang: {'ES' if IS_SPANISH else 'EN'} | Groq: GPT-OSS-120B")
print(f"{'='*50}\n")

update_status(1, "Starting...")

# Assets
voice = TEMP_DIR / "voice.mp3"
logo = TEMP_DIR / "logo.png"
if not download_asset(VOICE_PATH, voice):
    update_status(0, "Voice download failed", "failed"); exit(1)
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, logo)
    logo = str(logo) if os.path.exists(logo) else None
else: logo = None

# Script
update_status(5, "Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
if len(text) < 50:
    update_status(0, "Script too short", "failed"); exit(1)
print(f"  {len(text.split())} words")

# Queries
update_status(10, "Generating queries...")
n_clips = max(8, int(DURATION_MINS * 7))
AI_QUERIES = generate_queries(text, n_clips)

# Audio
update_status(15, "Audio generation...")
audio = TEMP_DIR / "audio.wav"
if not generate_audio(text, voice, audio):
    update_status(0, "Audio failed", "failed"); exit(1)

# Transcribe
update_status(48, "Transcribing...")
sentences = []
if ASSEMBLY_KEY:
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        tx = aai.Transcriber().transcribe(str(audio))
        for s in tx.get_sentences():
            sentences.append({"text": s.text, "start": s.start/1000, "end": s.end/1000})
        if sentences: sentences[-1]['end'] += 0.5
    except Exception as e:
        print(f"  Transcribe error: {e}")

if not sentences:
    import wave
    try:
        with wave.open(str(audio),'rb') as w: dur = w.getnframes()/float(w.getframerate())
    except: dur = len(text.split())/2.5
    wps = len(text.split())/dur if dur > 0 else 2.5
    t, words = 0, text.split()
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        d = len(chunk)/wps
        sentences.append({"text":' '.join(chunk), "start":t, "end":t+d})
        t += d

# Pad queries
while len(AI_QUERIES) < len(sentences):
    AI_QUERIES.append(random.choice(FALLBACK))

# Subtitles
update_status(50, "Creating subtitles...")
ass = TEMP_DIR / "subs.ass"
create_subtitles(sentences, ass)

# Render
update_status(52, "Processing video...")
o1 = OUTPUT_DIR / f"final_{JOB_ID}_NO_SUBS.mp4"
o2 = OUTPUT_DIR / f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o1, o2):
    update_status(93, "Uploading...")
    l1 = upload_drive(o1)
    l2 = upload_drive(o2)
    msg = "Done!\n"
    if l1: msg += f"No Subs: {l1}\n"
    if l2: msg += f"With Subs: {l2}\n"
    update_status(100, msg, "completed", l1 or l2)
    print(f"\n{'='*50}\n  {msg}{'='*50}")
else:
    update_status(0, "Render failed", "failed")

# Cleanup
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4","list.txt"]:
    if os.path.exists(f): os.remove(f)
print("\n--- DONE ---")
