"""
VIDEO FACTORY V4 - PRODUCTION ENGINE
======================================
- Groq GPT-OSS-120B: sentence-level visual matching (JSON approach)
- Chatterbox Multilingual TTS + Resemble Enhance (44.1kHz studio)
- GPU-accelerated encoding (h264_nvenc on Kaggle T4/P100)
- Ken Burns cinematic effects on clips
- Multiple subtitle presets
- Optimized for Kaggle GPU
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
    "transformers", "sentencepiece", "requests", "pydub", "numpy", "pillow",
    "deepspeed", "resemble-enhance", "librosa", "scipy"
], capture_output=True)
subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, capture_output=True)

import torch, torchaudio
import assemblyai as aai
import google.generativeai as genai

# Check if NVENC is available
def _has_nvenc():
    r = subprocess.run("ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc",
        shell=True, capture_output=True, text=True)
    return "h264_nvenc" in r.stdout

USE_GPU = _has_nvenc()
print(f"  GPU Encoding: {'h264_nvenc' if USE_GPU else 'libx264 (CPU)'}")

def _enc_args():
    """Return encoder args based on GPU availability"""
    if USE_GPU:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]



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

GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEY","").split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS","").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS","").split(",")
GROQ_KEY = os.environ.get("GROQ_API_KEY","")

OUTPUT_DIR = Path("output"); TEMP_DIR = Path("temp")
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True); TEMP_DIR.mkdir(exist_ok=True)

IS_SPANISH = LANGUAGE.lower().strip() in ["spanish","es","espanol"]
USED_URLS = set()
AI_QUERIES = []



# ==========================================
# 3. GROQ QUERY ENGINE (Sentence-Matched JSON)
# ==========================================
def generate_queries_for_sentences(sentences):
    """
    Send ALL sentences to Groq, get back a JSON mapping each sentence to its ideal visual.
    This is the key to submagic-level sync: every visual matches what's being said RIGHT NOW.
    """
    if not GROQ_KEY or not sentences:
        return [random.choice(FALLBACK) for _ in sentences]
    
    n = len(sentences)
    print(f"  Groq: matching {n} sentences to visuals...")
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        
        # Format sentences as numbered list
        numbered = "\n".join([f"{i+1}. {s['text'][:100]}" for i, s in enumerate(sentences)])
        
        r = client.chat.completions.create(
            messages=[
                {"role": "system", "content": """You are a professional video editor. For each numbered sentence from a script, you must provide the PERFECT stock footage search query that visually represents what is being said.

RULES:
- Each query must be in ENGLISH (even if script is in another language)
- Each query is 3-6 words describing what a CAMERA would film
- The visual MUST match the sentence content:
  * "Technology is advancing rapidly" -> "futuristic circuit board closeup"
  * "The ocean is vast and mysterious" -> "deep ocean underwater darkness"
  * "Cities are growing faster" -> "aerial cityscape construction cranes"
  * "Ancient civilizations built pyramids" -> "egyptian pyramids aerial sunset"
- NEVER default to generic nature unless the sentence is ABOUT nature
- NO people/faces/bodies, NO religion, NO violence, NO NSFW
- Return ONLY the queries, one per line, numbered to match"""},
                {"role": "user", "content": f"Match each sentence to a video search query:\n\n{numbered[:7000]}"}
            ],
            model="openai/gpt-oss-120b",
            max_tokens=2500,
            temperature=0.5
        )
        
        result = r.choices[0].message.content
        queries = []
        
        for line in result.strip().split('\n'):
            cleaned = re.sub(r'^[\d\.\)\-\*]+\s*', '', line).strip().strip('"\'')
            if 3 < len(cleaned) < 60 and _safe(cleaned):
                queries.append(cleaned)
        
        # Show matching for debug
        for i in range(min(3, len(queries), len(sentences))):
            print(f"    [{i+1}] '{sentences[i]['text'][:35]}...' -> '{queries[i]}'")
        
        while len(queries) < n:
            queries.append(random.choice(FALLBACK))
        return queries[:n]
        
    except Exception as e:
        print(f"  Groq error: {e}")
        return [random.choice(FALLBACK) for _ in sentences]

FALLBACK = [
    "technology data center servers", "futuristic city aerial night",
    "abstract digital particles", "space nebula stars 4k",
    "ocean waves aerial cinematic", "mountain landscape dramatic",
    "circuit board macro electronics", "laboratory scientific research",
    "architecture modern building glass", "sunrise golden hour landscape",
    "underwater coral reef fish", "clouds timelapse dramatic sky",
    "desert sand dunes aerial", "forest aerial cinematic fog",
    "lightning storm dramatic clouds", "volcano lava flow night"
]

def _safe(q):
    bad = ['woman','women','girl','female','bikini','nude','naked','sexy',
           'jesus','christ','church','mosque','temple','bible','buddha',
           'gun','weapon','war','blood','violence','kill','alcohol',
           'drug','gambling','pork','lgbtq','person','people','crowd']
    return not any(t in q.lower() for t in bad)



# ==========================================
# 4. SUBTITLE PRESETS (Variety)
# ==========================================
SUBTITLE_STYLES = {
    "cinema": {"name":"Cinema","font":"Arial","size":54,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&H80000000",
        "border":3,"outline":0,"shadow":0,"margin":40,"spacing":0.5},
    "modern_bold": {"name":"Modern Bold","font":"Arial Black","size":60,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00111111","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":48,"spacing":1.2},
    "neon_yellow": {"name":"Neon Yellow","font":"Arial Black","size":62,"bold":-1,
        "primary":"&H0000FFFF","outline_c":"&H00000044","back":"&H00000000",
        "border":1,"outline":5,"shadow":3,"margin":50,"spacing":1.5},
    "soft_white": {"name":"Soft White","font":"Arial","size":52,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00333333","back":"&H00000000",
        "border":1,"outline":3,"shadow":4,"margin":42,"spacing":0.8},
    "electric_cyan": {"name":"Electric Cyan","font":"Arial Black","size":58,"bold":-1,
        "primary":"&H00FFFF00","outline_c":"&H00663300","back":"&H00000000",
        "border":1,"outline":4,"shadow":3,"margin":46,"spacing":1},
}

def create_subtitles(sentences, ass_path, word_data=None):
    """
    Word-level highlighted subtitles (like Submagic/Captions.ai).
    If word_data is provided, each word lights up as it's spoken.
    Falls back to sentence-level if no word data.
    """
    key = random.choice(list(SUBTITLE_STYLES.keys()))
    s = SUBTITLE_STYLES[key]
    print(f"  Subtitle: {s['name']} {'(word-highlight)' if word_data else '(sentence)'}")
    
    # Highlight color (the word currently being spoken)
    highlight = "&H0000FFFF"  # Yellow highlight
    if "cyan" in key: highlight = "&H0000FF00"  # Green for cyan style
    if "yellow" in key: highlight = "&H00FFFFFF"  # White for yellow style
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n")
        f.write("WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{s['font']},{s['size']},{s['primary']},&H00FFFFFF,"
                f"{s['outline_c']},{s['back']},{s['bold']},0,0,0,100,100,"
                f"{s['spacing']},0,{s['border']},{s['outline']},{s['shadow']},"
                f"2,50,50,{s['margin']},1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        if word_data:
            # Word-level highlighting: group words into display chunks (4-6 words)
            # Each chunk shows all words, but highlights one at a time via karaoke-style
            chunk_size = 5
            for i in range(0, len(word_data), chunk_size):
                chunk_words = word_data[i:i+chunk_size]
                if not chunk_words: continue
                
                chunk_start = chunk_words[0]['start']
                chunk_end = chunk_words[-1]['end']
                
                # For each word in chunk, create a dialogue line where THAT word is highlighted
                for w_idx, word in enumerate(chunk_words):
                    w_start = _fmt(word['start'])
                    w_end = _fmt(word['end'])
                    
                    # Build text with highlight on current word
                    parts = []
                    for j, cw in enumerate(chunk_words):
                        if j == w_idx:
                            # Highlighted word (different color + slightly bigger)
                            parts.append(f"{{\\c{highlight}\\fscx110\\fscy110}}{cw['text']}{{\\r}}")
                        else:
                            parts.append(cw['text'])
                    
                    line = ' '.join(parts)
                    # 2-line wrap if too long
                    if len(' '.join([w['text'] for w in chunk_words])) > 40:
                        mid = len(chunk_words) // 2
                        p1 = []
                        p2 = []
                        for j, cw in enumerate(chunk_words):
                            txt = f"{{\\c{highlight}\\fscx110\\fscy110}}{cw['text']}{{\\r}}" if j == w_idx else cw['text']
                            if j < mid: p1.append(txt)
                            else: p2.append(txt)
                        line = ' '.join(p1) + "\\N" + ' '.join(p2)
                    
                    f.write(f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n")
        else:
            # Sentence-level fallback
            for sent in sentences:
                t1 = _fmt(sent['start']); t2 = _fmt(sent['end'])
                txt = sent['text'].strip().rstrip('.,;:')
                if len(txt) > 42:
                    w = txt.split(); mid = len(w)//2
                    txt = ' '.join(w[:mid]) + "\\N" + ' '.join(w[mid:])
                f.write(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{txt}\n")

def _fmt(sec):
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60); cs=int((sec%1)*100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"



# ==========================================
# 5. AUDIO ENGINE (TTS + Resemble Enhance)
# ==========================================
def generate_audio(text, ref_audio, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_path = TEMP_DIR / "raw_tts.wav"
    
    print(f"  TTS: {'Spanish (V3)' if IS_SPANISH else 'English'} on {device}")
    try:
        if IS_SPANISH:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        
        sr = model.sr
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Group 2-3 sentences per chunk
        chunks, buf, blen = [], [], 0
        for s in sents:
            if blen + len(s) > 160 and buf:
                chunks.append(' '.join(buf)); buf, blen = [s], len(s)
            else: buf.append(s); blen += len(s)+1
        if buf: chunks.append(' '.join(buf))
        
        print(f"  {len(sents)} sentences -> {len(chunks)} chunks")
        wavs = []
        for i, c in enumerate(chunks):
            if i%5==0: update_status(18+int((i/len(chunks))*27), f"TTS {i+1}/{len(chunks)}")
            try:
                with torch.no_grad():
                    if IS_SPANISH:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio), language_id="es")
                    else:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio), exaggeration=0.6, cfg_weight=0.4)
                    wavs.append(w.cpu())
                if i%8==0: torch.cuda.empty_cache()
            except: continue
        
        if not wavs: return False
        full = wavs[0]
        for w in wavs[1:]:
            full = torch.cat([full, torch.zeros((full.shape[0], int(0.15*sr))), w], dim=1)
        full = torch.cat([full, torch.zeros((full.shape[0], int(1.5*sr)))], dim=1)
        torchaudio.save(str(raw_path), full, sr)
        print(f"  TTS: {full.shape[1]/sr:.1f}s at {sr}Hz")
        del model, wavs, full; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"  TTS error: {e}"); return False
    
    # --- RESEMBLE ENHANCE ---
    print("  Enhancing audio...")
    try:
        from resemble_enhance.enhancer.inference import enhance as re_enhance
        dwav, osr = torchaudio.load(str(raw_path))
        chunk_s = 25 * osr; parts = []; esr = 44100
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
        print(f"  Enhance failed: {e}, using raw audio")
        shutil.copy2(str(raw_path), str(out_path))
        return True



# ==========================================
# 6. VIDEO ENGINE (GPU-Accelerated)
# ==========================================
def search_and_download(query, idx, duration):
    """Search + download + encode with GPU"""
    urls = []
    page = random.randint(1,3)
    
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization":key},
                params={"query":query,"per_page":15,"page":page,"orientation":"landscape"}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('videos',[]):
                    files = v.get('video_files',[])
                    hd = [f for f in files if f.get('quality')=='hd' and f.get('width',0)>=1280]
                    if not hd: hd = [f for f in files if f.get('quality') in ['hd','large']]
                    if hd:
                        url = random.choice(hd)['link']
                        if url not in USED_URLS: urls.append(url)
        except: pass
    
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key":key,"q":query,"per_page":15,"page":page}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('hits',[]):
                    vd = v.get('videos',{})
                    url = vd.get('large',vd.get('medium',{})).get('url')
                    if url and url not in USED_URLS: urls.append(url)
        except: pass
    
    for url in urls[:3]:
        try:
            raw = TEMP_DIR / f"raw_{idx}.mp4"
            out = TEMP_DIR / f"clip_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(raw) < 5000: continue
            
            # Encode with GPU (h264_nvenc) or CPU fallback
            vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30"
            cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",str(raw),"-t",str(duration),
                   "-vf",vf] + _enc_args() + ["-an",str(out)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            
            try: os.remove(raw)
            except: pass
            if os.path.exists(out) and os.path.getsize(out) > 2000:
                USED_URLS.add(url); return str(out)
        except: continue
    return None

def process_clip(args):
    i, sent, total = args
    dur = max(3.5, sent['end'] - sent['start'])
    query = AI_QUERIES[i] if i < len(AI_QUERIES) else random.choice(FALLBACK)
    
    for _ in range(3):
        clip = search_and_download(query, i, dur)
        if clip: return (i, clip)
        query = random.choice(FALLBACK)
        time.sleep(0.3)
    return (i, None)



# ==========================================
# 7. RENDER ENGINE (GPU-Accelerated)
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_nosub, out_sub):
    n = len(sentences)
    print(f"\n  Rendering {n} clips (5 workers)...")
    
    clips = [None]*n
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(process_clip, (i,s,n)): i for i,s in enumerate(sentences)}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            try:
                idx, path = f.result()
                if path: clips[idx]=path; done+=1
                update_status(55+int((done/n)*22), f"Clips {done}/{n}")
            except: pass
    
    # Fill gaps
    valid = [i for i,c in enumerate(clips) if c and os.path.exists(c)]
    if not valid: return False
    for i in range(n):
        if clips[i] and os.path.exists(clips[i]): continue
        nearest = min(valid, key=lambda x:abs(x-i))
        dur = max(3.5, sentences[i]['end']-sentences[i]['start'])
        gap = TEMP_DIR / f"gap_{i}.mp4"
        subprocess.run(["ffmpeg","-y","-stream_loop","-1","-i",clips[nearest],
            "-t",str(dur)] + _enc_args() + ["-an",str(gap)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips[i] = str(gap) if os.path.exists(gap) else clips[nearest]
    
    # Concat (stream copy)
    print("  Concatenating...")
    with open("list.txt","w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists("visual.mp4"):
        subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt" + 
            " -c:v libx264 -preset ultrafast -crf 18 visual.mp4",
            shell=True, capture_output=True)
    if not os.path.exists("visual.mp4"): return False
    
    # V1: No subs (900p, GPU)
    update_status(80, "Rendering V1 (900p)...")
    enc = _enc_args()
    if logo_path and os.path.exists(logo_path):
        filt = "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=160:-1[l];[bg][l]overlay=20:20[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a"] + enc + ["-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    else:
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(audio_path),
            "-vf","scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2"
            ] + enc + ["-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if not os.path.exists(out_nosub): return False
    print(f"  V1: {os.path.getsize(out_nosub)/(1024**2):.0f}MB")
    
    # V2: With subs (1080p, GPU)
    update_status(88, "Rendering V2 (1080p + subs)...")
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];[wl]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    else:
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","1:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if os.path.exists(out_sub): print(f"  V2: {os.path.getsize(out_sub)/(1024**2):.0f}MB")
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
    data = {"progress":progress,"message":message,"status":status,
            "logs":"\n".join(LOG_BUF),"timestamp":time.time()}
    if file_url: data["file_io_url"] = file_url
    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    headers = {"Authorization":f"token {token}","Accept":"application/vnd.github.v3+json"}
    try:
        gr = requests.get(url, headers=headers)
        sha = gr.json().get("sha") if gr.status_code==200 else None
        payload = {"message":f"s{progress}","content":base64.b64encode(json.dumps(data).encode()).decode(),"branch":"main"}
        if sha: payload["sha"]=sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization":f"token {token}","Accept":"application/vnd.github.v3.raw"})
        if r.status_code==200:
            with open(local,"wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def upload_drive(fp):
    if not os.path.exists(fp): return None
    print(f"  Uploading {os.path.basename(fp)}...")
    cid,cs,rt,fid = (os.environ.get(k) for k in ["OAUTH_CLIENT_ID","OAUTH_CLIENT_SECRET","OAUTH_REFRESH_TOKEN","GOOGLE_DRIVE_FOLDER_ID"])
    if not all([cid,cs,rt]): return None
    try: tok=requests.post("https://oauth2.googleapis.com/token",data={"client_id":cid,"client_secret":cs,"refresh_token":rt,"grant_type":"refresh_token"}).json()['access_token']
    except: return None
    sz=os.path.getsize(fp); meta={"name":os.path.basename(fp),"mimeType":"video/mp4"}
    if fid: meta["parents"]=[fid]
    resp=requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json","X-Upload-Content-Type":"video/mp4","X-Upload-Content-Length":str(sz)},json=meta)
    if resp.status_code!=200: return None
    with open(fp,"rb") as f: ur=requests.put(resp.headers["Location"],headers={"Content-Length":str(sz)},data=f)
    if ur.status_code in [200,201]:
        fid2=ur.json().get('id')
        requests.post(f"https://www.googleapis.com/drive/v3/files/{fid2}/permissions",
            headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},json={'role':'reader','type':'anyone'})
        link=f"https://drive.google.com/file/d/{fid2}/view?usp=sharing"; print(f"  -> {link}"); return link
    return None

def generate_script(topic, mins):
    words=int(mins*180); lang="Write in Spanish." if IS_SPANISH else "Write in English."
    prompt=f"Write a documentary narration about '{topic}'. {words} words.\n{lang}\nRules: Only narration. No brackets. Islamic guidelines. Family-friendly."
    random.shuffle(GEMINI_KEYS)
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            r=genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt)
            return re.sub(r'\[.*?\]','',r.text.replace("*","").replace("#","").strip())
        except: continue
    return ""



# ==========================================
# 9. MAIN EXECUTION
# ==========================================
print(f"\n{'='*50}")
print(f"  VIDEO FACTORY V4")
print(f"  Lang: {'ES' if IS_SPANISH else 'EN'} | GPU: {USE_GPU}")
print(f"{'='*50}\n")

update_status(1, "Starting...")
voice = TEMP_DIR/"voice.mp3"; logo = TEMP_DIR/"logo.png"
if not download_asset(VOICE_PATH, voice):
    update_status(0,"Voice failed","failed"); exit(1)
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, logo)
    logo = str(logo) if os.path.exists(logo) else None
else: logo = None

# Script
update_status(5, "Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE=="topic" else SCRIPT_TEXT
if len(text)<50: update_status(0,"Script too short","failed"); exit(1)
print(f"  {len(text.split())} words")

# Audio
update_status(12, "Audio...")
audio = TEMP_DIR/"audio.wav"
if not generate_audio(text, voice, audio):
    update_status(0,"Audio failed","failed"); exit(1)

# Transcribe
update_status(48, "Transcribing...")
sentences = []
word_data = []  # Word-level timestamps for highlighting
if ASSEMBLY_KEY:
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        tx = aai.Transcriber().transcribe(str(audio))
        for s in tx.get_sentences():
            sentences.append({"text":s.text,"start":s.start/1000,"end":s.end/1000})
        if sentences: sentences[-1]['end']+=0.5
        
        # Get word-level timestamps for subtitle highlighting
        for word in tx.words:
            word_data.append({"text": word.text, "start": word.start/1000, "end": word.end/1000})
        print(f"  Got {len(word_data)} word timestamps for highlighting")
    except Exception as e: print(f"  Transcribe err: {e}")

if not sentences:
    import wave
    try:
        with wave.open(str(audio),'rb') as w: dur=w.getnframes()/float(w.getframerate())
    except: dur=len(text.split())/2.5
    wps=len(text.split())/dur if dur>0 else 2.5; t=0
    for i in range(0,len(text.split()),8):
        chunk=text.split()[i:i+8]; d=len(chunk)/wps
        sentences.append({"text":' '.join(chunk),"start":t,"end":t+d}); t+=d

# Queries (sentence-matched via Groq JSON approach)
update_status(50, "Matching visuals to sentences...")
AI_QUERIES = generate_queries_for_sentences(sentences)

# Subtitles (word-level highlighting if available)
update_status(52, "Subtitles...")
ass = TEMP_DIR/"subs.ass"
create_subtitles(sentences, ass, word_data=word_data if word_data else None)

# Render
update_status(54, "Processing video...")
o1 = OUTPUT_DIR/f"final_{JOB_ID}_NO_SUBS.mp4"
o2 = OUTPUT_DIR/f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o1, o2):
    update_status(93, "Uploading...")
    l1 = upload_drive(o1); l2 = upload_drive(o2)
    msg = "Done!\n"
    if l1: msg += f"No Subs: {l1}\n"
    if l2: msg += f"With Subs: {l2}\n"
    update_status(100, msg, "completed", l1 or l2)
    print(f"\n  {msg}")
else:
    update_status(0, "Render failed", "failed")

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4","list.txt"]:
    if os.path.exists(f): os.remove(f)
print("--- DONE ---")
