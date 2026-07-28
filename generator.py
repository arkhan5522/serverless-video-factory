"""
AI VIDEO GENERATOR V3 - CLEAN EDITION
=======================================
1. Groq API for query generation (GPT-OSS 120B)
2. Chatterbox Multilingual TTS (English/Spanish)
3. Movie-style subtitles (bottom, 2 lines, clean)
4. Fast concat (no transitions overhead)
5. Optimized for Kaggle P100 16GB GPU
"""

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

print("--- Installing Dependencies ---")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "groq", "chatterbox-tts", "assemblyai", "google-generativeai",
        "transformers", "sentencepiece", "requests", "pydub", "numpy", "pillow"
    ])
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
LANGUAGE = """{{LANGUAGE_PLACEHOLDER}}"""

# Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Detect language
IS_SPANISH = LANGUAGE.lower().strip() in ["spanish", "es", "espanol"]



# ==========================================
# 3. GROQ QUERY GENERATION
# ==========================================

FALLBACK_QUERIES = [
    "forest trees cinematic 4k", "mountain landscape nature 4k",
    "waterfall nature cinematic", "river flowing nature 4k",
    "misty forest morning 4k", "lake reflection nature 4k",
    "sunset mountain landscape", "clouds sky timelapse 4k",
    "snow mountain landscape", "northern lights aurora",
    "sand dunes desert landscape", "ocean waves aerial cinematic",
    "autumn forest golden leaves", "spring meadow flowers bloom",
    "deep space nebula stars", "coral reef underwater 4k"
]

def generate_queries_groq(script_text, num_queries):
    """Generate ENGLISH video queries using Groq API (gpt-oss-120b). Always English even for Spanish scripts."""
    if not GROQ_API_KEY:
        print("  No Groq key, using Flan-T5...")
        return _generate_queries_t5(script_text, num_queries)
    
    print(f"  Generating {num_queries} queries via Groq...")
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        all_queries = []
        words = script_text.split()
        
        # Split into 2 batches
        mid = len(words) // 2
        segments = [' '.join(words[:mid])[:2000], ' '.join(words[mid:])[:2000]]
        
        for seg_idx, segment in enumerate(segments):
            needed = min(num_queries - len(all_queries), (num_queries + 1) // 2 + 5)
            if needed <= 0:
                break
            
            prompt = f"""Generate exactly {needed} English video stock footage search queries (3-5 words each).

RULES:
- ALL queries must be in ENGLISH regardless of script language
- Each query 3-5 words, visually descriptive
- NO people/faces/bodies, NO religion, NO violence, NO NSFW, NO alcohol/drugs/pork
- Focus: nature, landscapes, technology, architecture, space, underwater, aerial, animals, weather, textures
- Specific to the content, not generic
- One per line, no numbering

SCRIPT:
{segment}

Queries:"""
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                max_tokens=1000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                    all_queries.append(cleaned)
                if len(all_queries) >= num_queries:
                    break
            
            time.sleep(1)
        
        # Fill remaining if needed
        while len(all_queries) < num_queries:
            all_queries.append(random.choice(FALLBACK_QUERIES))
        
        print(f"  Groq generated {len(all_queries)} queries")
        return all_queries[:num_queries]
        
    except Exception as e:
        print(f"  Groq error: {e}, falling back to Flan-T5...")
        return _generate_queries_t5(script_text, num_queries)

def _generate_queries_t5(script_text, num_queries):
    """Fallback: Flan-T5 local query generation"""
    print(f"  Loading Flan-T5 ({num_queries} queries)...")
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        
        all_queries = []
        context = ' '.join(script_text.split()[:300]) if script_text else "nature documentary"
        
        prompt = f"Generate {num_queries} short English video search queries for stock footage about: {context[:500]}. No people, no religion. Focus: nature, technology, space, animals. Queries:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, num_beams=4, do_sample=True, temperature=0.8)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        for line in result.replace(',', '\n').split('\n'):
            cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line.strip()).strip()
            if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                all_queries.append(cleaned)
        
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        while len(all_queries) < num_queries:
            all_queries.append(random.choice(FALLBACK_QUERIES))
        return all_queries[:num_queries]
    except Exception as e:
        print(f"  T5 error: {e}")
        return [random.choice(FALLBACK_QUERIES) for _ in range(num_queries)]

def _is_query_safe(query):
    blocked = ['woman','women','girl','female','bikini','nude','naked','sexy','nsfw',
               'jesus','christ','church','mosque','temple','bible','quran','buddha',
               'gun','weapon','war','blood','violence','kill','alcohol','beer','wine',
               'drug','gambling','pork','lgbtq','person','people','crowd','human','man face']
    q = query.lower()
    return not any(t in q for t in blocked)



# ==========================================
# 4. CONTENT FILTER
# ==========================================

def is_content_appropriate(text):
    blocked = ['nude','naked','porn','nsfw','lgbtq','war','pork','bikini','violence','drugs','terror','gun','gambling',
               'jesus','christ','bible','church','crucifix','buddha','hindu','shiva','vishnu']
    t = text.lower()
    return not any(re.search(r'\b' + re.escape(b) + r'\b', t) for b in blocked)

# ==========================================
# 5. MOVIE-STYLE SUBTITLES
# ==========================================

def create_ass_file(sentences, ass_file, res_x=1920, res_y=1080):
    """
    Movie-style subtitles: clean, bottom of screen, max 2 lines.
    White text, thin dark outline, semi-transparent shadow.
    Compact and professional - like Netflix/cinema captions.
    """
    print(f"  Creating movie-style subtitles...")
    
    # Max chars per line - keeps text compact at bottom
    max_chars = 42
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write(f"PlayResX: {res_x}\n")
        f.write(f"PlayResY: {res_y}\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        # Clean movie style: white text, dark outline, bottom-center
        f.write("Style: Default,Arial,52,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
                "-1,0,0,0,100,100,0.5,0,1,3,1,2,40,40,35,1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = _format_ass_time(s['start'])
            end_time = _format_ass_time(s['end'])
            
            text = s['text'].strip()
            text = text.replace('\\', '').replace('\n', ' ')
            # Remove trailing punctuation for cleaner look
            text = text.rstrip('.,;:')
            
            # Word wrap to max 2 lines
            words = text.split()
            if len(' '.join(words)) <= max_chars:
                formatted = ' '.join(words)
            else:
                # Split into 2 lines at midpoint
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                # Rebalance if one line is much longer
                if len(line1) > max_chars:
                    mid = mid - 1
                    line1 = ' '.join(words[:mid])
                    line2 = ' '.join(words[mid:])
                formatted = f"{line1}\\N{line2}"
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted}\n")

def _format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"



# ==========================================
# 6. GOOGLE DRIVE UPLOAD
# ==========================================

def upload_to_google_drive(file_path):
    if not os.path.exists(file_path): return None
    print(f"  Uploading {os.path.basename(file_path)}...")
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not all([client_id, client_secret, refresh_token]): return None
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id, "client_secret": client_secret,
            "refresh_token": refresh_token, "grant_type": "refresh_token"
        })
        access_token = r.json()['access_token']
    except: return None
    
    file_size = os.path.getsize(file_path)
    metadata = {"name": os.path.basename(file_path), "mimeType": "video/mp4"}
    if folder_id: metadata["parents"] = [folder_id]
    
    resp = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json",
                 "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(file_size)},
        json=metadata)
    if resp.status_code != 200: return None
    
    with open(file_path, "rb") as f:
        upload_resp = requests.put(resp.headers["Location"], headers={"Content-Length": str(file_size)}, data=f)
    
    if upload_resp.status_code in [200, 201]:
        fid = upload_resp.json().get('id')
        requests.post(f"https://www.googleapis.com/drive/v3/files/{fid}/permissions",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'})
        link = f"https://drive.google.com/file/d/{fid}/view?usp=sharing"
        print(f"  Uploaded: {link}")
        return link
    return None



# ==========================================
# 7. VIDEO SEARCH & DOWNLOAD
# ==========================================

USED_URLS = set()
AI_QUERIES = []

def _search_videos(query, idx):
    page = random.randint(1, 3)
    results = []
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 15, "page": page, "orientation": "landscape"}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('videos', []):
                    files = v.get('video_files', [])
                    hd = [f for f in files if f.get('quality') == 'hd'] or files
                    if hd:
                        url = random.choice(hd)['link']
                        if url not in USED_URLS and is_content_appropriate(query):
                            results.append(url)
        except: pass
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key": key, "q": query, "per_page": 15, "page": page}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('hits', []):
                    vids = v.get('videos', {})
                    url = vids.get('large', vids.get('medium', {})).get('url')
                    if url and url not in USED_URLS and is_content_appropriate(query):
                        results.append(url)
        except: pass
    return results

def _download_clip(url, duration, idx):
    raw = TEMP_DIR / f"raw_{idx}.mp4"
    out = TEMP_DIR / f"clip_{idx}.mp4"
    try:
        r = requests.get(url, timeout=30, stream=True)
        with open(raw, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)
        if os.path.getsize(raw) < 1000: return None
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: os.remove(raw)
        except: pass
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            USED_URLS.add(url)
            return str(out)
    except: pass
    return None

def process_clip(args):
    i, sent, total = args
    duration = max(3.5, sent['end'] - sent['start'])
    query = AI_QUERIES[i] if i < len(AI_QUERIES) else random.choice(FALLBACK_QUERIES)
    print(f"  Clip {i+1}/{total}: '{query}'")
    for _ in range(4):
        results = _search_videos(query, i)
        if results:
            clip = _download_clip(results[0], duration, i)
            if clip: return (i, clip)
        query = random.choice(FALLBACK_QUERIES)
        time.sleep(0.3)
    return (i, None)



# ==========================================
# 8. STATUS & ASSETS
# ==========================================

LOG_BUF = []
def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")
    LOG_BUF.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(LOG_BUF) > 30: LOG_BUF.pop(0)
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return
    import base64
    data = {"progress": progress, "message": message, "status": status,
            "logs": "\n".join(LOG_BUF), "timestamp": time.time()}
    if file_url: data["file_io_url"] = file_url
    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        get_r = requests.get(url, headers=headers)
        sha = get_r.json().get("sha") if get_r.status_code == 200 else None
        payload = {"message": f"Update {progress}%", "content": base64.b64encode(json.dumps(data).encode()).decode(), "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"})
        if r.status_code == 200:
            with open(local, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 9. SCRIPT GENERATION
# ==========================================

def generate_script(topic, minutes):
    words = int(minutes * 180)
    lang_instruction = "Write in Spanish." if IS_SPANISH else "Write in English."
    prompt = f"""Write a documentary narration script about '{topic}'. {words} words.
{lang_instruction}
Rules: Only spoken text, no brackets, no stage directions.
Islamic guidelines: no alcohol, inappropriate content, gambling, pork.
Family-friendly and educational."""
    
    random.shuffle(GEMINI_KEYS)
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            result = model.generate_content(prompt).text.replace("*","").replace("#","").strip()
            return re.sub(r'\[.*?\]', '', result)
        except: continue
    return "Script generation failed."



# ==========================================
# 10. AUDIO GENERATION (Language-Aware)
# ==========================================

def generate_audio(text, ref_audio, out_path):
    """Generate voice clone - uses multilingual model for Spanish, English model for English."""
    print(f"--- AUDIO: {'Spanish' if IS_SPANISH else 'English'} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if IS_SPANISH:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            lang_id = "es"
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
            lang_id = None
        
        sr = model.sr
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Group into chunks (2-3 sentences, ~150 chars)
        chunks = []
        cur, cur_len = [], 0
        for s in sentences:
            if cur_len + len(s) > 150 and cur:
                chunks.append(' '.join(cur))
                cur, cur_len = [s], len(s)
            else:
                cur.append(s)
                cur_len += len(s) + 1
        if cur: chunks.append(' '.join(cur))
        
        print(f"  {len(sentences)} sentences -> {len(chunks)} chunks")
        all_wavs = []
        
        for i, chunk_text in enumerate(chunks):
            if i % 5 == 0:
                update_status(20 + int((i/len(chunks))*25), f"Voice {i}/{len(chunks)}")
            try:
                with torch.no_grad():
                    if IS_SPANISH:
                        wav = model.generate(chunk_text.replace('"',''),
                            audio_prompt_path=str(ref_audio), language_id=lang_id)
                    else:
                        wav = model.generate(chunk_text.replace('"',''),
                            audio_prompt_path=str(ref_audio), exaggeration=0.6, cfg_weight=0.4)
                    all_wavs.append(wav.cpu())
                if i % 10 == 0: torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Chunk {i} failed: {str(e)[:40]}")
                continue
        
        if not all_wavs:
            print("  TTS failed completely")
            return False
        
        # Simple concat with micro-pauses
        full = all_wavs[0]
        for w in all_wavs[1:]:
            pause = torch.zeros((full.shape[0], int(0.15 * sr)))
            full = torch.cat([full, pause, w], dim=1)
        
        # Add ending silence
        full = torch.cat([full, torch.zeros((full.shape[0], int(2.0 * sr)))], dim=1)
        torchaudio.save(str(out_path), full, sr)
        print(f"  Audio saved: {sr}Hz, {full.shape[1]/sr:.1f}s")
        
        del model, all_wavs, full
        torch.cuda.empty_cache()
        gc.collect()
        return True
        
    except Exception as e:
        print(f"  Audio error: {e}")
        return False



# ==========================================
# 11. VISUAL PROCESSING
# ==========================================

def process_visuals(sentences, audio_path, ass_file, logo_path, out_no_subs, out_with_subs):
    print(f"\n  Processing {len(sentences)} clips...")
    
    clips = [None] * len(sentences)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(process_clip, (i, s, len(sentences))): i for i, s in enumerate(sentences)}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                idx, path = f.result()
                if path: clips[idx] = path; done += 1
                update_status(55 + int((done/len(sentences))*25), f"Clips: {done}/{len(sentences)}")
            except: pass
    
    # Fill gaps with nearest clip
    valid = [i for i, c in enumerate(clips) if c and os.path.exists(c)]
    if not valid: return False
    for i in range(len(clips)):
        if clips[i] and os.path.exists(clips[i]): continue
        nearest = min(valid, key=lambda x: abs(x-i))
        clips[i] = clips[nearest]
    
    # Fast concat (no transitions, no re-encoding)
    print(f"  Concatenating {len(clips)} clips...")
    with open("list.txt", "w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, text=True, timeout=60)
    
    if not os.path.exists("visual.mp4"):
        subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c:v libx264 -preset ultrafast -crf 18 visual.mp4",
            shell=True, capture_output=True, text=True)
    if not os.path.exists("visual.mp4"): return False
    
    # Render V1: No subtitles (900p)
    update_status(82, "Rendering 900p...")
    if logo_path and os.path.exists(logo_path):
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex","[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=180:-1[l];[bg][l]overlay=25:25[v]",
            "-map","[v]","-map","2:a","-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","128k","-shortest",str(out_no_subs)]
    else:
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(audio_path),
            "-vf","scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","128k","-shortest",str(out_no_subs)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if not os.path.exists(out_no_subs): return False
    print(f"  V1: {os.path.getsize(out_no_subs)/(1024*1024):.1f}MB")
    
    # Render V2: With subtitles (1080p)
    update_status(88, "Rendering 1080p with subtitles...")
    ass_esc = str(ass_file).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=200:-1[l];[bg][l]overlay=25:25[wl];[wl]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","192k","-shortest",str(out_with_subs)]
    else:
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-i","visual.mp4","-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","1:a",
            "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","192k","-shortest",str(out_with_subs)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if not os.path.exists(out_with_subs):
        print("  V2 failed, V1 ok")
        return True
    print(f"  V2: {os.path.getsize(out_with_subs)/(1024*1024):.1f}MB")
    return True



# ==========================================
# 12. MAIN EXECUTION
# ==========================================

print("\n" + "="*50)
print("  VIDEO FACTORY V3")
print(f"  Language: {'Spanish' if IS_SPANISH else 'English'}")
print(f"  Queries: Groq (GPT-OSS 120B)")
print("="*50)

update_status(1, "Initializing...")

# Assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"
if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice download failed", "failed"); exit(1)
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if not os.path.exists(ref_logo): ref_logo = None
else: ref_logo = None

# Script
update_status(5, "Generating script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE == "topic" else SCRIPT_TEXT
if len(text) < 100:
    update_status(0, "Script too short", "failed"); exit(1)
print(f"  Script: {len(text.split())} words")

# Queries (always English)
update_status(10, "Generating queries via Groq...")
num_clips = max(10, int(DURATION_MINS * 8))
AI_QUERIES = generate_queries_groq(text, num_clips)

# Audio
update_status(15, "Generating audio...")
audio_out = TEMP_DIR / "audio.wav"
if not generate_audio(text, ref_voice, audio_out):
    update_status(0, "Audio failed", "failed"); exit(1)

# Transcribe
update_status(50, "Transcribing...")
sentences = []
if ASSEMBLY_KEY:
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        transcript = aai.Transcriber().transcribe(str(audio_out))
        for s in transcript.get_sentences():
            sentences.append({"text": s.text, "start": s.start/1000, "end": s.end/1000})
        if sentences: sentences[-1]['end'] += 1.0
    except Exception as e:
        print(f"  Transcription error: {e}")

if not sentences:
    words = text.split()
    import wave
    try:
        with wave.open(str(audio_out), 'rb') as w:
            total_dur = w.getnframes() / float(w.getframerate())
    except: total_dur = len(words) / 2.5
    wps = len(words) / total_dur if total_dur > 0 else 2.5
    t = 0
    for i in range(0, len(words), 10):
        chunk = words[i:i+10]
        d = len(chunk) / wps
        sentences.append({"text": ' '.join(chunk), "start": t, "end": t+d})
        t += d

# Fill queries to match sentences
while len(AI_QUERIES) < len(sentences):
    AI_QUERIES.append(random.choice(FALLBACK_QUERIES))

# Subtitles
ass_file = TEMP_DIR / "subs.ass"
create_ass_file(sentences, ass_file)

# Process
update_status(55, "Processing visuals...")
out1 = OUTPUT_DIR / f"final_{JOB_ID}_NO_SUBS.mp4"
out2 = OUTPUT_DIR / f"final_{JOB_ID}_WITH_SUBS.mp4"

if process_visuals(sentences, audio_out, ass_file, ref_logo, out1, out2):
    update_status(92, "Uploading...")
    link1 = upload_to_google_drive(out1)
    link2 = upload_to_google_drive(out2)
    
    msg = "Complete!\n"
    if link1: msg += f"No Subs: {link1}\n"
    if link2: msg += f"With Subs: {link2}\n"
    update_status(100, msg, "completed", link1 or link2)
    print(f"\n{'='*50}\n  {msg}{'='*50}")
else:
    update_status(0, "Processing failed", "failed")

# Cleanup
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4", "list.txt"]:
    if os.path.exists(f): os.remove(f)
print("\n--- COMPLETE ---")
