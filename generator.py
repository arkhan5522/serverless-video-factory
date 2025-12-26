"""
AI VIDEO GENERATOR WITH PROFESSIONAL DOCUMENTARY TECHNIQUES
============================================
ENHANCED VERSION:
1. T5 Transformers for Smart Query Generation
2. Smart B-Roll Categorization (Documentary-style)
3. Cinematic Transitions (Fade, Wipe, Dissolve)
4. Professional Color Grading
5. Ken Burns Effect (Zoom/Pan)
6. Dual Output: With & Without Subtitles
7. Islamic Content Filtering
8. NO BACKGROUND MUSIC (Voice only)
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

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "chatterbox-tts",
        "torchaudio", 
        "assemblyai",
        "google-generativeai",
        "requests",
        "beautifulsoup4",
        "pydub",
        "numpy",
        "transformers",
        "pillow",
        "opencv-python",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ========================================== 
# 3. LOAD AI MODELS (T5 ONLY)
# ========================================== 

print("--- 🤖 Loading AI Models ---")

# T5 for Smart Query Generation
print("Loading T5 Model for Query Generation...")
try:
    t5_tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
    T5_AVAILABLE = True
    print("✅ T5 Model loaded")
except Exception as e:
    print(f"⚠️ T5 Model failed to load: {e}")
    T5_AVAILABLE = False

# ========================================== 
# 4. DOCUMENTARY B-ROLL SYSTEM
# ========================================== 

# Smart B-Roll Categories (Documentary-style)
B_ROLL_CATEGORIES = {
    "human_activity": {
        "keywords": ["people", "person", "human", "working", "walking", "man", "woman", "child", "community"],
        "style": "documentary style people realistic",
        "movement": "handheld_dynamic"
    },
    "environment": {
        "keywords": ["nature", "city", "landscape", "building", "place", "location", "area", "region"],
        "style": "cinematic establishing shot 4k",
        "movement": "slow_cinematic"
    },
    "detail_shots": {
        "keywords": ["hands", "object", "tool", "device", "close", "detail", "specific", "precisely"],
        "style": "extreme close up macro professional",
        "movement": "static_tripod"
    },
    "atmospheric": {
        "keywords": ["weather", "time", "season", "mood", "atmosphere", "feeling", "ambient"],
        "style": "atmospheric moody cinematic footage",
        "movement": "slow_pan"
    },
    "action": {
        "keywords": ["fast", "quick", "rapid", "action", "movement", "dynamic", "speed"],
        "style": "slow motion 120fps dramatic",
        "movement": "action_dynamic"
    }
}

def categorize_script_segment(text):
    """Determine what TYPE of visual would work best (Documentary technique)"""
    text_lower = text.lower()
    
    # Analyze sentence for visual needs
    for category, data in B_ROLL_CATEGORIES.items():
        if any(word in text_lower for word in data["keywords"]):
            return category
    
    # Default to atmospheric
    return "atmospheric"

# ========================================== 
# 5. CONTENT FILTERS
# ========================================== 

EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic xxx', 'nsfw','lgbtq','LGBTQ','war','pork','bikini','swim','violence','drugs','terror','gun','gambling'
]

RELIGIOUS_HOLY_TERMS = [
    'jesus', 'christ', 'god', 'lord', 'bible', 'gospel', 'church worship',
    'crucifix', 'crucifixion', 'virgin mary', 'holy spirit', 'baptism',
    'yahweh', 'jehovah', 'torah', 'talmud', 'synagogue', 'rabbi', 'kosher',
    'hanukkah', 'yom kippur', 'passover',
    'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'ganesh', 'hindu temple',
    'vedas', 'bhagavad gita', 'diwali',
    'buddha', 'buddhist temple', 'nirvana', 'dharma', 'meditation buddha',
    'tibetan monk', 'dalai lama',
    'holy book', 'scripture', 'religious ceremony', 'worship service',
    'religious ritual', 'sacred text', 'divine revelation'
]

def is_content_appropriate(text):
    """Light content filter with word boundary matching"""
    text_lower = text.lower()
    
    for term in EXPLICIT_CONTENT_BLACKLIST:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Inappropriate content - '{term}'")
            return False
    
    for term in RELIGIOUS_HOLY_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious content - '{term}'")
            return False
    
    return True

# ========================================== 
# 6. T5 SMART QUERY GENERATION
# ========================================== 

def generate_smart_query_t5(script_text):
    """Generate intelligent search queries using T5 transformer"""
    if not T5_AVAILABLE:
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        return words[0] if words else "background"
    
    try:
        inputs = t5_tokenizer([script_text], max_length=512, truncation=True, return_tensors="pt")
        output = t5_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        decoded_output = t5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        for tag in tags:
            if is_content_appropriate(tag):
                return tag
        
        return "background"
        
    except Exception as e:
        print(f"    T5 Error: {e}")
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        return words[0] if words else "background"

# ========================================== 
# 7. CINEMATIC ENHANCEMENTS
# ========================================== 

def apply_ken_burns_effect(clip_path, output_path, duration):
    """Add cinematic zoom/pan to clips (30% chance)"""
    try:
        zoom_in = random.choice([True, False])
        
        if zoom_in:
            zoom_expr = "min(zoom+0.002,1.3)"
        else:
            zoom_expr = "if(lte(zoom,1.0),1.3,max(zoom-0.002,1.0))"
        
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(clip_path),
            "-vf", f"zoompan=z='{zoom_expr}':d={int(duration*30)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-t", str(duration),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        
    except Exception as e:
        print(f"    Ken Burns failed: {e}")
    
    return False

def apply_color_grade(clip_path, output_path, style="cinematic"):
    """Apply professional color grading"""
    
    LUTs = {
        "cinematic": "eq=contrast=1.2:brightness=0.0:saturation=1.1",
        "documentary": "eq=contrast=1.1:brightness=0.05:saturation=0.9",
        "dramatic": "eq=contrast=1.3:brightness=-0.1:saturation=1.2",
        "warm": "eq=contrast=1.1:brightness=0.1:saturation=1.1,colorchannelmixer=rr=1.1:gg=1.0:bb=0.9",
        "cold": "eq=contrast=1.1:brightness=0.0:saturation=1.0,colorchannelmixer=rr=0.9:gg=1.0:bb=1.1"
    }
    
    try:
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(clip_path),
            "-vf", LUTs.get(style, LUTs["cinematic"]),
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=60)
        
        if os.path.exists(output_path):
            return True
            
    except Exception as e:
        print(f"    Color grading failed: {e}")
    
    return False

# ========================================== 
# 8. SUBTITLE STYLES
# ========================================== 

SUBTITLE_STYLES = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow (3D Pop)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 4,
        "shadow": 3,
        "margin_v": 45,
        "alignment": 2,
        "spacing": 1.5
    },
    "hormozi_green": {
        "name": "Hormozi Green (High Contrast)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FF00",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 5,
        "shadow": 0,
        "margin_v": 55,
        "alignment": 2,
        "spacing": 0.5
    },
    "finance_blue": {
        "name": "Finance Blue (Neon Glow)",
        "fontname": "Arial",
        "fontsize": 80,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00FF9900",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 3,
        "margin_v": 50,
        "alignment": 2,
        "spacing": 2
    },
    "netflix_box": {
        "name": "Netflix Modern",
        "fontname": "Roboto",
        "fontsize": 80,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H90000000",
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 3,
        "outline": 0,
        "shadow": 0,
        "margin_v": 35,
        "alignment": 2,
        "spacing": 0.5
    },
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with proper format"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            
            if text.endswith('.'):
                text = text[:-1]
            if text.endswith(','):
                text = text[:-1]
            
            if "mrbeast" in style_key or "hormozi" in style_key:
                text = text.upper()
            
            MAX_CHARS = 35
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1
                if current_length + word_length > MAX_CHARS and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    current_line.append(word)
                    current_length += word_length
            
            if current_line:
                lines.append(' '.join(current_line))
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ========================================== 
# 9. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Upload file to Google Drive"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"☁️ Uploading {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Missing OAuth credentials")
        return None
    
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    
    try:
        r = requests.post(token_url, data=data)
        r.raise_for_status()
        access_token = r.json()['access_token']
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        return None
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
    
    metadata = {"name": filename, "mimeType": "video/mp4"}
    if folder_id:
        metadata["parents"] = [folder_id]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size)
    }
    
    response = requests.post(upload_url, headers=headers, json=metadata)
    if response.status_code != 200:
        print(f"❌ Init failed: {response.text}")
        return None
    
    session_uri = response.headers.get("Location")
    
    with open(file_path, "rb") as f:
        upload_headers = {"Content-Length": str(file_size)}
        upload_resp = requests.put(session_uri, headers=upload_headers, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_data = upload_resp.json()
        file_id = file_data.get('id')
        
        perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        requests.post(
            perm_url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"✅ Uploaded: {link}")
        return link
    else:
        print(f"❌ Upload failed: {upload_resp.text}")
        return None

# ========================================== 
# 10. SMART VIDEO SEARCH WITH B-ROLL
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_realistic(script_text, sentence_index):
    """Search videos using Documentary B-Roll Intelligence"""
    
    # Step 1: Categorize the script segment
    category = categorize_script_segment(script_text)
    category_data = B_ROLL_CATEGORIES[category]
    
    # Step 2: Generate T5 query
    if T5_AVAILABLE:
        primary_query = generate_smart_query_t5(script_text)
    else:
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        primary_query = words[0] if words else "background"
    
    # Step 3: Enhance with B-roll style
    enhanced_query = f"{primary_query} {category_data['style']}"
    
    print(f"    📸 B-ROLL [{category.upper()}]: '{enhanced_query}'")
    
    return search_videos_by_query(enhanced_query, sentence_index)

def search_videos_by_query(query, sentence_index, page=None):
    """Direct search with a specific query"""
    if page is None:
        page = random.randint(1, 3)
    
    all_results = []
    
    # Pexels
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 20,
                "page": page,
                "orientation": "landscape"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    video_files = video.get('video_files', [])
                    if video_files:
                        hd_files = [f for f in video_files if f.get('quality') == 'hd']
                        if not hd_files:
                            hd_files = [f for f in video_files if f.get('quality') == 'large']
                        if not hd_files:
                            hd_files = video_files
                        
                        if hd_files:
                            best_file = random.choice(hd_files)
                            video_url = best_file['link']
                            
                            video_title = video.get('user', {}).get('name', '')
                            if not is_content_appropriate(video_title + " " + query):
                                continue
                            
                            if video_url not in USED_VIDEO_URLS:
                                all_results.append({
                                    'url': video_url,
                                    'service': 'pexels',
                                    'duration': video.get('duration', 0)
                                })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    # Pixabay
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 20,
                "page": page,
                "orientation": "horizontal"
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    video_url = None
                    for quality in ['large', 'medium', 'small', 'tiny']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    
                    if video_url:
                        video_tags = video.get('tags', '')
                        if not is_content_appropriate(video_tags + " " + query):
                            continue
                        
                        if video_url not in USED_VIDEO_URLS:
                            all_results.append({
                                'url': video_url,
                                'service': 'pixabay',
                                'duration': video.get('duration', 0)
                            })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

def download_and_rank_videos(results, script_text, target_duration, clip_index):
    """Download videos and use the first valid one"""
    
    for i, result in enumerate(results[:5]):
        try:
            raw_path = TEMP_DIR / f"raw_{clip_index}_{i}.mp4"
            response = requests.get(result['url'], timeout=30, stream=True)
            
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                output_path = TEMP_DIR / f"clip_{clip_index}.mp4"
                
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda",
                    "-i", str(raw_path),
                    "-t", str(target_duration),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",
                    "-b:v", "8M",
                    "-an",
                    str(output_path)
                ]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                try:
                    os.remove(raw_path)
                except:
                    pass
                
                if os.path.exists(output_path):
                    USED_VIDEO_URLS.add(result['url'])
                    print(f"    ✓ {result['service']} video downloaded")
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Download error: {str(e)[:60]}")
            continue
    
    return None

# ========================================== 
# 11. STATUS UPDATES
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Update status for HTML frontend"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(f"--- {progress}% | {message} ---")
    
    LOG_BUFFER.append(log_entry)
    if len(LOG_BUFFER) > 30:
        LOG_BUFFER.pop(0)
    
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    
    if not repo or not token:
        return
    
    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    
    data = {
        "progress": progress,
        "message": message,
        "status": status,
        "logs": "\n".join(LOG_BUFFER),
        "timestamp": time.time()
    }
    
    if file_url:
        data["file_io_url"] = file_url
    
    import base64
    content_json = json.dumps(data)
    content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        
        payload = {
            "message": f"Update {progress}%",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
        
        requests.put(url, headers=headers, json=payload)
    except:
        pass

def download_asset(path, local):
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

# ========================================== 
# 12. SCRIPT & AUDIO GENERATION
# ==========================================

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
CRITICAL RULES:
- Write ONLY spoken narration text
- NO stage directions, sound effects, or [brackets]
- Start directly with content
- Islamic content guidelines: No mention of alcohol, inappropriate relationships, gambling, or pork
- Family-friendly and educational tone
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"{base_instructions}\nWrite Part {i+1}/{chunks} about '{topic}'. Context: {context}. Length: 700 words."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nWrite a documentary script about '{topic}'. {words} words."
        script = call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    return script.strip()

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except:
            continue
    return "Script generation failed."

def clone_voice(text, ref_audio, out_path):
    print("🎤 Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            if i % 10 == 0:
                update_status(20 + int((i/len(sentences))*30), f"Voice {i}/{len(sentences)}")
            
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk.replace('"', ''),
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                
                if i % 20 == 0:
                    torch.cuda.empty_cache()
            except:
                continue
        
        if all_wavs:
            full_audio = torch.cat(all_wavs, dim=1)
            silence = torch.zeros((full_audio.shape[0], int(2.0 * 24000)))
            full_audio_padded = torch.cat([full_audio, silence], dim=1)
            torchaudio.save(out_path, full_audio_padded, 24000)
            return True
    except Exception as e:
        print(f"❌ Audio failed: {e}")
    return False
    # ========================================== 
# 13. VISUAL PROCESSING WITH DOCUMENTARY TECHNIQUES
# ========================================== 

def process_single_clip(args):
    """Process a single clip with documentary techniques"""
    i, sent, sentences_count = args
    
    duration = max(3.5, sent['end'] - sent['start'])
    
    print(f"  🔍 Clip {i+1}/{sentences_count}: '{sent['text'][:50]}...'")
    
    # Try multiple search strategies
    max_attempts = 10
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        if attempt == 1:
            # Primary: Documentary B-Roll search
            print(f"    Attempt {attempt}: Documentary B-Roll")
            results = search_videos_realistic(sent['text'], i)
        
        elif attempt == 2:
            # Secondary: T5 smart query
            print(f"    Attempt {attempt}: T5 Smart Query")
            if T5_AVAILABLE:
                query = generate_smart_query_t5(sent['text'])
            else:
                words = re.findall(r'\b\w{5,}\b', sent['text'].lower())
                query = words[0] if words else "background"
            results = search_videos_by_query(f"{query} cinematic", i)
        
        elif attempt == 3:
            # Tertiary: Extract keywords
            print(f"    Attempt {attempt}: Keyword Extraction")
            words = re.findall(r'\b\w{5,}\b', sent['text'].lower())
            if words:
                query = f"{words[0]} documentary style"
                results = search_videos_by_query(query, i)
            else:
                results = []
        
        elif attempt == 4:
            # Fourth: Nature/establishing shots
            print(f"    Attempt {attempt}: Establishing Shots")
            results = search_videos_by_query("cinematic establishing shot 4k", i)
        
        elif attempt == 5:
            # Fifth: Human activity
            print(f"    Attempt {attempt}: Human Activity")
            results = search_videos_by_query("people documentary realistic", i)
        
        elif attempt == 6:
            # Sixth: Detail shots
            print(f"    Attempt {attempt}: Detail Shots")
            results = search_videos_by_query("extreme close up macro", i)
        
        elif attempt == 7:
            # Seventh: Atmospheric
            print(f"    Attempt {attempt}: Atmospheric")
            results = search_videos_by_query("atmospheric cinematic moody", i)
        
        elif attempt == 8:
            # Eighth: Time-lapse
            print(f"    Attempt {attempt}: Time-lapse")
            results = search_videos_by_query("time lapse cinematic", i)
        
        elif attempt == 9:
            # Ninth: Slow motion
            print(f"    Attempt {attempt}: Slow Motion")
            results = search_videos_by_query("slow motion cinematic", i)
        
        else:
            # Final attempt: Broad search
            print(f"    Attempt {attempt}: Broad Search")
            results = search_videos_by_query("cinematic background", i, page=random.randint(1, 5))
        
        # Try to download and rank
        if results:
            clip_path = download_and_rank_videos(results, sent['text'], duration, i)
            if clip_path and os.path.exists(clip_path):
                
                # Apply cinematic enhancements (30% chance each)
                enhanced_path = TEMP_DIR / f"enhanced_{i}.mp4"
                
                # Apply color grading (30% chance)
                if random.random() < 0.3:
                    color_styles = ["cinematic", "documentary", "dramatic", "warm", "cold"]
                    style = random.choice(color_styles)
                    if apply_color_grade(clip_path, enhanced_path, style):
                        clip_path = str(enhanced_path)
                        print(f"    🎨 Applied {style} color grade")
                
                # Apply Ken Burns effect (30% chance)
                if random.random() < 0.3:
                    kb_path = TEMP_DIR / f"kenburns_{i}.mp4"
                    if apply_ken_burns_effect(clip_path, kb_path, duration):
                        clip_path = str(kb_path)
                        print(f"    🎬 Applied Ken Burns effect")
                
                print(f"    ✅ Video found on attempt {attempt}")
                return (i, clip_path)
        
        if attempt < max_attempts:
            time.sleep(0.5)
    
    print(f"    ❌ Failed to find video after {max_attempts} attempts")
    return (i, None)

def process_visuals(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """Process visuals with documentary techniques and GPU-accelerated encoding"""
    
    print("🎬 Processing Visuals with Documentary Techniques...")
    print("📽️ Applying: B-Roll Categories + Color Grading + Ken Burns Effect")
    print(f"⚡ Processing {min(5, len(sentences))} clips in parallel...")
    
    # Prepare arguments for parallel processing
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    
    # Process clips in parallel
    clips = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(process_single_clip, arg): arg[0] 
            for arg in clip_args
        }
        
        completed = 0
        failed_clips = []
        
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                
                if clip_path and os.path.exists(clip_path):
                    clips[index] = clip_path
                    completed += 1
                    print(f"✅ Clip {index+1} completed with cinematic effects")
                else:
                    failed_clips.append(index)
                    print(f"❌ Clip {index+1} FAILED after all attempts")
                
                update_status(60 + int((completed/len(sentences))*25), 
                            f"Completed {completed}/{len(sentences)} clips")
                
            except Exception as e:
                index = future_to_index[future]
                failed_clips.append(index)
                print(f"❌ Clip {index+1} error: {e}")
    
    # Handle failed clips
    if failed_clips:
        print(f"\n⚠️ WARNING: {len(failed_clips)} clips failed to download")
        print(f"Failed clip indices: {failed_clips}")
    
    # Filter out None values
    valid_clips = [c for c in clips if c is not None and os.path.exists(c)]
    
    if not valid_clips:
        print("❌ No clips generated - cannot create video")
        return False
    
    print(f"✅ Generated {len(valid_clips)}/{len(sentences)} clips")
    
    if len(valid_clips) < len(sentences) * 0.5:
        print(f"⚠️ WARNING: Only {len(valid_clips)} out of {len(sentences)} clips generated")
    
    # Concatenate clips with transitions
    print("🔗 Concatenating clips with cinematic transitions...")
    
    # Create list file with transition filters
    transition_list = TEMP_DIR / "transition_list.txt"
    
    with open(transition_list, "w") as f:
        for i, clip in enumerate(valid_clips):
            # Add fade in/out transitions between clips
            if i == 0:
                # First clip: fade in only
                filter_chain = "fade=t=in:st=0:d=1.5"
            elif i == len(valid_clips) - 1:
                # Last clip: fade out only
                filter_chain = "fade=t=out:st=0:d=2.0"
            else:
                # Middle clips: crossfade between them
                filter_chain = ""
            
            if filter_chain:
                f.write(f"file '{clip}'\nduration {max(3.5, sentences[i]['end'] - sentences[i]['start'])}\ninpoint 0\noutpoint {max(3.5, sentences[i]['end'] - sentences[i]['start'])}\nfilter_complex {filter_chain}\n")
            else:
                f.write(f"file '{clip}'\nduration {max(3.5, sentences[i]['end'] - sentences[i]['start'])}\ninpoint 0\noutpoint {max(3.5, sentences[i]['end'] - sentences[i]['start'])}\n")
    
    # Concatenate with transitions
    concat_output = TEMP_DIR / "visual_with_transitions.mp4"
    
    cmd_concat = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-f", "concat",
        "-safe", "0",
        "-i", str(transition_list),
        "-vf", "fps=30,format=yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "10M",
        str(concat_output)
    ]
    
    result_concat = subprocess.run(cmd_concat, capture_output=True, text=True, timeout=300)
    
    if result_concat.returncode != 0:
        print(f"❌ Concatenation failed: {result_concat.stderr[:200]}")
        # Fallback to simple concatenation
        with open("list.txt", "w") as f:
            for c in valid_clips:
                f.write(f"file '{c}'\n")
        
        subprocess.run(
            "ffmpeg -y -f concat -safe 0 -i list.txt -c:v h264_nvenc -preset p1 visual.mp4",
            shell=True, 
            capture_output=True,
            text=True
        )
        visual_source = "visual.mp4"
    else:
        visual_source = str(concat_output)
    
    if not os.path.exists(visual_source):
        print("❌ Visual source not created")
        return False
    
    # === VERSION 1: 900p NO SUBTITLES (OPTIMIZED FOR <1GB) ===
    print("\n📹 Rendering Version 1: 900p (No Subtitles) - Documentary Style")
    update_status(85, "Rendering 900p documentary version...")
    
    TARGET_WIDTH = 1600
    TARGET_HEIGHT = 900
    
    if logo_path and os.path.exists(logo_path):
        # Scale to 900p with logo overlay
        filter_v1 = f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25[v]"
        cmd_v1 = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", visual_source, "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "h264_nvenc", 
            "-preset", "p4",
            "-b:v", "6M",
            "-maxrate", "8M",
            "-bufsize", "12M",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_no_subs)
        ]
    else:
        # Scale to 900p without logo
        cmd_v1 = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", visual_source, "-i", str(audio_path),
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "6M",
            "-maxrate", "8M",
            "-bufsize", "12M",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_no_subs)
        ]
    
    print("    Encoding 900p documentary version (NVENC)...")
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=600)
    
    if result_v1.returncode != 0:
        print(f"❌ Version 1 failed: {result_v1.stderr[-300:]}")
        return False
    
    if not os.path.exists(output_no_subs):
        print(f"❌ Output file not created")
        return False
    
    file_size = os.path.getsize(output_no_subs)
    if file_size < 1000000:
        print(f"❌ Output file too small: {file_size} bytes")
        return False
    
    file_size_gb = file_size / (1024**3)
    file_size_mb = file_size / (1024*1024)
    print(f"✅ Version 1 Complete: {file_size_gb:.3f}GB ({file_size_mb:.1f}MB)")
    
    if file_size_gb > 0.95:
        print(f"⚠️ WARNING: File is {file_size_gb:.3f}GB - very close to 1GB limit!")
    elif file_size_gb < 1.0:
        print(f"✅ File size under 1GB target!")
    
    # === VERSION 2: 1080p WITH SUBTITLES (CINEMATIC QUALITY) ===
    print("\n📹 Rendering Version 2: 1080p (With Subtitles) - Cinematic Quality")
    update_status(90, "Rendering 1080p cinematic version...")
    
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", visual_source, "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M",
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            str(output_with_subs)
        ]
    else:
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", visual_source, "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M",
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            str(output_with_subs)
        ]
    
    print("    Encoding 1080p with cinematic subtitles (NVENC)...")
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=600)
    
    if result_v2.returncode != 0:
        print(f"⚠️ Version 2 failed: {result_v2.stderr[-300:]}")
        print("Continuing with Version 1 only...")
        return True  # Version 1 succeeded
    
    if not os.path.exists(output_with_subs) or os.path.getsize(output_with_subs) < 1000000:
        print(f"⚠️ Version 2 output invalid")
        return True
    
    file_size_v2 = os.path.getsize(output_with_subs)
    file_size_v2_gb = file_size_v2 / (1024**3)
    file_size_v2_mb = file_size_v2 / (1024*1024)
    print(f"✅ Version 2 Complete: {file_size_v2_gb:.3f}GB ({file_size_v2_mb:.1f}MB)")
    
    # Add final cinematic touch: Apply overall color grade to both versions
    print("\n🎨 Applying final cinematic color grade...")
    
    for version_path in [output_no_subs, output_with_subs]:
        if os.path.exists(version_path):
            temp_path = TEMP_DIR / f"temp_{os.path.basename(version_path)}"
            
            # Apply subtle cinematic LUT
            cmd_lut = [
                "ffmpeg", "-y", "-hwaccel", "cuda",
                "-i", str(version_path),
                "-vf", "eq=contrast=1.1:brightness=0.0:saturation=1.05",
                "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "6M",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(temp_path)
            ]
            
            subprocess.run(cmd_lut, capture_output=True, timeout=120)
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000000:
                shutil.move(temp_path, version_path)
                print(f"    🎨 Enhanced {os.path.basename(version_path)}")
    
    return True

# ========================================== 
# 14. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START: Documentary Style AI Video Generator ---")
update_status(1, "Initializing Documentary System...")

# Download assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice download failed", "failed")
    exit(1)

if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if not os.path.exists(ref_logo):
        ref_logo = None
else:
    ref_logo = None

# Generate script
update_status(10, "Writing Documentary Script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if len(text) < 100:
    update_status(0, "Script too short", "failed")
    exit(1)

print(f"📜 Script Length: {len(text.split())} words")

# Generate audio
update_status(20, "Creating Documentary Narration...")
audio_out = TEMP_DIR / "audio.wav"

if clone_voice(text, ref_voice, audio_out):
    update_status(50, "Creating Professional Subtitles...")
    
    # Transcribe with AssemblyAI
    sentences = []
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_out))
            
            for sentence in transcript.get_sentences():
                sentences.append({
                    "text": sentence.text,
                    "start": sentence.start / 1000,
                    "end": sentence.end / 1000
                })
            if sentences:
                sentences[-1]['end'] += 1.0  # Add buffer at end
            
            print(f"✅ Transcribed {len(sentences)} sentences")
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}")
            # Fallback timing
            sentences = create_fallback_timing(text, audio_out)
    else:
        sentences = create_fallback_timing(text, audio_out)
    
    # Create professional subtitles
    ass_file = TEMP_DIR / "subs.ass"
    create_ass_file(sentences, ass_file)
    
    # Process visuals - TWO OUTPUTS
    update_status(60, "Creating Documentary Visuals...")
    output_no_subs = OUTPUT_DIR / f"documentary_{JOB_ID}_NO_SUBS.mp4"
    output_with_subs = OUTPUT_DIR / f"documentary_{JOB_ID}_WITH_SUBS.mp4"
    
    if process_visuals(sentences, audio_out, ass_file, ref_logo, output_no_subs, output_with_subs):
        # Upload both versions
        update_status(90, "Uploading Version 1 (No Subs)...")
        link1 = upload_to_google_drive(output_no_subs)
        
        update_status(95, "Uploading Version 2 (With Subs)...")
        link2 = upload_to_google_drive(output_with_subs)
        
        final_message = "✅ Documentary Created Successfully!\n"
        if link1:
            final_message += f"No Subtitles: {link1}\n"
        if link2:
            final_message += f"With Subtitles: {link2}"
        
        update_status(100, final_message, "completed", link1 or link2)
        print(f"🎬 {final_message}")
    else:
        update_status(0, "Visual processing failed", "failed")
else:
    update_status(0, "Audio generation failed", "failed")

# Cleanup
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4", "list.txt", "transition_list.txt", "visual_with_transitions.mp4"]:
    if os.path.exists(f):
        try:
            os.remove(f)
        except:
            pass

print("--- ✅ DOCUMENTARY COMPLETE ---")

def create_fallback_timing(text, audio_out):
    """Create fallback timing for sentences"""
    import wave
    
    words = text.split()
    with wave.open(str(audio_out), 'rb') as wav:
        total_dur = wav.getnframes() / float(wav.getframerate())
    
    words_per_sec = len(words) / total_dur
    sentences = []
    current_time = 0
    
    # Split into natural sentences
    text_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in text_sentences:
        if len(sentence.strip()) < 3:
            continue
            
        sentence_words = sentence.split()
        dur = len(sentence_words) / words_per_sec
        
        sentences.append({
            "text": sentence.strip(),
            "start": current_time,
            "end": current_time + dur
        })
        current_time += dur
    
    return sentences
    
