"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
ENHANCED VERSION:
1. T5 Transformers for Smart Query Generation
2. CLIP Model for Exact Visual Matching
3. Dual Output: With & Without Subtitles
4. Islamic Content Filtering
5. Subtitle Design Implementation (ASS format)
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
# 4. CONTENT FILTERS (Separate categories)
# ========================================== 

# Explicit inappropriate content filter
EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic xxx', 'nsfw'
]

# Religious/Holy terms filter (to avoid mixing religious content with random videos)
RELIGIOUS_HOLY_TERMS = [
    # Christian terms
    'jesus', 'christ', 'god', 'lord', 'bible', 'gospel', 'church worship',
    'crucifix', 'crucifixion', 'virgin mary', 'holy spirit', 'baptism',
    
    # Jewish terms
    'yahweh', 'jehovah', 'torah', 'talmud', 'synagogue', 'rabbi', 'kosher',
    'hanukkah', 'yom kippur', 'passover',
    
    # Hindu terms
    'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'ganesh', 'hindu temple',
    'vedas', 'bhagavad gita', 'diwali',
    
    # Buddhist terms
    'buddha', 'buddhist temple', 'nirvana', 'dharma', 'meditation buddha',
    'tibetan monk', 'dalai lama',
    
    # General religious
    'holy book', 'scripture', 'religious ceremony', 'worship service',
    'religious ritual', 'sacred text', 'divine revelation'
]

def is_content_appropriate(text):
    """
    Light content filter with two separate checks:
    1. Block explicit inappropriate content
    2. Block religious/holy terms to avoid conflicts
    
    Uses word boundary matching to avoid false positives
    (e.g., "drama" won't be blocked when "rama" is in the list)
    """
    text_lower = text.lower()
    
    # Check 1: Explicit content (word boundary matching)
    for term in EXPLICIT_CONTENT_BLACKLIST:
        # Use word boundary regex to match whole words only
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Inappropriate content - '{term}'")
            return False
    
    # Check 2: Religious/Holy terms (word boundary matching)
    for term in RELIGIOUS_HOLY_TERMS:
        # Use word boundary regex to match whole words only
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious content - '{term}' (avoiding religious conflicts)")
            return False
    
    return True

# ========================================== 
# 6. T5 SMART QUERY GENERATION
# ========================================== 

def generate_smart_query_t5(script_text):
    """Generate intelligent search queries using T5 transformer"""
    if not T5_AVAILABLE:
        # Fallback to keyword extraction
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        return words[0] if words else "background"
    
    try:
        # Prepare input
        inputs = t5_tokenizer([script_text], max_length=512, truncation=True, return_tensors="pt")
        
        # Generate tags
        output = t5_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode
        decoded_output = t5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        # Return first appropriate tag
        for tag in tags:
            if is_content_appropriate(tag):
                return tag
        
        return "background"
        
    except Exception as e:
        print(f"    T5 Error: {e}")
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        return words[0] if words else "background"


# ========================================== 
# 7. SUBTITLE STYLES
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
# 8. GOOGLE DRIVE UPLOAD
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
    
    # Get access token
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
    
    # Upload
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
        
        # Make public
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
# 10. VIDEO SEARCH WITH T5
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_smart(script_text, sentence_index):
    """Search videos using Flan-T5 query generation only"""
    
    # Use only T5 for query generation
    if T5_AVAILABLE:
        primary_query = generate_smart_query_t5(script_text)
        print(f"    🧠 Flan-T5 Query: '{primary_query}'")
    else:
        # Fallback: extract keywords
        words = re.findall(r'\b\w{5,}\b', script_text.lower())
        primary_query = words[0] if words else "background"
        primary_query += " 4k cinematic"
        print(f"    📝 Fallback Query: '{primary_query}'")
    
    # Use the generic search function
    return search_videos_by_query(primary_query, sentence_index)

def download_and_rank_videos(results, script_text, target_duration, clip_index):
    """Download videos and use the first valid one"""
    
    for i, result in enumerate(results[:5]):  # Try top 5 results
        try:
            raw_path = TEMP_DIR / f"raw_{clip_index}_{i}.mp4"
            response = requests.get(result['url'], timeout=30, stream=True)
            
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                # Process the video
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
                
                # Cleanup raw file
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
# 13. VISUAL PROCESSING WITH PARALLEL PROCESSING
# ========================================== 

def process_single_clip(args):
    """Process a single clip - used for parallel processing"""
    i, sent, sentences_count = args
    
    duration = max(3.5, sent['end'] - sent['start'])
    
    print(f"  🔍 Clip {i+1}/{sentences_count}: '{sent['text'][:50]}...'")
    
    # Try multiple search strategies until video found
    max_attempts = 10  # Increased attempts
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        if attempt == 1:
            # Primary: T5 generated query
            print(f"    Attempt {attempt}: T5 Smart Query")
            results = search_videos_smart(sent['text'], i)
        
        elif attempt == 2:
            # Secondary: Extract main keywords from sentence
            print(f"    Attempt {attempt}: Keyword Extraction")
            words = re.findall(r'\b\w{5,}\b', sent['text'].lower())
            if words:
                query = f"{words[0]} cinematic 4k"
                results = search_videos_by_query(query, i)
            else:
                results = []
        
        elif attempt == 3:
            # Tertiary: Generic visual terms
            print(f"    Attempt {attempt}: Visual Terms")
            visual_terms = ['cinematic', 'motion', 'animation', 'stock footage', 
                           'background video', 'b-roll', 'visual effects']
            query = f"{random.choice(visual_terms)} hd"
            results = search_videos_by_query(query, i)
        
        elif attempt == 4:
            # Fourth: Nature/landscape
            print(f"    Attempt {attempt}: Nature/Landscape")
            nature_terms = ['nature', 'landscape', 'scenery', 'outdoors',
                          'mountain', 'ocean', 'forest', 'sky']
            query = f"{random.choice(nature_terms)} cinematic"
            results = search_videos_by_query(query, i)
        
        elif attempt == 5:
            # Fifth: Abstract/patterns
            print(f"    Attempt {attempt}: Abstract Patterns")
            abstract_terms = ['abstract', 'pattern', 'geometric', 'light',
                            'particles', 'flow', 'motion graphics']
            query = f"{random.choice(abstract_terms)} 4k"
            results = search_videos_by_query(query, i)
        
        elif attempt == 6:
            # Sixth: City/urban
            print(f"    Attempt {attempt}: City/Urban")
            urban_terms = ['city', 'urban', 'architecture', 'building',
                          'skyline', 'street', 'traffic', 'lights']
            query = f"{random.choice(urban_terms)} time lapse"
            results = search_videos_by_query(query, i)
        
        elif attempt == 7:
            # Seventh: Technology
            print(f"    Attempt {attempt}: Technology")
            tech_terms = ['technology', 'digital', 'futuristic', 'innovation',
                         'data', 'network', 'circuit', 'code']
            query = f"{random.choice(tech_terms)} background"
            results = search_videos_by_query(query, i)
        
        elif attempt == 8:
            # Eighth: Time-lapse/motion
            print(f"    Attempt {attempt}: Time-lapse")
            motion_terms = ['time lapse', 'slow motion', 'hyperlapse',
                          'drone footage', 'aerial view', 'moving']
            query = f"{random.choice(motion_terms)}"
            results = search_videos_by_query(query, i)
        
        elif attempt == 9:
            # Ninth: Generic safe terms
            print(f"    Attempt {attempt}: Generic Safe Terms")
            safe_terms = ['background', 'video', 'footage', 'stock',
                         'cinematic', 'shot', 'scene', 'view']
            query = f"{random.choice(safe_terms)} hd"
            results = search_videos_by_query(query, i)
        
        else:
            # Final attempt: Completely random page from broad search
            print(f"    Attempt {attempt}: Random Broad Search")
            broad_terms = ['background', 'cinematic', 'nature', 'city', 
                         'technology', 'abstract', 'motion', 'light']
            query = random.choice(broad_terms)
            results = search_videos_by_query(query, i, page=random.randint(1, 10))
        
        # Try to download and rank
        if results:
            clip_path = download_and_rank_videos(results, sent['text'], duration, i)
            if clip_path and os.path.exists(clip_path):
                print(f"    ✅ Video found on attempt {attempt}")
                return (i, clip_path)
        
        # Wait a bit before next attempt to avoid rate limits
        if attempt < max_attempts:
            time.sleep(0.5)
    
    # If we reach here after all attempts, something is very wrong
    print(f"    ❌ Failed to find video after {max_attempts} attempts")
    print(f"    ⚠️ This should not happen - check API keys and internet connection")
    
    # Return None to indicate failure
    return (i, None)

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
                "per_page": 20,  # Increased from 15
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
                            hd_files = video_files  # Accept any quality
                        
                        if hd_files:
                            best_file = random.choice(hd_files)
                            video_url = best_file['link']
                            
                            # Content check
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
                "per_page": 20,  # Increased from 15
                "page": page,
                "orientation": "horizontal"
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    # Try all quality levels
                    video_url = None
                    for quality in ['large', 'medium', 'small', 'tiny']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    
                    if video_url:
                        # Content check
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

def process_visuals(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """Optimized for Kaggle P100 GPU - FIXED NVENC Version"""
    
    print("🎬 Processing Visuals with T5 ONLY + PARALLEL PROCESSING...")
    print(f"⚡ Processing up to 20 clips in parallel for maximum speed!")
    print("🎥 NO CATEGORY LOCK - Using pure T5 queries only!")
    
    # Prepare arguments for parallel processing
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    
    # Process clips in parallel (UP TO 20 at a time)
    clips = [None] * len(sentences)
    
    # Determine optimal worker count
    optimal_workers = min(20, len(sentences), 20)
    
    print(f"🚀 Using {optimal_workers} parallel workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        # Submit all tasks
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
                    print(f"✅ Clip {index+1} completed successfully ({completed}/{len(sentences)})")
                else:
                    failed_clips.append(index)
                    print(f"❌ Clip {index+1} FAILED after all attempts")
                
                update_status(60 + int((completed/len(sentences))*25), f"Completed {completed}/{len(sentences)} clips")
                
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
    
    # Concatenate
    print("🔗 Concatenating clips...")
    with open("list.txt", "w") as f:
        for c in valid_clips:
            if os.path.exists(c):
                f.write(f"file '{c}'\n")
    
    # Use copy for concatenation (fast, no re-encoding)
    concat_result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", 
         "-c", "copy", "visual.mp4"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if concat_result.returncode != 0:
        print(f"❌ Concatenation failed: {concat_result.stderr[-300:]}")
        return False
    
    if not os.path.exists("visual.mp4"):
        print("❌ visual.mp4 not created")
        return False
    
    visual_size = os.path.getsize("visual.mp4")
    if visual_size < 100000:
        print(f"❌ visual.mp4 too small: {visual_size} bytes")
        return False
    
    print(f"✅ visual.mp4: {visual_size / (1024*1024):.1f}MB")
    
    # === VERSION 1: 900p NO SUBTITLES (OPTIMIZED FOR <1GB) ===
    print("\n📹 Rendering Version 1: 900p (No Subtitles) - Optimized for size")
    update_status(85, "Rendering 900p version without subtitles...")
    
    TARGET_WIDTH = 1600
    TARGET_HEIGHT = 900
    
    # Build command for P100 with FIXED NVENC pipeline
    if logo_path and os.path.exists(logo_path):
        # Download from GPU to CPU for overlay, then use NVENC for encoding
        filter_v1 = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"hwdownload,format=yuv420p[bg];"
            f"[1:v]scale=200:-1[logo];"
            f"[bg][logo]overlay=25:25[v]"
        )
        cmd_v1 = [
            "ffmpeg", "-y",
            "-vsync", "0",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", "visual.mp4",
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "21",
            "-b:v", "8M",
            "-maxrate", "12M",
            "-bufsize", "16M",
            "-profile:v", "high",
            "-level", "4.2",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_no_subs)
        ]
    else:
        # No logo - simpler filter, download from GPU before encoding
        filter_v1 = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"hwdownload,format=yuv420p[v]"
        )
        cmd_v1 = [
            "ffmpeg", "-y",
            "-vsync", "0",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", "visual.mp4",
            "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "21",
            "-b:v", "8M",
            "-maxrate", "12M",
            "-bufsize", "16M",
            "-profile:v", "high",
            "-level", "4.2",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_no_subs)
        ]
    
    print("    Encoding with NVENC (P100 optimized)...")
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=900)
    
    if result_v1.returncode != 0:
        print(f"❌ NVENC failed, trying CPU fallback...")
        print(f"Error (last 500 chars): {result_v1.stderr[-500:]}")
        
        # CPU Fallback (libx264)
        if logo_path and os.path.exists(logo_path):
            filter_fallback = (
                f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2[bg];"
                f"[1:v]scale=200:-1[logo];"
                f"[bg][logo]overlay=25:25[v]"
            )
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", "visual.mp4",
                "-i", str(logo_path),
                "-i", str(audio_path),
                "-filter_complex", filter_fallback,
                "-map", "[v]",
                "-map", "2:a",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-profile:v", "high",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_no_subs)
            ]
        else:
            filter_fallback = (
                f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2[v]"
            )
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", "visual.mp4",
                "-i", str(audio_path),
                "-filter_complex", filter_fallback,
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-profile:v", "high",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_no_subs)
            ]
        
        result_v1 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=1200)
        
        if result_v1.returncode != 0:
            print(f"❌ CPU fallback also failed: {result_v1.stderr[-500:]}")
            return False
    
    # Validate output
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
    
    # === VERSION 2: 1080p WITH SUBTITLES (MAXIMUM QUALITY) ===
    print("\n📹 Rendering Version 2: 1080p (With Subtitles) - Maximum Quality")
    update_status(90, "Rendering with subtitles...")
    
    # Prepare subtitle path (escape for FFmpeg)
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        # Download from GPU, apply overlays and subtitles on CPU, encode with NVENC
        filter_v2 = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"hwdownload,format=yuv420p[bg];"
            f"[1:v]scale=230:-1[logo];"
            f"[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles='{ass_path}'[v]"
        )
        cmd_v2 = [
            "ffmpeg", "-y",
            "-vsync", "0",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", "visual.mp4",
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "18",          # Better quality
            "-b:v", "12M",        # Higher bitrate
            "-maxrate", "18M",    # Higher max
            "-bufsize", "24M",    # Larger buffer
            "-profile:v", "high",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "256k",       # Better audio
            str(output_with_subs)
        ]
    else:
        # No logo - download from GPU, add subs on CPU, encode with NVENC
        filter_v2 = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"hwdownload,format=yuv420p[bg];"
            f"[bg]subtitles='{ass_path}'[v]"
        )
        cmd_v2 = [
            "ffmpeg", "-y",
            "-vsync", "0",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", "visual.mp4",
            "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "12M",
            "-maxrate", "18M",
            "-bufsize", "24M",
            "-profile:v", "high",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "256k",
            str(output_with_subs)
        ]
    
    print("    Encoding 1080p with subtitles...")
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=900)
    
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
    
    return True
    # 14. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START: Enhanced T5 ONLY Version ---")
update_status(1, "Initializing...")

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
update_status(10, "Scripting...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if len(text) < 100:
    update_status(0, "Script too short", "failed")
    exit(1)

# Generate audio
update_status(20, "Audio Synthesis...")
audio_out = TEMP_DIR / "audio.wav"

if clone_voice(text, ref_voice, audio_out):
    update_status(50, "Creating Subtitles...")
    
    # Transcribe
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_out))
            
            sentences = []
            for sentence in transcript.get_sentences():
                sentences.append({
                    "text": sentence.text,
                    "start": sentence.start / 1000,
                    "end": sentence.end / 1000
                })
            if sentences:
                sentences[-1]['end'] += 1.0
        except:
            # Fallback timing
            words = text.split()
            import wave
            with wave.open(str(audio_out), 'rb') as wav:
                total_dur = wav.getnframes() / float(wav.getframerate())
            
            words_per_sec = len(words) / total_dur
            sentences = []
            current_time = 0
            
            for i in range(0, len(words), 12):
                chunk = words[i:i+12]
                dur = len(chunk) / words_per_sec
                sentences.append({
                    "text": ' '.join(chunk),
                    "start": current_time,
                    "end": current_time + dur
                })
                current_time += dur
    else:
        # Fallback
        words = text.split()
        import wave
        with wave.open(str(audio_out), 'rb') as wav:
            total_dur = wav.getnframes() / float(wav.getframerate())
        
        words_per_sec = len(words) / total_dur
        sentences = []
        current_time = 0
        
        for i in range(0, len(words), 12):
            chunk = words[i:i+12]
            dur = len(chunk) / words_per_sec
            sentences.append({
                "text": ' '.join(chunk),
                "start": current_time,
                "end": current_time + dur
            })
            current_time += dur
    
    # Create subtitles
    ass_file = TEMP_DIR / "subs.ass"
    create_ass_file(sentences, ass_file)
    
    # Process visuals - TWO OUTPUTS
    update_status(60, "Processing Visuals...")
    output_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_NO_SUBS.mp4"
    output_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_WITH_SUBS.mp4"
    
    if process_visuals(sentences, audio_out, ass_file, ref_logo, output_no_subs, output_with_subs):
        # Upload both versions
        update_status(90, "Uploading Version 1 (No Subs)...")
        link1 = upload_to_google_drive(output_no_subs)
        
        update_status(95, "Uploading Version 2 (With Subs)...")
        link2 = upload_to_google_drive(output_with_subs)
        
        final_message = "✅ Success!\n"
        if link1:
            final_message += f"No Subs: {link1}\n"
        if link2:
            final_message += f"With Subs: {link2}"
        
        update_status(100, final_message, "completed", link1 or link2)
        print(f"🎉 {final_message}")
    else:
        update_status(0, "Processing failed", "failed")
else:
    update_status(0, "Audio failed", "failed")

# Cleanup
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4", "list.txt"]:
    if os.path.exists(f):
        os.remove(f)

print("--- ✅ COMPLETE ---")
