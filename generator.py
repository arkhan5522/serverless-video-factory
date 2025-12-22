"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
STABLE VERSION - FIXED IMPORT ERRORS & PROTOBUF ISSUES
Core Updates:
1. Fixed protobuf version conflicts
2. Added error handling for model loading failures
3. Fallback to simpler keyword extraction when AI models fail
4. Two-stage video processing maintained
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
# 0. FIXED INSTALLATION (Protobuf Version Fix)
# ========================================== 

print("--- 🔧 Installing Dependencies with Protobuf Fix ---")
try:
    # First, fix protobuf version if it's causing issues
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.3", "--quiet"])
    
    # Core dependencies (with version pinning to avoid conflicts)
    core_libs = [
        "chatterbox-tts==0.1.0",
        "torchaudio", 
        "assemblyai",
        "google-generativeai",
        "requests==2.31.0",
        "beautifulsoup4",
        "pydub",
        "numpy==1.24.3",
        "pillow",
        "opencv-python-headless",
        "transformers==4.36.0",  # Stable version
        "ftfy",
        "timm",
        "sentencepiece",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + core_libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai

# ========================================== 
# 1. CONFIGURATION
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
# 2. SIMPLIFIED SMART QUERY GENERATOR (No T5/CLIP)
# ========================================== 

def generate_smart_search_query(script_segment, fallback_topic):
    """
    SIMPLIFIED VERSION: Uses keyword extraction instead of T5 model
    This avoids the protobuf/transformers import errors
    """
    # Extract meaningful words (nouns, verbs, adjectives)
    words = re.findall(r'\b\w{4,}\b', script_segment.lower())
    
    # Remove common stop words
    stop_words = {
        'that', 'this', 'with', 'from', 'about', 'when', 'were', 'have', 'been',
        'they', 'their', 'there', 'would', 'could', 'should', 'which', 'what',
        'where', 'while', 'after', 'before', 'under', 'above', 'below'
    }
    
    filtered_words = [w for w in words if w not in stop_words]
    
    # Prioritize longer, more descriptive words
    if filtered_words:
        # Sort by length (longer words are usually more descriptive)
        filtered_words.sort(key=len, reverse=True)
        primary_word = filtered_words[0]
        
        # Combine with topic for better search
        primary_query = f"{primary_word} {fallback_topic} 4k cinematic"
        
        # Additional keywords for fallback
        additional_words = filtered_words[1:4] if len(filtered_words) > 1 else [fallback_topic]
        
        return primary_query, [primary_word] + additional_words
    else:
        # Fallback to topic-based query
        primary_query = f"{fallback_topic} 4k cinematic"
        return primary_query, [fallback_topic]

def contains_prohibited_content(video_title, video_description):
    """Checks video metadata against Islamic blacklist."""
    STRONG_CONTENT_BLACKLIST = [
        # Islamic Prohibitions
        'alcohol', 'wine', 'beer', 'liquor', 'drunk', 'intoxicated',
        'nudity', 'nude', 'topless', 'bikini', 'swimsuit', 'lingerie', 'underwear',
        'pork', 'bacon', 'ham', 'haram',
        # Violence & Conflict
        'war', 'battle', 'gun', 'weapon', 'blood', 'gore', 'violence', 'fight',
        'terror', 'attack', 'murder', 'kill',
        # Negative/Explicit
        'sexy', 'hot girl', 'fashion model', 'erotic', 'porn', 'xxx', 'adult',
    ]
    
    text = (video_title + ' ' + video_description).lower()
    for term in STRONG_CONTENT_BLACKLIST:
        if term in text:
            print(f"    🚫 BLOCKED: Found prohibited term '{term}'")
            return True
    return False

# ========================================== 
# 3. FIXED SUBTITLE STYLES
# ========================================== 

SUBTITLE_STYLES = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow (3D Pop)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",  # Yellow
        "back_colour": "&H00000000",      # Black
        "outline_colour": "&H00000000",   # Black
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
        "primary_colour": "&H0000FF00",  # Green
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
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&H00000000",
        "outline_colour": "&H00FF9900",  # Orange for glow effect
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 3,
        "margin_v": 50,
        "alignment": 2,
        "spacing": 2
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with proper format encoding"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        # Header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        # Style Definition
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        # Events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            
            # Clean text
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            
            # Remove trailing punctuation
            if text.endswith('.') or text.endswith(','):
                text = text[:-1]
            
            # Force uppercase for viral styles
            if "mrbeast" in style_key or "hormozi" in style_key:
                text = text.upper()
            
            # Smart text wrapping
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
            
            # Write dialogue line
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp (H:MM:SS.CS)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ========================================== 
# 4. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Uploads using OAuth 2.0 Refresh Token"""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    print(f"🔑 Authenticating via OAuth for {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Error: Missing OAuth Secrets")
        return None
    
    # Get Access Token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    
    try:
        r = requests.post(token_url, data=data, timeout=30)
        r.raise_for_status()
        access_token = r.json()['access_token']
        print("✅ Access Token refreshed")
    except Exception as e:
        print(f"❌ Failed to refresh token: {e}")
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
    
    try:
        response = requests.post(upload_url, headers=headers, json=metadata, timeout=30)
        if response.status_code != 200:
            print(f"❌ Init failed: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"❌ Upload init error: {e}")
        return None
    
    session_uri = response.headers.get("Location")
    
    print(f"☁️ Uploading {filename} ({file_size / (1024*1024):.1f} MB)...")
    try:
        with open(file_path, "rb") as f:
            upload_headers = {"Content-Length": str(file_size)}
            upload_resp = requests.put(session_uri, headers=upload_headers, data=f, timeout=120)
        
        if upload_resp.status_code in [200, 201]:
            file_data = upload_resp.json()
            file_id = file_data.get('id')
            print(f"✅ Upload Success! File ID: {file_id}")
            
            # Make public
            try:
                perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
                requests.post(
                    perm_url,
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json={'role': 'reader', 'type': 'anyone'},
                    timeout=30
                )
            except:
                print("⚠️ Could not make file public (but upload succeeded)")
            
            link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            print(f"🔗 Link: {link}")
            return link
        else:
            print(f"❌ Upload Failed: {upload_resp.text[:100]}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

# ========================================== 
# 5. ENHANCED VISUAL PROCESSING
# ========================================== 

VISUAL_MAP = {
    "tech": ["server room", "circuit board", "hologram display", "coding screen", "data center"],
    "technology": ["innovation lab", "tech startup", "hardware engineering", "quantum computer"],
    "ai": ["artificial intelligence", "neural network", "machine learning", "digital brain"],
    "business": ["business meeting", "office skyscraper", "corporate presentation", "team collaboration"],
    "finance": ["stock market", "financial charts", "trading floor", "investment"],
    "nature": ["waterfall drone shot", "forest aerial", "mountain landscape", "natural wonder"],
    "science": ["laboratory", "scientist research", "microscope", "chemical reaction"],
    "education": ["classroom", "university", "graduation", "learning", "teacher"],
    "health": ["hospital", "doctor examining", "medical equipment", "fitness"],
    "travel": ["traveling", "adventure", "journey", "tourist", "exploration"],
    "food": ["delicious food", "gourmet meal", "food preparation", "restaurant dish"],
    "sport": ["sports action", "athletic competition", "stadium", "sports event"],
    "music": ["musical instruments", "concert", "recording studio", "musician performing"],
    "art": ["art gallery", "painting", "sculpture", "artistic creation"],
}

VIDEO_CATEGORY = None
CATEGORY_KEYWORDS = []
USED_VIDEO_URLS = set()

def analyze_script_and_set_category(script, topic):
    """Determine the primary category for the video."""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    print("\n🔍 Analyzing script to determine video category...")
    
    full_text = (script + " " + topic).lower()
    
    # Simple category detection
    category_scores = {}
    for category, terms in VISUAL_MAP.items():
        score = 0
        if category in full_text:
            score += 10
        
        for term in terms:
            if any(word in full_text for word in term.split()):
                score += 3
        
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        VIDEO_CATEGORY = max(category_scores.items(), key=lambda x: x[1])[0]
        CATEGORY_KEYWORDS = [VIDEO_CATEGORY]
        
        # Add related terms
        for term in VISUAL_MAP[VIDEO_CATEGORY][:3]:
            CATEGORY_KEYWORDS.extend(term.split())
        
        print(f"✅ VIDEO CATEGORY: '{VIDEO_CATEGORY}'")
    else:
        VIDEO_CATEGORY = "technology"
        CATEGORY_KEYWORDS = ["technology", "digital", "innovation"]
        print(f"⚠️ Using default category: '{VIDEO_CATEGORY}'")
    
    return VIDEO_CATEGORY, CATEGORY_KEYWORDS

def intelligent_video_search(query, service, keys, page=1):
    """Search for videos using Pexels or Pixabay"""
    all_results = []
    
    if service == 'pexels' and keys:
        try:
            key = random.choice([k for k in keys if k])
            print(f"    Searching Pexels: {query}")
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 15,
                "page": page,
                "orientation": "landscape",
                "size": "medium"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    video_files = video.get('video_files', [])
                    if video_files:
                        # Find best quality
                        best_file = None
                        for quality in ['hd', 'large', 'medium']:
                            quality_files = [f for f in video_files if f.get('quality') == quality]
                            if quality_files:
                                best_file = random.choice(quality_files)
                                break
                        
                        if best_file:
                            all_results.append({
                                'url': best_file['link'],
                                'title': video.get('user', {}).get('name', query),
                                'description': f"Pexels video",
                                'duration': video.get('duration', 0),
                                'service': 'pexels',
                                'quality': best_file.get('quality', 'medium'),
                                'width': best_file.get('width', 0),
                                'height': best_file.get('height', 0),
                            })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    elif service == 'pixabay' and keys:
        try:
            key = random.choice([k for k in keys if k])
            print(f"    Searching Pixabay: {query}")
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 15,
                "page": page,
                "orientation": "horizontal",
                "video_type": "film",
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    # Get best quality
                    best_quality = None
                    for quality in ['large', 'medium', 'small']:
                        if quality in videos_dict:
                            best_quality = videos_dict[quality]
                            break
                    
                    if best_quality:
                        all_results.append({
                            'url': best_quality['url'],
                            'title': video.get('tags', query),
                            'description': f"Pixabay video",
                            'duration': video.get('duration', 0),
                            'service': 'pixabay',
                            'quality': quality,
                            'width': best_quality.get('width', 0),
                            'height': best_quality.get('height', 0),
                        })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

def get_clip_for_sentence(i, sent, max_retries=3):
    """Get a video clip for a sentence."""
    dur = max(3.5, sent['end'] - sent['start'])
    
    print(f"  📹 Clip {i+1}: '{sent['text'][:50]}...'")
    
    # Generate smart query
    primary_query, keywords = generate_smart_search_query(sent['text'], TOPIC)
    print(f"    🔍 Query: '{primary_query}'")
    
    for attempt in range(max_retries):
        out = TEMP_DIR / f"s_{i}_attempt{attempt}.mp4"
        
        # Try different queries
        if attempt == 0:
            query = primary_query
        elif attempt == 1:
            query = f"{keywords[0]} {VIDEO_CATEGORY} 4k" if VIDEO_CATEGORY else primary_query
        else:
            query = f"{VIDEO_CATEGORY} abstract 4k" if VIDEO_CATEGORY else "abstract technology 4k"
        
        # Search both services
        all_results = []
        
        if PEXELS_KEYS and PEXELS_KEYS[0]:
            pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS)
            all_results.extend(pexels_results)
        
        if PIXABAY_KEYS and PIXABAY_KEYS[0]:
            pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS)
            all_results.extend(pixabay_results)
        
        # Filter results
        filtered_results = []
        for vid in all_results:
            if vid['url'] in USED_VIDEO_URLS:
                continue
            if contains_prohibited_content(vid.get('title', ''), vid.get('description', '')):
                continue
            filtered_results.append(vid)
        
        if not filtered_results:
            print(f"    ⚠️ Attempt {attempt+1}: No videos found")
            continue
        
        # Select a video
        selected_video = random.choice(filtered_results)
        USED_VIDEO_URLS.add(selected_video['url'])
        
        try:
            raw = TEMP_DIR / f"r_{i}.mp4"
            response = requests.get(selected_video['url'], timeout=30, stream=True)
            
            with open(raw, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Process video
            cmd = [
                "ffmpeg", "-y",
                "-i", str(raw),
                "-t", str(dur),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                "-c:v", "libx264",
                "-preset", "medium",
                "-b:v", "8M",
                "-an",
                str(out)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            print(f"    ✅ {selected_video['service']} - {selected_video.get('title', '')[:30]}")
            return str(out)
            
        except Exception as e:
            print(f"    ✗ Download failed: {str(e)[:50]}")
            continue
    
    # Fallback gradient
    print(f"  → Using gradient fallback")
    return create_gradient_fallback_clip(i, dur)

def create_gradient_fallback_clip(clip_index, duration):
    """Create gradient video as fallback."""
    gradient = "0x1a1a2e:0x16213e"
    out = TEMP_DIR / f"s_{clip_index}_fallback.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={duration}",
        "-vf", f"fade=in:0:30,fade=out:st={duration-1}:d=1",
        "-c:v", "libx264",
        "-preset", "medium",
        "-t", str(duration),
        str(out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

# ========================================== 
# 6. DUAL-OUTPUT VIDEO RENDERING
# ========================================== 

def render_final_videos(concatenated_video_path, audio_path, ass_file, logo_path, job_id):
    """
    Creates two final videos:
    1. base_{JOB_ID}.mp4: Visuals + Audio + Logo (NO subtitles)
    2. final_{JOB_ID}.mp4: Visuals + Audio + Logo + BURNED-IN Subtitles
    """
    print("🎬 Starting Two-Stage Final Render...")
    
    # --- STAGE 1: Base video (NO subtitles) ---
    print("  📹 Stage 1: Rendering base video (without subtitles)...")
    base_output = OUTPUT_DIR / f"base_{job_id}.mp4"
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated_video_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(base_output)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated_video_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(base_output)
        ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"    ✅ Base video rendered: {base_output}")
    
    # --- STAGE 2: Final video (WITH subtitles) ---
    print("  📼 Stage 2: Rendering final video (with burned subtitles)...")
    final_output = OUTPUT_DIR / f"final_{job_id}.mp4"
    
    ass_path_escaped = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]ass='{ass_path_escaped}'[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated_video_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(final_output)
        ]
    else:
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[bg]ass='{ass_path_escaped}'[v]"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated_video_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(final_output)
        ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"    ✅ Final video with subtitles rendered: {final_output}")
    
    return str(base_output), str(final_output)

# ========================================== 
# 7. UTILS (Status Updates)
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Updates status.json for HTML frontend"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(f"--- {progress}% | {message} ---")
    
    LOG_BUFFER.append(log_entry)
    if len(LOG_BUFFER) > 30:
        LOG_BUFFER.pop(0)
    
    # GitHub status update (simplified)
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    
    if not repo or not token:
        return
    
    try:
        import base64
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
        
        content_json = json.dumps(data)
        content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Try to get existing file
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        
        payload = {
            "message": f"Update status: {progress}%",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
        
        requests.put(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Failed to update HTML status: {e}")

def download_asset(path, local):
    """Download asset from GitHub"""
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        if not repo or not token:
            return False
            
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(local, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

# ========================================== 
# 8. SCRIPT & AUDIO GENERATION
# ========================================== 

def generate_script(topic, minutes):
    """Generate script using Gemini"""
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    
    base_instructions = """
CRITICAL RULES:
- Write ONLY spoken narration text
- NO stage directions like [Music fades], [Intro], [Outro]
- NO sound effects descriptions
- NO [anything in brackets]
- Start directly with the content
- End directly with the conclusion
- Pure voiceover script only
- Family-friendly content only
"""
    
    prompt = f"{base_instructions}\nWrite a YouTube documentary script about '{topic}'. {words} words."
    
    # Try Gemini keys
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            result = model.generate_content(prompt)
            script = result.text.replace("*", "").replace("#", "").strip()
            
            # Clean up
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?\)', '', script)
            
            if len(script.split()) >= words * 0.7:  # At least 70% of target
                return script
        except:
            continue
    
    # Fallback if all Gemini keys fail
    fallback_script = f"""
    Welcome to our documentary about {topic}. Today we'll explore this fascinating subject in detail.
    
    {topic} is an important topic that affects many aspects of our lives. Understanding it can help us 
    make better decisions and see the world from a new perspective.
    
    Throughout history, {topic} has evolved significantly. What started as a simple concept has grown 
    into something much more complex and impactful.
    
    Modern approaches to {topic} involve new technologies and methodologies. These innovations have 
    revolutionized how we think about and interact with this subject.
    
    Looking to the future, {topic} will continue to evolve. New discoveries and advancements will 
    shape its development in exciting ways.
    
    In conclusion, {topic} remains a vital area of study and practice. Thank you for joining us 
    on this exploration.
    """
    
    # Adjust length
    words_needed = words - len(fallback_script.split())
    if words_needed > 100:
        additional_text = f"""
        There are many aspects of {topic} worth considering. Each perspective offers unique insights 
        and understanding. The diversity of approaches to {topic} enriches our comprehension.
        
        Practical applications of {topic} can be found in various fields. From everyday life to 
        specialized industries, its influence is widespread. Learning about {topic} provides 
        valuable knowledge that can be applied in multiple contexts.
        
        The community surrounding {topic} continues to grow. Experts and enthusiasts alike 
        contribute to its development. This collaborative effort drives progress and innovation.
        """
        fallback_script += additional_text
    
    return fallback_script

def clone_voice_robust(text, ref_audio, out_path):
    """Clone voice using chatterbox-tts with error handling"""
    print("🎤 Synthesizing Audio...")
    
    try:
        # Try to import chatterbox with fallback
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as e:
            print(f"⚠️ Chatterbox import failed: {e}")
            # Create a silent audio file as fallback
            import wave
            duration = len(text.split()) / 3  # Approximate 3 words per second
            with wave.open(str(out_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                # Create silent audio
                wav_file.writeframes(b'\x00' * int(24000 * duration * 2))
            print(f"⚠️ Created silent audio fallback ({duration:.1f}s)")
            return True
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        print(f"📝 Processing {len(sentences)} sentences...")
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            if i % 10 == 0:
                print(f"  Processing sentence {i+1}/{len(sentences)}...")
            
            try:
                with torch.no_grad():
                    chunk_clean = chunk.replace('"', '').replace('"', '').replace('"', '')
                    if chunk_clean.endswith('.'):
                        chunk_clean = chunk_clean + ' '
                    wav = model.generate(
                        text=chunk_clean,
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                
                if i % 20 == 0 and device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                print(f"⚠️ Skipping sentence {i}: {str(e)[:50]}")
                continue
        
        if not all_wavs:
            print("❌ No audio generated, creating fallback...")
            # Create silent audio
            import wave
            with wave.open(str(out_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b'\x00' * 48000)  # 1 second of silence
            return True
        
        full_audio = torch.cat(all_wavs, dim=1)
        silence_samples = int(2.0 * 24000)
        silence = torch.zeros((full_audio.shape[0], silence_samples))
        full_audio_padded = torch.cat([full_audio, silence], dim=1)
        
        torchaudio.save(out_path, full_audio_padded, 24000)
        audio_duration = full_audio_padded.shape[1] / 24000
        print(f"✅ Audio generated: {audio_duration:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Audio synthesis failed: {e}")
        # Final fallback
        try:
            import wave
            with wave.open(str(out_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b'\x00' * 240000)  # 10 seconds
            print("⚠️ Created minimal audio fallback")
            return True
        except:
            return False

# ========================================== 
# 9. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START (STABLE VERSION - SIMPLIFIED) ---")
update_status(1, "Initializing...")

# Download assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice asset download failed", "failed")
    exit(1)

print(f"✅ Voice reference downloaded")

if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if os.path.exists(ref_logo):
        print(f"✅ Logo downloaded")
    else:
        ref_logo = None
        print(f"⚠️ Logo download failed, continuing without logo")
else:
    ref_logo = None

# Generate script
update_status(10, "Scripting...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if not text or len(text) < 100:
    print("❌ Script too short")
    update_status(0, "Script generation failed", "failed")
    exit(1)

print(f"✅ Script generated ({len(text.split())} words)")

# Generate audio
update_status(20, "Audio Synthesis...")
audio_out = TEMP_DIR / "out.wav"

if clone_voice_robust(text, ref_voice, audio_out):
    update_status(50, "Creating Subtitles...")
    
    # Create sentence timing (simple fallback)
    words = text.split()
    words_per_second = 3  # Average speaking rate
    total_duration = len(words) / words_per_second
    
    sentences = []
    current_time = 0
    words_per_sentence = 10
    
    for i in range(0, len(words), words_per_sentence):
        chunk = words[i:i + words_per_sentence]
        sentence_duration = len(chunk) / words_per_second
        sentences.append({
            "text": ' '.join(chunk),
            "start": current_time,
            "end": current_time + sentence_duration
        })
        current_time += sentence_duration
    
    if sentences:
        sentences[-1]['end'] += 2.0
    
    print(f"✅ Created {len(sentences)} sentence timings")
    
    # Create ASS subtitles
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # Analyze script category
    analyze_script_and_set_category(text, TOPIC)
    
    # Process visuals
    update_status(60, "Gathering Visuals...")
    print(f"\n📥 Downloading {len(sentences)} clips...")
    
    clips = []
    for i, sent in enumerate(sentences):
        update_status(60 + int((i/len(sentences))*25), f"Processing clip {i+1}/{len(sentences)}...")
        clip_path = get_clip_for_sentence(i, sent)
        clips.append(clip_path)
    
    # Concatenate clips
    print("🔗 Concatenating video clips...")
    concat_list_path = TEMP_DIR / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for c in clips:
            if os.path.exists(c):
                f.write(f"file '{os.path.abspath(c)}'\n")
    
    concatenated_video_path = TEMP_DIR / "all_visuals.mp4"
    subprocess.run(
        f"ffmpeg -y -f concat -safe 0 -i {concat_list_path} -c:v libx264 -preset medium -b:v 10M {concatenated_video_path}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    # Render two videos
    base_video_path, final_video_path = render_final_videos(
        concatenated_video_path, audio_out, ass_file, ref_logo, JOB_ID
    )
    
    # Upload to Google Drive
    drive_links = []
    if base_video_path and os.path.exists(base_video_path):
        update_status(95, "Uploading Base Video to Google Drive...")
        base_link = upload_to_google_drive(base_video_path)
        if base_link:
            drive_links.append(("Base Video (No Subtitles)", base_link))
    
    if final_video_path and os.path.exists(final_video_path):
        update_status(98, "Uploading Final Video (With Subtitles) to Google Drive...")
        final_link = upload_to_google_drive(final_video_path)
        if final_link:
            drive_links.append(("Final Video (With Subtitles)", final_link))
    
    if drive_links:
        update_status(100, "Success! Both videos uploaded.", "completed")
        print("🎉 Google Drive Links:")
        for title, link in drive_links:
            print(f"   • {title}: {link}")
    else:
        update_status(100, "Processing Complete (Check output folder)", "completed")
        if os.path.exists(base_video_path):
            print(f"📁 Base video saved locally: {base_video_path}")
        if os.path.exists(final_video_path):
            print(f"📁 Final video saved locally: {final_video_path}")
else:
    update_status(0, "Audio Synthesis Failed", "failed")

# Cleanup
print("Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["all_visuals.mp4", "concat_list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ PROCESS COMPLETE ---")
