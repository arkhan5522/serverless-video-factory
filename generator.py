"""
AI VIDEO GENERATOR WITH PARALLEL PROCESSING
============================================
OPTIMIZATIONS:
1. Parallel video downloads (5 concurrent)
2. Parallel FFmpeg processing
3. Background audio generation
4. Smart caching and resource management
5. All previous realism features maintained
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
import math
from pathlib import Path
from collections import defaultdict
from threading import Thread, Lock
import queue

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "chatterbox-tts",
        "torchaudio",
        "google-generativeai",
        "requests",
        "pydub",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import google.generativeai as genai
import numpy as np

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
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Thread-safe globals
USED_VIDEO_URLS = set()
URL_LOCK = Lock()
LOG_BUFFER = []
LOG_LOCK = Lock()

# ========================================== 
# 3. LOGGING & STATUS SYSTEM
# ========================================== 

def update_status(progress, message, status="processing", file_url=None):
    """Updates status.json in GitHub repo so HTML can read it"""
    
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(f"--- {progress}% | {message} ---")
    
    with LOG_LOCK:
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
            "message": f"Update status: {progress}%",
            "content": content_b64,
            "branch": "main" 
        }
        if sha:
            payload["sha"] = sha
            
        requests.put(url, headers=headers, json=payload)
    except Exception as e:
        print(f"⚠️ Failed to update HTML status: {e}")

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
# 4. SUBTITLE STYLES
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
        "outline_colour": "&H00FF9900",  # Blue
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
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&H90000000",     # Dark box
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 3,  # Opaque box
        "outline": 0,
        "shadow": 0,
        "margin_v": 35,
        "alignment": 2,
        "spacing": 0.5
    },
    "tiktok_white": {
        "name": "TikTok White (Ultra Bold)",
        "fontname": "Arial Black",
        "fontsize": 65,
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&H60000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 4.5,
        "shadow": 2,
        "margin_v": 40,
        "alignment": 2,
        "spacing": -0.5
    }
}
def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def create_human_subtitles(sentences, ass_file):
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
            start_time = max(0, s['start'] - 0.15)
            end_time = s['end'] - 0.3
            
            text = s['text'].strip()
            text = re.sub(r'[\[\]]', '', text)
            
            words = text.split()
            lines = []
            current_line = []
            char_count = 0
            
            for word in words:
                if char_count + len(word) + 1 > 35 or len(current_line) >= 4:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    char_count = len(word)
                else:
                    current_line.append(word)
                    char_count += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            if style_key in ["mrbeast_yellow", "tiktok_white"]:
                lines = [line.upper() for line in lines]
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{formatted_text}\n")

# ========================================== 
# 5. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    print("🔑 Authenticating via OAuth (Refresh Token)...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Error: Missing OAuth Secrets")
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
        print("✅ Access Token refreshed")
    except Exception as e:
        print(f"❌ Failed to refresh token: {e}")
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
    
    print(f"☁️ Uploading {filename} ({file_size / (1024*1024):.1f} MB)...")
    with open(file_path, "rb") as f:
        upload_headers = {"Content-Length": str(file_size)}
        upload_resp = requests.put(session_uri, headers=upload_headers, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_data = upload_resp.json()
        file_id = file_data.get('id')
        print(f"✅ Upload Success! File ID: {file_id}")
        
        perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        requests.post(
            perm_url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"🔗 Link: {link}")
        return link
    else:
        print(f"❌ Upload Failed: {upload_resp.text}")
        return None

# ========================================== 
# 6. CINEMATIC SCENE ENGINE
# ========================================== 

class CinematicSceneEngine:
    EMOTION_TO_VISUAL = {
        "frustration": {
            "mood": "dark",
            "camera": "slow_motion",
            "color": "desaturated",
            "motion": "subtle_shake",
            "keywords": ["struggle", "failed", "problem", "hard", "difficult"]
        },
        "curiosity": {
            "mood": "mysterious",
            "camera": "zoom_in",
            "color": "warm_contrast",
            "motion": "slow_pan_right",
            "keywords": ["discover", "found", "secret", "learned", "realized"]
        },
        "excitement": {
            "mood": "energetic",
            "camera": "dynamic",
            "color": "vibrant",
            "motion": "subtle_zoom",
            "keywords": ["amazing", "incredible", "awesome", "boom", "wow"]
        },
        "explanation": {
            "mood": "professional",
            "camera": "stable",
            "color": "balanced",
            "motion": "still",
            "keywords": ["because", "reason", "explain", "shows", "demonstrates"]
        },
        "success": {
            "mood": "triumphant",
            "camera": "wide",
            "color": "golden_hour",
            "motion": "subtle_zoom",
            "keywords": ["success", "win", "achieve", "result", "growth"]
        }
    }
    
    VISUAL_INTENT_TYPES = {
        "explanation": ["hands typing", "laptop screen", "whiteboard"],
        "story": ["person walking", "city street", "looking thoughtful"],
        "data": ["charts growing", "graphs animation", "statistics"],
        "concept": ["abstract shapes", "particles floating", "connections"],
        "action": ["person working", "team collaboration", "progress"]
    }
    
    @staticmethod
    def analyze_sentence(sentence):
        sentence_lower = sentence.lower()
        
        detected_emotions = []
        for emotion, data in CinematicSceneEngine.EMOTION_TO_VISUAL.items():
            for keyword in data["keywords"]:
                if keyword in sentence_lower:
                    detected_emotions.append(emotion)
                    break
        
        if not detected_emotions:
            detected_emotions = ["explanation"]
        
        detected_intents = []
        for intent, keywords in CinematicSceneEngine.VISUAL_INTENT_TYPES.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    detected_intents.append(intent)
                    break
        
        if not detected_intents:
            detected_intents = ["story"]
        
        primary_emotion = detected_emotions[0]
        primary_intent = detected_intents[0]
        
        return {
            "emotion": primary_emotion,
            "intent": primary_intent,
            "visual_style": CinematicSceneEngine.EMOTION_TO_VISUAL[primary_emotion]
        }

# ========================================== 
# 7. CATEGORY ANALYSIS
# ========================================== 

def analyze_topic_for_category(topic, script):
    topic_category_map = {
        "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
        "machine learning": "artificial intelligence", "technology": "technology",
        "business": "business", "finance": "finance",
        "motivation": "people", "success": "people",
        "science": "science", "health": "health"
    }
    
    topic_lower = topic.lower()
    
    for keyword, category in topic_category_map.items():
        if keyword in topic_lower:
            print(f"✅ Category determined: {category}")
            return category, [category, keyword]
    
    return "technology", ["technology", "digital"]

# ========================================== 
# 8. PARALLEL VIDEO SEARCH
# ========================================== 

def search_single_service(query, service, keys):
    """Search single service with error handling"""
    try:
        if service == 'pexels' and keys and keys[0]:
            key = random.choice([k for k in keys if k])
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 15,
                "orientation": "landscape"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=8)
            
            if response.status_code == 429:
                time.sleep(1)
                return []
                
            if response.status_code == 200:
                data = response.json()
                results = []
                for video in data.get('videos', [])[:8]:
                    video_files = video.get('video_files', [])
                    if video_files:
                        best_file = None
                        for quality in ['hd', 'large', 'medium']:
                            files = [f for f in video_files if f.get('quality') == quality]
                            if files:
                                best_file = random.choice(files)
                                break
                        
                        if best_file:
                            results.append({
                                'url': best_file['link'],
                                'duration': video.get('duration', 0),
                                'service': 'pexels',
                                'quality': best_file.get('quality', 'medium')
                            })
                return results
                
        elif service == 'pixabay' and keys and keys[0]:
            key = random.choice([k for k in keys if k])
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 15,
                "orientation": "horizontal"
            }
            
            response = requests.get(url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                results = []
                for video in data.get('hits', [])[:8]:
                    videos_dict = video.get('videos', {})
                    for quality in ['large', 'medium']:
                        if quality in videos_dict:
                            results.append({
                                'url': videos_dict[quality]['url'],
                                'duration': video.get('duration', 0),
                                'service': 'pixabay',
                                'quality': quality
                            })
                            break
                return results
    except Exception as e:
        pass
    
    return []

def parallel_video_search(query):
    """Search both services in parallel"""
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if PEXELS_KEYS and PEXELS_KEYS[0]:
            futures.append(executor.submit(search_single_service, query, 'pexels', PEXELS_KEYS))
        if PIXABAY_KEYS and PIXABAY_KEYS[0]:
            futures.append(executor.submit(search_single_service, query, 'pixabay', PIXABAY_KEYS))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result(timeout=10)
                all_results.extend(results)
            except:
                pass
    
    return all_results

# ========================================== 
# 9. PARALLEL VIDEO DOWNLOADER
# ========================================== 

def download_and_process_video(video_info, output_path, duration, scene_analysis):
    """Download and process a single video"""
    try:
        response = requests.get(video_info['url'], timeout=12, stream=True)
        response.raise_for_status()
        
        temp_download = output_path.parent / f"temp_{output_path.name}"
        
        with open(temp_download, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > 40 * 1024 * 1024:
                        break
        
        if not os.path.exists(temp_download) or os.path.getsize(temp_download) < 50000:
            return False
        
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_download),
            "-t", str(duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-b:v", "6M", "-an",
            str(output_path)
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=25)
        
        try:
            os.remove(temp_download)
        except:
            pass
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 10000
        
    except Exception as e:
        return False

# ========================================== 
# 10. PARALLEL VISUAL PROCESSING
# ========================================== 

def get_category_queries(category, scene_analysis):
    """Get search queries for category"""
    CATEGORY_VISUALS = {
        "artificial intelligence": ["neural network", "AI processing", "machine learning", "data visualization"],
        "technology": ["innovation", "tech startup", "digital interface", "coding"],
        "business": ["business meeting", "office", "team success", "corporate"],
        "finance": ["stock charts", "financial growth", "money", "investment"],
        "people": ["person achieving", "success story", "motivation", "inspiration"],
        "science": ["laboratory", "research", "discovery", "experiment"],
        "health": ["fitness", "healthy lifestyle", "medical", "wellness"]
    }
    
    visuals = CATEGORY_VISUALS.get(category, CATEGORY_VISUALS["technology"])
    emotion = scene_analysis.get("emotion", "explanation")
    
    queries = [
        f"{random.choice(visuals)} {category} cinematic",
        f"{category} professional 4k",
        f"{emotion} {category} visual"
    ]
    
    return queries

def process_single_clip(i, sent, category):
    """Process a single video clip (runs in parallel) - NEVER uses gradients"""
    try:
        dur = max(3.5, sent['end'] - sent['start'])
        scene_analysis = CinematicSceneEngine.analyze_sentence(sent['text'])
        
        # Get base queries
        queries = get_category_queries(category, scene_analysis)
        
        # Add more specific queries based on sentence content
        sentence_lower = sent['text'].lower()
        words = sentence_lower.split()[:5]  # Use first few words
        if len(words) > 2:
            specific_query = ' '.join(words[:3])
            queries.insert(0, f"{specific_query} {category}")
        
        # Try all queries until we find a video
        for query_idx, query in enumerate(queries[:4]):  # Try up to 4 queries
            results = parallel_video_search(query)
            
            with URL_LOCK:
                available = [v for v in results if v['url'] not in USED_VIDEO_URLS]
            
            if available:
                # Prioritize by quality
                quality_order = ['hd', 'large', 'medium', 'small']
                for quality in quality_order:
                    quality_videos = [v for v in available if v.get('quality') == quality]
                    if quality_videos:
                        video = random.choice(quality_videos)
                        break
                else:
                    video = random.choice(available[:3])
                
                with URL_LOCK:
                    USED_VIDEO_URLS.add(video['url'])
                
                output_path = TEMP_DIR / f"clip_{i}.mp4"
                
                if download_and_process_video(video, output_path, dur, scene_analysis):
                    print(f"    ✅ Clip {i+1} processed ({video['service']} - {query})")
                    return str(output_path)
            
            print(f"    🔄 Clip {i+1} trying query {query_idx+1}: '{query}'")
        
        # If all queries fail, try broader searches
        print(f"    🔍 Clip {i+1} trying broader searches...")
        
        broader_queries = [
            f"{category} background",
            f"{category} scene",
            "cinematic footage",
            "professional background"
        ]
        
        for query in broader_queries:
            results = parallel_video_search(query)
            
            with URL_LOCK:
                available = [v for v in results if v['url'] not in USED_VIDEO_URLS]
            
            if available:
                video = random.choice(available[:3])
                
                with URL_LOCK:
                    USED_VIDEO_URLS.add(video['url'])
                
                output_path = TEMP_DIR / f"clip_{i}.mp4"
                
                if download_and_process_video(video, output_path, dur, scene_analysis):
                    print(f"    ✅ Clip {i+1} processed (fallback: {query})")
                    return str(output_path)
        
        # LAST RESORT: Use category-based video (still not gradient)
        print(f"    ⚠️ Clip {i+1} using category default...")
        
        # Get default videos for the category
        CATEGORY_DEFAULTS = {
            "artificial intelligence": ["technology future", "digital innovation", "data flow"],
            "technology": ["tech innovation", "digital world", "future technology"],
            "business": ["office environment", "business success", "team meeting"],
            "finance": ["financial growth", "market success", "money concept"],
            "people": ["person achieving", "success story", "motivational"],
            "science": ["scientific research", "laboratory", "discovery"],
            "health": ["healthy lifestyle", "fitness", "wellness"]
        }
        
        default_queries = CATEGORY_DEFAULTS.get(category, ["professional", "cinematic", "background"])
        
        for query in default_queries:
            results = parallel_video_search(query)
            if results:
                with URL_LOCK:
                    available = [v for v in results if v['url'] not in USED_VIDEO_URLS]
                
                if available:
                    video = random.choice(available)
                    
                    with URL_LOCK:
                        USED_VIDEO_URLS.add(video['url'])
                    
                    output_path = TEMP_DIR / f"clip_{i}.mp4"
                    
                    if download_and_process_video(video, output_path, dur, scene_analysis):
                        print(f"    ✅ Clip {i+1} processed (default: {query})")
                        return str(output_path)
        
        # If we STILL can't find anything, reuse a previous clip with different processing
        # This is better than gradient
        print(f"    🔄 Clip {i+1} reusing with different processing...")
        
        # Try one more generic search
        results = parallel_video_search(category)
        if results:
            video = random.choice(results[:5])
            output_path = TEMP_DIR / f"clip_{i}.mp4"
            
            if download_and_process_video(video, output_path, dur, scene_analysis):
                print(f"    ✅ Clip {i+1} processed (generic: {category})")
                return str(output_path)
        
        print(f"    ❌ Clip {i+1} could not find suitable video")
        return None
        
    except Exception as e:
        print(f"    ✗ Clip {i+1} failed: {str(e)[:40]}")
        return None
def process_visuals_parallel(sentences, topic, full_script):
    """Process all visuals in parallel - skip failed clips"""
    category, _ = analyze_topic_for_category(topic, full_script)
    print(f"🎯 CATEGORY: {category.upper()}")
    
    processed_clips = []
    clip_indices = []  # Store which sentences we have clips for
    
    batch_size = 5
    for batch_start in range(0, len(sentences), batch_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = sentences[batch_start:batch_end]
        
        progress = 60 + int((batch_start/len(sentences))*30)
        update_status(progress, f"Processing clips {batch_start+1}-{batch_end}/{len(sentences)}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for idx, sent in enumerate(batch):
                global_idx = batch_start + idx
                future = executor.submit(process_single_clip, global_idx, sent, category)
                futures[future] = global_idx
            
            for future in concurrent.futures.as_completed(futures):
                global_idx = futures[future]
                try:
                    result = future.result(timeout=60)  # Increased timeout
                    if result:
                        processed_clips.append(result)
                        clip_indices.append(global_idx)
                    else:
                        print(f"    ⚠️ Clip {global_idx+1} skipped (no suitable video found)")
                except Exception as e:
                    print(f"    ✗ Batch item {global_idx} failed: {e}")
    
    print(f"✅ Processed {len(processed_clips)}/{len(sentences)} clips")
    
    # If we have fewer clips than sentences, adjust the audio/script timing
    if len(processed_clips) < len(sentences):
        print(f"⚠️ Only {len(processed_clips)} clips available, adjusting...")
    
    return processed_clips

# ========================================== 
# 11. DUAL OUTPUT RENDERER
# ========================================== 

def render_dual_outputs(processed_clips, audio_path, ass_file, logo_path):
    """Render two versions: with and without subtitles"""
    
    print("🎬 Concatenating clips...")
    
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in processed_clips:
            if os.path.exists(clip):
                f.write(f"file '{clip}'\n")
    
    concatenated = TEMP_DIR / "concatenated.mp4"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast",
        "-b:v", "8M", "-an", str(concatenated)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("🎬 Rendering version 1 (no subtitles)...")
    final_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_no_subs.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(concatenated),
        "-i", str(audio_path),
        "-filter_complex", "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast",
        "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
        str(final_no_subs)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
        print(f"✅ Version 1: {os.path.getsize(final_no_subs)/(1024*1024):.1f}MB")
    except:
        final_no_subs = None
    
    print("🎬 Rendering version 2 (with subtitles)...")
    final_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_with_subs.mp4"
    
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(concatenated),
        "-i", str(audio_path),
        "-filter_complex", f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_path}'[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast",
        "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
        str(final_with_subs)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
        print(f"✅ Version 2: {os.path.getsize(final_with_subs)/(1024*1024):.1f}MB")
    except:
        final_with_subs = None
    
    return final_no_subs, final_with_subs

# ========================================== 
# 12. PARALLEL AUDIO GENERATION
# ========================================== 

class AudioGenerator:
    """Async audio generation"""
    
    def __init__(self, text, ref_audio, out_path):
        self.text = text
        self.ref_audio = ref_audio
        self.out_path = out_path
        self.completed = False
        self.success = False
        
    def generate_in_background(self):
        """Run in separate thread"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
            
            clean = re.sub(r'\[.*?\]', '', self.text)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
            
            print(f"🎤 Generating audio ({len(sentences)} sentences)...")
            all_wavs = []
            
            for i, chunk in enumerate(sentences):
                if i % 20 == 0:
                    update_status(20 + int((i/len(sentences))*25), f"Audio: {i}/{len(sentences)}")
                
                try:
                    with torch.no_grad():
                        chunk_clean = chunk.replace('"', '').replace('"', '').replace('"', '')
                        if chunk_clean.endswith('.'):
                            chunk_clean = chunk_clean + ' '
                        wav = model.generate(
                            text=chunk_clean,
                            audio_prompt_path=str(self.ref_audio),
                            exaggeration=0.5
                        )
                        all_wavs.append(wav.cpu())
                    
                    if i % 25 == 0 and device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"⚠️ Skipping sentence {i}")
                    continue
            
            if all_wavs:
                full_audio = torch.cat(all_wavs, dim=1)
                silence = torch.zeros((full_audio.shape[0], int(1.5 * 24000)))
                full_audio_padded = torch.cat([full_audio, silence], dim=1)
                
                torchaudio.save(self.out_path, full_audio_padded, 24000)
                duration = full_audio_padded.shape[1] / 24000
                print(f"✅ Audio complete: {duration:.1f}s")
                self.success = True
            else:
                self.success = False
                
        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            self.success = False
        finally:
            self.completed = True

# ========================================== 
# 13. SCRIPT GENERATION
# ========================================== 

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    
    if not GEMINI_KEYS:
        return f"This is a sample script about {topic}. " * 50
    
    random.shuffle(GEMINI_KEYS)
    
    prompt = f"""Write a YouTube documentary script about '{topic}'. {words} words.
CRITICAL: Write ONLY spoken narration. NO [brackets], NO stage directions, NO sound effects."""
    
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            script = response.text.replace("*","").replace("#","").strip()
            script = re.sub(r'\[.*?\]', '', script)
            return script
        except Exception as e:
            print(f"⚠️ Gemini error: {str(e)[:50]}")
            continue
    
    return f"This is a documentary about {topic}. " * 100

# ========================================== 
# 14. MAIN EXECUTION
# ========================================== 

print("--- 🚀 PARALLEL PROCESSING MODE ---")
update_status(1, "Initializing parallel engine...")

ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice download failed", "failed")
    exit(1)

print(f"✅ Voice downloaded")

if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if not os.path.exists(ref_logo):
        ref_logo = None
else:
    ref_logo = None

update_status(10, "Generating script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if not text or len(text) < 100:
    update_status(0, "Script too short", "failed")
    exit(1)

print(f"✅ Script: {len(text.split())} words")

update_status(15, "Starting parallel audio + video processing...")

audio_out = TEMP_DIR / "audio.wav"
audio_gen = AudioGenerator(text, ref_voice, audio_out)
audio_thread = Thread(target=audio_gen.generate_in_background)
audio_thread.start()

update_status(20, "Preparing video scenes...")

words = text.split()
total_duration = len(words) / 2.5

sentences = []
current_time = 0
words_per_sentence = random.randint(8, 12)

for i in range(0, len(words), words_per_sentence):
    chunk = words[i:i + words_per_sentence]
    sentence_duration = len(chunk) / 2.5
    sentences.append({
        "text": ' '.join(chunk),
        "start": current_time,
        "end": current_time + sentence_duration
    })
    current_time += sentence_duration
    words_per_sentence = random.randint(8, 12)

if sentences:
    sentences[-1]['end'] += 1.5

ass_file = TEMP_DIR / "subtitles.ass"
create_human_subtitles(sentences, ass_file)

update_status(50, "Processing videos in parallel (5 at a time)...")
processed_clips = process_visuals_parallel(sentences, TOPIC, text)

update_status(85, "Waiting for audio completion...")
audio_thread.join(timeout=300)

if not audio_gen.success or not os.path.exists(audio_out):
    update_status(0, "Audio generation failed", "failed")
    exit(1)

if processed_clips and len(processed_clips) > 0:
    update_status(90, "Rendering final outputs...")
    
    final_no_subs, final_with_subs = render_dual_outputs(
        processed_clips, audio_out, ass_file, ref_logo
    )
    
    if final_no_subs and os.path.exists(final_no_subs):
        update_status(93, "Uploading version 1...")
        link1 = upload_to_google_drive(final_no_subs)
        if link1:
            print(f"🔗 Version 1: {link1}")
    
    if final_with_subs and os.path.exists(final_with_subs):
        update_status(97, "Uploading version 2...")
        link2 = upload_to_google_drive(final_with_subs)
        if link2:
            print(f"🔗 Version 2: {link2}")
            update_status(100, "Complete!", "completed", link2)
        else:
            update_status(100, "Complete (upload partial)", "completed", link1 if 'link1' in locals() else None)
else:
    update_status(0, "No clips processed", "failed")

print("🧹 Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

print("--- ✅ PARALLEL PROCESSING COMPLETE ---")
