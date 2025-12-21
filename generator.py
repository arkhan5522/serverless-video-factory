"""
AI VIDEO GENERATOR WITH PARALLEL PROCESSING
============================================
OPTIMIZATIONS:
1. Parallel video downloads (5 concurrent)
2. Parallel FFmpeg processing
3. Background audio generation
4. Smart caching and resource management
5. Islamic content filtering
6. Contextual query enhancement
7. All previous realism features maintained
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

# ISLAMIC CONTENT FILTERING
FORBIDDEN_KEYWORDS = {
    'sexual': ['nude', 'nudity', 'sexy', 'erotic', 'sexual', 'bikini', 'swimsuit', 'lingerie', 'porn'],
    'haram': ['alcohol', 'wine', 'beer', 'drunk', 'pork', 'bacon', 'gambling', 'casino'],
    'inappropriate': ['violence', 'blood', 'gore', 'weapon', 'gun', 'war', 'terror'],
    'immoral': ['drugs', 'marijuana', 'cocaine', 'heroin', 'smoking']
}

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
# 6. CINEMATIC SCENE ENGINE (ENHANCED)
# ========================================== 

class CinematicSceneEngine:
    EMOTION_TO_VISUAL = {
        "frustration": {
            "mood": "dark",
            "camera": "slow_motion",
            "color": "desaturated",
            "motion": "subtle_shake",
            "keywords": ["struggle", "failed", "problem", "hard", "difficult", "challenge", "obstacle"]
        },
        "curiosity": {
            "mood": "mysterious",
            "camera": "zoom_in",
            "color": "warm_contrast",
            "motion": "slow_pan_right",
            "keywords": ["discover", "found", "secret", "learned", "realized", "question", "mystery"]
        },
        "excitement": {
            "mood": "energetic",
            "camera": "dynamic",
            "color": "vibrant",
            "motion": "subtle_zoom",
            "keywords": ["amazing", "incredible", "awesome", "boom", "wow", "unbelievable", "breakthrough"]
        },
        "explanation": {
            "mood": "professional",
            "camera": "stable",
            "color": "balanced",
            "motion": "still",
            "keywords": ["because", "reason", "explain", "shows", "demonstrates", "means", "indicates"]
        },
        "success": {
            "mood": "triumphant",
            "camera": "wide",
            "color": "golden_hour",
            "motion": "subtle_zoom",
            "keywords": ["success", "win", "achieve", "result", "growth", "victory", "accomplish"]
        },
        "inspiration": {
            "mood": "uplifting",
            "camera": "rising",
            "color": "bright",
            "motion": "slow_rise",
            "keywords": ["inspire", "motivate", "encourage", "hope", "dream", "aspire", "ambition"]
        },
        "analysis": {
            "mood": "serious",
            "camera": "steady",
            "color": "cool",
            "motion": "slow_pan_left",
            "keywords": ["analyze", "study", "research", "examine", "investigate", "evaluate", "assess"]
        }
    }
    
    VISUAL_INTENT_TYPES = {
        "explanation": ["hands typing", "laptop screen", "whiteboard", "presentation", "diagram", "flowchart"],
        "story": ["person walking", "city street", "looking thoughtful", "journey", "path", "road"],
        "data": ["charts growing", "graphs animation", "statistics", "numbers", "metrics", "dashboard"],
        "concept": ["abstract shapes", "particles floating", "connections", "network", "idea", "lightbulb"],
        "action": ["person working", "team collaboration", "progress", "construction", "building", "creating"],
        "nature": ["landscape", "mountains", "ocean", "forest", "sunrise", "sky"],
        "technology": ["circuit board", "processor", "server room", "robotics", "innovation", "future"]
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
    
    @staticmethod
    def generate_contextual_queries(sentence, category, emotion, intent):
        """Generate highly contextual queries based on sentence content"""
        queries = []
        words = sentence.lower().split()[:6]
        key_terms = [w for w in words if len(w) > 3][:3]
        
        # Base queries
        queries.append(f"{category} {emotion} {intent}")
        queries.append(f"{' '.join(key_terms)} {category} cinematic")
        
        # Emotion-specific queries
        if emotion == "success":
            queries.extend([
                f"{category} achievement celebration",
                f"{category} victory moment",
                f"{category} goal accomplishment"
            ])
        elif emotion == "curiosity":
            queries.extend([
                f"{category} discovery reveal",
                f"{category} mystery solving",
                f"{category} learning process"
            ])
        elif emotion == "inspiration":
            queries.extend([
                f"{category} motivational moment",
                f"{category} uplifting scene",
                f"{category} hope optimism"
            ])
        
        # Intent-specific queries
        if intent == "data":
            queries.extend([
                f"{category} data visualization",
                f"{category} charts graphs",
                f"{category} statistics analysis"
            ])
        elif intent == "technology":
            queries.extend([
                f"{category} tech innovation",
                f"{category} digital future",
                f"{category} advanced technology"
            ])
        elif intent == "nature":
            queries.extend([
                f"{category} natural beauty",
                f"{category} peaceful landscape",
                f"{category} scenic view"
            ])
        
        # Remove duplicates and filter
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries[:8]  # Return top 8 queries

# ========================================== 
# 7. CONTENT FILTERING
# ========================================== 

def filter_islamic_content(query):
    """Filter out haram and inappropriate content from queries"""
    query_lower = query.lower()
    
    # Check for forbidden keywords
    for category, keywords in FORBIDDEN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                print(f"⚠️ Filtered query '{query}' (contains {category} keyword)")
                return None
    
    # Filter specific problematic patterns
    problematic_patterns = [
        r'bikini|swimsuit',
        r'sexy.*woman|woman.*sexy',
        r'alcohol|drunk|wine|beer',
        r'violence|blood|gore',
        r'drugs|marijuana'
    ]
    
    for pattern in problematic_patterns:
        if re.search(pattern, query_lower):
            print(f"⚠️ Filtered query '{query}' (matches forbidden pattern)")
            return None
    
    return query

def filter_video_results(results):
    """Filter video results for Islamic compliance"""
    if not results:
        return []
    
    filtered = []
    for result in results:
        # Check URL for problematic terms
        url_lower = result.get('url', '').lower()
        safe = True
        
        for category, keywords in FORBIDDEN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in url_lower:
                    safe = False
                    break
        
        if safe:
            filtered.append(result)
    
    return filtered

# ========================================== 
# 8. EXPANDED CATEGORIES
# ========================================== 

def analyze_topic_for_category(topic, script):
    """Enhanced category detection with more categories"""
    
    expanded_category_map = {
        # Technology & Digital
        "ai": "artificial intelligence", 
        "artificial intelligence": "artificial intelligence",
        "machine learning": "artificial intelligence",
        "deep learning": "artificial intelligence",
        "neural network": "artificial intelligence",
        "technology": "technology",
        "digital": "technology",
        "programming": "technology",
        "coding": "technology",
        "software": "technology",
        "computer": "technology",
        "internet": "technology",
        
        # Business & Finance
        "business": "business",
        "startup": "business",
        "entrepreneur": "business",
        "marketing": "business",
        "finance": "finance",
        "money": "finance",
        "investment": "finance",
        "stock": "finance",
        "crypto": "finance",
        "bitcoin": "finance",
        "wealth": "finance",
        
        # Education & Knowledge
        "education": "education",
        "learning": "education",
        "school": "education",
        "university": "education",
        "study": "education",
        "knowledge": "education",
        "science": "science",
        "physics": "science",
        "chemistry": "science",
        "biology": "science",
        "research": "science",
        
        # Lifestyle & Personal Development
        "motivation": "personal development",
        "success": "personal development",
        "self-improvement": "personal development",
        "productivity": "personal development",
        "habits": "personal development",
        "health": "health",
        "fitness": "health",
        "wellness": "health",
        "nutrition": "health",
        "mental health": "health",
        
        # Creativity & Arts
        "art": "creativity",
        "design": "creativity",
        "creative": "creativity",
        "music": "creativity",
        "photography": "creativity",
        "writing": "creativity",
        
        # Nature & Environment
        "nature": "nature",
        "environment": "nature",
        "climate": "nature",
        "sustainability": "nature",
        "animal": "nature",
        "wildlife": "nature",
        
        # Society & Culture
        "history": "society",
        "culture": "society",
        "philosophy": "society",
        "psychology": "society",
        "society": "society",
        "community": "society"
    }
    
    topic_lower = topic.lower()
    script_lower = script.lower()
    
    # Check topic first
    for keyword, category in expanded_category_map.items():
        if keyword in topic_lower:
            print(f"✅ Category determined from topic: {category}")
            return category, [category, keyword]
    
    # Check script content if topic not found
    word_freq = {}
    for word in script_lower.split():
        word = re.sub(r'[^\w\s]', '', word)
        if len(word) > 4:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    for word, freq in sorted_words[:10]:
        for keyword, category in expanded_category_map.items():
            if keyword in word.lower():
                print(f"✅ Category determined from script: {category} (keyword: {word})")
                return category, [category, keyword]
    
    print(f"⚠️ Using default category: technology")
    return "technology", ["technology", "digital"]

# ========================================== 
# 9. ENHANCED PARALLEL VIDEO SEARCH
# ========================================== 

def search_single_service(query, service, keys):
    """Search single service with Islamic filtering"""
    try:
        # Filter query first
        filtered_query = filter_islamic_content(query)
        if not filtered_query:
            return []
        
        if service == 'pexels' and keys and keys[0]:
            key = random.choice([k for k in keys if k])
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": filtered_query,
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
                
                # Apply Islamic filtering
                return filter_video_results(results)
                
        elif service == 'pixabay' and keys and keys[0]:
            key = random.choice([k for k in keys if k])
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": filtered_query,
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
                
                # Apply Islamic filtering
                return filter_video_results(results)
    except Exception as e:
        pass
    
    return []

def parallel_video_search(query):
    """Search both services in parallel with Islamic filtering"""
    filtered_query = filter_islamic_content(query)
    if not filtered_query:
        return []
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if PEXELS_KEYS and PEXELS_KEYS[0]:
            futures.append(executor.submit(search_single_service, filtered_query, 'pexels', PEXELS_KEYS))
        if PIXABAY_KEYS and PIXABAY_KEYS[0]:
            futures.append(executor.submit(search_single_service, filtered_query, 'pixabay', PIXABAY_KEYS))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result(timeout=10)
                all_results.extend(results)
            except:
                pass
    
    return all_results

# ========================================== 
# 10. ENHANCED VIDEO DOWNLOADER
# ========================================== 

def download_and_process_video(video_info, output_path, duration, scene_analysis):
    """Download and process a single video with better error handling"""
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
        
        # Apply visual style based on scene analysis
        filters = ["scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"]
        
        if scene_analysis["visual_style"]["color"] == "desaturated":
            filters.append("colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3")
        elif scene_analysis["visual_style"]["color"] == "vibrant":
            filters.append("eq=saturation=1.2:brightness=0.05")
        elif scene_analysis["visual_style"]["color"] == "warm_contrast":
            filters.append("colorbalance=rs=.1:gs=0:bs=-.1")
        
        filter_chain = ','.join(filters)
        
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_download),
            "-t", str(duration),
            "-vf", filter_chain,
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
# 11. ENHANCED VISUAL PROCESSING
# ========================================== 

def get_category_queries(category, scene_analysis, sentence_text):
    """Get highly contextual search queries for category"""
    
    EXPANDED_CATEGORY_VISUALS = {
        "artificial intelligence": [
            "neural network visualization", "AI processing data", "machine learning concept",
            "data visualization abstract", "artificial intelligence future", "digital brain network",
            "algorithm visualization", "smart technology innovation"
        ],
        "technology": [
            "innovation technology", "tech startup office", "digital interface screens",
            "coding programming computer", "future technology concept", "digital transformation",
            "smart devices connectivity", "tech innovation background"
        ],
        "business": [
            "business meeting success", "office team collaboration", "corporate success growth",
            "entrepreneur startup journey", "business strategy planning", "professional office environment",
            "successful business presentation", "team achievement celebration"
        ],
        "finance": [
            "stock market growth", "financial charts success", "money investment concept",
            "wealth accumulation strategy", "financial planning analysis", "economic growth visualization",
            "digital currency future", "investment portfolio management"
        ],
        "personal development": [
            "personal growth journey", "self improvement motivation", "success mindset development",
            "goal achievement progress", "life transformation story", "positive change inspiration",
            "habit formation process", "mindset shift breakthrough"
        ],
        "health": [
            "healthy lifestyle fitness", "wellness mental health", "nutrition exercise balance",
            "medical research innovation", "fitness motivation journey", "healthy living habits",
            "wellbeing mindfulness practice", "healthcare advancement"
        ],
        "education": [
            "learning education process", "knowledge acquisition journey", "study research academic",
            "classroom online learning", "educational technology innovation", "skill development training",
            "lifelong learning concept", "educational inspiration"
        ],
        "science": [
            "scientific research discovery", "laboratory experiment innovation", "space exploration technology",
            "nature science research", "physics chemistry biology", "scientific breakthrough moment",
            "research development progress", "science technology advancement"
        ],
        "creativity": [
            "creative art design", "innovation imagination concept", "artistic expression process",
            "design thinking creation", "creative problem solving", "artistic inspiration moment",
            "creative process workflow", "design innovation visualization"
        ],
        "nature": [
            "natural landscape beauty", "environment sustainability concept", "wildlife nature conservation",
            "peaceful nature scenery", "environmental protection effort", "nature photography landscape",
            "ecological balance harmony", "nature inspiration calm"
        ],
        "society": [
            "community culture diversity", "historical cultural heritage", "social innovation change",
            "human connection society", "cultural exchange understanding", "social progress development",
            "community building teamwork", "cultural diversity celebration"
        ]
    }
    
    # Get base visuals for category
    visuals = EXPANDED_CATEGORY_VISUALS.get(category, EXPANDED_CATEGORY_VISUALS["technology"])
    emotion = scene_analysis.get("emotion", "explanation")
    intent = scene_analysis.get("intent", "story")
    
    # Generate contextual queries
    queries = CinematicSceneEngine.generate_contextual_queries(
        sentence_text, category, emotion, intent
    )
    
    # Add high-quality generic queries as fallback
    if len(queries) < 4:
        queries.extend([
            f"{category} {emotion} cinematic 4k",
            f"{category} professional background",
            f"{intent} {category} visual",
            f"{category} documentary footage"
        ])
    
    # Ensure Islamic compliance
    filtered_queries = []
    for query in queries:
        filtered = filter_islamic_content(query)
        if filtered:
            filtered_queries.append(filtered)
    
    return filtered_queries[:6]  # Return top 6 filtered queries

def process_single_clip(i, sent, category):
    """Process a single video clip with enhanced contextual queries"""
    try:
        dur = max(3.5, sent['end'] - sent['start'])
        scene_analysis = CinematicSceneEngine.analyze_sentence(sent['text'])
        
        # Get highly contextual queries
        queries = get_category_queries(category, scene_analysis, sent['text'])
        
        print(f"    🔍 Clip {i+1} queries: {', '.join(queries[:3])}")
        
        # Try all queries until we find a video
        for query_idx, query in enumerate(queries):
            results = parallel_video_search(query)
            
            with URL_LOCK:
                available = [v for v in results if v['url'] not in USED_VIDEO_URLS]
            
            if available:
                # Prioritize by quality and relevance
                quality_order = ['hd', 'large', 'medium', 'small']
                selected_video = None
                
                for quality in quality_order:
                    quality_videos = [v for v in available if v.get('quality') == quality]
                    if quality_videos:
                        selected_video = random.choice(quality_videos)
                        break
                
                if not selected_video and available:
                    selected_video = random.choice(available[:3])
                
                with URL_LOCK:
                    USED_VIDEO_URLS.add(selected_video['url'])
                
                output_path = TEMP_DIR / f"clip_{i}.mp4"
                
                if download_and_process_video(selected_video, output_path, dur, scene_analysis):
                    print(f"    ✅ Clip {i+1} processed ({selected_video['service']} - {query[:40]}...)")
                    return str(output_path)
            
            print(f"    🔄 Clip {i+1} trying query {query_idx+1}/{len(queries)}: '{query[:50]}...'")
        
        # If all queries fail, try broader but safe searches
        print(f"    🔍 Clip {i+1} trying broader safe searches...")
        
        broader_safe_queries = [
            f"{category} professional",
            "cinematic footage 4k",
            "documentary background",
            "professional video background"
        ]
        
        # Filter for Islamic compliance
        safe_queries = []
        for query in broader_safe_queries:
            filtered = filter_islamic_content(query)
            if filtered:
                safe_queries.append(filtered)
        
        for query in safe_queries[:3]:
            results = parallel_video_search(query)
            
            with URL_LOCK:
                available = [v for v in results if v['url'] not in USED_VIDEO_URLS]
            
            if available:
                video = random.choice(available[:3])
                
                with URL_LOCK:
                    USED_VIDEO_URLS.add(video['url'])
                
                output_path = TEMP_DIR / f"clip_{i}.mp4"
                
                if download_and_process_video(video, output_path, dur, scene_analysis):
                    print(f"    ✅ Clip {i+1} processed (safe fallback: {query})")
                    return str(output_path)
        
        print(f"    ⚠️ Clip {i+1} could not find suitable video")
        return None
        
    except Exception as e:
        print(f"    ✗ Clip {i+1} failed: {str(e)[:40]}")
        return None

def process_visuals_parallel(sentences, topic, full_script):
    """Process all visuals in parallel with progress tracking"""
    category, _ = analyze_topic_for_category(topic, full_script)
    print(f"🎯 CATEGORY: {category.upper()}")
    
    processed_clips = []
    
    batch_size = 5
    total_batches = math.ceil(len(sentences) / batch_size)
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
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
                    result = future.result(timeout=60)
                    if result:
                        processed_clips.append(result)
                        print(f"    ✓ Clip {global_idx+1} completed successfully")
                    else:
                        print(f"    ⚠️ Clip {global_idx+1} skipped (no suitable video)")
                except Exception as e:
                    print(f"    ✗ Clip {global_idx+1} timed out or failed: {e}")
    
    print(f"✅ Processed {len(processed_clips)}/{len(sentences)} clips successfully")
    
    if len(processed_clips) < len(sentences) * 0.7:  # Less than 70% success
        print(f"⚠️ Warning: Only {len(processed_clips)}/{len(sentences)} clips processed")
    
    return processed_clips

# ========================================== 
# 12. DUAL OUTPUT RENDERER (ENHANCED)
# ========================================== 

# ========================================== 
# 12. DUAL OUTPUT RENDERER (FIXED FFMPEG)
# ========================================== 

def render_dual_outputs(processed_clips, audio_path, ass_file, logo_path):
    """Render two versions: with and without subtitles - FIXED FFMPEG"""
    
    print("🎬 Concatenating clips...")
    
    # Ensure we have clips
    if not processed_clips:
        print("❌ No clips to render")
        return None, None
    
    # Filter out None values
    valid_clips = [c for c in processed_clips if c and os.path.exists(c)]
    
    if not valid_clips:
        print("❌ No valid clips found")
        return None, None
    
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in valid_clips:
            f.write(f"file '{clip}'\n")
    
    concatenated = TEMP_DIR / "concatenated.mp4"
    
    # FIXED: Use proper FFmpeg settings from working version
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", "8M", "-an", str(concatenated)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
        print(f"✅ Concatenated {len(valid_clips)} clips")
    except subprocess.TimeoutExpired:
        print("⚠️ Concatenation timed out, trying simpler settings")
        # Try simpler concatenation
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(concatenated)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(concatenated) or os.path.getsize(concatenated) < 10000:
        print("❌ Concatenation failed")
        return None, None
    
    print("🎬 Rendering version 1 (no subtitles)...")
    final_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_no_subs.mp4"
    
    # Get audio duration for proper timing
    audio_duration = None
    try:
        cmd_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                       "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
        result = subprocess.run(cmd_duration, capture_output=True, text=True)
        if result.returncode == 0:
            audio_duration = float(result.stdout.strip())
            print(f"🎵 Audio duration: {audio_duration:.1f}s")
    except:
        pass
    
    # FIXED: Use same FFmpeg settings as working version
    if logo_path and os.path.exists(logo_path):
        # Version with logo
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[1:v]scale=230:-1[logo];"
            f"[bg][logo]overlay=30:30[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",  # Changed from fast to medium
            "-b:v", "10M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest"  # Ensure video matches audio length
        ]
    else:
        # Version without logo
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",  # Changed from fast to medium
            "-b:v", "10M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest"  # Ensure video matches audio length
        ]
    
    if audio_duration:
        cmd.extend(["-t", str(audio_duration)])
    
    cmd.append(str(final_no_subs))
    
    try:
        print(f"🚀 Rendering version 1...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print(f"✅ Version 1: {os.path.getsize(final_no_subs)/(1024*1024):.1f}MB")
        else:
            print(f"❌ Version 1 failed: {result.stderr[:200]}")
            final_no_subs = None
    except Exception as e:
        print(f"❌ Version 1 exception: {e}")
        final_no_subs = None
    
    print("🎬 Rendering version 2 (with subtitles)...")
    final_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_with_subs.mp4"
    
    # CRITICAL FIX: Properly escape ASS file path for FFmpeg
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        # Version with logo AND subtitles
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[1:v]scale=230:-1[logo];"
            f"[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles='{ass_path}':force_style='Fontsize=24'[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",  # Changed from fast to medium
            "-b:v", "10M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest"
        ]
    else:
        # Version with subtitles only
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[bg]subtitles='{ass_path}':force_style='Fontsize=24'[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",  # Changed from fast to medium
            "-b:v", "10M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest"
        ]
    
    if audio_duration:
        cmd.extend(["-t", str(audio_duration)])
    
    cmd.append(str(final_with_subs))
    
    try:
        print(f"🚀 Rendering version 2...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print(f"✅ Version 2: {os.path.getsize(final_with_subs)/(1024*1024):.1f}MB")
        else:
            print(f"❌ Version 2 failed: {result.stderr[:200]}")
            final_with_subs = None
    except Exception as e:
        print(f"❌ Version 2 exception: {e}")
        final_with_subs = None
    
    return final_no_subs, final_with_subs


# ========================================== 
# 13. MAIN EXECUTION (WITH FIXED CONTINUATION)
# ========================================== 

print("--- 🚀 PARALLEL PROCESSING MODE ---")
print("🔒 Islamic Content Filtering: ACTIVE")
update_status(1, "Initializing parallel engine...")

# Setup directories
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

# Download voice sample
if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice download failed", "failed")
    exit(1)

print(f"✅ Voice downloaded: {os.path.getsize(ref_voice)/(1024*1024):.1f} MB")

# Download logo if provided
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if not os.path.exists(ref_logo):
        ref_logo = None
        print("⚠️ Logo not found, continuing without logo")
else:
    ref_logo = None

# Generate or use script
update_status(10, "Generating script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if not text or len(text) < 100:
    update_status(0, "Script too short", "failed")
    exit(1)

print(f"✅ Script: {len(text.split())} words")

# Split script into timed sentences
update_status(15, "Preparing audio and video timeline...")

words = text.split()
total_duration = len(words) / 2.5  # Approximate speaking rate

sentences = []
current_time = 0
base_words_per_sentence = random.randint(8, 12)

for i in range(0, len(words), base_words_per_sentence):
    chunk = words[i:i + base_words_per_sentence]
    sentence_duration = len(chunk) / 2.5
    sentences.append({
        "text": ' '.join(chunk),
        "start": current_time,
        "end": current_time + sentence_duration
    })
    current_time += sentence_duration
    base_words_per_sentence = random.randint(8, 12)  # Vary sentence length

# Add ending pause
if sentences:
    sentences[-1]['end'] += 1.5

print(f"📊 Timeline: {len(sentences)} segments, {current_time:.1f} seconds total")

# Create subtitles
ass_file = TEMP_DIR / "subtitles.ass"
create_human_subtitles(sentences, ass_file)
print(f"✅ Subtitles created: {ass_file}")

# Start audio generation in background
update_status(20, "Starting parallel audio generation...")
audio_out = TEMP_DIR / "audio.wav"
audio_gen = AudioGenerator(text, ref_voice, audio_out)
audio_thread = Thread(target=audio_gen.generate_in_background)
audio_thread.start()

# Process videos in parallel while audio generates
update_status(25, "Starting parallel video processing...")
category, _ = analyze_topic_for_category(TOPIC, text)

# Start video processing
video_start_time = time.time()
processed_clips = process_visuals_parallel(sentences, TOPIC, text)
video_elapsed = time.time() - video_start_time
print(f"⏱️ Video processing took {video_elapsed:.1f} seconds")

# Wait for audio completion with timeout
update_status(85, "Waiting for audio completion...")
audio_thread.join(timeout=300)  # 5 minute timeout

if not audio_gen.completed:
    print("⚠️ Audio generation timed out, checking for partial output")
    
if not audio_gen.success:
    print(f"❌ Audio generation failed: {audio_gen.error}")
    # Try to continue with audio file if it exists
    if not os.path.exists(audio_out) or os.path.getsize(audio_out) < 10000:
        update_status(0, "Audio generation failed", "failed")
        exit(1)

# Verify audio file
if not os.path.exists(audio_out) or os.path.getsize(audio_out) < 10000:
    print("❌ Audio file not found or too small")
    update_status(0, "Audio file invalid", "failed")
    exit(1)

audio_duration = os.path.getsize(audio_out) / (24000 * 2)  # Approximate duration
print(f"✅ Audio ready: {audio_duration:.1f}s")

# Check if we have clips
if not processed_clips or len(processed_clips) == 0:
    print("❌ No video clips processed")
    update_status(0, "No video content", "failed")
    exit(1)

# Render final videos using FIXED FFmpeg settings
update_status(90, "Rendering final outputs...")
final_no_subs, final_with_subs = render_dual_outputs(
    processed_clips, audio_out, ass_file, ref_logo
)

# Upload to Google Drive
uploaded_links = []

if final_no_subs and os.path.exists(final_no_subs):
    update_status(93, "Uploading version 1 (no subtitles)...")
    link1 = upload_to_google_drive(final_no_subs)
    if link1:
        uploaded_links.append({"type": "no_subs", "url": link1})
        print(f"🔗 Version 1: {link1}")

if final_with_subs and os.path.exists(final_with_subs):
    update_status(97, "Uploading version 2 (with subtitles)...")
    link2 = upload_to_google_drive(final_with_subs)
    if link2:
        uploaded_links.append({"type": "with_subs", "url": link2})
        print(f"🔗 Version 2: {link2}")

# Final status update
if uploaded_links:
    final_url = uploaded_links[-1]["url"]  # Use last successful upload
    update_status(100, "Complete! Videos uploaded successfully.", "completed", final_url)
    print(f"🎉 Successfully uploaded {len(uploaded_links)} videos")
else:
    update_status(100, "Complete (no uploads)", "completed")
    print("⚠️ Processing complete but no videos were uploaded")

# Cleanup
print("🧹 Cleaning up temporary files...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
        print("✅ Temporary files cleaned")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

print("--- ✅ PARALLEL PROCESSING COMPLETE ---")
print(f"📊 Summary:")
print(f"   Script: {len(text.split())} words")
print(f"   Audio: {audio_duration:.1f}s")
print(f"   Clips: {len(processed_clips)} processed")
print(f"   Videos: {len(uploaded_links)} uploaded")
