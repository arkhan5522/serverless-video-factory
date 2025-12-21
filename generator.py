"""
AI VIDEO GENERATOR WITH REALISM UPGRADES
========================================
FIXED VERSION:
1. Fixed Category Lock System
2. Cinematic Scene Engine
3. Stock Video Clustering
4. Fake Camera Movement
5. Color & Lighting Unification
6. Two Output Versions (with/without subtitles)
7. Enhanced Audio Realism
8. Human-like Subtitles
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

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    # Install only essential packages
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

# ========================================== 
# 3. LOGGING & STATUS SYSTEM
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Updates status.json in GitHub repo so HTML can read it"""
    
    # 1. Print to Kaggle Console
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(f"--- {progress}% | {message} ---")
    
    # 2. Add to Log Buffer (Keep last 30 lines)
    LOG_BUFFER.append(log_entry)
    if len(LOG_BUFFER) > 30:
        LOG_BUFFER.pop(0)

    # 3. Get GitHub Credentials
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    
    # If running locally without secrets, stop here
    if not repo or not token: 
        return

    # 4. Prepare Data for HTML
    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    
    data = {
        "progress": progress,
        "message": message,
        "status": status,
        "logs": "\n".join(LOG_BUFFER), # Send logs to HTML
        "timestamp": time.time()
    }
    
    # If we have a Google Drive link, send it
    if file_url: 
        data["file_io_url"] = file_url
    
    # 5. Send to GitHub API
    import base64
    content_json = json.dumps(data)
    content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # Step A: Get existing file SHA (required to update a file)
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        
        # Step B: Upload new status
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
    """Format seconds to ASS timestamp (H:MM:SS.CS)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def create_human_subtitles(sentences, ass_file):
    """Create human-like subtitles with proper timing"""
    
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
        
        # Styles
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        # Events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            # Human-like timing: appear slightly before, disappear before end
            start_time = max(0, s['start'] - 0.15)  # Appear 150ms early
            end_time = s['end'] - 0.3  # Disappear 300ms early
            
            text = s['text'].strip()
            text = re.sub(r'[\[\]]', '', text)
            
            # Human reading speed: 3-4 words per line
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
            
            # Uppercase for certain styles
            if style_key in ["mrbeast_yellow", "hormozi_green", "tiktok_white"]:
                lines = [line.upper() for line in lines]
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{formatted_text}\n")

# ========================================== 
# 5. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Uploads using OAuth 2.0 Refresh Token"""
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
    
    # Get Access Token
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
    
    print(f"☁️ Uploading {filename} ({file_size / (1024*1024):.1f} MB)...")
    with open(file_path, "rb") as f:
        upload_headers = {"Content-Length": str(file_size)}
        upload_resp = requests.put(session_uri, headers=upload_headers, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_data = upload_resp.json()
        file_id = file_data.get('id')
        print(f"✅ Upload Success! File ID: {file_id}")
        
        # Make public
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
    """Converts script into cinematic beats with emotional mapping"""
    
    EMOTION_TO_VISUAL = {
        # Emotional states
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
            "motion": "slow_pan",
            "keywords": ["discover", "found", "secret", "learned", "realized"]
        },
        "excitement": {
            "mood": "energetic",
            "camera": "dynamic",
            "color": "vibrant",
            "motion": "quick_cuts",
            "keywords": ["amazing", "incredible", "awesome", "boom", "wow"]
        },
        "explanation": {
            "mood": "professional",
            "camera": "stable",
            "color": "balanced",
            "motion": "still",
            "keywords": ["because", "reason", "explain", "shows", "demonstrates"]
        },
        "warning": {
            "mood": "tense",
            "camera": "close_up",
            "color": "cold",
            "motion": "static",
            "keywords": ["danger", "warning", "careful", "avoid", "mistake"]
        },
        "success": {
            "mood": "triumphant",
            "camera": "wide",
            "color": "golden_hour",
            "motion": "slow_rise",
            "keywords": ["success", "win", "achieve", "result", "growth"]
        },
        "calm": {
            "mood": "peaceful",
            "camera": "smooth",
            "color": "soft",
            "motion": "gentle_pan",
            "keywords": ["peace", "calm", "quiet", "still", "relax"]
        }
    }
    
    VISUAL_INTENT_TYPES = {
        "explanation": ["hands typing", "laptop screen", "whiteboard", "diagram", "flowchart"],
        "story": ["person walking", "city street", "looking thoughtful", "coffee shop", "window"],
        "data": ["charts growing", "graphs animation", "numbers increasing", "statistics", "dashboard"],
        "concept": ["abstract shapes", "particles floating", "light trails", "neural network", "connections"],
        "action": ["person working", "building something", "team collaboration", "construction", "progress"],
        "result": ["celebration", "high five", "trophy", "finished project", "before after"]
    }
    
    @staticmethod
    def analyze_sentence(sentence):
        """Analyze sentence for emotional content and visual intent"""
        sentence_lower = sentence.lower()
        
        # Detect emotion
        detected_emotions = []
        for emotion, data in CinematicSceneEngine.EMOTION_TO_VISUAL.items():
            for keyword in data["keywords"]:
                if keyword in sentence_lower:
                    detected_emotions.append(emotion)
                    break
        
        # Default to explanation if no emotion detected
        if not detected_emotions:
            detected_emotions = ["explanation"]
        
        # Detect visual intent
        detected_intents = []
        for intent, keywords in CinematicSceneEngine.VISUAL_INTENT_TYPES.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    detected_intents.append(intent)
                    break
        
        # Default intent
        if not detected_intents:
            if len(sentence.split()) > 15:
                detected_intents = ["explanation"]
            else:
                detected_intents = ["story"]
        
        # Get primary emotion/intent
        primary_emotion = detected_emotions[0]
        primary_intent = detected_intents[0]
        
        return {
            "emotion": primary_emotion,
            "intent": primary_intent,
            "visual_style": CinematicSceneEngine.EMOTION_TO_VISUAL[primary_emotion],
            "duration_factor": 1.0 if primary_emotion in ["explanation", "calm"] else 0.8,
            "pace": "slow" if primary_emotion in ["calm", "explanation"] else "normal"
        }

# ========================================== 
# 7. STOCK VIDEO SEARCH
# ========================================== 

USED_VIDEO_URLS = set()

def intelligent_video_search(query, service, keys, page=1):
    """Search for videos using Pexels or Pixabay"""
    all_results = []
    
    if service == 'pexels' and keys and keys[0]:
        try:
            key = random.choice([k for k in keys if k])
            print(f"    Searching Pexels: {query}")
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 20,
                "page": page,
                "orientation": "landscape",
                "size": "medium_large"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    video_files = video.get('video_files', [])
                    if video_files:
                        hd_files = [f for f in video_files if f.get('quality') == 'hd' and f.get('width', 0) >= 1280]
                        large_files = [f for f in video_files if f.get('quality') == 'large']
                        medium_files = [f for f in video_files if f.get('quality') == 'medium']
                        
                        best_file = None
                        if hd_files:
                            best_file = random.choice(hd_files)
                        elif large_files:
                            best_file = random.choice(large_files)
                        elif medium_files:
                            best_file = random.choice(medium_files)
                        
                        if best_file:
                            all_results.append({
                                'url': best_file['link'],
                                'title': video.get('user', {}).get('name', query),
                                'description': f"Pexels video by {video.get('user', {}).get('name', '')}",
                                'duration': video.get('duration', 0),
                                'service': 'pexels',
                                'quality': best_file.get('quality', 'medium'),
                                'width': best_file.get('width', 0),
                                'height': best_file.get('height', 0),
                                'license': 'free'
                            })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    elif service == 'pixabay' and keys and keys[0]:
        try:
            key = random.choice([k for k in keys if k])
            print(f"    Searching Pixabay: {query}")
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 20,
                "page": page,
                "orientation": "horizontal",
                "video_type": "film",
                "min_width": 1280
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    # Check landscape
                    width = videos_dict.get('large', {}).get('width', 0)
                    height = videos_dict.get('large', {}).get('height', 0)
                    
                    if height > width:
                        continue
                    
                    best_quality = None
                    for quality in ['large', 'medium', 'small']:
                        if quality in videos_dict:
                            best_quality = videos_dict[quality]
                            break
                    
                    if best_quality:
                        all_results.append({
                            'url': best_quality['url'],
                            'title': video.get('tags', query),
                            'description': f"Pixabay video ID: {video.get('id', '')}",
                            'duration': video.get('duration', 0),
                            'service': 'pixabay',
                            'quality': quality,
                            'width': best_quality.get('width', 0),
                            'height': best_quality.get('height', 0),
                            'license': 'free'
                        })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

# ========================================== 
# 8. CATEGORY ANALYSIS
# ========================================== 

def analyze_topic_for_category(topic, script):
    """Intelligently determine the REAL category from topic and script"""
    
    # Topic-based category mapping
    topic_category_map = {
        # AI & Tech
        "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
        "machine learning": "artificial intelligence", "deep learning": "artificial intelligence",
        "neural network": "artificial intelligence", "chatgpt": "artificial intelligence",
        "openai": "artificial intelligence", "llm": "artificial intelligence",
        "technology": "technology", "tech": "technology",
        "programming": "software", "coding": "software",
        "software": "software", "developer": "software",
        
        # Business & Finance
        "business": "business", "entrepreneur": "business",
        "startup": "business", "company": "business",
        "finance": "finance", "money": "finance",
        "invest": "finance", "stock": "finance",
        "trading": "finance", "wealth": "finance",
        
        # Motivation & Personal Development
        "motivation": "people", "motivational": "people",
        "inspire": "people", "inspirational": "people",
        "success": "people", "achieve": "people",
        "goal": "people", "dream": "people",
        "mindset": "people", "personal development": "people",
        "self improvement": "people", "growth": "people",
        
        # Science & Education
        "science": "science", "scientific": "science",
        "research": "science", "discovery": "science",
        "education": "education", "learn": "education",
        "study": "education", "knowledge": "education",
        
        # Health & Fitness
        "health": "health", "fitness": "health",
        "exercise": "health", "workout": "health",
        "diet": "health", "nutrition": "health",
        
        # Nature & Environment
        "nature": "nature", "environment": "nature",
        "climate": "nature", "earth": "nature",
        "animal": "animal", "wildlife": "animal",
        
        # Creative
        "art": "art", "creative": "art",
        "design": "art", "music": "music",
        "film": "movie", "photography": "photo"
    }
    
    # Analyze topic first
    topic_lower = topic.lower()
    script_lower = script.lower()[:2000]  # First 2000 chars
    
    # Check for exact matches in topic
    for keyword, category in topic_category_map.items():
        if keyword in topic_lower:
            print(f"✅ Category determined by topic keyword '{keyword}': {category}")
            return category, [category, keyword]
    
    # Check for topic phrases
    topic_words = topic_lower.split()
    for i in range(len(topic_words) - 1):
        phrase = f"{topic_words[i]} {topic_words[i+1]}"
        if phrase in topic_category_map:
            category = topic_category_map[phrase]
            print(f"✅ Category determined by topic phrase '{phrase}': {category}")
            return category, [category, phrase]
    
    # Analyze script content
    script_words = re.findall(r'\b\w{4,}\b', script_lower)
    word_freq = defaultdict(int)
    
    for word in script_words:
        word_freq[word] += 1
    
    # Check script words against category map
    for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
        if word in topic_category_map:
            category = topic_category_map[word]
            print(f"✅ Category determined by script word '{word}' (appears {count} times): {category}")
            return category, [category, word]
    
    # Default fallback based on common content types
    if any(word in script_lower for word in ["money", "profit", "revenue", "business", "invest"]):
        return "finance", ["finance", "business"]
    elif any(word in script_lower for word in ["ai", "machine", "algorithm", "data", "tech"]):
        return "artificial intelligence", ["artificial intelligence", "technology"]
    elif any(word in script_lower for word in ["mindset", "success", "goal", "achieve", "motivation"]):
        return "people", ["people", "motivation"]
    else:
        return "technology", ["technology", "digital"]

# ========================================== 
# 9. CATEGORY-FILTERED QUERIES
# ========================================== 

def get_category_filtered_queries(category, scene_analysis, sentence_text):
    """Get queries filtered by category and enhanced by scene analysis"""
    
    # Expanded visual dictionary with category focus
    CATEGORY_FOCUSED_VISUALS = {
        "artificial intelligence": {
            "primary": ["neural network visualization", "AI processing", "data center", 
                       "machine learning algorithm", "digital brain", "robot assembly",
                       "hologram display", "circuit board", "coding screen"],
            "emotion_map": {
                "frustration": ["glitching screen", "error message", "broken circuit"],
                "curiosity": ["data visualization", "network connections", "discovery"],
                "excitement": ["breakthrough moment", "light speed data", "innovation"],
                "explanation": ["whiteboard diagram", "presentation screen", "teaching"],
                "success": ["achievement unlocked", "growth chart", "trophy"]
            }
        },
        "technology": {
            "primary": ["innovation lab", "tech startup", "silicon valley", 
                       "hardware engineering", "microchip", "quantum computer",
                       "server room", "circuit board", "digital interface"],
            "emotion_map": {
                "frustration": ["bug fixing", "system error", "broken device"],
                "curiosity": ["new invention", "experiment", "research"],
                "excitement": ["product launch", "breakthrough", "celebration"],
                "explanation": ["technical diagram", "instruction manual", "demo"],
                "success": ["finished product", "award ceremony", "team celebration"]
            }
        },
        "people": {
            "primary": ["person achieving goal", "personal growth journey", 
                       "success story", "motivational moment", "overcoming challenge",
                       "celebration victory", "team accomplishment", "inspirational scene"],
            "emotion_map": {
                "frustration": ["struggle moment", "facing obstacle", "difficult task"],
                "curiosity": ["learning new skill", "discovery moment", "exploration"],
                "excitement": ["breakthrough moment", "achievement", "celebration"],
                "explanation": ["teaching moment", "sharing knowledge", "mentoring"],
                "success": ["trophy moment", "graduation", "promotion"]
            }
        },
        "finance": {
            "primary": ["stock market charts", "financial growth", "money visualization",
                       "investment success", "wealth building", "economic growth"],
            "emotion_map": {
                "frustration": ["market crash", "financial loss", "debt"],
                "curiosity": ["market analysis", "research", "planning"],
                "excitement": ["bull market", "profit surge", "success"],
                "explanation": ["financial chart", "budget planning", "strategy"],
                "success": ["wealth achievement", "retirement goal", "financial freedom"]
            }
        },
        "business": {
            "primary": ["business meeting", "corporate success", "startup growth",
                       "entrepreneur journey", "office achievement", "team success"],
            "emotion_map": {
                "frustration": ["business failure", "rejection", "bankruptcy"],
                "curiosity": ["market research", "opportunity", "innovation"],
                "excitement": ["deal closing", "funding secured", "expansion"],
                "explanation": ["business plan", "strategy meeting", "presentation"],
                "success": ["ipo celebration", "award ceremony", "profit milestone"]
            }
        }
    }
    
    # Default category visuals
    category_data = CATEGORY_FOCUSED_VISUALS.get(category, CATEGORY_FOCUSED_VISUALS["technology"])
    
    # Get emotion-specific visuals
    emotion = scene_analysis.get("emotion", "explanation")
    emotion_visuals = category_data["emotion_map"].get(emotion, category_data["primary"])
    
    # Get intent-specific visuals
    intent = scene_analysis.get("intent", "explanation")
    if intent in CinematicSceneEngine.VISUAL_INTENT_TYPES:
        intent_visuals = CinematicSceneEngine.VISUAL_INTENT_TYPES[intent]
    else:
        intent_visuals = category_data["primary"]
    
    # Combine visuals
    all_visuals = list(set(category_data["primary"] + emotion_visuals + intent_visuals))
    
    # Extract keywords from sentence
    sentence_lower = sentence_text.lower()
    sentence_keywords = re.findall(r'\b\w{5,}\b', sentence_lower)[:3]
    
    # Build queries
    queries = []
    
    # Primary query: Category + emotion + intent visual
    primary_visual = random.choice(emotion_visuals)
    primary_query = f"{primary_visual} {category}"
    queries.append(f"{primary_query} 4k cinematic")
    
    # Secondary query: Intent visual + category
    if intent_visuals:
        secondary_visual = random.choice(intent_visuals)
        queries.append(f"{secondary_visual} {category} cinematic")
    
    # Tertiary query: Sentence keyword + category visual
    if sentence_keywords:
        for keyword in sentence_keywords:
            if len(keyword) > 4:
                queries.append(f"{keyword} {category} visual")
                break
    
    # Fallback queries
    for visual in random.sample(all_visuals, min(3, len(all_visuals))):
        if visual not in primary_query:
            queries.append(f"{visual} cinematic")
    
    # Ensure we have enough queries
    while len(queries) < 4:
        queries.append(f"{category} abstract cinematic")
    
    print(f"    🎬 Scene: {emotion.upper()} | Intent: {intent.upper()}")
    print(f"    📌 Queries: {queries[:3]}")
    
    return queries, emotion, intent

# ========================================== 
# 10. REALISM ENHANCEMENTS
# ========================================== 

class RealismEnhancer:
    """Adds cinematic realism to videos"""
    
    @staticmethod
    def add_camera_motion(input_path, output_path, motion_type="subtle_zoom"):
        """Add subtle fake camera movement"""
        motions = {
            "subtle_zoom": "zoompan=z='min(zoom+0.0003,1.03)':d=125:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
            "slow_pan_right": "zoompan=z=1:x='if(lte(on,1),0,x+1)':y='0':d=125",
            "gentle_float": "zoompan=z=1:x='iw/2-(iw/zoom/2)+sin(on/25)*5':y='ih/2-(ih/zoom/2)+cos(on/30)*3':d=125",
            "micro_rotation": "rotate=angle=0.1*sin(2*PI*t/10):ow=hypot(iw,ih):oh=ow",
            "subtle_shake": "crop=iw-10:ih-10:5+2*sin(2*PI*t):5+2*cos(2*PI*t)",
            "slow_motion": "setpts=2.0*PTS",
            "quick_cuts": "setpts=0.5*PTS",
            "still": "null"  # No motion
        }
        
        filter_complex = motions.get(motion_type, motions["subtle_zoom"])
        
        if filter_complex == "null":
            # Just copy if no motion
            shutil.copy(input_path, output_path)
            return True
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            # If filter fails, just copy
            shutil.copy(input_path, output_path)
            return False
    
    @staticmethod
    def apply_color_unification(input_path, output_path, style="cinematic"):
        """Apply consistent color grading"""
        color_luts = {
            "cinematic": "colorbalance=rs=0.05:gs=0:bs=-0.05,eq=saturation=1.05:contrast=1.05",
            "warm_contrast": "colorbalance=rs=0.1:gs=0.05:bs=-0.1,eq=saturation=1.1:contrast=1.1",
            "cool_moody": "colorbalance=rs=-0.05:gs=0:bs=0.05,eq=saturation=0.95:contrast=1.15",
            "golden_hour": "colorbalance=rs=0.15:gs=0.1:bs=-0.1,eq=saturation=1.15:brightness=0.02",
            "desaturated": "eq=saturation=0.8:contrast=1.1",
            "vibrant": "eq=saturation=1.2:contrast=1.05",
            "balanced": "eq=saturation=1.0:contrast=1.0",
            "soft": "eq=saturation=0.9:contrast=0.95:brightness=0.02"
        }
        
        lut = color_luts.get(style, color_luts["cinematic"])
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", lut,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            shutil.copy(input_path, output_path)
            return False

# ========================================== 
# 11. VISUAL PROCESSING WITH REALISM
# ========================================== 

def process_visuals_with_realism(sentences, audio_path, ass_file, logo_path, topic, full_script):
    """Process visuals with cinematic realism and category locking"""
    
    # Determine category ONCE
    category, category_keywords = analyze_topic_for_category(topic, full_script)
    print(f"🎯 VIDEO CATEGORY LOCKED: {category.upper()}")
    print(f"   Keywords: {', '.join(category_keywords)}")
    
    processed_clips = []
    
    for i, sent in enumerate(sentences):
        dur = max(3.5, sent['end'] - sent['start'])
        update_status(60 + int((i/len(sentences))*30), 
                     f"Scene {i+1}/{len(sentences)}: {category[:20]}...")
        
        # Analyze scene
        scene_analysis = CinematicSceneEngine.analyze_sentence(sent['text'])
        
        # Get category-filtered queries
        queries, emotion, intent = get_category_filtered_queries(
            category, scene_analysis, sent['text']
        )
        
        # Try each query
        video_found = False
        for query_idx, query in enumerate(queries):
            if video_found:
                break
                
            print(f"  🔍 Clip {i+1}: Query {query_idx+1} - '{query}'")
            
            # Search videos
            all_candidates = []
            
            # Search Pexels
            if PEXELS_KEYS and PEXELS_KEYS[0]:
                pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS)
                all_candidates.extend(pexels_results)
            
            # Search Pixabay
            if PIXABAY_KEYS and PIXABAY_KEYS[0]:
                pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS)
                all_candidates.extend(pixabay_results)
            
            # Remove duplicates and used URLs
            unique_candidates = []
            seen_urls = set()
            for vid in all_candidates:
                if vid['url'] not in USED_VIDEO_URLS and vid['url'] not in seen_urls:
                    seen_urls.add(vid['url'])
                    unique_candidates.append(vid)
            
            if unique_candidates:
                # Select random candidate
                best_video = random.choice(unique_candidates[:3])  # Pick from top 3
                USED_VIDEO_URLS.add(best_video['url'])
                
                print(f"    ✅ Selected: {best_video.get('service', 'unknown')} "
                      f"({best_video.get('duration', 0):.1f}s) - {emotion}/{intent}")
                
                # Download
                raw_path = TEMP_DIR / f"raw_{i}.mp4"
                try:
                    response = requests.get(best_video['url'], timeout=30, stream=True)
                    with open(raw_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Apply camera motion based on emotion
                    motion_enhanced = TEMP_DIR / f"motion_{i}.mp4"
                    motion_type = scene_analysis['visual_style']['motion']
                    RealismEnhancer.add_camera_motion(raw_path, motion_enhanced, motion_type)
                    
                    # Apply color unification
                    color_enhanced = TEMP_DIR / f"color_{i}.mp4"
                    color_style = scene_analysis['visual_style']['color']
                    RealismEnhancer.apply_color_unification(motion_enhanced, color_enhanced, color_style)
                    
                    # Trim to duration
                    final_clip = TEMP_DIR / f"clip_{i}.mp4"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(color_enhanced),
                        "-t", str(dur),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-b:v", "8M",
                        "-an",
                        str(final_clip)
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    processed_clips.append(str(final_clip))
                    video_found = True
                    break
                    
                except Exception as e:
                    print(f"    ✗ Download/processing failed: {str(e)[:50]}")
                    continue
        
        # Fallback gradient
        if not video_found:
            print(f"  ⚠️ Using category-themed gradient fallback")
            
            # Category-specific colors
            category_colors = {
                "artificial intelligence": ["0x0f3460:0x1a1a2e", "0x533483:0x16213e"],
                "technology": ["0x1a1a2e:0x0f3460", "0x16213e:0x533483"],
                "people": ["0x2c3e50:0x3498db", "0x2c3e50:0xe74c3c"],
                "finance": ["0x27ae60:0x2ecc71", "0x2980b9:0x3498db"],
                "business": ["0x2c3e50:0x34495e", "0x16a085:0x27ae60"],
                "science": ["0x1e3a5f:0x3498db", "0x2c3e50:0x1abc9c"],
                "education": ["0x2c3e50:0x3498db", "0x2980b9:0x1abc9c"],
                "health": ["0x27ae60:0x2ecc71", "0x16a085:0x1abc9c"],
                "nature": ["0x27ae60:0x2ecc71", "0x16a085:0x27ae60"],
                "art": ["0x8e44ad:0x9b59b6", "0x2c3e50:0x3498db"]
            }
            
            gradient = category_colors.get(category, ["0x1a1a2e:0x16213e"])[0]
            fallback_clip = TEMP_DIR / f"gradient_{i}.mp4"
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={dur}",
                "-vf", f"gradients=s=1920x1080:x0=0:y0=0:x1=1920:y1=1080:c0={gradient.split(':')[0]}:c1={gradient.split(':')[1]},fade=in:0:30,fade=out:st={dur-1}:d=1",
                "-c:v", "libx264",
                "-preset", "medium",
                "-t", str(dur),
                str(fallback_clip)
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processed_clips.append(str(fallback_clip))
    
    return processed_clips

# ========================================== 
# 12. DUAL OUTPUT RENDERER
# ========================================== 

def render_dual_outputs(processed_clips, audio_path, ass_file, logo_path):
    """Render two versions: with and without subtitles"""
    
    print("🎬 Concatenating all clips...")
    
    # Create concatenation list
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in processed_clips:
            if os.path.exists(clip):
                f.write(f"file '{clip}'\n")
    
    # Concatenate all clips
    concatenated = TEMP_DIR / "concatenated.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", "10M",
        "-an",
        str(concatenated)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Get audio duration
    audio_duration = None
    try:
        import wave
        with wave.open(str(audio_path), 'rb') as wav_file:
            audio_duration = wav_file.getnframes() / float(wav_file.getframerate())
    except:
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]
            audio_duration = float(subprocess.check_output(cmd).decode().strip())
        except:
            audio_duration = None
    
    # Render Version 1: WITHOUT subtitles (just audio + logo)
    print("🎬 Rendering version 1 (no subtitles)...")
    final_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_no_subs.mp4"
    
    base_cmd = [
        "ffmpeg", "-y",
        "-i", str(concatenated),
        "-i", str(audio_path),
    ]
    
    filter_complex = []
    map_cmds = []
    
    if logo_path and os.path.exists(logo_path):
        base_cmd.extend(["-i", str(logo_path)])
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[2:v]scale=200:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    else:
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v]"
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    
    full_cmd = base_cmd + ["-filter_complex", ";".join(filter_complex)] + map_cmds + [
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "256k"
    ]
    
    if audio_duration:
        full_cmd.extend(["-t", str(audio_duration)])
    
    full_cmd.append(str(final_no_subs))
    
    try:
        subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Version 1 rendered: {final_no_subs}")
    except Exception as e:
        print(f"❌ Version 1 failed: {e}")
        final_no_subs = None
    
    # Render Version 2: WITH subtitles
    print("🎬 Rendering version 2 (with subtitles)...")
    final_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_with_subs.mp4"
    
    # Escape ASS file path
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    base_cmd = [
        "ffmpeg", "-y",
        "-i", str(concatenated),
        "-i", str(audio_path),
    ]
    
    filter_complex = []
    map_cmds = []
    
    if logo_path and os.path.exists(logo_path):
        base_cmd.extend(["-i", str(logo_path)])
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[2:v]scale=200:-1[logo];"
            "[bg][logo]overlay=30:30[withlogo];"
            "[withlogo]subtitles='{ass_path}'[v]".format(ass_path=ass_path)
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    else:
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[bg]subtitles='{ass_path}'[v]".format(ass_path=ass_path)
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    
    full_cmd = base_cmd + ["-filter_complex", ";".join(filter_complex)] + map_cmds + [
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "256k"
    ]
    
    if audio_duration:
        full_cmd.extend(["-t", str(audio_duration)])
    
    full_cmd.append(str(final_with_subs))
    
    try:
        subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Version 2 rendered: {final_with_subs}")
    except Exception as e:
        print(f"❌ Version 2 failed: {e}")
        final_with_subs = None
    
    return final_no_subs, final_with_subs

# ========================================== 
# 13. AUDIO ENHANCEMENT
# ========================================== 

def enhance_audio_realism(audio_path, output_path):
    """Add subtle audio enhancements"""
    try:
        # Simple enhancement: normalize and add slight compression
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,compand=attacks=0.002:decays=0.050:points=-80/-80|-30/-10|0/0",
            "-ar", "44100",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ Audio enhanced")
        return True
    except Exception as e:
        print(f"⚠️ Audio enhancement failed: {e}")
        # Fallback: copy original
        shutil.copy(audio_path, output_path)
        return False

# ========================================== 
# 14. SCRIPT & AUDIO GENERATION
# ========================================== 

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    
    if not GEMINI_KEYS:
        print("❌ No Gemini API keys found")
        return f"This is a sample script about {topic}. " * 50
    
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
CRITICAL RULES:
- Write ONLY spoken narration text
- NO stage directions like [Music fades], [Intro], [Outro]
- NO sound effects descriptions
- NO [anything in brackets]
- Start directly with the content
- End directly with the conclusion
- Pure voiceover script only
"""
    
    prompt = f"{base_instructions}\nWrite a YouTube documentary script about '{topic}'. {words} words."
    
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            script = response.text.replace("*","").replace("#","").strip()
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?music.*?\)', '', script, flags=re.IGNORECASE)
            return script
        except Exception as e:
            print(f"⚠️ Gemini API error with key: {str(e)[:50]}")
            continue
    
    # Fallback
    return f"This is a documentary about {topic}. " * 100

def clone_voice_robust(text, ref_audio, out_path):
    print("🎤 Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        print(f"📝 Processing {len(sentences)} sentences...")
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            if i % 10 == 0:
                update_status(20 + int((i/len(sentences))*30), f"Voice Gen {i}/{len(sentences)}")
            
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
            print("❌ No audio generated")
            return False
        
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
        return False

# ========================================== 
# 15. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START (CINEMATIC REALISM EDITION) ---")
update_status(1, "Initializing cinematic engine...")

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
raw_audio = TEMP_DIR / "raw_audio.wav"
enhanced_audio = TEMP_DIR / "enhanced_audio.wav"

if clone_voice_robust(text, ref_voice, raw_audio):
    # Enhance audio
    update_status(25, "Enhancing audio realism...")
    enhance_audio_realism(raw_audio, enhanced_audio)
    audio_out = enhanced_audio if os.path.exists(enhanced_audio) else raw_audio
    
    update_status(30, "Creating human-like subtitles...")
    
    # Create sentence timing (simplified)
    words = text.split()
    total_duration = len(words) / 2.5  # Rough estimate: 2.5 words per second
    
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
    
    # Create ASS subtitles with human timing
    ass_file = TEMP_DIR / "subtitles.ass"
    create_human_subtitles(sentences, ass_file)
    
    # Process visuals with cinematic realism
    update_status(40, "Creating cinematic visuals...")
    processed_clips = process_visuals_with_realism(
        sentences, audio_out, ass_file, ref_logo, TOPIC, text
    )
    
    if processed_clips:
        update_status(80, "Rendering final videos...")
        
        # Render both versions
        final_no_subs, final_with_subs = render_dual_outputs(
            processed_clips, audio_out, ass_file, ref_logo
        )
        
        if final_no_subs and os.path.exists(final_no_subs):
            file_size = os.path.getsize(final_no_subs) / (1024 * 1024)
            print(f"✅ Version 1 (no subs): {file_size:.1f} MB")
            
            update_status(90, "Uploading Version 1 to Google Drive...")
            drive_link1 = upload_to_google_drive(final_no_subs)
            
            if drive_link1:
                print(f"🔗 Version 1 Link: {drive_link1}")
            else:
                print("⚠️ Version 1 upload failed")
        
        if final_with_subs and os.path.exists(final_with_subs):
            file_size = os.path.getsize(final_with_subs) / (1024 * 1024)
            print(f"✅ Version 2 (with subs): {file_size:.1f} MB")
            
            update_status(95, "Uploading Version 2 to Google Drive...")
            drive_link2 = upload_to_google_drive(final_with_subs)
            
            if drive_link2:
                print(f"🔗 Version 2 Link: {drive_link2}")
            
            # Update final status with both links
            update_status(100, "Both versions completed!", "completed", 
                         drive_link2 if drive_link2 else drive_link1)
        else:
            update_status(100, "One version completed", "completed", 
                         drive_link1 if drive_link1 else None)
    else:
        update_status(0, "Visual processing failed", "failed")
else:
    update_status(0, "Audio synthesis failed", "failed")

# Cleanup
print("🧹 Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["list.txt", "concat_list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ CINEMATIC PROCESS COMPLETE ---")
