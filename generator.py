"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
ENHANCED VERSION WITH T5 QUERIES, CLIP MATCHING & DUAL OUTPUT
FIXED VERSION:
1. T5 Transformer for Intelligent Search Query Generation
2. CLIP Model for Visual-Script Alignment (Exact Visual Matching)
3. Two-Stage Processing for Videos Without/With Subtitles
4. Stronger Ethical & Islamic Content Filters
5. Comprehensive Google Drive Upload for Both Files
6. Fixed Subtitle Design Implementation
7. Enhanced 100% Context-Aligned Scoring System
8. Pixabay & Pexels with Visual Map
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
import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel

# ========================================== 
# 0. INSTALLATION & SETUP
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    # Core dependencies
    core_libs = [
        "chatterbox-tts",
        "torchaudio", 
        "assemblyai",
        "google-generativeai",
        "requests",
        "beautifulsoup4",
        "pydub",
        "numpy",
        "pillow",
        "opencv-python",
        "transformers",
        "ftfy",
        "timm",
        "sentencepiece",
        "protobuf",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + core_libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

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
# 2. LOAD AI MODELS (T5 & CLIP)
# ========================================== 

print("--- 🤖 Loading AI Models for Search & Matching ---")

# Initialize T5 Model for Smart Query Generation
T5_TOKENIZER = None
T5_MODEL = None
try:
    T5_TOKENIZER = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    T5_MODEL = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
    print("✅ T5 Model loaded successfully")
except Exception as e:
    print(f"⚠️ T5 Model loading failed: {e}")

# Initialize CLIP Model for Visual Matching
CLIP_MODEL = None
CLIP_PROCESSOR = None
try:
    CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_MODEL.to(CLIP_DEVICE)
    print(f"✅ CLIP Model loaded successfully (Device: {CLIP_DEVICE})")
except Exception as e:
    print(f"⚠️ CLIP Model loading failed: {e}")

# ========================================== 
# 3. STRONG CONTENT FILTER (ISLAMIC SAFE)
# ========================================== 

STRONG_CONTENT_BLACKLIST = [
    # Islamic Prohibitions - Alcohol & Intoxicants
    'alcohol', 'wine', 'beer', 'liquor', 'whiskey', 'vodka', 'rum', 'gin',
    'drunk', 'intoxicated', 'intoxication', 'drinking', 'bar', 'pub',
    'cocktail', 'champagne', 'brewery', 'distillery',
    
    # Islamic Prohibitions - Nudity & Immodesty
    'nudity', 'nude', 'topless', 'bikini', 'swimsuit', 'lingerie', 'underwear',
    'bathing suit', 'swimwear', 'model', 'sexy', 'hot girl', 'erotic',
    'porn', 'xxx', 'adult', 'explicit', 'seductive',
    
    # Islamic Prohibitions - Haram Foods
    'pork', 'bacon', 'ham', 'sausage', 'pepperoni', 'salami',
    
    # Violence & Conflict
    'war', 'battle', 'gun', 'weapon', 'blood', 'gore', 'violence', 'fight',
    'terror', 'attack', 'murder', 'kill', 'shoot', 'bomb', 'explosion',
    'assault', 'combat',
    
    # Negative/Explicit Content
    'fashion model', 'catwalk', 'runway',
    'halloween', 'witch', 'ghost', 'demon', 'satan',
    'gambling', 'casino', 'poker', 'betting',
    
    # General Avoidance for Family Content
    'christmas', 'easter', 'valentine',  # Religious/cultural avoidance
    'horror', 'scary', 'fear'
]

def contains_prohibited_content(video_title, video_description):
    """Checks video metadata against the strong Islamic blacklist."""
    text = (video_title + ' ' + video_description).lower()
    for term in STRONG_CONTENT_BLACKLIST:
        if term in text:
            print(f"    🚫 BLOCKED: Found prohibited term '{term}'")
            return True
    return False

# ========================================== 
# 4. T5 SMART QUERY GENERATOR
# ========================================== 

def generate_smart_search_query(script_segment, fallback_topic):
    """
    Uses the T5 model to generate a contextually relevant, visual search query
    from a segment of the script.
    """
    # Fallback if T5 model not loaded
    if not T5_MODEL or not T5_TOKENIZER:
        # Use simple keyword extraction as fallback
        words = re.findall(r'\b\w{4,}\b', script_segment.lower())
        filtered_words = [w for w in words if w not in ['that', 'this', 'with', 'from', 'about']]
        if filtered_words:
            primary_query = f"{filtered_words[0]} {fallback_topic} 4k cinematic"
        else:
            primary_query = f"{fallback_topic} 4k cinematic"
        return primary_query, filtered_words[:3]
    
    # Prepare text: combine segment with topic for context
    input_text = f"{script_segment[:200]}. Topic: {fallback_topic}"
    inputs = T5_TOKENIZER([input_text], max_length=512, truncation=True, return_tensors="pt")

    try:
        # Generate tags (asking for 3-5 tags)
        with torch.no_grad():
            output = T5_MODEL.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )

        # Decode the result
        decoded_output = T5_TOKENIZER.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        # Form the primary query: Use the first tag + "4k cinematic"
        if tags:
            primary_query = f"{tags[0]} 4k cinematic"
        else:
            primary_query = f"{fallback_topic} 4k cinematic"
        
        return primary_query, tags
    except Exception as e:
        print(f"    ⚠️ T5 query generation failed: {e}")
        return f"{fallback_topic} 4k cinematic", [fallback_topic]

# ========================================== 
# 5. CLIP VISUAL MATCHING FUNCTIONS
# ========================================== 

def get_middle_frame(video_path):
    """Extracts a single RGB image from the middle of a video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
            
        middle_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"    ⚠️ Error extracting frame: {e}")
    return None

def rank_videos_by_clip_match(script_sentence, downloaded_video_paths):
    """
    Uses CLIP to rank a list of downloaded videos by how well their middle frame
    matches the script sentence. Returns the path of the best match.
    """
    if not downloaded_video_paths or not CLIP_MODEL:
        if downloaded_video_paths:
            return downloaded_video_paths[0]  # Fallback to first video
        return None

    images = []
    valid_paths = []
    
    print(f"    🤔 CLIP Ranking {len(downloaded_video_paths)} candidate videos...")

    # Extract middle frame from each video
    for vid_path in downloaded_video_paths:
        frame = get_middle_frame(vid_path)
        if frame is not None:
            try:
                pil_image = Image.fromarray(frame)
                images.append(pil_image)
                valid_paths.append(vid_path)
            except Exception as e:
                print(f"    ⚠️ Error converting frame: {e}")
                continue

    if not images:
        if downloaded_video_paths:
            return downloaded_video_paths[0]
        return None

    try:
        # Run CLIP comparison (Text vs Images)
        with torch.no_grad():
            inputs = CLIP_PROCESSOR(
                text=[script_sentence[:200]],  # Truncate long sentences
                images=images, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}
            outputs = CLIP_MODEL(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0)
            best_idx = probs.argmax().item()

        best_video = valid_paths[best_idx]
        confidence = probs[best_idx].item() * 100
        print(f"    ✅ Best CLIP Match: {os.path.basename(best_video)} (Confidence: {confidence:.1f}%)")
        return best_video
    except Exception as e:
        print(f"    ⚠️ CLIP matching failed: {e}")
        if valid_paths:
            return valid_paths[0]  # Fallback
        return None

# ========================================== 
# 6. FIXED SUBTITLE STYLES
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

def create_ass_file(sentences, ass_file):
    """FIXED: Create ASS subtitle file with proper format encoding"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']} (Size: {style['fontsize']}px)")
    
    # CRITICAL FIX: Use UTF-8-BOM encoding for proper ASS rendering
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
        
        # FIXED: Proper style formatting with all parameters
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
            if text.endswith('.'):
                text = text[:-1]
            if text.endswith(','):
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
# 7. GOOGLE DRIVE UPLOAD
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
            perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
            try:
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
# 8. VISUAL DICTIONARY (700+ TOPICS)
# ========================================== 

VISUAL_MAP = {
    # TECH & AI
    "tech": ["server room", "circuit board", "hologram display", "robot assembly", "coding screen", "data center", "fiber optics", "microchip manufacturing"],
    "technology": ["innovation lab", "tech startup", "silicon valley", "hardware engineering", "semiconductor", "quantum computer"],
    "ai": ["artificial intelligence", "neural network visualization", "machine learning", "deep learning", "robot face", "digital brain", "AI processing"],
    "artificial intelligence": ["neural pathways", "AI algorithm", "computer vision", "natural language processing"],
    "robot": ["humanoid robot", "industrial robot", "robot arm", "android", "automated warehouse", "robot manufacturing"],
    "automation": ["automated factory", "robotic assembly line", "conveyor belt", "industrial automation"],
    "computer": ["workstation setup", "gaming PC", "laptop coding", "computer lab", "PC building"],
    "software": ["code editor", "programming", "software development", "agile team", "debugging"],
    "coding": ["python code", "javascript", "github", "IDE interface", "terminal commands"],
    "programming": ["algorithm visualization", "code review", "pair programming", "hackathon"],
    "data": ["big data visualization", "database", "data analytics dashboard", "data mining", "cloud storage"],
    "database": ["SQL query", "server racks", "data warehouse", "NoSQL"],
    "cloud": ["cloud computing", "AWS datacenter", "virtual machines", "cloud storage"],
    "internet": ["fiber optic cables", "network switches", "wifi signal", "5G tower", "undersea cable"],
    "network": ["mesh network", "router", "ethernet cables", "network topology"],
    "cyber": ["cybersecurity", "firewall", "encryption", "security operations center"],
    "security": ["lock and key", "biometric scanner", "security camera", "access control"],
    "hacker": ["hoodie hacker", "matrix code", "dark web", "penetration testing"],
    "crypto": ["bitcoin mining", "blockchain", "ethereum", "cryptocurrency exchange", "digital wallet", "crypto trading"],
    "blockchain": ["distributed ledger", "smart contract", "crypto nodes", "decentralized network"],
    "bitcoin": ["bitcoin logo", "crypto mining rig", "bitcoin transaction", "BTC chart"],
    "nft": ["digital art", "NFT marketplace", "crypto art", "metaverse"],
    "quantum": ["quantum computing", "quantum mechanics", "quantum physics", "subatomic particles"],
    "algorithm": ["algorithm visualization", "code algorithm", "sorting algorithm", "AI algorithm"],
    "server": ["server room", "data server", "server rack", "cloud server"],
    "virtual": ["virtual reality", "VR headset", "virtual world", "VR gaming"],
    "metaverse": ["metaverse concept", "digital universe", "virtual metaverse"],
    
    # SCIENCE & SPACE
    "science": ["laboratory", "scientist research", "microscope", "chemical reaction", "petri dish", "beaker", "scientific equipment"],
    "research": ["research lab", "data analysis", "experiment", "clinical trial"],
    "chemistry": ["chemical formula", "periodic table", "molecule structure", "chemical lab"],
    "physics": ["particle accelerator", "quantum mechanics", "physics experiment", "Newton's cradle"],
    "biology": ["DNA helix", "cell division", "microscopic organisms", "genetics lab", "human anatomy"],
    "dna": ["double helix", "gene editing", "DNA sequence", "genetic code"],
    "medicine": ["hospital", "surgery", "medical equipment", "doctor examining", "pharmaceutical lab"],
    "health": ["fitness tracker", "healthy food", "yoga", "meditation", "hospital ward"],
    "space": ["galaxy stars", "planet earth from space", "astronaut floating", "black hole", "nebula", "space station", "rocket launch"],
    "astronomy": ["telescope", "observatory", "star field", "planetary system"],
    "nasa": ["mission control", "space shuttle", "mars rover", "ISS"],
    "planet": ["earth rotation", "mars surface", "jupiter storm", "saturn rings"],
    "universe": ["cosmic web", "expanding universe", "multiverse concept", "big bang"],
    "satellite": ["satellite orbit", "GPS satellites", "communication satellite"],
    "telescope": ["space telescope", "observatory telescope", "astronomical observation"],
    "microscope": ["electron microscope", "lab microscope", "microscopy"],
    "experiment": ["scientific experiment", "lab experiment", "research experiment"],
    "laboratory": ["research lab", "science lab", "lab equipment", "laboratory research"],
    "chemical": ["chemical reaction", "chemical formula", "chemistry lab"],
    "molecule": ["molecular structure", "3D molecule", "molecular biology"],
    "atom": ["atomic structure", "atom model", "atomic particles"],
    "particle": ["particle collision", "subatomic particle", "particle physics"],
    "gravity": ["gravity illustration", "gravitational force", "zero gravity"],
    "magnetism": ["magnetic field", "magnet", "electromagnetic"],
    "electricity": ["electrical current", "electric power", "lightning strike"],
    "radiation": ["radiation waves", "electromagnetic radiation"],
    "laser": ["laser beam", "laser light", "laser technology"],
    
    # BUSINESS & FINANCE
    "business": ["business meeting", "handshake deal", "office skyscraper", "corporate presentation", "team collaboration"],
    "company": ["startup office", "company headquarters", "corporate culture", "board meeting"],
    "entrepreneur": ["startup founder", "pitch presentation", "business plan", "entrepreneur working"],
    "startup": ["startup workspace", "brainstorming session", "product launch", "venture capital"],
    "office": ["modern office", "cubicles", "open workspace", "coworking space"],
    "meeting": ["conference room", "video call", "team discussion", "presentation"],
    "money": ["cash counting", "gold bars", "falling coins", "paper money", "wealth", "safe full of money"],
    "finance": ["stock market", "financial district", "trading floor", "financial charts"],
    "bank": ["bank vault", "banking hall", "ATM", "investment banking"],
    "investment": ["stock portfolio", "investment chart", "bull market", "dividend"],
    "stock": ["stock ticker", "trading screen", "NYSE floor", "candlestick chart"],
    "trading": ["day trader", "forex trading", "commodity trading", "options trading"],
    "economy": ["economic growth", "GDP chart", "inflation graph", "economic indicators"],
    "market": ["financial market", "market analysis", "bull and bear", "market volatility"],
    "success": ["mountain peak success", "trophy", "celebration", "achievement", "victory", "winning"],
    "growth": ["plant growing timelapse", "chart rising", "skyscraper construction", "upward trend"],
    "profit": ["profit margin", "revenue growth", "earnings report", "profit chart"],
    "wealth": ["luxury mansion", "private yacht", "expensive cars", "luxury lifestyle"],
    "ceo": ["ceo portrait", "chief executive", "business leader"],
    "manager": ["team manager", "project manager", "management"],
    "executive": ["business executive", "corporate executive", "c-suite"],
    "board": ["board meeting", "board of directors", "corporate board"],
    "shareholder": ["shareholders meeting", "investor meeting"],
    "dividend": ["dividend payment", "stock dividend"],
    "merger": ["company merger", "business merger", "corporate merger"],
    "acquisition": ["business acquisition", "company takeover"],
    "ipo": ["initial public offering", "stock market ipo", "company going public"],
    "bond": ["financial bond", "treasury bond", "bond market"],
    "commodity": ["commodity trading", "raw materials", "commodities"],
    "forex": ["foreign exchange", "currency trading", "forex market"],
    "hedge": ["hedge fund", "hedging strategy", "risk management"],
    "venture": ["venture capital", "vc funding", "startup investment"],
    "angel": ["angel investor", "angel funding", "early stage investment"],
    "crowdfund": ["crowdfunding campaign", "kickstarter", "crowdfund platform"],
    "loan": ["bank loan", "mortgage", "lending"],
    "credit": ["credit card", "credit score", "credit rating"],
    "debt": ["debt burden", "financial debt", "debt management"],
    "interest": ["interest rate", "compound interest", "interest payment"],
    "inflation": ["inflation chart", "rising prices", "economic inflation"],
    "deflation": ["deflation chart", "price decline"],
    "recession": ["economic recession", "financial crisis", "downturn"],
    "depression": ["great depression", "economic depression"],
    "boom": ["economic boom", "growth period", "prosperity"],
    "bubble": ["economic bubble", "market bubble", "speculative bubble"],
    "crash": ["stock market crash", "financial crash", "market collapse"],
    
    # NATURE & ENVIRONMENT
    "nature": ["waterfall drone shot", "forest aerial", "mountain landscape", "natural wonder", "wilderness", "pristine nature"],
    "forest": ["rainforest", "pine forest", "autumn forest", "forest path", "woodland"],
    "tree": ["oak tree", "redwood forest", "bonsai", "tree rings", "deforestation"],
    "ocean": ["ocean waves", "coral reef", "deep sea", "ocean sunset", "sea creatures"],
    "sea": ["stormy sea", "calm sea", "beach waves", "sailing"],
    "water": ["water droplets", "river flowing", "lake reflection", "hydropower"],
    "rain": ["rain on window", "rainstorm", "rain forest", "rain droplets"],
    "weather": ["storm clouds", "lightning", "tornado", "weather patterns"],
    "climate": ["climate change", "melting glacier", "drought", "extreme weather"],
    "environment": ["environmental conservation", "pollution", "recycling", "green energy"],
    "animal": ["wildlife", "safari animals", "zoo", "endangered species", "animal migration"],
    "wildlife": ["lion pride", "elephant herd", "wolf pack", "bear"],
    "bird": ["eagle soaring", "flock of birds", "hummingbird", "penguin colony"],
    "fish": ["school of fish", "tropical fish", "shark", "aquarium"],
    "mountain": ["mountain peak", "mountain range", "himalayan mountains", "mountain climbing"],
    "desert": ["sahara desert", "sand dunes", "desert sunset", "cactus"],
    "flower": ["blooming flower timelapse", "flower garden", "rose", "sunflower field"],
    "plant": ["plant growth", "indoor plants", "greenhouse", "botanical garden"],
    "jungle": ["tropical jungle", "rainforest jungle", "dense jungle"],
    "safari": ["african safari", "wildlife safari", "safari animals"],
    "canyon": ["grand canyon", "canyon landscape", "desert canyon"],
    "volcano": ["volcanic eruption", "active volcano", "lava flow"],
    "earthquake": ["seismic activity", "earthquake damage", "tremor"],
    "tsunami": ["ocean tsunami", "tidal wave", "tsunami wave"],
    "hurricane": ["hurricane storm", "cyclone", "tropical storm"],
    "tornado": ["tornado funnel", "twister", "tornado destruction"],
    "avalanche": ["snow avalanche", "mountain avalanche"],
    "glacier": ["ice glacier", "glacier melting", "arctic glacier"],
    "iceberg": ["floating iceberg", "arctic iceberg", "ice formation"],
    "coral": ["coral reef", "coral ecosystem", "underwater coral"],
    "whale": ["humpback whale", "whale breach", "ocean whale"],
    "dolphin": ["dolphins swimming", "dolphin pod", "jumping dolphin"],
    "shark": ["great white shark", "shark swimming", "ocean predator"],
    "eagle": ["bald eagle", "eagle flying", "bird of prey"],
    "lion": ["lion pride", "male lion", "lion roar"],
    "elephant": ["elephant herd", "african elephant", "elephant trunk"],
    "tiger": ["bengal tiger", "tiger hunting", "wild tiger"],
    "bear": ["grizzly bear", "polar bear", "bear fishing"],
    "wolf": ["wolf pack", "gray wolf", "howling wolf"],
    "deer": ["deer in forest", "white-tailed deer", "deer grazing"],
    "monkey": ["monkey swinging", "primate", "monkey troop"],
    "gorilla": ["mountain gorilla", "silverback gorilla"],
    "butterfly": ["butterfly flying", "monarch butterfly", "butterfly wings"],
    "bee": ["honey bee", "bee pollinating", "bee hive"],
    "spider": ["spider web", "spider spinning", "arachnid"],
    
    # URBAN & CITY
    "city": ["city skyline", "urban aerial", "city lights night", "metropolitan", "downtown"],
    "urban": ["urban development", "city street", "urban planning", "high-rise buildings"],
    "building": ["skyscraper", "architecture", "construction site", "modern building"],
    "architecture": ["architectural design", "famous buildings", "modern architecture", "interior design"],
    "street": ["busy street", "night street", "empty street", "street market"],
    "traffic": ["traffic jam", "highway traffic", "time lapse traffic", "traffic lights"],
    "car": ["sports car", "electric car", "car driving", "luxury car", "car manufacturing"],
    "vehicle": ["truck", "bus", "motorcycle", "autonomous vehicle"],
    "transportation": ["public transport", "train", "airplane", "shipping"],
    "train": ["bullet train", "subway", "freight train", "train station"],
    "airport": ["airport terminal", "airplane takeoff", "air traffic control"],
    "bridge": ["suspension bridge", "brooklyn bridge", "golden gate", "bridge construction"],
    "automobile": ["car", "vehicle", "automobile manufacturing"],
    "truck": ["semi truck", "pickup truck", "delivery truck"],
    "bus": ["city bus", "school bus", "public bus"],
    "bicycle": ["bike riding", "cycling", "mountain bike"],
    "motorcycle": ["motorbike", "biker", "motorcycle riding"],
    "scooter": ["electric scooter", "motor scooter", "mobility scooter"],
    "helicopter": ["helicopter flying", "chopper", "helicopter rescue"],
    "airplane": ["commercial airplane", "jet aircraft", "plane flying"],
    "jet": ["fighter jet", "private jet", "jet plane"],
    "drone": ["aerial drone", "drone flying", "quadcopter"],
    "ship": ["cargo ship", "cruise ship", "naval ship"],
    "boat": ["sailing boat", "speedboat", "fishing boat"],
    "submarine": ["underwater submarine", "military submarine"],
    "ferry": ["passenger ferry", "car ferry", "ferry boat"],
    "cruise": ["cruise ship", "ocean cruise", "cruise vacation"],
    "cargo": ["cargo container", "freight", "cargo ship"],
    "freight": ["freight train", "cargo freight", "shipping"],
    "delivery": ["package delivery", "delivery truck", "courier"],
    "logistics": ["logistics warehouse", "supply chain", "distribution"],
    "warehouse": ["storage warehouse", "distribution center", "fulfillment"],
    "container": ["shipping container", "cargo container", "containerization"],
    "port": ["shipping port", "harbor", "container port"],
    "harbor": ["harbor view", "marina", "port harbor"],
    "dock": ["loading dock", "ship dock", "docking"],
    
    # PEOPLE & EMOTION (Family Friendly)
    "people": ["crowd", "diverse people", "community", "team", "human connection"],
    "person": ["portrait", "individual", "human face", "person walking"],
    "human": ["human anatomy", "human evolution", "humanity", "human achievement"],
    "face": ["facial expressions", "close up face", "emotional face", "smiling face"],
    "happy": ["people laughing", "celebration", "joy", "friends having fun", "party"],
    "smile": ["genuine smile", "child smiling", "happy person", "laughter"],
    "family": ["family together", "parents and children", "family dinner", "family portrait"],
    "child": ["children playing", "kid learning", "child laughing", "childhood"],
    "woman": ["strong woman", "businesswoman", "woman portrait", "female empowerment"],
    "man": ["businessman", "man working", "male portrait", "gentleman"],
    "friend": ["friends together", "friendship", "best friends", "social gathering"],
    
    # WORK & CAREER
    "work": ["person working", "workplace", "hard work", "productivity"],
    "job": ["job interview", "employment", "career", "professional work"],
    "career": ["career path", "professional growth", "career success", "promotion"],
    "employee": ["office worker", "team member", "staff meeting", "workplace"],
    "worker": ["construction worker", "factory worker", "essential worker", "labor"],
    "professional": ["business professional", "expert", "specialist", "consultant"],
    "skill": ["learning skills", "training", "expertise", "talent development"],
    "education": ["classroom", "university", "graduation", "learning", "teacher", "students studying"],
    "school": ["school building", "schoolyard", "classroom learning", "school bus"],
    "university": ["campus", "lecture hall", "university life", "college students"],
    "student": ["student studying", "library", "note taking", "exam"],
    "teacher": ["teaching", "classroom instruction", "professor", "mentor"],
    "learning": ["e-learning", "online course", "knowledge", "study"],
    
    # CREATIVE & ART
    "art": ["art gallery", "painting", "sculpture", "artistic creation", "artist working"],
    "creative": ["creative process", "brainstorming", "artistic expression", "innovation"],
    "design": ["graphic design", "designer workspace", "design thinking", "creative design"],
    "music": ["musical instruments", "concert", "recording studio", "musician performing", "music notes"],
    "song": ["singing", "songwriter", "music production", "vocals"],
    "dance": ["ballet", "modern dance", "dancing", "choreography"],
    "film": ["movie production", "cinema", "film set", "director", "camera crew"],
    "movie": ["movie theater", "film premiere", "cinematography", "movie making"],
    "photo": ["photography", "camera", "photographer", "photo shoot"],
    "camera": ["professional camera", "filming", "lens", "DSLR"],
    "paint": ["painting process", "artist painting", "paint strokes", "canvas"],
    "draw": ["drawing", "sketch", "illustration", "digital art"],
    "theater": ["theater stage", "movie theater", "theatrical performance"],
    "stage": ["concert stage", "theater stage", "performance stage"],
    "concert": ["live concert", "music concert", "concert crowd"],
    "performance": ["live performance", "stage performance", "artistic performance"],
    "actor": ["actor performing", "movie actor", "theater actor"],
    "actress": ["actress portrait", "female actor", "movie star"],
    "director": ["film director", "movie director", "directing"],
    "producer": ["film producer", "music producer", "production"],
    "screenplay": ["script writing", "screenplay", "movie script"],
    "cinema": ["movie theater", "cinema hall", "cinematography"],
    "animation": ["animated movie", "animation studio", "cartoon animation"],
    "cartoon": ["cartoon character", "animated cartoon", "animation"],
    "comic": ["comic book", "comic art", "graphic novel"],
    "sculpture": ["stone sculpture", "art sculpture", "sculptor"],
    "statue": ["bronze statue", "monument statue", "stone statue"],
    "monument": ["historical monument", "memorial monument", "famous monument"],
    "museum": ["art museum", "museum exhibit", "gallery"],
    "gallery": ["art gallery", "photo gallery", "exhibition"],
    "exhibition": ["art exhibition", "museum exhibition", "exhibit"],
    "portrait": ["portrait painting", "portrait photography", "face portrait"],
    "landscape": ["landscape painting", "scenic landscape", "landscape photo"],
    "abstract": ["abstract art", "abstract painting", "modern art"],
    "modern": ["modern art", "contemporary art", "modern design"],
    "contemporary": ["contemporary art", "modern contemporary", "current art"],
    "classic": ["classical art", "classic painting", "traditional art"],
    "vintage": ["vintage style", "retro vintage", "antique vintage"],
    "antique": ["antique furniture", "antique items", "vintage antique"],
    "craft": ["handicraft", "artisan craft", "craftsmanship"],
    "pottery": ["ceramic pottery", "clay pottery", "potter"],
    "ceramic": ["ceramic art", "ceramic pottery", "ceramics"],
    "textile": ["fabric textile", "textile art", "woven textile"],
    
    # FOOD & COOKING (Halal Focused)
    "food": ["delicious food", "gourmet meal", "food preparation", "culinary", "restaurant dish"],
    "cooking": ["chef cooking", "kitchen", "recipe", "culinary arts"],
    "chef": ["professional chef", "restaurant kitchen", "chef preparing", "culinary expert"],
    "restaurant": ["fine dining", "restaurant service", "food service", "dining experience"],
    "eat": ["eating food", "meal time", "dining", "food consumption"],
    "drink": ["beverage", "juice", "coffee", "pouring drink"],
    "coffee": ["coffee brewing", "coffee shop", "espresso", "coffee beans"],
    "meal": ["family meal", "dinner", "lunch", "breakfast"],
    "kitchen": ["modern kitchen", "restaurant kitchen", "home kitchen"],
    "recipe": ["cooking recipe", "recipe book", "recipe card"],
    "ingredient": ["fresh ingredients", "cooking ingredients", "raw ingredients"],
    "spice": ["spices", "spice market", "spice rack"],
    "herb": ["fresh herbs", "herb garden", "cooking herbs"],
    "vegetable": ["fresh vegetables", "vegetable market", "organic vegetables"],
    "fruit": ["fresh fruit", "fruit bowl", "tropical fruit"],
    "bread": ["fresh bread", "bakery bread", "bread loaf"],
    "bakery": ["bakery shop", "baked goods", "bakery display"],
    "cake": ["birthday cake", "wedding cake", "cake decorating"],
    "dessert": ["dessert plate", "sweet dessert", "dessert menu"],
    "pastry": ["french pastry", "pastry shop", "sweet pastry"],
    "chocolate": ["chocolate bar", "chocolate making", "cocoa chocolate"],
    "ice": ["ice cream", "ice cubes", "frozen ice"],
    "cream": ["whipped cream", "ice cream", "dairy cream"],
    "cheese": ["cheese platter", "cheese making", "artisan cheese"],
    "juice": ["fresh juice", "fruit juice", "juice making"],
    "tea": ["tea ceremony", "tea cup", "tea plantation"],
    "breakfast": ["breakfast table", "morning breakfast", "breakfast food"],
    "lunch": ["lunch plate", "midday meal", "lunch break"],
    "dinner": ["dinner table", "evening meal", "family dinner"],
    "buffet": ["buffet table", "food buffet", "buffet restaurant"],
    "feast": ["feast table", "banquet", "festive meal"],
    "picnic": ["outdoor picnic", "picnic basket", "picnic setting"],
    
    # SPORTS & FITNESS
    "sport": ["sports action", "athletic competition", "stadium", "sports event"],
    "fitness": ["gym workout", "exercise", "fitness training", "healthy lifestyle"],
    "exercise": ["exercising", "cardio", "strength training", "workout"],
    "gym": ["gym equipment", "weight training", "fitness center", "workout space"],
    "run": ["running", "marathon", "jogging", "sprint"],
    "soccer": ["soccer match", "football game", "soccer stadium", "goal"],
    "basketball": ["basketball game", "NBA", "dunk", "basketball court"],
    "athlete": ["professional athlete", "sports athlete", "athletic performance"],
    "training": ["sports training", "athletic training", "workout training"],
    "competition": ["sports competition", "athletic competition", "tournament"],
    "championship": ["championship game", "title match", "championship trophy"],
    "olympic": ["olympic games", "olympic sports", "olympic athlete"],
    "medal": ["gold medal", "olympic medal", "award medal"],
    "trophy": ["championship trophy", "sports trophy", "winner trophy"],
    "stadium": ["sports stadium", "football stadium", "arena"],
    "arena": ["sports arena", "basketball arena", "indoor arena"],
    "field": ["sports field", "playing field", "athletic field"],
    "court": ["tennis court", "basketball court", "sports court"],
    "swimming": ["swimming pool", "swimmer", "competitive swimming"],
    "tennis": ["tennis match", "tennis player", "tennis court"],
    "golf": ["golf course", "golfer", "golf swing"],
    "baseball": ["baseball game", "baseball diamond", "baseball player"],
    "hockey": ["ice hockey", "hockey game", "hockey rink"],
    "volleyball": ["volleyball game", "beach volleyball", "volleyball court"],
    "skiing": ["snow skiing", "ski resort", "downhill skiing"],
    "yoga": ["yoga pose", "yoga class", "yoga practice"],
    "meditation": ["meditation practice", "zen meditation", "mindfulness"],
    "climbing": ["rock climbing", "mountain climbing", "climber"],
    "hiking": ["mountain hiking", "trail hiking", "hiker"],
    "camping": ["outdoor camping", "campsite", "camping tent"],
    "cycling": ["road cycling", "cyclist", "bike race"],
    
    # MEDICAL & HEALTH
    "surgery": ["surgical operation", "operating room", "surgery procedure"],
    "operation": ["medical operation", "surgical procedure"],
    "diagnosis": ["medical diagnosis", "diagnostic imaging", "patient diagnosis"],
    "treatment": ["medical treatment", "therapy", "patient care"],
    "therapy": ["physical therapy", "rehabilitation", "therapy session"],
    "rehabilitation": ["rehab center", "recovery therapy", "physical rehabilitation"],
    "emergency": ["emergency room", "medical emergency", "ambulance"],
    "ambulance": ["ambulance vehicle", "emergency medical", "paramedic"],
    "paramedic": ["emt", "emergency medical technician", "first responder"],
    "pharmacy": ["drug store", "pharmacy counter", "pharmacist"],
    "prescription": ["prescription medication", "rx", "medical prescription"],
    "injection": ["medical injection", "vaccine shot", "syringe"],
    "syringe": ["medical syringe", "needle", "injection device"],
    "stethoscope": ["doctor stethoscope", "medical exam", "cardiac exam"],
    "xray": ["x-ray image", "radiograph", "medical imaging"],
    "mri": ["mri scan", "magnetic resonance imaging", "brain mri"],
    "scan": ["ct scan", "body scan", "medical scan"],
    "ultrasound": ["ultrasound imaging", "sonogram", "medical ultrasound"],
    "cardiac": ["heart health", "cardiac care", "heart monitor"],
    "cancer": ["cancer cells", "oncology", "cancer treatment"],
    "tumor": ["brain tumor", "cancer tumor", "tumor cells"],
    "organ": ["human organ", "organ transplant", "vital organs"],
    "transplant": ["organ transplant", "transplant surgery"],
    "donor": ["organ donor", "blood donor", "donation"],
    "immune": ["immune system", "white blood cells", "immunity"],
    "antibody": ["antibody response", "immune antibody"],
    "infection": ["bacterial infection", "viral infection", "infectious disease"],
    "hospital": ["hospital corridor", "medical facility", "emergency room", "patient care"],
    "doctor": ["physician", "medical examination", "doctor consulting", "healthcare provider"],
    "nurse": ["nursing", "patient care", "medical professional", "hospital staff"],
    "patient": ["medical patient", "hospital bed", "treatment", "care"],
    "disease": ["illness", "pathogen", "epidemic", "medical condition"],
    "virus": ["viral infection", "microscopic virus", "pandemic", "contagion"],
    "bacteria": ["microorganism", "bacterial culture", "microbe", "germ"],
    "vaccine": ["vaccination", "syringe", "immunization", "medical injection"],
    "drug": ["medication", "pills", "pharmaceuticals", "medicine"],
    "pill": ["tablets", "capsules", "prescription", "medication"],
    "blood": ["blood cells", "blood test", "circulation", "donation"],
    "heart": ["heart beating", "cardiac", "heart health", "cardiovascular"],
    "lung": ["respiratory", "breathing", "pulmonary", "airways"],
    "muscle": ["muscular system", "muscle tissue", "bodybuilding", "strength"],
    "bone": ["skeleton", "bone structure", "x-ray", "skeletal"],
    "skin": ["skin texture", "dermatology", "skin care", "complexion"],
    
    # TIME & HISTORY
    "time": ["clock ticking", "hourglass", "time lapse", "calendar", "watch"],
    "history": ["ancient ruins", "historical site", "museum", "old photographs", "historical events"],
    "ancient": ["ancient civilization", "pyramids", "roman ruins", "archaeological site"],
    "old": ["vintage", "antique", "aged", "retro"],
    "past": ["nostalgia", "memories", "flashback", "history"],
    "future": ["futuristic city", "future technology", "sci-fi", "tomorrow", "innovation"],
    
    # ABSTRACT & CONCEPTS
    "idea": ["light bulb moment", "brainstorming", "innovation", "creative thinking", "eureka"],
    "think": ["thinking person", "contemplation", "problem solving", "deep thought"],
    "brain": ["brain scan", "neuroscience", "mental activity", "cognitive"],
    "mind": ["mindfulness", "psychology", "consciousness", "mental health"],
    "dream": ["dreaming", "surreal", "dreamscape", "imagination"],
    "hope": ["hopeful", "optimism", "aspiration", "looking forward"],
    "peace": ["peaceful scene", "meditation", "tranquility", "harmony"],
    "freedom": ["liberation", "free spirit", "independence", "liberty"],
    "power": ["powerful imagery", "strength", "force", "energy"],
    "energy": ["energy flow", "electricity", "solar power", "renewable energy"],
    "light": ["light rays", "illumination", "bright light", "glow"],
    "color": ["vibrant colors", "color spectrum", "rainbow", "colorful"],
    
    # TRAVEL & PLACES
    "travel": ["traveling", "adventure", "journey", "tourist", "exploration"],
    "journey": ["road trip", "voyage", "expedition", "passage"],
    "adventure": ["adventurous", "explorer", "adventure sports", "thrill"],
    "tourist": ["tourism", "sightseeing", "vacation", "holiday"],
    "vacation": ["beach vacation", "resort", "getaway", "leisure travel"],
    "world": ["globe", "world map", "planet earth", "worldwide", "global"],
    "country": ["countryside", "rural area", "farmland", "nation"],
    "continent": ["continental map", "geographic regions", "world continents"],
    "america": ["USA landmarks", "american flag", "north america", "american cities"],
    "europe": ["european architecture", "european cities", "EU", "european landmarks"],
    "asia": ["asian culture", "asian cities", "far east", "asian landscape"],
    "africa": ["african wildlife", "african landscape", "safari", "african culture"],
    
    # SOCIAL & POLITICAL (Neutral)
    "social": ["social interaction", "community", "society", "social media"],
    "community": ["community gathering", "neighborhood", "local community", "together"],
    "society": ["social structure", "civilization", "culture", "societal"],
    "culture": ["cultural diversity", "traditions", "cultural heritage", "customs"],
    "government": ["capitol building", "government offices", "administration", "federal"],
    "law": ["courthouse", "legal", "justice", "legislation"],
    "justice": ["scales of justice", "court", "legal system", "judiciary"],
    "vote": ["voting booth", "election", "ballot", "democracy"],
    
    # MISC COMMON TERMS
    "new": ["brand new", "innovation", "fresh start", "latest"],
    "change": ["transformation", "evolution", "transition", "shifting"],
    "life": ["living", "lifestyle", "life journey", "existence"],
    "birth": ["newborn baby", "childbirth", "beginning", "new life"],
    "age": ["aging", "elderly", "life stages", "generations"],
    "young": ["youth", "teenagers", "young adults", "youthful"],
    "communication": ["talking", "conversation", "dialogue", "networking"],
    "language": ["languages", "translation", "linguistics", "communication"],
    "book": ["reading", "library", "bookshelf", "literature"],
    "read": ["person reading", "studying", "book lover", "reading time"],
    "write": ["writing", "author", "pen and paper", "manuscript"],
    "story": ["storytelling", "narrative", "tale", "fiction"],
    "game": ["gaming", "video games", "board games", "playing"],
    "play": ["playing", "playground", "fun", "recreation"],
    "phone": ["smartphone", "mobile phone", "phone call", "telephone"],
    "screen": ["computer screen", "display", "monitor", "digital screen"],
    "hand": ["hands", "handshake", "holding hands", "helping hand"],
    "eye": ["eyes close-up", "looking", "vision", "eye contact", "staring"],
    "window": ["window view", "looking through window", "window light", "open window"],
    "door": ["doorway", "opening door", "entrance", "doorstep"],
    "home": ["house exterior", "cozy home", "family home", "residential"],
    "house": ["suburban house", "modern house", "home architecture", "real estate"],
    "room": ["living room", "bedroom", "interior room", "empty room"],
    "snow": ["snowfall", "snowy landscape", "winter snow", "snowflakes"],
    "sun": ["sunrise", "sunset", "sun rays", "solar", "sunshine"],
    "moon": ["full moon", "lunar", "moonlight", "crescent moon"],
    "star": ["starry sky", "stars twinkling", "star trails", "constellation"],
    "sky": ["blue sky", "cloudy sky", "dramatic sky", "sky time lapse"],
    "cloud": ["clouds moving", "storm clouds", "fluffy clouds", "cloudscape"],
    "rainbow": ["rainbow arc", "double rainbow", "rainbow after rain"],
    "night": ["night city", "starry night", "nighttime", "nocturnal"],
    "day": ["daytime", "sunny day", "midday", "daylight"],
    "morning": ["early morning", "sunrise", "morning routine", "dawn"],
    "evening": ["evening sky", "dusk", "evening city", "twilight"],
    "season": ["four seasons", "seasonal change", "autumn leaves", "spring bloom"],
    "summer": ["summer beach", "hot summer", "summer activities", "sunny summer"],
    "winter": ["winter landscape", "snowy winter", "winter activities", "cold winter"],
    "spring": ["spring flowers", "spring awakening", "blooming spring", "fresh spring"],
    "autumn": ["fall colors", "autumn leaves", "harvest", "fall season"],
    "map": ["world map", "navigation", "treasure map", "cartography"],
    "direction": ["compass", "navigation", "arrow pointing", "guidance"],
    "machine": ["machinery", "mechanical", "industrial machine", "engine"],
    "tool": ["tools", "workshop", "equipment", "hardware"],
    "factory": ["manufacturing plant", "production line", "industrial factory", "assembly"],
    "industry": ["industrial sector", "heavy industry", "manufacturing", "production"],
    "farm": ["farmland", "agriculture", "barn", "farming"],
    "agriculture": ["crop field", "tractor", "harvest", "agricultural"],
    "garden": ["flower garden", "vegetable garden", "gardening", "botanical"],
    "park": ["city park", "national park", "park scenery", "public park"],
    "beach": ["sandy beach", "beach waves", "tropical beach", "seaside"],
    "island": ["tropical island", "island paradise", "remote island", "archipelago"],
    "lake": ["mountain lake", "calm lake", "lake reflection", "lakeside"],
    "river": ["river flowing", "riverbank", "river landscape", "stream"],
    "valley": ["mountain valley", "green valley", "valley view", "scenic valley"],
    "hill": ["rolling hills", "hillside", "green hills", "hill landscape"],
    "cave": ["cave interior", "cave exploration", "underground cave", "cavern"],
    "rock": ["rock formation", "rocky terrain", "stone", "boulder"],
    "sand": ["sand dunes", "sandy beach", "desert sand", "sand texture"],
    "stone": ["stone texture", "cobblestone", "rock pile", "stonework"],
    "metal": ["metal texture", "metallic", "steel", "iron"],
    "wood": ["wooden texture", "lumber", "wood grain", "timber"],
    "glass": ["glass surface", "transparent glass", "broken glass", "glass reflection"],
    "paper": ["paper stack", "origami", "paper texture", "documents"],
    "fabric": ["textile", "cloth texture", "fabric pattern", "material"],
    "beauty": ["beautiful scenery", "beauty products", "gorgeous", "aesthetic beauty"],
    "body": ["human body", "fitness body", "anatomy", "physique"],
    "sound": ["audio waves", "sound system", "acoustics", "noise"],
    "memory": ["memories", "remembering", "nostalgia", "brain memory"],
    "smartphone": ["mobile phone", "iphone", "android phone", "smartphone screen"],
    "laptop": ["macbook", "laptop computer", "portable computer"],
    "tablet": ["ipad", "digital tablet", "tablet device"],
    "keyboard": ["mechanical keyboard", "typing", "computer keyboard"],
    "monitor": ["computer monitor", "display screen", "4k monitor"],
    "processor": ["CPU", "computer processor", "microprocessor"],
    "circuit": ["circuit board", "electronic circuit", "printed circuit"],
    "chip": ["computer chip", "silicon chip", "microchip"],
    "fiber": ["fiber optic cable", "fiber optic network", "fiber optics"],
    "5g": ["5g tower", "5g network", "wireless 5g"],
    "wifi": ["wifi signal", "wireless internet", "wifi router"],
    "router": ["network router", "wifi router", "internet router"],
    "storage": ["data storage", "hard drive", "cloud storage"],
    "backup": ["data backup", "backup server", "file backup"],
    "recovery": ["data recovery", "disaster recovery", "system recovery"]
}

# ========================================== 
# 9. INTELLIGENT CATEGORY-LOCKED SYSTEM
# ========================================== 

# Global variable to lock the video category throughout the entire video
VIDEO_CATEGORY = None
CATEGORY_KEYWORDS = []

def analyze_script_and_set_category(script, topic):
    """
    CRITICAL: Analyze the ENTIRE script once and determine the PRIMARY category
    This locks the visual theme for the entire video to prevent off-topic clips
    """
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    print("\n🔍 Analyzing script to determine video category...")
    
    full_text = (script + " " + topic).lower()
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has',
                  'this', 'that', 'will', 'can', 'could', 'would', 'should', 'may', 'might'}
    
    words = [w for w in re.findall(r'\b\w+\b', full_text) if len(w) >= 4 and w not in stop_words]
    
    # Count category matches across the ENTIRE script
    category_scores = {}
    for category, terms in VISUAL_MAP.items():
        score = 0
        # Check if category name appears in script
        if category in full_text:
            score += full_text.count(category) * 10
        
        # Check if any category terms appear
        for term in terms:
            term_words = term.split()
            for term_word in term_words:
                if len(term_word) > 3 and term_word in words:
                    score += 3
        
        # Check for related keywords
        for word in words[:50]:  # Check first 50 meaningful words
            if word in category or category in word:
                score += 5
        
        if score > 0:
            category_scores[category] = score
    
    # Sort categories by score
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_categories:
        VIDEO_CATEGORY = sorted_categories[0][0]
        category_score = sorted_categories[0][1]
        
        # Extract all keywords related to this category
        CATEGORY_KEYWORDS = [VIDEO_CATEGORY]
        
        # Add related words from script
        for word in words[:30]:
            if word in VIDEO_CATEGORY or VIDEO_CATEGORY in word:
                CATEGORY_KEYWORDS.append(word)
            # Check if word appears in category terms
            for term in VISUAL_MAP[VIDEO_CATEGORY]:
                if word in term:
                    CATEGORY_KEYWORDS.append(word)
        
        # Remove duplicates
        CATEGORY_KEYWORDS = list(set(CATEGORY_KEYWORDS))[:10]
        
        print(f"✅ VIDEO CATEGORY LOCKED: '{VIDEO_CATEGORY}' (score: {category_score})")
        print(f"📋 Category Keywords: {', '.join(CATEGORY_KEYWORDS[:5])}")
        
        # Show top 3 categories for transparency
        print(f"📊 Top Categories Detected:")
        for i, (cat, score) in enumerate(sorted_categories[:3]):
            print(f"   {i+1}. {cat}: {score} points")
    else:
        # Default fallback categories based on common themes
        VIDEO_CATEGORY = "technology"
        CATEGORY_KEYWORDS = ["technology", "tech", "digital", "innovation"]
        print(f"⚠️ No clear category detected. Using default: '{VIDEO_CATEGORY}'")
    
    return VIDEO_CATEGORY, CATEGORY_KEYWORDS

def get_category_locked_query(text, sentence_index, total_sentences):
    """
    INTELLIGENT: Generate queries that:
    1. STAY WITHIN the locked category (prevents off-topic clips)
    2. USE SEGMENT-SPECIFIC KEYWORDS (makes each clip contextually relevant)
    """
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    if not VIDEO_CATEGORY:
        print("⚠️ Category not set! Using generic query")
        return "abstract technology", ["digital background", "data visualization"], ["technology"]
    
    text_lower = text.lower()
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has',
                  'this', 'that', 'these', 'those', 'will', 'would', 'could', 'should'}
    
    # Extract meaningful words from THIS specific segment
    sentence_words = [w for w in re.findall(r'\b\w+\b', text_lower) if len(w) >= 4 and w not in stop_words]
    
    print(f"    🔎 Segment analysis: '{text[:60]}...'")
    print(f"       Keywords found: {sentence_words[:5]}")
    
    # Get category-specific terms
    category_terms = VISUAL_MAP.get(VIDEO_CATEGORY, [])
    safe_category_terms = [term for term in category_terms if len(term.split()) <= 4]
    
    # === STRATEGY 1: EXACT MATCH - Segment keyword matches category term ===
    matched_terms = []
    for word in sentence_words:
        for term in category_terms:
            if word in term or (len(word) > 5 and term in word):
                matched_terms.append(term)
                print(f"       ✓ Exact match: '{word}' → '{term}'")
                break
    
    # === STRATEGY 2: SEMANTIC MATCH - Segment keyword is category-relevant ===
    keyword_matches = []
    for word in sentence_words:
        if word in CATEGORY_KEYWORDS:
            keyword_matches.append(word)
            print(f"       ✓ Category keyword: '{word}'")
    
    # === STRATEGY 3: CONTEXTUAL MATCH - Use segment's most important words ===
    # Extract nouns and important terms (6+ letters, not too common)
    important_words = [w for w in sentence_words if len(w) >= 6][:3]
    
    # Check if these words relate to ANY category in VISUAL_MAP
    segment_relevant_terms = []
    for word in important_words:
        for cat, terms in VISUAL_MAP.items():
            # Check if word relates to any term in the category
            for term in terms:
                if word in term or term.split()[0] == word:
                    segment_relevant_terms.append(term)
                    break
            if segment_relevant_terms:
                break
    
    # === BUILD PRIMARY QUERY ===
    if matched_terms:
        # BEST CASE: Direct match between segment and category
        segment_keyword = matched_terms[0]
        primary = f"{segment_keyword}"
        keywords = [VIDEO_CATEGORY, segment_keyword] + sentence_words[:2]
        print(f"       → Strategy: Exact match ('{segment_keyword}')")
        
    elif keyword_matches:
        # GOOD CASE: Segment keyword is in category keywords
        main_keyword = keyword_matches[0]
        related_terms = [term for term in safe_category_terms if main_keyword in term]
        if related_terms:
            segment_keyword = random.choice(related_terms)
            primary = f"{segment_keyword}"
        else:
            segment_keyword = f"{main_keyword} {VIDEO_CATEGORY}"
            primary = segment_keyword
        keywords = [VIDEO_CATEGORY, main_keyword] + sentence_words[:2]
        print(f"       → Strategy: Keyword match ('{main_keyword}')")
        
    elif segment_relevant_terms:
        # DECENT CASE: Segment's important word relates to some category
        segment_keyword = segment_relevant_terms[0]
        primary = f"{segment_keyword} {VIDEO_CATEGORY}"
        keywords = [VIDEO_CATEGORY, segment_keyword] + important_words[:2]
        print(f"       → Strategy: Contextual match ('{segment_keyword}')")
        
    elif important_words:
        # FALLBACK 1: Use segment's most important word + category
        segment_keyword = important_words[0]
        # Check if this word is safe and visual
        if segment_keyword in ['problem', 'issue', 'thing', 'situation', 'question']:
            # Skip generic words, use category directly
            segment_keyword = random.choice(safe_category_terms) if safe_category_terms else VIDEO_CATEGORY
        primary = f"{segment_keyword} {VIDEO_CATEGORY}"
        keywords = [VIDEO_CATEGORY, segment_keyword]
        print(f"       → Strategy: Important word ('{segment_keyword}')")
        
    else:
        # FALLBACK 2: Pure category terms with variation
        # Use different category terms for visual variety
        term_index = sentence_index % len(safe_category_terms)
        segment_keyword = safe_category_terms[term_index] if safe_category_terms else VIDEO_CATEGORY
        primary = segment_keyword
        keywords = [VIDEO_CATEGORY]
        print(f"       → Strategy: Category rotation (term #{term_index})")
    
    # === BUILD FALLBACK QUERIES (also segment-aware) ===
    fallbacks = []
    
    # Fallback 1: Another matched term or category term
    if len(matched_terms) > 1:
        fallbacks.append(matched_terms[1])
    elif safe_category_terms:
        fallbacks.append(random.choice(safe_category_terms))
    else:
        fallbacks.append(VIDEO_CATEGORY)
    
    # Fallback 2: Category keyword + segment word
    if keyword_matches and sentence_words:
        fallbacks.append(f"{keyword_matches[0]} {sentence_words[0]}")
    elif safe_category_terms:
        fallbacks.append(random.choice(safe_category_terms))
    else:
        fallbacks.append(f"{VIDEO_CATEGORY} abstract")
    
    # Fallback 3: Pure category term (different from primary)
    if safe_category_terms:
        alt_terms = [t for t in safe_category_terms if t != primary]
        if alt_terms:
            fallbacks.append(random.choice(alt_terms))
        else:
            fallbacks.append(safe_category_terms[0])
    else:
        fallbacks.append(f"{VIDEO_CATEGORY} landscape")
    
    # === SAFETY CHECK: Ensure category is always present ===
    if VIDEO_CATEGORY not in primary.lower():
        primary = f"{primary} {VIDEO_CATEGORY}"
    
    # Add quality indicators
    primary = f"{primary} 4k"
    fallbacks = [f"{fb} cinematic" for fb in fallbacks]
    
    print(f"    📌 Final Query: '{primary}'")
    print(f"       Fallbacks: {fallbacks}")
    
    return primary, fallbacks, keywords

# ========================================== 
# 10. ENHANCED SCORING SYSTEM (100% Context Aligned)
# ========================================== 

def calculate_enhanced_relevance_score(video, query, sentence_text, context_keywords, full_script="", topic=""):
    """
    SMART SCORING: Intelligent relevance without being overly strict
    """
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    score = 0
    
    # Prepare text
    video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
    sentence_lower = sentence_text.lower()
    query_lower = query.lower()
    
    # === SMART CATEGORY VALIDATION ===
    category_trust_score = 0
    
    # Check 1: Does video mention category directly?
    if VIDEO_CATEGORY and VIDEO_CATEGORY in video_text:
        category_trust_score += 25
        print(f"      ✓ Category '{VIDEO_CATEGORY}' in video")
    
    # Check 2: Does video mention any category keywords?
    keyword_matches = 0
    for keyword in CATEGORY_KEYWORDS[:5]:  # Check top 5 keywords
        if keyword in video_text:
            keyword_matches += 1
            category_trust_score += 8
    
    if keyword_matches > 0:
        print(f"      ✓ {keyword_matches} category keywords matched")
    
    # Check 3: CRITICAL - Does our QUERY contain category context?
    query_has_category = False
    if VIDEO_CATEGORY in query_lower:
        query_has_category = True
        category_trust_score += 20
        print(f"      ✓ Query contains category - trusting search API")
    
    # Check if query has any category keywords
    for keyword in CATEGORY_KEYWORDS[:3]:
        if keyword in query_lower:
            query_has_category = True
            category_trust_score += 10
            break
    
    # Check 4: Does query match category TERMS from VISUAL_MAP?
    if VIDEO_CATEGORY in VISUAL_MAP:
        category_terms = VISUAL_MAP[VIDEO_CATEGORY]
        for term in category_terms[:10]:  # Check first 10 terms
            # Check if any words from the term appear in query
            term_words = term.split()
            for term_word in term_words:
                if len(term_word) > 3 and term_word in query_lower:
                    category_trust_score += 5
                    query_has_category = True
                    break
    
    # === SMART PENALTY LOGIC ===
    if category_trust_score > 0:
        # Video or query has category context - GOOD!
        score += category_trust_score
    elif not query_has_category:
        # Video has NO category match AND our query was generic - apply mild penalty
        score -= 15  # Reduced from -60
        print(f"      ⚠️ Weak category match (mild penalty)")
    else:
        # Query has category, video doesn't mention it explicitly
        # But that's OK - search API already filtered for us
        score += 10  # Small bonus for being in search results
        print(f"      → Query-based trust (no penalty)")
    
    # === 1. QUERY MATCH (30 points) ===
    query_terms = [t for t in query_lower.split() if len(t) > 3 and t not in ['landscape', 'cinematic', 'abstract']]
    query_match_count = 0
    
    for term in query_terms:
        if term in video_text:
            query_match_count += 1
            score += 10
    
    # Bonus for exact phrase
    if len(query_terms) >= 2:
        query_phrase = ' '.join(query_terms[:2])
        if query_phrase in video_text:
            score += 15
    
    # === 2. CONTEXT KEYWORDS MATCH (25 points) ===
    context_match_count = 0
    for keyword in context_keywords:
        if len(keyword) > 3:
            if keyword in video_text:
                context_match_count += 1
                score += 8
            if keyword in sentence_lower:
                score += 3
    
    # === 3. SEMANTIC RELEVANCE (20 points) ===
    # Check if video relates to the overall topic/theme
    topic_lower = topic.lower()
    topic_words = [w for w in re.findall(r'\b\w{5,}\b', topic_lower)][:5]
    
    semantic_matches = 0
    for word in topic_words:
        if word in video_text:
            semantic_matches += 1
            score += 4
    
    # === 4. VIDEO QUALITY (15 points) ===
    quality = video.get('quality', '').lower()
    if '4k' in quality or 'uhd' in quality:
        score += 15
    elif 'hd' in quality or 'high' in quality or 'large' in quality:
        score += 12
    elif 'medium' in quality:
        score += 8
    else:
        score += 4  # Give some points even for SD
    
    duration = video.get('duration', 0)
    if duration >= 15:
        score += 5
    elif duration >= 10:
        score += 3
    elif duration >= 5:
        score += 1
    
    # === 5. LANDSCAPE VERIFICATION (10 points + penalties) ===
    landscape_indicators = ['landscape', 'horizontal', 'wide', 'panoramic', 'widescreen', '16:9']
    portrait_indicators = ['vertical', 'portrait', '9:16', 'instagram', 'tiktok', 'reel', 'story']
    
    if any(indicator in video_text for indicator in landscape_indicators):
        score += 10
    
    # Check for explicit portrait indicators
    portrait_detected = False
    for indicator in portrait_indicators:
        if indicator in video_text:
            portrait_detected = True
            break
    
    if portrait_detected:
        score -= 40  # Penalty for confirmed portrait
    
    # Check dimensions
    width = video.get('width', 0)
    height = video.get('height', 0)
    if width and height:
        aspect_ratio = width / height
        if aspect_ratio >= 1.5:  # Landscape (16:9 or wider)
            score += 10
        elif aspect_ratio >= 1.2:  # Slightly landscape
            score += 5
        elif aspect_ratio < 1.0:  # Portrait
            score -= 50
            print(f"      ✗ Portrait aspect ratio detected")
    
    # === 6. ISLAMIC CONTENT FILTER CHECK ===
    for term in STRONG_CONTENT_BLACKLIST:
        if term in video_text:
            score -= 100  # Instant disqualification
            print(f"      🚫 BLOCKED: Prohibited term '{term}'")
            return 0  # Return immediately
    
    # === 7. PLATFORM & SOURCE QUALITY ===
    service = video.get('service', '')
    if service == 'pexels':
        score += 5
    elif service == 'pixabay':
        score += 3
    
    # === 8. BONUS: Multi-factor Perfect Match ===
    if query_match_count >= 2 and context_match_count >= 1 and quality in ['hd', 'large', '4k', 'uhd']:
        score += 10
        print(f"      ⭐ Perfect match bonus")
    
    # Small random factor for variety
    score += random.randint(0, 2)
    
    # Cap between 0-100
    final_score = min(100, max(0, score))
    
    return final_score

# ========================================== 
# 11. VIDEO SEARCH (Pixabay & Pexels)
# ========================================== 

USED_VIDEO_URLS = set()

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
    
    elif service == 'pixabay' and keys:
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
# 12. UTILS (FIXED: Updates GitHub Status)
# ==========================================

# Buffer to store logs for the HTML frontend
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
# 13. SCRIPT & AUDIO
# ========================================== 

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
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
- Ensure content is family-friendly and appropriate for all audiences
- Avoid any references to alcohol, violence, or inappropriate content
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"{base_instructions}\nWrite Part {i+1}/{chunks} of a documentary about '{topic}'. Context: {context}. Length: 700 words."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nWrite a YouTube documentary script about '{topic}'. {words} words. Ensure it's educational and family-friendly."
        script = call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'\(.*?music.*?\)', '', script, flags=re.IGNORECASE)
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
# 14. ENHANCED VISUAL PROCESSING WITH T5 & CLIP
# ========================================== 

def enhanced_get_clip_for_sentence(i, sent, max_retries=4):
    """
    NEW Pipeline: T5 Query -> Search APIs -> Download N candidates -> CLIP Rank -> Best Match.
    """
    dur = max(3.5, sent['end'] - sent['start'])
    temp_clips_dir = TEMP_DIR / f"clip_{i}_candidates"
    temp_clips_dir.mkdir(exist_ok=True)

    print(f"  🔍 Clip {i+1}/{len(sentences)}: '{sent['text'][:60]}...'")

    # STEP 1: Generate intelligent search query using T5
    primary_query, query_tags = generate_smart_search_query(sent['text'], TOPIC)
    print(f"    🧠 T5 Smart Query: '{primary_query}' (Tags: {query_tags})")

    for attempt in range(max_retries):
        # STEP 2: Search on Pexels/Pixabay
        all_search_results = []
        page = random.randint(1, 3)

        if PEXELS_KEYS and PEXELS_KEYS[0]:
            pexels_results = intelligent_video_search(primary_query, 'pexels', PEXELS_KEYS, page)
            all_search_results.extend(pexels_results)
        if PIXABAY_KEYS and PIXABAY_KEYS[0]:
            pixabay_results = intelligent_video_search(primary_query, 'pixabay', PIXABAY_KEYS, page)
            all_search_results.extend(pixabay_results)

        # Filter: Remove used, prohibited, low-scoring, and portrait videos
        filtered_results = []
        for vid in all_search_results:
            if vid['url'] in USED_VIDEO_URLS:
                continue
            if contains_prohibited_content(vid.get('title', ''), vid.get('description', '')):
                continue
            # Apply the existing scoring (but with relaxed thresholds for CLIP stage)
            relevance = calculate_enhanced_relevance_score(vid, primary_query, sent['text'], query_tags, full_script=SCRIPT_TEXT, topic=TOPIC)
            vid['relevance_score'] = relevance
            if relevance >= 20:  # Lower threshold to get more candidates for CLIP
                filtered_results.append(vid)

        if not filtered_results:
            print(f"    ⚠️ Attempt {attempt+1}: No suitable videos found.")
            continue

        # Sort and pick top N candidates for download
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        candidates_to_download = filtered_results[:5]  # Top 5 for CLIP evaluation

        # STEP 3: Download candidate videos
        downloaded_paths = []
        for idx, cand in enumerate(candidates_to_download):
            try:
                raw_path = temp_clips_dir / f"candidate_{idx}.mp4"
                response = requests.get(cand['url'], timeout=30, stream=True)
                with open(raw_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                downloaded_paths.append(str(raw_path))
            except Exception as e:
                print(f"    Could not download candidate {idx}: {e}")
                continue

        if not downloaded_paths:
            print(f"    ⚠️ Could not download any candidates.")
            continue

        # STEP 4: CLIP Ranking - Find the visually best match
        best_video_path = rank_videos_by_clip_match(sent['text'], downloaded_paths)

        if best_video_path and os.path.exists(best_video_path):
            # STEP 5: Process the selected video (trim, scale)
            # Find which candidate this was to mark URL as used
            for cand in candidates_to_download:
                if str(best_video_path).endswith(os.path.basename(cand['url']).split('.')[0] + '.mp4'):
                    USED_VIDEO_URLS.add(cand['url'])
                    break
            
            final_clip_path = TEMP_DIR / f"s_{i}_final.mp4"
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
                "-i", best_video_path,
                "-t", str(dur),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
                "-preset", "p4" if torch.cuda.is_available() else "medium",
                "-b:v", "8M",
                "-an",
                str(final_clip_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # Cleanup candidate directory
            shutil.rmtree(temp_clips_dir, ignore_errors=True)
            return str(final_clip_path)

    # FALLBACK: If all retries fail, use gradient
    print(f"  → Clip {i}: All attempts failed. Using category-themed gradient.")
    shutil.rmtree(temp_clips_dir, ignore_errors=True)
    return create_gradient_fallback_clip(i, dur)

def create_gradient_fallback_clip(clip_index, duration):
    """Creates a simple gradient video as fallback."""
    category_gradients = {
        "tech": ["0x1a1a2e:0x0f3460", "0x16213e:0x0f3460"],
        "technology": ["0x1a1a2e:0x0f3460", "0x16213e:0x0f3460"],
        "internet": ["0x0f3460:0x533483", "0x16213e:0x533483"],
        "digital": ["0x16213e:0x533483", "0x0f3460:0x16213e"],
        "ai": ["0x0f3460:0x16213e", "0x1a1a2e:0x533483"],
        "computer": ["0x1a1a2e:0x16213e", "0x0f3460:0x1a1a2e"],
        "business": ["0x1e3a5f:0x2a2d34", "0x1a1a2e:0x2a2d34"],
        "finance": ["0x0f3460:0x1e3a5f", "0x16213e:0x1e3a5f"],
        "nature": ["0x1e4d2b:0x2d5016", "0x1a3a1e:0x2d5016"],
        "science": ["0x1e3a5f:0x0f3460", "0x16213e:0x0f3460"],
    }
    
    gradient = category_gradients.get(VIDEO_CATEGORY, ["0x1a1a2e:0x16213e"])[0] if VIDEO_CATEGORY else "0x1a1a2e:0x16213e"
    
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
# 15. DUAL-OUTPUT VIDEO RENDERING
# ========================================== 

def render_final_videos(concatenated_video_path, audio_path, ass_file, logo_path, job_id):
    """
    Creates two final videos:
    1. base_{JOB_ID}.mp4: Visuals + Audio + Logo (NO subtitles).
    2. final_{JOB_ID}.mp4: Visuals + Audio + Logo + BURNED-IN Subtitles.
    Both are uploaded to Google Drive.
    """
    print("🎬 Starting Two-Stage Final Render...")

    # --- STAGE 1: Render BASE video (NO subtitles) ---
    print("  📹 Stage 1: Rendering base video (without subtitles)...")
    base_output = OUTPUT_DIR / f"base_{job_id}.mp4"

    # Use the same complex filter but WITHOUT the subtitles filter
    if logo_path and os.path.exists(logo_path):
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(concatenated_video_path),  # Visuals
            "-i", str(logo_path),          # Logo
            "-i", str(audio_path),         # Audio
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
            "-preset", "p4" if torch.cuda.is_available() else "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(base_output)
        ]
    else:
        # No logo version
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(concatenated_video_path),
            "-i", str(audio_path),
            "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
            "-preset", "p4" if torch.cuda.is_available() else "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(base_output)
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    print(f"    ✅ Base video rendered: {base_output}")

    # --- STAGE 2: Render FINAL video (WITH burned subtitles) ---
    print("  📼 Stage 2: Rendering final video (with burned subtitles)...")
    final_output = OUTPUT_DIR / f"final_{job_id}.mp4"

    # Use the `ass` filter to burn ASS format subtitles
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
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(concatenated_video_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
            "-preset", "p4" if torch.cuda.is_available() else "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(final_output)
        ]
    else:
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[bg]ass='{ass_path_escaped}'[v]"
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(concatenated_video_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
            "-preset", "p4" if torch.cuda.is_available() else "medium",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            str(final_output)
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    print(f"    ✅ Final video with subtitles rendered: {final_output}")

    return str(base_output), str(final_output)

# ========================================== 
# 16. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START (ENHANCED: T5 + CLIP + Dual Output + Islamic Safe) ---")
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
    
    # Transcribe for subtitles
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            print("📝 Transcribing audio...")
            transcript = transcriber.transcribe(str(audio_out))
            
            if transcript.status == aai.TranscriptStatus.completed:
                sentences = []
                for sentence in transcript.get_sentences():
                    sentences.append({
                        "text": sentence.text,
                        "start": sentence.start / 1000,
                        "end": sentence.end / 1000
                    })
                if sentences:
                    sentences[-1]['end'] += 1.0
                print(f"✅ Transcription complete: {len(sentences)} sentences")
            else:
                raise Exception("Transcription failed")
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}. Using fallback...")
            # Fallback timing logic
            words = text.split()
            import wave
            with wave.open(str(audio_out), 'rb') as wav_file:
                total_duration = wav_file.getnframes() / float(wav_file.getframerate())
            
            words_per_second = len(words) / total_duration
            sentences = []
            current_time = 0
            words_per_sentence = 12
            
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
                sentences[-1]['end'] += 1.5
    else:
        # Fallback if no AssemblyAI
        words = text.split()
        import wave
        with wave.open(str(audio_out), 'rb') as wav_file:
            total_duration = wav_file.getnframes() / float(wav_file.getframerate())
        
        words_per_second = len(words) / total_duration
        sentences = []
        current_time = 0
        words_per_sentence = 12
        
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
            sentences[-1]['end'] += 1.5
    
    # Create ASS subtitles
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # CRITICAL: Analyze script ONCE and lock category for entire video
    analyze_script_and_set_category(text, TOPIC)
    
    # Process visuals using the NEW enhanced pipeline (which uses T5 and CLIP)
    update_status(60, "Gathering Visuals with T5 & CLIP Intelligence...")
    print(f"\n📥 Downloading {len(sentences)} clips (Category: {VIDEO_CATEGORY})...")

    clips = []
    for i, sent in enumerate(sentences):
        update_status(60 + int((i/len(sentences))*30), f"Processing clip {i+1}/{len(sentences)}...")
        # Use the NEW function that implements the T5->CLIP pipeline
        clip_path = enhanced_get_clip_for_sentence(i, sent)
        clips.append(clip_path)

    # Concatenate all clips into one visual track
    print("🔗 Concatenating video clips...")
    concat_list_path = TEMP_DIR / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for c in clips:
            if os.path.exists(c):
                f.write(f"file '{os.path.abspath(c)}'\n")

    concatenated_video_path = TEMP_DIR / "all_visuals.mp4"
    subprocess.run(
        f"ffmpeg -y -f concat -safe 0 -i {concat_list_path} -c:v h264_nvenc -preset p1 -b:v 10M {concatenated_video_path}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # RENDER TWO VIDEOS: base (no subtitles) and final (with subtitles)
    base_video_path, final_video_path = render_final_videos(
        concatenated_video_path, audio_out, ass_file, ref_logo, JOB_ID
    )

    # UPLOAD BOTH VIDEOS TO GOOGLE DRIVE
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
        update_status(100, "Processing Complete (Upload Failed)", "completed")
else:
    update_status(0, "Audio Synthesis Failed", "failed")

# Cleanup
print("Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["visual.mp4", "list.txt", "all_visuals.mp4"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ PROCESS COMPLETE ---")
