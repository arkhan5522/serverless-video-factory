"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
ENHANCED VERSION:
1. T5 Transformer for Smart Search Query Generation
2. CLIP Model for Visual-Script Matching
3. Dual Output: With & Without Subtitles
4. Islamic Content Filter
5. Fixed Subtitle Design (ASS format)
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
        "sentencepiece",
        "opencv-python-headless",
        "pillow",
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
import cv2
from PIL import Image

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
CLIPS_DIR = TEMP_DIR / "clips"

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

# ========================================== 
# 3. ISLAMIC CONTENT FILTER
# ========================================== 

FORBIDDEN_TERMS = [
    # Explicit content
    'nude', 'nudity', 'naked', 'sexy', 'sex', 'porn', 'erotic', 'sensual',
    'bikini', 'lingerie', 'underwear', 'swimsuit', 'bra', 'panties',
    'stripper', 'strip club', 'prostitute', 'escort',
    
    # Alcohol & Intoxicants
    'alcohol', 'beer', 'wine', 'whiskey', 'vodka', 'rum', 'cocktail',
    'drunk', 'drinking alcohol', 'bar', 'pub', 'nightclub', 'brewery',
    'champagne', 'liquor', 'booze', 'intoxicated', 'hangover',
    
    # Gambling
    'casino', 'gambling', 'poker', 'slot machine', 'betting', 'lottery',
    'blackjack', 'roulette',
    
    # Violence & War
    'war', 'warfare', 'combat', 'battle', 'military attack', 'bombing',
    'explosion', 'terrorist', 'terrorism', 'massacre', 'genocide',
    'blood', 'gore', 'murder', 'killing', 'death', 'corpse', 'dead body',
    'gun violence', 'shooting', 'stabbing',
    
    # Pork & Haram food
    'pork', 'bacon', 'ham', 'pig', 'swine', 'sausage pork',
    
    # Idolatry & Religious
    'idol', 'statue worship', 'pagan', 'occult', 'witchcraft', 'satan',
    'devil worship', 'black magic', 'voodoo',
    
    # Drugs
    'drugs', 'cocaine', 'heroin', 'marijuana', 'weed', 'cannabis',
    'smoking weed', 'drug use', 'overdose',
    
    # Inappropriate relationships
    'affair', 'adultery', 'cheating spouse', 'one night stand',
    'hookup', 'dating app', 'tinder',
    
    # LGBTQ+ (as per Islamic guidelines)
    'gay', 'lesbian', 'transgender', 'drag queen', 'pride parade',
    
    # Other
    'tattoo', 'piercing', 'rave', 'party girls', 'hot girls', 'hot women',
    'belly dancer', 'pole dance'
]

def is_content_halal(text):
    """Check if content is appropriate according to Islamic guidelines"""
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in text_lower:
            return False, term
    return True, None

def filter_forbidden_from_query(query):
    """Remove any forbidden terms from search query"""
    query_lower = query.lower()
    for term in FORBIDDEN_TERMS:
        if term in query_lower:
            query_lower = query_lower.replace(term, '')
    return query_lower.strip()

# ========================================== 
# 4. T5 TRANSFORMER FOR SMART QUERIES
# ========================================== 

T5_MODEL = None
T5_TOKENIZER = None

def load_t5_model():
    """Load T5 model for tag generation"""
    global T5_MODEL, T5_TOKENIZER
    
    if T5_MODEL is not None:
        return True
    
    try:
        print("🤖 Loading T5 Model for Smart Query Generation...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        T5_TOKENIZER = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
        T5_MODEL = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            T5_MODEL = T5_MODEL.cuda()
        
        print("✅ T5 Model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ T5 Model loading failed: {e}")
        return False

def generate_smart_query_t5(script_text, num_tags=5):
    """
    Use T5 to generate intelligent search tags from script text
    Returns list of relevant visual search terms
    """
    global T5_MODEL, T5_TOKENIZER
    
    if T5_MODEL is None:
        if not load_t5_model():
            return None
    
    try:
        # Truncate text if too long
        text = script_text[:500]
        
        # Prepare input
        inputs = T5_TOKENIZER(
            [text], 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate tags
        with torch.no_grad():
            output = T5_MODEL.generate(
                **inputs,
                max_length=64,
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True
            )
        
        # Decode
        decoded = T5_TOKENIZER.batch_decode(output, skip_special_tokens=True)[0]
        
        # Parse tags
        tags = [tag.strip() for tag in decoded.split(",") if tag.strip()]
        tags = list(dict.fromkeys(tags))  # Remove duplicates while preserving order
        
        # Filter forbidden terms
        clean_tags = []
        for tag in tags[:num_tags]:
            is_halal, _ = is_content_halal(tag)
            if is_halal:
                clean_tags.append(tag)
        
        return clean_tags if clean_tags else None
        
    except Exception as e:
        print(f"⚠️ T5 generation error: {e}")
        return None

# ========================================== 
# 5. CLIP MODEL FOR VISUAL MATCHING
# ========================================== 

CLIP_MODEL = None
CLIP_PROCESSOR = None

def load_clip_model():
    """Load CLIP model for visual-text matching"""
    global CLIP_MODEL, CLIP_PROCESSOR
    
    if CLIP_MODEL is not None:
        return True
    
    try:
        print("🎯 Loading CLIP Model for Visual Matching...")
        from transformers import CLIPProcessor, CLIPModel
        
        CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if torch.cuda.is_available():
            CLIP_MODEL = CLIP_MODEL.cuda()
        
        print("✅ CLIP Model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ CLIP Model loading failed: {e}")
        return False

def extract_video_frame(video_path, position='middle'):
    """Extract a frame from video for CLIP analysis"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if position == 'middle':
            target_frame = frame_count // 2
        elif position == 'start':
            target_frame = min(30, frame_count - 1)
        else:
            target_frame = max(0, frame_count - 30)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    except Exception as e:
        print(f"⚠️ Frame extraction error: {e}")
        return None

def calculate_clip_similarity(text, image):
    """Calculate CLIP similarity score between text and image"""
    global CLIP_MODEL, CLIP_PROCESSOR
    
    if CLIP_MODEL is None:
        if not load_clip_model():
            return 0.0
    
    try:
        inputs = CLIP_PROCESSOR(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = CLIP_MODEL(**inputs)
        
        # Get similarity score
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.softmax(dim=1)[0][0].item()
        
        return similarity * 100  # Convert to percentage
        
    except Exception as e:
        print(f"⚠️ CLIP similarity error: {e}")
        return 0.0

def rank_videos_by_clip(script_text, video_candidates):
    """
    Rank downloaded video candidates using CLIP
    Returns sorted list with CLIP scores
    """
    global CLIP_MODEL
    
    if not video_candidates:
        return []
    
    if CLIP_MODEL is None:
        if not load_clip_model():
            # Return original order if CLIP unavailable
            return video_candidates
    
    print(f"    🎯 CLIP ranking {len(video_candidates)} candidates...")
    
    scored_videos = []
    
    for video in video_candidates:
        video_path = video.get('local_path')
        if not video_path or not os.path.exists(video_path):
            video['clip_score'] = 0
            scored_videos.append(video)
            continue
        
        # Extract frame
        frame = extract_video_frame(video_path, 'middle')
        if frame is None:
            video['clip_score'] = 0
            scored_videos.append(video)
            continue
        
        # Calculate CLIP similarity
        clip_score = calculate_clip_similarity(script_text, frame)
        video['clip_score'] = clip_score
        scored_videos.append(video)
    
    # Sort by CLIP score (highest first)
    scored_videos.sort(key=lambda x: x.get('clip_score', 0), reverse=True)
    
    if scored_videos:
        best = scored_videos[0]
        print(f"    ✓ Best CLIP match: {best.get('clip_score', 0):.1f}% - {best.get('title', '')[:40]}")
    
    return scored_videos

# ========================================== 
# 6. SUBTITLE STYLES
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
    "tiktok_white": {
        "name": "TikTok White (Ultra Bold)",
        "fontname": "Arial Black",
        "fontsize": 65,
        "primary_colour": "&H00FFFFFF",
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
    """Create ASS subtitle file with proper format encoding"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']} (Size: {style['fontsize']}px)")
    
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
# 7. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path, custom_name=None):
    """Uploads using OAuth 2.0 Refresh Token"""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    print(f"🔑 Uploading: {os.path.basename(file_path)}...")
    
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
    except Exception as e:
        print(f"❌ Failed to refresh token: {e}")
        return None
    
    # Upload
    filename = custom_name if custom_name else os.path.basename(file_path)
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
# 8. VISUAL DICTIONARY (HALAL CATEGORIES)
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
    "business": ["business meeting", "handshake deal", "office skyscrapers", "corporate presentation", "team collaboration"],
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
    "luxury": ["luxury hotel", "first class", "designer brands", "champagne"],
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
    "yacht": ["luxury yacht", "sailing yacht", "motor yacht"],
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
    
    # PEOPLE & EMOTION
    "people": ["crowd", "diverse people", "community", "team", "human connection"],
    "person": ["portrait", "individual", "human face", "person walking"],
    "human": ["human anatomy", "human evolution", "humanity", "human achievement"],
    "face": ["facial expressions", "close up face", "emotional face", "smiling face"],
    "happy": ["people laughing", "celebration", "joy", "friends having fun", "party"],
    "smile": ["genuine smile", "child smiling", "happy person", "laughter"],
    "sad": ["crying", "sadness", "lonely person", "depression", "grief"],
    "emotion": ["emotional expression", "feelings", "mood", "empathy"],
    "love": ["couple in love", "romance", "wedding", "heart shape", "kiss"],
    "family": ["family together", "parents and children", "family dinner", "family portrait"],
    "child": ["children playing", "kid learning", "child laughing", "childhood"],
    "woman": ["strong woman", "businesswoman", "woman portrait", "female empowerment"],
    "man": ["businessman", "man working", "male portrait", "gentleman"],
    "fear": ["scared person", "horror", "anxiety", "panic", "nightmare"],
    "anger": ["angry person", "rage", "frustration", "conflict"],
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
    "manga": ["japanese manga", "manga art", "anime manga"],
    "anime": ["anime art", "japanese animation", "anime character"],
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
    "weaving": ["fabric weaving", "textile weaving", "loom weaving"],
    "embroidery": ["embroidered fabric", "needlework", "embroidery art"],
    "knitting": ["knitting needles", "knitted fabric", "yarn knitting"],
    "sewing": ["sewing machine", "tailoring", "seamstress"],
    
    # FOOD & COOKING
    "food": ["delicious food", "gourmet meal", "food preparation", "culinary", "restaurant dish"],
    "cooking": ["chef cooking", "kitchen", "recipe", "culinary arts"],
    "chef": ["professional chef", "restaurant kitchen", "chef preparing", "culinary expert"],
    "restaurant": ["fine dining", "restaurant service", "food service", "dining experience"],
    "eat": ["eating food", "meal time", "dining", "food consumption"],
    "drink": ["beverage", "cocktail", "coffee", "pouring drink"],
    "coffee": ["coffee brewing", "coffee shop", "espresso", "coffee beans"],
    "meal": ["family meal", "dinner", "lunch", "breakfast"],
    "kitchen": ["modern kitchen", "restaurant kitchen", "home kitchen"],
    "recipe": ["cooking recipe", "recipe book", "recipe card"],
    "ingredient": ["fresh ingredients", "cooking ingredients", "raw ingredients"],
    "spice": ["spices", "spice market", "spice rack"],
    "herb": ["fresh herbs", "herb garden", "cooking herbs"],
    "vegetable": ["fresh vegetables", "vegetable market", "organic vegetables"],
    "fruit": ["fresh fruit", "fruit bowl", "tropical fruit"],
    "meat": ["raw meat", "butcher", "meat preparation"],
    "seafood": ["seafood platter", "ocean seafood", "shellfish"],
    "bread": ["fresh bread", "bakery bread", "bread loaf"],
    "bakery": ["bakery shop", "baked goods", "bakery display"],
    "cake": ["birthday cake", "wedding cake", "cake decorating"],
    "dessert": ["dessert plate", "sweet dessert", "dessert menu"],
    "pastry": ["french pastry", "pastry shop", "sweet pastry"],
    "chocolate": ["chocolate bar", "chocolate making", "cocoa chocolate"],
    "ice": ["ice cream", "ice cubes", "frozen ice"],
    "cream": ["whipped cream", "ice cream", "dairy cream"],
    "cheese": ["cheese platter", "cheese making", "artisan cheese"],
    "wine": ["wine bottle", "wine tasting", "vineyard wine"],
    "beer": ["beer glass", "brewery", "craft beer"],
    "cocktail": ["cocktail drink", "bartender", "mixed drink"],
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
    "football": ["american football", "NFL", "touchdown", "football stadium"],
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
    "diving": ["diving board", "scuba diving", "cliff diving"],
    "tennis": ["tennis match", "tennis player", "tennis court"],
    "golf": ["golf course", "golfer", "golf swing"],
    "baseball": ["baseball game", "baseball diamond", "baseball player"],
    "hockey": ["ice hockey", "hockey game", "hockey rink"],
    "rugby": ["rugby match", "rugby player", "rugby tackle"],
    "cricket": ["cricket match", "cricket bat", "cricket field"],
    "volleyball": ["volleyball game", "beach volleyball", "volleyball court"],
    "skiing": ["snow skiing", "ski resort", "downhill skiing"],
    "snowboard": ["snowboarding", "snowboard trick", "snow sport"],
    "skating": ["ice skating", "figure skating", "skate rink"],
    "surfing": ["ocean surfing", "surfer", "surf wave"],
    "boxing": ["boxing match", "boxer", "boxing ring"],
    "wrestling": ["wrestling match", "wrestler", "wrestling ring"],
    "martial": ["martial arts", "karate", "taekwondo"],
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
    "epidemic": ["disease epidemic", "outbreak", "epidemic spread"],
    "pandemic": ["global pandemic", "pandemic response", "worldwide outbreak"],
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
    "pain": ["suffering", "injury", "ache", "discomfort"],
    "blood": ["blood cells", "blood test", "circulation", "donation"],
    "heart": ["heart beating", "cardiac", "heart health", "cardiovascular"],
    "lung": ["respiratory", "breathing", "pulmonary", "airways"],
    "muscle": ["muscular system", "muscle tissue", "bodybuilding", "strength"],
    "bone": ["skeleton", "bone structure", "x-ray", "skeletal"],
    "skin": ["skin texture", "dermatology", "skin care", "complexion"],
    "tooth": ["teeth", "dental", "dentist", "smile"],
    
    # TIME & HISTORY
    "time": ["clock ticking", "hourglass", "time lapse", "calendar", "watch"],
    "history": ["ancient ruins", "historical site", "museum", "old photographs", "historical events"],
    "ancient": ["ancient civilization", "pyramids", "roman ruins", "archaeological site"],
    "old": ["vintage", "antique", "aged", "retro"],
    "past": ["nostalgia", "memories", "flashback", "history"],
    "future": ["futuristic city", "future technology", "sci-fi", "tomorrow", "innovation"],
    
    # WAR & CONFLICT
    "war": ["battlefield", "military operation", "war memorial", "conflict zone"],
    "military": ["military training", "soldiers", "army", "defense"],
    "soldier": ["soldier portrait", "troops", "military personnel", "veteran"],
    "weapon": ["weapons", "armory", "military equipment", "defense system"],
    "fight": ["combat", "fighting", "battle", "confrontation"],
    "conflict": ["war zone", "tension", "dispute", "crisis"],
    
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
    "dark": ["darkness", "shadows", "night", "mysterious"],
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
    
    # RELIGION & SPIRITUALITY
    "religion": ["religious symbols", "worship", "faith", "spiritual"],
    "spiritual": ["spirituality", "meditation", "zen", "enlightenment"],
    "church": ["cathedral", "religious building", "worship place", "chapel"],
    "temple": ["buddhist temple", "hindu temple", "sacred temple", "shrine"],
    "pray": ["prayer", "praying hands", "worship", "devotion"],
    "god": ["divine", "deity", "religious art", "heavenly"],
    
    # SOCIAL & POLITICAL
    "social": ["social interaction", "community", "society", "social media"],
    "community": ["community gathering", "neighborhood", "local community", "together"],
    "society": ["social structure", "civilization", "culture", "societal"],
    "culture": ["cultural diversity", "traditions", "cultural heritage", "customs"],
    "political": ["politics", "government", "political rally", "democracy"],
    "government": ["capitol building", "government offices", "administration", "federal"],
    "law": ["courthouse", "legal", "justice", "legislation"],
    "justice": ["scales of justice", "court", "legal system", "judiciary"],
    "vote": ["voting booth", "election", "ballot", "democracy"],
    "election": ["election campaign", "polling", "voters", "electoral"],
    
    # MISC COMMON TERMS
    "new": ["brand new", "innovation", "fresh start", "latest"],
    "change": ["transformation", "evolution", "transition", "shifting"],
    "life": ["living", "lifestyle", "life journey", "existence"],
    "death": ["cemetery", "memorial", "end of life", "mortality"],
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
    "toy": ["toys", "playtime", "children's toys", "action figures"],
    "phone": ["smartphone", "mobile phone", "phone call", "telephone"],
    "screen": ["computer screen", "display", "monitor", "digital screen"],
    "hand": ["hands", "handshake", "holding hands", "helping hand"],
    "eye": ["eyes close-up", "looking", "vision", "eye contact", "staring"],
    "window": ["window view", "looking through window", "window light", "open window"],
    "door": ["doorway", "opening door", "entrance", "doorstep"],
    "home": ["house exterior", "cozy home", "family home", "residential"],
    "house": ["suburban house", "modern house", "home architecture", "real estate"],
    "room": ["living room", "bedroom", "interior room", "empty room"],
    "fire": ["flames", "campfire", "wildfire", "fireplace"],
    "smoke": ["smoke rising", "smoky atmosphere", "fog", "mist"],
    "snow": ["snowfall", "snowy landscape", "winter snow", "snowflakes"],
    "wind": ["windy day", "wind turbine", "trees in wind", "air movement"],
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
    "fast": ["speed", "racing", "quick motion", "velocity"],
    "slow": ["slow motion", "slowness", "gradual", "leisurely"],
    "big": ["large scale", "gigantic", "enormous", "massive structure"],
    "small": ["tiny", "miniature", "microscopic", "little"],
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
    "plastic": ["plastic material", "polymer", "plastic waste", "synthetic"],
    "paper": ["paper stack", "origami", "paper texture", "documents"],
    "fabric": ["textile", "cloth texture", "fabric pattern", "material"],
    "fashion": ["fashion show", "designer clothes", "style", "runway"],
    "style": ["stylish", "trendy", "fashion style", "aesthetic"],
    "beauty": ["beautiful scenery", "beauty products", "gorgeous", "aesthetic beauty"],
    "makeup": ["cosmetics", "makeup application", "beauty routine", "lipstick"],
    "hair": ["hairstyle", "hair salon", "flowing hair", "hair care"],
    "body": ["human body", "fitness body", "anatomy", "physique"],
    "smell": ["fragrance", "aroma", "scent", "perfume"],
    "taste": ["tasting food", "flavor", "taste buds", "culinary experience"],
    "touch": ["tactile", "feeling texture", "sense of touch", "contact"],
    "sound": ["audio waves", "sound system", "acoustics", "noise"],
    "hear": ["listening", "hearing", "ear", "audio"],
    "see": ["vision", "seeing", "eyesight", "visual"],
    "feel": ["feeling", "emotion", "sensation", "touch"],
    "sense": ["five senses", "sensory", "perception", "awareness"],
    "memory": ["memories", "remembering", "nostalgia", "brain memory"],
    "smartphone": ["mobile phone", "iphone", "android phone", "smartphone screen"],
    "laptop": ["macbook", "laptop computer", "portable computer"],
    "tablet": ["ipad", "digital tablet", "tablet device"],
    "keyboard": ["mechanical keyboard", "typing", "computer keyboard"],
    "mouse": ["computer mouse", "wireless mouse", "gaming mouse"],
    "monitor": ["computer monitor", "display screen", "4k monitor"],
    "processor": ["CPU", "computer processor", "microprocessor"],
    "circuit": ["circuit board", "electronic circuit", "printed circuit"],
    "chip": ["computer chip", "silicon chip", "microchip"],
    "semiconductor": ["semiconductor wafer", "chip manufacturing"],
    "fiber": ["fiber optic cable", "fiber optic network", "fiber optics"],
    "5g": ["5g tower", "5g network", "wireless 5g"],
    "wifi": ["wifi signal", "wireless internet", "wifi router"],
    "router": ["network router", "wifi router", "internet router"],
    "modem": ["cable modem", "internet modem"],
    "storage": ["data storage", "hard drive", "cloud storage"],
    "backup": ["data backup", "backup server", "file backup"],
    "recovery": ["data recovery", "disaster recovery", "system recovery"]
}


# ========================================== 
# 9. CATEGORY ANALYSIS
# ========================================== 

VIDEO_CATEGORY = None
CATEGORY_KEYWORDS = []

def analyze_script_and_set_category(script, topic):
    """Analyze script and determine primary category"""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    print("\n🔍 Analyzing script to determine video category...")
    
    full_text = (script + " " + topic).lower()
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has',
                  'this', 'that', 'will', 'can', 'could', 'would', 'should', 'may', 'might'}
    
    words = [w for w in re.findall(r'\b\w+\b', full_text) if len(w) >= 4 and w not in stop_words]
    
    category_scores = {}
    for category, terms in VISUAL_MAP.items():
        score = 0
        if category in full_text:
            score += full_text.count(category) * 10
        
        for term in terms:
            term_words = term.split()
            for term_word in term_words:
                if len(term_word) > 3 and term_word in words:
                    score += 3
        
        for word in words[:50]:
            if word in category or category in word:
                score += 5
        
        if score > 0:
            category_scores[category] = score
    
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_categories:
        VIDEO_CATEGORY = sorted_categories[0][0]
        category_score = sorted_categories[0][1]
        
        CATEGORY_KEYWORDS = [VIDEO_CATEGORY]
        
        for word in words[:30]:
            if word in VIDEO_CATEGORY or VIDEO_CATEGORY in word:
                CATEGORY_KEYWORDS.append(word)
            for term in VISUAL_MAP[VIDEO_CATEGORY]:
                if word in term:
                    CATEGORY_KEYWORDS.append(word)
        
        CATEGORY_KEYWORDS = list(set(CATEGORY_KEYWORDS))[:10]
        
        print(f"✅ VIDEO CATEGORY LOCKED: '{VIDEO_CATEGORY}' (score: {category_score})")
        print(f"📋 Category Keywords: {', '.join(CATEGORY_KEYWORDS[:5])}")
        
        print(f"📊 Top Categories Detected:")
        for i, (cat, score) in enumerate(sorted_categories[:3]):
            print(f"   {i+1}. {cat}: {score} points")
    else:
        VIDEO_CATEGORY = "technology"
        CATEGORY_KEYWORDS = ["technology", "tech", "digital", "innovation"]
        print(f"⚠️ No clear category detected. Using default: '{VIDEO_CATEGORY}'")
    
    return VIDEO_CATEGORY, CATEGORY_KEYWORDS

# ========================================== 
# 10. INTELLIGENT QUERY GENERATION
# ========================================== 

def get_smart_query(text, sentence_index, total_sentences, full_script=""):
    """
    Generate search query using:
    1. T5 Transformer (primary)
    2. Category-locked fallback
    """
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    print(f"    🧠 Generating smart query for: '{text[:50]}...'")
    
    # Try T5 first
    t5_tags = generate_smart_query_t5(text)
    
    if t5_tags and len(t5_tags) > 0:
        # Filter T5 tags for halal content
        clean_tags = []
        for tag in t5_tags:
            is_halal, forbidden = is_content_halal(tag)
            if is_halal:
                clean_tags.append(tag)
            else:
                print(f"    ⚠️ Filtered forbidden tag: '{tag}' (contains: {forbidden})")
        
        if clean_tags:
            primary_query = clean_tags[0]
            
            # Add category context if not present
            if VIDEO_CATEGORY and VIDEO_CATEGORY not in primary_query.lower():
                primary_query = f"{primary_query} {VIDEO_CATEGORY}"
            
            fallbacks = clean_tags[1:4] if len(clean_tags) > 1 else []
            
            # Add category-based fallbacks
            if VIDEO_CATEGORY in VISUAL_MAP:
                category_terms = VISUAL_MAP[VIDEO_CATEGORY]
                for term in category_terms[:2]:
                    if term not in fallbacks:
                        fallbacks.append(term)
            
            print(f"    ✓ T5 Query: '{primary_query}'")
            print(f"      T5 Tags: {clean_tags}")
            
            return f"{primary_query} 4k", [f"{fb} cinematic" for fb in fallbacks[:3]], clean_tags
    
    # Fallback to category-based query
    print(f"    → Using category-based fallback")
    return get_category_locked_query(text, sentence_index, total_sentences)

def get_category_locked_query(text, sentence_index, total_sentences):
    """Fallback query generation using category system"""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    if not VIDEO_CATEGORY:
        return "abstract technology", ["digital background", "data visualization"], ["technology"]
    
    text_lower = text.lower()
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has',
                  'this', 'that', 'these', 'those', 'will', 'would', 'could', 'should'}
    
    sentence_words = [w for w in re.findall(r'\b\w+\b', text_lower) if len(w) >= 4 and w not in stop_words]
    
    category_terms = VISUAL_MAP.get(VIDEO_CATEGORY, [])
    safe_category_terms = [term for term in category_terms if len(term.split()) <= 4]
    
    # Find matches
    matched_terms = []
    for word in sentence_words:
        for term in category_terms:
            if word in term or (len(word) > 5 and term in word):
                matched_terms.append(term)
                break
    
    keyword_matches = [w for w in sentence_words if w in CATEGORY_KEYWORDS]
    important_words = [w for w in sentence_words if len(w) >= 6][:3]
    
    # Build query
    if matched_terms:
        primary = matched_terms[0]
        keywords = [VIDEO_CATEGORY, matched_terms[0]] + sentence_words[:2]
    elif keyword_matches:
        main_keyword = keyword_matches[0]
        related_terms = [term for term in safe_category_terms if main_keyword in term]
        if related_terms:
            primary = random.choice(related_terms)
        else:
            primary = f"{main_keyword} {VIDEO_CATEGORY}"
        keywords = [VIDEO_CATEGORY, main_keyword] + sentence_words[:2]
    elif important_words:
        segment_keyword = important_words[0]
        if segment_keyword in ['problem', 'issue', 'thing', 'situation', 'question']:
            segment_keyword = random.choice(safe_category_terms) if safe_category_terms else VIDEO_CATEGORY
        primary = f"{segment_keyword} {VIDEO_CATEGORY}"
        keywords = [VIDEO_CATEGORY, segment_keyword]
    else:
        term_index = sentence_index % len(safe_category_terms) if safe_category_terms else 0
        primary = safe_category_terms[term_index] if safe_category_terms else VIDEO_CATEGORY
        keywords = [VIDEO_CATEGORY]
    
    # Build fallbacks
    fallbacks = []
    if safe_category_terms:
        fallbacks = random.sample(safe_category_terms, min(3, len(safe_category_terms)))
    else:
        fallbacks = [f"{VIDEO_CATEGORY} abstract", f"{VIDEO_CATEGORY} background"]
    
    if VIDEO_CATEGORY not in primary.lower():
        primary = f"{primary} {VIDEO_CATEGORY}"
    
    primary = f"{primary} 4k"
    fallbacks = [f"{fb} cinematic" for fb in fallbacks]
    
    return primary, fallbacks, keywords

# ========================================== 
# 11. VIDEO SEARCH WITH HALAL FILTER
# ========================================== 

USED_VIDEO_URLS = set()

def intelligent_video_search(query, service, keys, page=1):
    """Search for videos with halal content filter"""
    
    # Filter query first
    query = filter_forbidden_from_query(query)
    
    all_results = []
    
    if service == 'pexels' and keys:
        try:
            key = random.choice([k for k in keys if k])
            print(f"    Searching Pexels: {query}")
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 25,
                "page": page,
                "orientation": "landscape",
                "size": "medium"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    # Check if video content is halal
                    video_text = f"{video.get('user', {}).get('name', '')} {video.get('url', '')}"
                    is_halal, forbidden = is_content_halal(video_text)
                    if not is_halal:
                        continue
                    
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
                "per_page": 25,
                "page": page,
                "orientation": "horizontal",
                "video_type": "film",
                "min_width": 1280,
                "safesearch": "true"  # Enable safe search
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    # Check if video content is halal
                    video_text = f"{video.get('tags', '')} {video.get('user', '')}"
                    is_halal, forbidden = is_content_halal(video_text)
                    if not is_halal:
                        continue
                    
                    videos_dict = video.get('videos', {})
                    
                    width = videos_dict.get('large', {}).get('width', 0)
                    height = videos_dict.get('large', {}).get('height', 0)
                    
                    if height > width:
                        continue
                    
                    best_quality = None
                    quality_name = 'medium'
                    for q in ['large', 'medium', 'small']:
                        if q in videos_dict:
                            best_quality = videos_dict[q]
                            quality_name = q
                            break
                    
                    if best_quality:
                        all_results.append({
                            'url': best_quality['url'],
                            'title': video.get('tags', query),
                            'description': f"Pixabay video ID: {video.get('id', '')}",
                            'duration': video.get('duration', 0),
                            'service': 'pixabay',
                            'quality': quality_name,
                            'width': best_quality.get('width', 0),
                            'height': best_quality.get('height', 0),
                            'license': 'free'
                        })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

# ========================================== 
# 12. ENHANCED SCORING WITH CLIP
# ========================================== 

def calculate_relevance_score(video, query, sentence_text, context_keywords, full_script="", topic=""):
    """Calculate relevance score for video"""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    score = 0
    
    video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
    query_lower = query.lower()
    
    # Halal check
    is_halal, forbidden = is_content_halal(video_text)
    if not is_halal:
        return -100  # Instant disqualification
    
    # Category match
    if VIDEO_CATEGORY and VIDEO_CATEGORY in video_text:
        score += 25
    
    for keyword in CATEGORY_KEYWORDS[:5]:
        if keyword in video_text:
            score += 8
    
    # Query match
    query_terms = [t for t in query_lower.split() if len(t) > 3 and t not in ['landscape', 'cinematic', 'abstract', '4k']]
    for term in query_terms:
        if term in video_text:
            score += 10
    
    # Context keywords
    for keyword in context_keywords:
        if len(keyword) > 3 and keyword in video_text:
            score += 8
    
    # Quality bonus
    quality = video.get('quality', '').lower()
    if '4k' in quality or 'uhd' in quality:
        score += 15
    elif 'hd' in quality or 'large' in quality:
        score += 12
    elif 'medium' in quality:
        score += 8
    else:
        score += 4
    
    # Duration bonus
    duration = video.get('duration', 0)
    if duration >= 15:
        score += 5
    elif duration >= 10:
        score += 3
    
    # Landscape check
    width = video.get('width', 0)
    height = video.get('height', 0)
    if width and height:
        aspect_ratio = width / height
        if aspect_ratio >= 1.5:
            score += 10
        elif aspect_ratio >= 1.2:
            score += 5
        elif aspect_ratio < 1.0:
            score -= 50
    
    # Random factor for variety
    score += random.randint(0, 3)
    
    return min(100, max(0, score))

# ========================================== 
# 13. STATUS UPDATES
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None, file_url_no_subs=None):
    """Updates status.json in GitHub repo"""
    
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
    if file_url_no_subs:
        data["file_url_no_subtitles"] = file_url_no_subs
    
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
# 14. SCRIPT & AUDIO GENERATION
# ========================================== 

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = f"""
CRITICAL RULES:
- Write ONLY spoken narration text
- NO stage directions like [Music fades], [Intro], [Outro]
- NO sound effects descriptions
- NO [anything in brackets]
- Start directly with the content
- End directly with the conclusion
- Pure voiceover script only
- AVOID any mention of: {', '.join(FORBIDDEN_TERMS[:20])}
- Keep content family-friendly and appropriate
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
        prompt = f"{base_instructions}\nWrite a YouTube documentary script about '{topic}'. {words} words."
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
# 15. VISUAL PROCESSING WITH T5 + CLIP
# ========================================== 

def download_video_candidate(video_info, index, attempt):
    """Download a single video candidate for CLIP evaluation"""
    try:
        local_path = CLIPS_DIR / f"candidate_{index}_{attempt}.mp4"
        
        response = requests.get(video_info['url'], timeout=30, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 10000:
                video_info['local_path'] = str(local_path)
                return video_info
    except Exception as e:
        print(f"      Download error: {str(e)[:40]}")
    
    return None

def get_best_clip_with_t5_and_clip(i, sent, total_sentences, full_script, topic, max_retries=4):
    """
    Get best video clip using:
    1. T5 for smart query generation
    2. Traditional search
    3. CLIP for visual-text matching
    """
    dur = max(3.5, sent['end'] - sent['start'])
    
    # Generate smart query using T5
    primary_query, fallback_queries, context_keywords = get_smart_query(
        sent['text'], i, total_sentences, full_script
    )
    
    print(f"  🔍 Clip {i+1}/{total_sentences}: '{sent['text'][:50]}...'")
    
    for attempt in range(max_retries):
        # Select query
        if attempt == 0:
            query = primary_query
        elif attempt < len(fallback_queries) + 1:
            query = fallback_queries[attempt - 1]
        else:
            query = f"{VIDEO_CATEGORY} abstract 4k"
        
        print(f"    Attempt {attempt+1}: '{query}'")
        
        # Search both services
        all_results = []
        page = random.randint(1, 3)
        
        if PEXELS_KEYS and PEXELS_KEYS[0]:
            pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS, page)
            for video in pexels_results:
                if video['url'] not in USED_VIDEO_URLS:
                    relevance = calculate_relevance_score(
                        video, query, sent['text'], context_keywords, full_script, topic
                    )
                    video['relevance_score'] = relevance
                    all_results.append(video)
        
        if PIXABAY_KEYS and PIXABAY_KEYS[0]:
            pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS, page)
            for video in pixabay_results:
                if video['url'] not in USED_VIDEO_URLS:
                    relevance = calculate_relevance_score(
                        video, query, sent['text'], context_keywords, full_script, topic
                    )
                    video['relevance_score'] = relevance
                    all_results.append(video)
        
        # Sort by initial relevance
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Take top candidates for CLIP evaluation
        top_candidates = all_results[:5]
        
        if top_candidates:
            # Download candidates for CLIP evaluation
            downloaded_candidates = []
            for idx, candidate in enumerate(top_candidates):
                downloaded = download_video_candidate(candidate, i, idx)
                if downloaded:
                    downloaded_candidates.append(downloaded)
                if len(downloaded_candidates) >= 3:  # Limit to 3 for speed
                    break
            
            if downloaded_candidates:
                # Use CLIP to rank
                clip_ranked = rank_videos_by_clip(sent['text'], downloaded_candidates)
                
                if clip_ranked:
                    best_video = clip_ranked[0]
                    
                    # Check minimum scores
                    min_relevance = 30 if attempt < 2 else 15
                    min_clip = 20 if attempt < 2 else 10
                    
                    if (best_video.get('relevance_score', 0) >= min_relevance or 
                        best_video.get('clip_score', 0) >= min_clip):
                        
                        USED_VIDEO_URLS.add(best_video['url'])
                        local_path = best_video.get('local_path')
                        
                        if local_path and os.path.exists(local_path):
                            # Process video
                            out = TEMP_DIR / f"s_{i}.mp4"
                            
                            cmd = [
                                "ffmpeg", "-y", "-hwaccel", "cuda",
                                "-i", str(local_path),
                                "-t", str(dur),
                                "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                                "-c:v", "h264_nvenc",
                                "-preset", "p4",
                                "-b:v", "8M",
                                "-an",
                                str(out)
                            ]
                            
                            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            if result.returncode == 0 and os.path.exists(out):
                                print(f"    ✓ Selected: {best_video['service']} | Relevance: {best_video.get('relevance_score', 0):.0f} | CLIP: {best_video.get('clip_score', 0):.1f}%")
                                
                                # Cleanup candidates
                                for cand in downloaded_candidates:
                                    cand_path = cand.get('local_path')
                                    if cand_path and cand_path != local_path and os.path.exists(cand_path):
                                        try:
                                            os.remove(cand_path)
                                        except:
                                            pass
                                
                                return str(out)
            
            # Cleanup failed candidates
            for cand in downloaded_candidates if 'downloaded_candidates' in dir() else []:
                cand_path = cand.get('local_path')
                if cand_path and os.path.exists(cand_path):
                    try:
                        os.remove(cand_path)
                    except:
                        pass
    
    # Fallback - category-themed gradient
    print(f"  → Clip {i}: Using category-themed fallback")
    
    category_gradients = {
        "tech": "0x1a1a2e:0x0f3460",
        "technology": "0x1a1a2e:0x0f3460",
        "internet": "0x0f3460:0x533483",
        "digital": "0x16213e:0x533483",
        "ai": "0x0f3460:0x16213e",
        "computer": "0x1a1a2e:0x16213e",
        "business": "0x1e3a5f:0x2a2d34",
        "finance": "0x0f3460:0x1e3a5f",
        "nature": "0x1e4d2b:0x2d5016",
        "science": "0x1e3a5f:0x0f3460",
    }
    
    gradient = category_gradients.get(VIDEO_CATEGORY, "0x1a1a2e:0x16213e")
    
    out = TEMP_DIR / f"s_{i}_fallback.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={dur}",
        "-vf", f"fade=in:0:30,fade=out:st={dur-1}:d=1",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-t", str(dur),
        str(out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

def process_visuals_dual_output(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs, full_script="", topic=""):
    """
    Process visuals and create TWO outputs:
    1. Video WITHOUT subtitles
    2. Video WITH subtitles
    """
    print("🎬 Visual Processing with T5 + CLIP Intelligence...")
    
    # Analyze script and set category
    analyze_script_and_set_category(full_script, topic)
    
    # Load AI models
    print("\n🤖 Loading AI Models...")
    load_t5_model()
    load_clip_model()
    
    # Process all clips
    print(f"\n📥 Downloading {len(sentences)} clips (Category: {VIDEO_CATEGORY})...")
    clips = []
    
    for i, sent in enumerate(sentences):
        update_status(55 + int((i/len(sentences))*25), f"Processing clip {i+1}/{len(sentences)}...")
        clip_path = get_best_clip_with_t5_and_clip(i, sent, len(sentences), full_script, topic)
        clips.append(clip_path)
    
    # Concatenate clips
    print("\n🔗 Concatenating video clips...")
    with open("list.txt", "w") as f:
        for c in clips:
            if os.path.exists(c):
                f.write(f"file '{c}'\n")
    
    concat_video = TEMP_DIR / "concat.mp4"
    subprocess.run(
        f"ffmpeg -y -f concat -safe 0 -i list.txt -c:v h264_nvenc -preset p1 -b:v 10M {concat_video}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    if not os.path.exists(concat_video):
        print("❌ Concatenation failed")
        return False, False
    
    # Get audio duration
    import wave
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            audio_duration = wav_file.getnframes() / float(wav_file.getframerate())
            print(f"🎵 Audio duration: {audio_duration:.1f} seconds")
    except:
        audio_duration = None
    
    # ========================================
    # STEP 1: Create video WITHOUT subtitles
    # ========================================
    print("\n🎬 Rendering video WITHOUT subtitles...")
    update_status(82, "Rendering video without subtitles...")
    
    if logo_path and os.path.exists(logo_path):
        filter_no_subs = "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[v]"
        cmd_no_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(concat_video),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_no_subs,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    else:
        filter_no_subs = "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v]"
        cmd_no_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(concat_video),
            "-i", str(audio_path),
            "-filter_complex", filter_no_subs,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    
    if audio_duration:
        cmd_no_subs.extend(["-t", str(audio_duration)])
    
    cmd_no_subs.append(str(output_no_subs))
    
    result_no_subs = subprocess.run(cmd_no_subs, capture_output=True, text=True)
    success_no_subs = result_no_subs.returncode == 0 and os.path.exists(output_no_subs)
    
    if success_no_subs:
        print(f"✅ Video WITHOUT subtitles: {output_no_subs}")
    else:
        print(f"❌ Failed to render video without subtitles: {result_no_subs.stderr[:200]}")
    
    # ========================================
    # STEP 2: Create video WITH subtitles
    # ========================================
    print("\n🎬 Rendering video WITH subtitles...")
    update_status(90, "Burning subtitles into video...")
    
    # Escape ASS file path for FFmpeg
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_with_subs = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        cmd_with_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(concat_video),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_with_subs,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    else:
        filter_with_subs = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[bg]subtitles='{ass_path}'[v]"
        cmd_with_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(concat_video),
            "-i", str(audio_path),
            "-filter_complex", filter_with_subs,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    
    if audio_duration:
        cmd_with_subs.extend(["-t", str(audio_duration)])
    
    cmd_with_subs.append(str(output_with_subs))
    
    result_with_subs = subprocess.run(cmd_with_subs, capture_output=True, text=True)
    success_with_subs = result_with_subs.returncode == 0 and os.path.exists(output_with_subs)
    
    if success_with_subs:
        print(f"✅ Video WITH subtitles: {output_with_subs}")
    else:
        print(f"❌ Failed to render video with subtitles: {result_with_subs.stderr[:200]}")
    
    return success_no_subs, success_with_subs

# ========================================== 
# 16. MAIN EXECUTION
# ========================================== 

# ========================================== 
# 16. MAIN EXECUTION - FIXED MODE HANDLING
# ========================================== 

def main():
    """Main execution with dual output"""
    update_status(0, "Starting AI Video Generator...")
    
    # Set random seed for reproducibility
    random.seed(int(time.time()))
    
    # Clean previous temp files
    for f in TEMP_DIR.glob("*"):
        try:
            f.unlink()
        except:
            pass
    
    # Process based on mode
    if MODE == "topic":
        print(f"📝 Mode: Generate from Topic (Topic: {TOPIC}, Duration: {DURATION_MINS} min)")
        update_status(5, "Generating script from topic...")
        
        # Generate script from topic
        script = generate_script(TOPIC, DURATION_MINS)
        
        if not script or len(script) < 100:
            update_status(100, "Script generation failed", "failed")
            print("❌ Script generation failed")
            return
        
        print(f"📄 Script generated from topic ({len(script.split())} words)")
        
    elif MODE == "script":
        print(f"📝 Mode: Use Provided Script")
        update_status(5, "Processing provided script...")
        
        if not SCRIPT_TEXT or len(SCRIPT_TEXT.strip()) < 100:
            update_status(100, "Script text is too short or empty", "failed")
            print("❌ Script text is too short")
            return
        
        script = SCRIPT_TEXT
        print(f"📄 Using provided script ({len(script.split())} words)")
    
    else:
        print(f"❌ Unknown mode: {MODE}")
        update_status(100, f"Unknown mode: {MODE}", "failed")
        return
    
    # Now continue with voice cloning and video generation
    # Step 1: Clone voice
    update_status(20, "Cloning voice...")
    audio_path = TEMP_DIR / "audio.wav"
    
    # Check if voice file exists
    if not VOICE_PATH or not os.path.exists(VOICE_PATH):
        print("❌ Voice file not found or not provided")
        
        # Try to download from GitHub if it's a path
        if VOICE_PATH and ("/" in VOICE_PATH or "\\" in VOICE_PATH):
            print(f"🔄 Trying to download voice file: {VOICE_PATH}")
            voice_downloaded = TEMP_DIR / "voice_reference.mp3"
            if download_asset(VOICE_PATH, voice_downloaded):
                VOICE_PATH = str(voice_downloaded)
                print(f"✅ Voice file downloaded from GitHub")
            else:
                update_status(100, "Voice reference missing", "failed")
                return
        else:
            update_status(100, "Voice reference missing", "failed")
            return
    
    # Clone voice
    if os.path.exists(VOICE_PATH):
        success = clone_voice_robust(script, VOICE_PATH, audio_path)
        if not success:
            print("⚠️ Voice cloning failed")
            update_status(100, "Voice cloning failed", "failed")
            return
    else:
        print("❌ Voice file not found after download attempt")
        update_status(100, "Voice reference missing", "failed")
        return
    
    if not os.path.exists(audio_path):
        print("❌ Audio file not created")
        update_status(100, "Audio creation failed", "failed")
        return
    
    # Step 2: Transcribe audio for timing
    update_status(52, "Transcribing audio for timing...")
    
    # Try AssemblyAI first
    sentences = []
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            print("📝 Transcribing audio...")
            transcript = transcriber.transcribe(str(audio_path))
            
            if transcript.status == aai.TranscriptStatus.completed:
                for utterance in transcript.utterances:
                    sentences.append({
                        'text': utterance.text,
                        'start': utterance.start / 1000.0,
                        'end': utterance.end / 1000.0
                    })
                print(f"✅ Transcription complete: {len(sentences)} sentences")
            else:
                raise Exception("Transcription failed")
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}. Using fallback...")
            sentences = []
    
    # Fallback if no sentences from AssemblyAI
    if not sentences:
        # Fallback timing logic
        words = script.split()
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                total_duration = wav_file.getnframes() / float(wav_file.getframerate())
            
            words_per_second = len(words) / total_duration
            words_per_sentence = 12
            
            for i in range(0, len(words), words_per_sentence):
                chunk = words[i:i + words_per_sentence]
                sentence_duration = len(chunk) / words_per_second
                start_time = i / len(words) * total_duration
                end_time = min(start_time + sentence_duration, total_duration)
                sentences.append({
                    "text": ' '.join(chunk),
                    "start": start_time,
                    "end": end_time
                })
            
            if sentences:
                sentences[-1]['end'] += 1.5
        except:
            # Ultimate fallback
            script_sentences = re.split(r'(?<=[.!?])\s+', script)
            avg_duration = len(script) / len(script_sentences) / 10
            for i, sent_text in enumerate(script_sentences):
                if sent_text.strip():
                    sentences.append({
                        "text": sent_text.strip(),
                        "start": i * 5.0,
                        "end": (i + 1) * 5.0
                    })
    
    print(f"⏱️  {len(sentences)} sentences with timing")
    
    # Step 3: Create subtitles
    update_status(55, "Creating subtitle file...")
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # Step 4: Process visuals (dual output)
    output_no_subs = OUTPUT_DIR / f"video_{JOB_ID}_no_subs.mp4"
    output_with_subs = OUTPUT_DIR / f"video_{JOB_ID}_with_subs.mp4"
    
    # Check for logo
    logo_path = None
    if LOGO_PATH and LOGO_PATH.strip() not in ["", "None", "null", "undefined"]:
        if os.path.exists(LOGO_PATH):
            logo_path = LOGO_PATH
        else:
            # Try to download from GitHub
            logo_downloaded = TEMP_DIR / "logo.png"
            if download_asset(LOGO_PATH, logo_downloaded):
                logo_path = str(logo_downloaded)
                print(f"✅ Logo downloaded from GitHub")
            else:
                print(f"⚠️ Logo not found: {LOGO_PATH}")
    
    # Call the process_visuals_dual_output function (make sure it exists!)
    success_no_subs, success_with_subs = process_visuals_dual_output(
        sentences, audio_path, ass_file, logo_path, 
        output_no_subs, output_with_subs, script, TOPIC
    )
    
    # Step 5: Upload to Google Drive
    drive_link_no_subs = None
    drive_link_with_subs = None
    
    if success_no_subs and os.environ.get("OAUTH_CLIENT_ID"):
        update_status(95, "Uploading video without subtitles...")
        drive_link_no_subs = upload_to_google_drive(
            output_no_subs, 
            f"{TOPIC.replace(' ', '_')}_no_subs_{JOB_ID}.mp4"
        )
    
    if success_with_subs and os.environ.get("OAUTH_CLIENT_ID"):
        update_status(97, "Uploading video with subtitles...")
        drive_link_with_subs = upload_to_google_drive(
            output_with_subs,
            f"{TOPIC.replace(' ', '_')}_with_subs_{JOB_ID}.mp4"
        )
    
    # Step 6: Final status
    if success_no_subs or success_with_subs:
        update_status(
            100, 
            "Video generation complete!", 
            "complete",
            drive_link_with_subs if drive_link_with_subs else drive_link_no_subs,
            drive_link_no_subs
        )
        print("\n" + "="*50)
        print("🎉 VIDEO GENERATION COMPLETE!")
        print("="*50)
        
        if success_no_subs:
            print(f"📹 Video WITHOUT subtitles: {output_no_subs}")
            if drive_link_no_subs:
                print(f"☁️  Drive link (no subs): {drive_link_no_subs}")
        
        if success_with_subs:
            print(f"📹 Video WITH subtitles: {output_with_subs}")
            if drive_link_with_subs:
                print(f"☁️  Drive link (with subs): {drive_link_with_subs}")
        
        # Get file sizes
        if os.path.exists(output_no_subs):
            size_mb = os.path.getsize(output_no_subs) / (1024*1024)
            print(f"📦 File size (no subs): {size_mb:.1f} MB")
        
        if os.path.exists(output_with_subs):
            size_mb = os.path.getsize(output_with_subs) / (1024*1024)
            print(f"📦 File size (with subs): {size_mb:.1f} MB")
    else:
        update_status(100, "Video generation failed", "failed")
        print("❌ Video generation failed")
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except:
        pass
    
    # Clear AI models from memory
    global T5_MODEL, CLIP_MODEL
    T5_MODEL = None
    CLIP_MODEL = None
    CLIP_PROCESSOR = None
    T5_TOKENIZER = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()
    print("✅ Cleanup complete")
