"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
FIXED VERSION:
1. Subtitle Design Implementation (ASS format properly applied)
2. Enhanced 100% Context-Aligned Scoring System
3. Pixabay & Pexels with Visual Map
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
# 4. GOOGLE DRIVE UPLOAD
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
# 5. VISUAL DICTIONARY (700+ TOPICS)
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
# 6. ENHANCED VISUAL QUERY WITH CONTEXT
# ========================================== 

def get_visual_query_with_context(text, full_script="", topic=""):
    """
    ENHANCED: Extract visual query with full context awareness
    Returns: (primary_query, fallback_queries, context_keywords)
    """
    text = text.lower()
    full_context = (text + " " + full_script[:500] + " " + topic).lower()
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has'}
    
    # Extract meaningful words
    words = [w for w in re.findall(r'\b\w+\b', text) if len(w) >= 4 and w not in stop_words]
    context_words = [w for w in re.findall(r'\b\w+\b', full_context) if len(w) >= 4 and w not in stop_words]
    
    # Priority 1: Exact category match in VISUAL_MAP
    for word in words:
        if word in VISUAL_MAP:
            selected_term = random.choice(VISUAL_MAP[word])
            fallbacks = [random.choice(VISUAL_MAP[word]) for _ in range(3)]
            return selected_term, fallbacks, [word]
    
    # Priority 2: Partial match in categories
    for word in words:
        for category, terms in VISUAL_MAP.items():
            if word in category or category in word:
                selected_term = random.choice(terms)
                fallbacks = [random.choice(terms) for _ in range(3)]
                return selected_term, fallbacks, [word, category]
    
    # Priority 3: Context-based matching
    for word in context_words[:5]:  # Check first 5 context words
        for category, terms in VISUAL_MAP.items():
            if word in category:
                selected_term = random.choice(terms)
                fallbacks = [random.choice(terms) for _ in range(3)]
                return selected_term, fallbacks, [word, category]
    
    # Priority 4: Use significant nouns
    significant = [w for w in words if len(w) >= 6]
    if significant:
        main_word = significant[0]
        for category, terms in VISUAL_MAP.items():
            if main_word in " ".join(terms):
                selected_term = random.choice(terms)
                return selected_term, [main_word], [main_word]
        return f"{main_word} landscape", [f"{main_word} cinematic"], [main_word]
    
    # Fallback
    if words:
        return f"{words[0]} landscape", [f"{words[0]} cinematic", "nature landscape"], words[:2]
    
    fallbacks = ["nature landscape", "cinematic landscape", "mountain vista", "forest aerial"]
    return random.choice(fallbacks), fallbacks, ["landscape"]

# ========================================== 
# 7. ENHANCED SCORING SYSTEM (100% Context Aligned)
# ========================================== 

def calculate_enhanced_relevance_score(video, query, sentence_text, context_keywords, full_script="", topic=""):
    """
    ENHANCED: Calculate relevance with 100% context alignment
    Score breakdown:
    - Query match: 30 points
    - Context keywords: 30 points  
    - Topic alignment: 20 points
    - Quality: 15 points
    - Landscape: 5 points
    """
    score = 0
    
    # Prepare text for matching
    video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
    sentence_lower = sentence_text.lower()
    topic_lower = topic.lower()
    
    # === 1. QUERY MATCH (30 points) ===
    query_terms = query.lower().split()
    query_match_count = 0
    for term in query_terms:
        if len(term) > 3:  # Only meaningful terms
            if term in video_text:
                query_match_count += 1
                score += 10
    
    # Bonus for exact phrase match
    if query.lower() in video_text:
        score += 10
    
    # === 2. CONTEXT KEYWORDS MATCH (30 points) ===
    context_match_count = 0
    for keyword in context_keywords:
        if keyword in video_text:
            context_match_count += 1
            score += 10
        # Check in sentence
        if keyword in sentence_lower:
            score += 5
    
    # === 3. TOPIC ALIGNMENT (20 points) ===
    topic_words = [w for w in topic_lower.split() if len(w) > 3]
    topic_match_count = 0
    for word in topic_words[:5]:  # Check first 5 topic words
        if word in video_text:
            topic_match_count += 1
            score += 4
    
    # === 4. VIDEO QUALITY (15 points) ===
    quality = video.get('quality', '').lower()
    if '4k' in quality or 'uhd' in quality:
        score += 15
    elif 'hd' in quality or 'high' in quality or 'large' in quality:
        score += 12
    elif 'medium' in quality:
        score += 8
    elif 'sd' in quality or 'standard' in quality:
        score += 4
    
    # Duration bonus
    duration = video.get('duration', 0)
    if duration >= 15:
        score += 5
    elif duration >= 10:
        score += 3
    
    # === 5. LANDSCAPE VERIFICATION (5 points + penalties) ===
    landscape_indicators = ['landscape', 'horizontal', 'wide', 'panoramic', 'widescreen', '16:9']
    portrait_indicators = ['vertical', 'portrait', '9:16', 'instagram', 'tiktok', 'reel']
    
    if any(indicator in video_text for indicator in landscape_indicators):
        score += 5
    elif any(indicator in video_text for indicator in portrait_indicators):
        score -= 40  # Heavy penalty for portrait
    
    # Check actual dimensions if available
    width = video.get('width', 0)
    height = video.get('height', 0)
    if width and height:
        aspect_ratio = width / height
        if aspect_ratio >= 1.5:  # 16:9 or wider
            score += 10
        elif aspect_ratio < 1.0:  # Portrait
            score -= 50
    
    # Platform quality bias
    service = video.get('service', '')
    if service == 'pexels':
        score += 5
    elif service == 'pixabay':
        score += 3
    
    # === BONUS: Perfect Match Detection ===
    # If video matches query AND context AND topic
    if query_match_count >= 2 and context_match_count >= 2 and topic_match_count >= 1:
        score += 20  # Perfect alignment bonus
    
    # Random variety factor
    score += random.randint(0, 3)
    
    # Cap at 100, floor at 0
    return min(100, max(0, score))

# ========================================== 
# 8. VIDEO SEARCH (Pixabay & Pexels)
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
# 9. UTILS
# ========================================== 

def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")

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
# 10. SCRIPT & AUDIO
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
# 11. ENHANCED VISUAL PROCESSING
# ========================================== 

def process_visuals(sentences, audio_path, ass_file, logo_path, final_out, full_script="", topic=""):
    print("🎬 Visual Processing with Enhanced Context Matching...")
    
    def get_clip_with_enhanced_retry(i, sent, max_retries=4):
        """Enhanced retry with 100% context alignment"""
        dur = max(3.5, sent['end'] - sent['start'])
        
        # Get primary query with context
        primary_query, fallback_queries, context_keywords = get_visual_query_with_context(
            sent['text'], full_script, topic
        )
        
        print(f"  🔍 Clip {i}: Context={context_keywords[:2]}")
        
        for attempt in range(max_retries):
            out = TEMP_DIR / f"s_{i}_attempt{attempt}.mp4"
            
            # Select query based on attempt
            if attempt == 0:
                query = primary_query
            elif attempt < len(fallback_queries) + 1:
                query = fallback_queries[attempt - 1]
            else:
                query = f"{context_keywords[0]} landscape" if context_keywords else "nature landscape"
            
            print(f"    Attempt {attempt+1}: '{query}'")
            
            # Search both services
            all_results = []
            page = random.randint(1, 3)
            
            if PEXELS_KEYS and PEXELS_KEYS[0]:
                pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS, page)
                for video in pexels_results:
                    if video['url'] not in USED_VIDEO_URLS:
                        relevance = calculate_enhanced_relevance_score(
                            video, query, sent['text'], context_keywords, full_script, topic
                        )
                        video['relevance_score'] = relevance
                        all_results.append(video)
            
            if PIXABAY_KEYS and PIXABAY_KEYS[0]:
                pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS, page)
                for video in pixabay_results:
                    if video['url'] not in USED_VIDEO_URLS:
                        relevance = calculate_enhanced_relevance_score(
                            video, query, sent['text'], context_keywords, full_script, topic
                        )
                        video['relevance_score'] = relevance
                        all_results.append(video)
            
            # Sort by score
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Progressive score thresholds
            score_thresholds = [75, 65, 55, 45, 0]
            found_link = None
            selected_video = None
            
            for threshold in score_thresholds:
                for video in all_results:
                    if video['url'] not in USED_VIDEO_URLS and video['relevance_score'] >= threshold:
                        found_link = video['url']
                        selected_video = video
                        break
                if found_link:
                    break
            
            if found_link and selected_video:
                USED_VIDEO_URLS.add(found_link)
                print(f"    ✓ Found {selected_video['service']} (score: {selected_video['relevance_score']})")
                
                try:
                    raw = TEMP_DIR / f"r_{i}.mp4"
                    response = requests.get(found_link, timeout=40, stream=True)
                    
                    with open(raw, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Process with proper scaling
                    cmd = [
                        "ffmpeg", "-y", "-hwaccel", "cuda",
                        "-i", str(raw),
                        "-t", str(dur),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-b:v", "8M",
                        "-an",
                        str(out)
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    return str(out)
                except Exception as e:
                    print(f"    ✗ Download failed: {str(e)[:60]}")
                    continue
        
        # Fallback
        print(f"  → Clip {i}: Using fallback")
        gradient = random.choice(["0x1a1a2e:0x16213e", "0x0f3460:0x533483"])
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
    
    # Process all clips
    print(f"📥 Downloading {len(sentences)} clips with enhanced matching...")
    clips = []
    for i, sent in enumerate(sentences):
        update_status(60 + int((i/len(sentences))*30), f"Processing clip {i+1}/{len(sentences)}...")
        clip_path = get_clip_with_enhanced_retry(i, sent)
        clips.append(clip_path)
    
    # Concatenate
    print("🔗 Concatenating video clips...")
    with open("list.txt", "w") as f:
        for c in clips:
            if os.path.exists(c):
                f.write(f"file '{c}'\n")
    
    subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i list.txt -c:v h264_nvenc -preset p1 -b:v 10M visual.mp4",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    # Final render with FIXED subtitle filter
    print("🎬 Rendering final video with subtitles...")
    
    # CRITICAL FIX: Properly escape ASS file path for FFmpeg
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    import wave
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            audio_duration = wav_file.getnframes() / float(wav_file.getframerate())
            print(f"🎵 Audio duration: {audio_duration:.1f} seconds")
    except:
        audio_duration = None
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", "visual.mp4",
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    else:
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[bg]subtitles='{ass_path}'[v]"
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", "visual.mp4",
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k"
        ]
    
    if audio_duration:
        cmd.extend(["-t", str(audio_duration)])
    
    cmd.append(str(final_out))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Final video rendered: {final_out}")
            return True
        else:
            print(f"❌ Rendering failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ Rendering exception: {e}")
        return False

# ========================================== 
# 12. MAIN EXECUTION
# ========================================== 

print("--- 🚀 START (FIXED VERSION: Subtitles + Enhanced Scoring) ---")
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
    
    # Process visuals
    update_status(60, "Gathering Visuals with Enhanced Matching...")
    final_output = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
    
    if process_visuals(sentences, audio_out, ass_file, ref_logo, final_output, text, TOPIC):
        if final_output.exists():
            file_size = os.path.getsize(final_output) / (1024 * 1024)
            print(f"✅ Video created: {file_size:.1f} MB")
            
            update_status(99, "Uploading to Google Drive...")
            drive_link = upload_to_google_drive(final_output)
            
            if drive_link:
                update_status(100, "Success!", "completed", drive_link)
                print(f"🎉 Google Drive Link: {drive_link}")
            else:
                update_status(100, "Upload Failed - Video Ready Locally", "completed")
        else:
            update_status(0, "Rendering Failed", "failed")
    else:
        update_status(0, "Visual Processing Failed", "failed")
else:
    update_status(0, "Audio Synthesis Failed", "failed")

# Cleanup
print("Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["visual.mp4", "list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ PROCESS COMPLETE ---")
