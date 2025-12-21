"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD
============================================
FIXED VERSION WITH IMPROVEMENTS:
1. Islamic content filtering
2. Context-aware queries using sentence transformers
3. Universal topic support (not just tech/motivation)
4. Dual video output (with/without subtitles)
5. Enhanced audio controls
6. Improved realism with cinematic effects
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
        "sentence-transformers",
        "transformers",
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

# Try to load sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️ Sentence transformers not available, using fallback")

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

# NEW HTML CONTROLS
SUBTITLE_STYLE = """{{SUBTITLE_STYLE_PLACEHOLDER}}"""  # Default: "mrbeast_yellow"
AUDIO_SPEED = float("""{{AUDIO_SPEED_PLACEHOLDER}}""")  # Default: 1.0
AUDIO_PITCH = float("""{{AUDIO_PITCH_PLACEHOLDER}}""")  # Default: 1.0
AUDIO_EXAGGERATION = float("""{{AUDIO_EXAG_PLACEHOLDER}}""")  # Default: 0.5

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

# Global variables
VIDEO_NO_SUBS_URL = None
VIDEO_WITH_SUBS_URL = None
USED_VIDEO_URLS = set()

# ========================================== 
# 3. ISLAMIC CONTENT FILTER
# ========================================== 

ISLAMIC_BLACKLIST = {
    # Sexual content
    "sexual": ["nude", "naked", "bare", "topless", "undress", "exposed", "sex", "porn", "xxx", "adult", "erotic"],
    "body_parts": ["breast", "butt", "ass", "cleavage", "bikini", "lingerie", "underwear"],
    "suggestive": ["sexy", "hot", "seduce", "flirt", "provocative", "suggestive"],
    
    # Violence & Weapons
    "violence": ["violence", "violent", "fight", "fighting", "battle", "war", "combat"],
    "weapons": ["gun", "rifle", "pistol", "weapon", "knife", "sword", "bomb"],
    "blood": ["blood", "gore", "bloody", "brutal", "horror", "terror"],
    
    # Haram Activities
    "alcohol": ["alcohol", "beer", "wine", "whiskey", "vodka", "drunk", "bar", "pub"],
    "drugs": ["drug", "marijuana", "cocaine", "heroin", "addict"],
    "gambling": ["gambling", "casino", "poker", "bet", "lottery"],
    
    # Religious Prohibitions
    "idols": ["idol", "statue", "worship", "temple", "church", "cross", "buddha"],
    "magic": ["magic", "witch", "sorcerer", "spell", "curse", "fortune"],
    
    # Immoral content
    "lgbt": ["gay", "lesbian", "transgender", "lgbt", "queer", "homosexual"],
}

def is_islamic_compliant(text):
    """Check if text is Islamic compliant"""
    if not text:
        return True
    
    text_lower = text.lower()
    
    for category, terms in ISLAMIC_BLACKLIST.items():
        for term in terms:
            if term in text_lower:
                print(f"🚫 Islamic Filter: Blocked '{term}'")
                return False
    
    return True

def filter_islamic_queries(queries):
    """Filter queries for Islamic compliance"""
    filtered = []
    for query in queries:
        if is_islamic_compliant(query):
            filtered.append(query)
        else:
            # Replace with safe alternative
            safe_query = re.sub(r'\b(sexy|hot|nude|violence|gun)\b', 'safe', query, flags=re.IGNORECASE)
            if is_islamic_compliant(safe_query):
                filtered.append(safe_query)
    
    return filtered if filtered else ["nature", "landscape", "education"]

# ========================================== 
# 4. CONTEXT-AWARE QUERY GENERATION
# ========================================== 

class ContextAnalyzer:
    """Analyze context using sentence transformers"""
    
    def __init__(self):
        self.sentence_model = None
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer loaded")
            except:
                print("⚠️ Failed to load sentence transformer")
    
    def extract_key_concepts(self, text, context_history=[]):
        """Extract key concepts from text"""
        text_lower = text.lower()
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract meaningful words
        words = re.findall(r'\b\w{4,}\b', text_lower)
        meaningful_words = [w for w in words if w not in stop_words]
        
        # Get top concepts
        from collections import Counter
        word_freq = Counter(meaningful_words)
        top_words = [word for word, _ in word_freq.most_common(5)]
        
        return top_words
    
    def get_semantic_category(self, text, visual_map):
        """Determine semantic category"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, terms in visual_map.items():
            score = 0
            
            # Direct keyword matching
            if category in text_lower:
                score += 10
            
            # Check category terms
            for term in terms[:15]:
                term_words = term.split()
                for term_word in term_words:
                    if len(term_word) > 3 and term_word in text_lower:
                        score += 5
            
            if score > 0:
                category_scores[category] = score
        
        # Return top categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_categories[:3]]

# ========================================== 
# 5. ENHANCED SUBTITLE STYLES
# ========================================== 

SUBTITLE_STYLES = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow (3D Pop)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",  # Yellow
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

def create_ass_file(sentences, ass_file, style_key="mrbeast_yellow"):
    """Create ASS subtitle file with selected style"""
    style = SUBTITLE_STYLES.get(style_key, SUBTITLE_STYLES["mrbeast_yellow"])
    
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
            # Improved timing: appear slightly before, disappear slightly early
            start_time = max(0, s['start'] - 0.1)
            end_time = max(start_time + 0.5, s['end'] - 0.05)
            
            start_str = format_ass_time(start_time)
            end_str = format_ass_time(end_time)
            
            # Clean text
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            
            # Remove trailing punctuation
            if text.endswith('.') or text.endswith(','):
                text = text[:-1]
            
            # Force uppercase for viral styles
            if "mrbeast" in style_key or "hormozi" in style_key:
                text = text.upper()
            
            # Smart word wrapping (2-4 words per line)
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(current_line) >= 4 and len(lines) < 2:  # Max 2 lines
                    lines.append(' '.join(current_line))
                    current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{formatted_text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ========================================== 
# 6. GOOGLE DRIVE UPLOAD (DUAL VIDEOS)
# ========================================== 

def upload_to_google_drive(file_path, custom_name=None):
    """Uploads using OAuth 2.0 Refresh Token"""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    print(f"📤 Uploading {os.path.basename(file_path)} to Google Drive...")
    
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
        
        # Make public
        perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        requests.post(
            perm_url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"✅ Upload Success! Link: {link}")
        return link
    else:
        print(f"❌ Upload Failed: {upload_resp.text}")
        return None

# ========================================== 
# 7. VISUAL DICTIONARY (700+ TOPICS) - KEEP ORIGINAL
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
    # ISLAMIC CATEGORIES ADDED
    "islam": ["mosque", "quran", "kaaba", "islamic architecture", "prayer mat", "calligraphy"],
    "religion": ["religious symbols", "peaceful worship", "spiritual", "meditation", "faith"],
    "peace": ["dove", "olive branch", "calm waters", "sunrise", "harmony"],
    
    # UNIVERSAL CATEGORIES
    "abstract": ["colorful abstract", "fluid art", "digital background", "motion graphics"],
    "background": ["neutral background", "blurred background", "simple background"],
    "concept": ["idea visualization", "concept art", "metaphorical", "symbolic"],
}

# ========================================== 
# 8. CONTEXT-AWARE QUERY GENERATION
# ========================================== 

def get_contextual_query(text, sentence_index, total_sentences, context_history=[], visual_map=VISUAL_MAP):
    """
    Generate intelligent queries based on context analysis
    Uses sentence transformers if available, otherwise keyword matching
    """
    context_analyzer = ContextAnalyzer()
    
    # Extract key concepts
    concepts = context_analyzer.extract_key_concepts(text, context_history)
    print(f"    🔍 Concepts: {concepts[:3]}")
    
    # Get semantic categories
    categories = context_analyzer.get_semantic_category(text, visual_map)
    if not categories:
        categories = ["abstract", "concept", "background"]
    
    print(f"    📂 Categories: {categories[:3]}")
    
    # Primary query: Combine concept with category
    primary_category = categories[0]
    
    if concepts:
        # Find most visual concept
        visual_concepts = []
        for concept in concepts:
            visual_terms = ["land", "scape", "view", "scene", "shot", "pan", "zoom"]
            if any(vt in concept for vt in visual_terms):
                visual_concepts.append(concept)
        
        if visual_concepts:
            primary_concept = visual_concepts[0]
        else:
            primary_concept = concepts[0]
        
        primary_query = f"{primary_concept} {primary_category}"
    else:
        # Use category terms
        category_terms = visual_map.get(primary_category, [])
        if category_terms:
            primary_query = random.choice(category_terms[:5])
        else:
            primary_query = f"{primary_category} cinematic"
    
    # Islamic compliance check
    if not is_islamic_compliant(primary_query):
        primary_query = re.sub(r'\b(sexy|hot|violence|gun)\b', 'professional', primary_query, flags=re.IGNORECASE)
    
    # Fallback queries
    fallbacks = []
    
    # Fallback 1: Alternate category
    if len(categories) > 1:
        alt_category = categories[1]
        alt_terms = visual_map.get(alt_category, [])
        if alt_terms:
            fallbacks.append(random.choice(alt_terms[:3]))
    
    # Fallback 2: Concept + alternate category
    if concepts and len(categories) > 1:
        fallbacks.append(f"{concepts[0]} {categories[1]}")
    
    # Fallback 3: Generic cinematic
    fallbacks.append(f"{primary_category} cinematic 4k")
    fallbacks.append("abstract background cinematic")
    
    # Filter for Islamic compliance
    fallbacks = filter_islamic_queries(fallbacks)
    
    # Add quality indicators
    primary_query = f"{primary_query} 4k cinematic"
    fallbacks = [f"{fb} hd" for fb in fallbacks]
    
    print(f"    📌 Primary Query: '{primary_query}'")
    print(f"    🔄 Fallbacks: {fallbacks[:2]}")
    
    return primary_query, fallbacks[:3], concepts

# ========================================== 
# 9. ENHANCED SCORING SYSTEM
# ========================================== 

def calculate_contextual_score(video, query, sentence_text, concepts, full_script="", topic=""):
    """
    Enhanced scoring with context awareness
    """
    score = 0
    
    video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
    sentence_lower = sentence_text.lower()
    query_lower = query.lower()
    
    # === 1. QUERY MATCH (30 points) ===
    query_terms = [t for t in query_lower.split() if len(t) > 3 and t not in ['landscape', 'cinematic', 'abstract']]
    query_match_count = 0
    
    for term in query_terms:
        if term in video_text:
            query_match_count += 1
            score += 8
    
    # Bonus for exact phrase
    if len(query_terms) >= 2:
        query_phrase = ' '.join(query_terms[:2])
        if query_phrase in video_text:
            score += 10
    
    # === 2. CONCEPT MATCH (25 points) ===
    concept_match_count = 0
    for concept in concepts:
        if len(concept) > 3:
            if concept in video_text:
                concept_match_count += 1
                score += 5
            if concept in sentence_lower:
                score += 2
    
    # === 3. SEMANTIC RELEVANCE (20 points) ===
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
        score += 4
    
    duration = video.get('duration', 0)
    if duration >= 15:
        score += 5
    elif duration >= 10:
        score += 3
    
    # === 5. LANDSCAPE VERIFICATION (10 points) ===
    landscape_indicators = ['landscape', 'horizontal', 'wide', 'panoramic', 'widescreen']
    portrait_indicators = ['vertical', 'portrait', '9:16', 'instagram', 'tiktok']
    
    if any(indicator in video_text for indicator in landscape_indicators):
        score += 10
    
    # Check for portrait
    portrait_detected = False
    for indicator in portrait_indicators:
        if indicator in video_text:
            portrait_detected = True
            break
    
    if portrait_detected:
        score -= 40
    
    # Check dimensions
    width = video.get('width', 0)
    height = video.get('height', 0)
    if width and height:
        aspect_ratio = width / height
        if aspect_ratio >= 1.5:  # Landscape
            score += 10
        elif aspect_ratio < 1.0:  # Portrait
            score -= 50
            print(f"      ✗ Portrait aspect ratio detected")
    
    # === 6. ISLAMIC COMPLIANCE CHECK ===
    if not is_islamic_compliant(video_text):
        score -= 1000
        print(f"      🚫 Non-Islamic content detected")
    
    # === 7. AVOID INAPPROPRIATE CONTENT ===
    blacklist = ['sexy', 'bikini', 'underwear', 'lingerie', 'violence', 'blood', 'gore', 'weapon', 'gun']
    for term in blacklist:
        if term in video_text:
            score -= 1000
            print(f"      🚫 Blacklisted: '{term}'")
            break
    
    # === 8. PLATFORM QUALITY ===
    service = video.get('service', '')
    if service == 'pexels':
        score += 5
    elif service == 'pixabay':
        score += 3
    
    # Small random factor
    score += random.randint(0, 2)
    
    # Cap between 0-100
    final_score = min(100, max(0, score))
    
    return final_score

# ========================================== 
# 10. VIDEO SEARCH WITH FILTERS
# ========================================== 

def intelligent_video_search(query, service, keys, page=1):
    """Search for videos with Islamic filtering"""
    
    if not is_islamic_compliant(query):
        print(f"    🚫 Query blocked: '{query}'")
        return []
    
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
                    # Islamic compliance check
                    title = video.get('user', {}).get('name', '').lower()
                    if not is_islamic_compliant(title):
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
                                'title': title,
                                'description': f"Pexels video",
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
                    # Islamic compliance check
                    tags = video.get('tags', '').lower()
                    if not is_islamic_compliant(tags):
                        continue
                    
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
                            'title': tags,
                            'description': f"Pixabay video",
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
# 11. STATUS UPDATES (ENHANCED FOR DUAL VIDEOS)
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", **kwargs):
    """Updates status.json with support for dual video links"""
    
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
        "logs": "\n".join(LOG_BUFFER[-10:]),
        "timestamp": time.time()
    }
    
    # Add video URLs if provided
    for key, value in kwargs.items():
        if value:
            data[key] = value
    
    import base64
    content_json = json.dumps(data, indent=2)
    content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        
        payload = {
            "message": f"Update: {progress}% - {message[:50]}...",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
            
        response = requests.put(url, headers=headers, json=payload)
        
        if response.status_code not in [200, 201]:
            print(f"⚠️ Status update failed: {response.status_code}")
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
# 12. ENHANCED AUDIO PROCESSING
# ========================================== 

def clone_voice_robust(text, ref_audio, out_path, speed=1.0, pitch=1.0, exaggeration=0.5):
    """Enhanced audio synthesis with speed and pitch controls"""
    print(f"🎤 Synthesizing Audio (Speed: {speed}, Pitch: {pitch})...")
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
                        exaggeration=exaggeration
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
        
        # Save raw audio
        raw_audio_path = TEMP_DIR / "raw_audio.wav"
        torchaudio.save(str(raw_audio_path), full_audio, 24000)
        
        # Apply speed and pitch changes if needed
        if speed != 1.0 or pitch != 1.0:
            print(f"🎚️ Applying audio effects: Speed={speed}, Pitch={pitch}")
            
            filter_parts = []
            if speed != 1.0:
                tempo = max(0.5, min(2.0, speed))
                filter_parts.append(f"atempo={tempo}")
            
            if pitch != 1.0:
                new_rate = int(24000 * pitch)
                filter_parts.append(f"asetrate={new_rate}")
                filter_parts.append("aresample=24000")
            
            if filter_parts:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_audio_path),
                    "-af", ",".join(filter_parts),
                    str(out_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                shutil.copy(raw_audio_path, out_path)
        else:
            shutil.copy(raw_audio_path, out_path)
        
        # Add 1s silence at end
        final_path = TEMP_DIR / "audio_with_silence.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(out_path),
            "-af", "apad=pad_dur=1",
            str(final_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        shutil.move(final_path, out_path)
        
        audio_duration = len(all_wavs) * 3  # Approximate duration
        print(f"✅ Audio generated: {audio_duration:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Audio synthesis failed: {e}")
        return False

# ========================================== 
# 13. CINEMATIC PROCESSING
# ========================================== 

def apply_cinematic_effects(input_path, output_path, duration):
    """Apply cinematic effects to video clip"""
    # Ken Burns zoom effect
    zoom_filter = "zoompan=z='min(zoom+0.0003,1.03)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    
    # Color grading (subtle)
    color_filter = "eq=contrast=1.05:saturation=0.98:brightness=0.01"
    
    # Combine filters
    filter_complex = f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,{zoom_filter},{color_filter}"
    
    frames = int(duration * 30)
    filter_complex = filter_complex.format(frames=frames)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_complex,
        "-t", str(duration),
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "8M",
        "-an",
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ========================================== 
# 14. DUAL VIDEO RENDERING
# ========================================== 

def create_two_videos(base_video_path, audio_path, logo_path, subtitle_path, job_id):
    """
    Create TWO versions:
    1. WITHOUT subtitles (clean version)
    2. WITH subtitles (normal version)
    """
    
    print("🎬 Creating dual video versions...")
    
    # Version 1: WITHOUT SUBTITLES
    video_no_subs = OUTPUT_DIR / f"{job_id}_no_subs.mp4"
    print(f"  Creating: {video_no_subs.name}")
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(video_no_subs)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(video_no_subs)
        ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Version 2: WITH SUBTITLES
    video_with_subs = OUTPUT_DIR / f"{job_id}_with_subs.mp4"
    print(f"  Creating: {video_with_subs.name}")
    
    # Escape ASS path for FFmpeg
    ass_path = str(subtitle_path).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles='{ass_path}'[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(video_with_subs)
        ]
    else:
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[bg]subtitles='{ass_path}'[v]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "12M",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(video_with_subs)
        ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return str(video_no_subs), str(video_with_subs)

# ========================================== 
# 15. ENHANCED VISUAL PROCESSING
# ========================================== 

def process_visuals(sentences, audio_path, ass_file, logo_path, job_id, full_script="", topic=""):
    print("🎬 Processing visuals with context-aware intelligence...")
    
    def get_clip_with_context(i, sent, max_retries=4):
        """Get clip based on sentence context"""
        dur = max(3.5, sent['end'] - sent['start'])
        
        # Get contextual query
        primary_query, fallback_queries, concepts = get_contextual_query(
            sent['text'], i, len(sentences)
        )
        
        print(f"  🔍 Clip {i+1}/{len(sentences)}: '{sent['text'][:50]}...'")
        
        for attempt in range(max_retries):
            out = TEMP_DIR / f"s_{i}_attempt{attempt}.mp4"
            
            # Select query based on attempt
            if attempt == 0:
                query = primary_query
            elif attempt < len(fallback_queries) + 1:
                query = fallback_queries[attempt - 1]
            else:
                query = "abstract cinematic background"
            
            print(f"    Attempt {attempt+1}: '{query}'")
            
            # Search both services
            all_results = []
            page = random.randint(1, 3)
            
            if PEXELS_KEYS and PEXELS_KEYS[0]:
                pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS, page)
                for video in pexels_results:
                    if video['url'] not in USED_VIDEO_URLS:
                        relevance = calculate_contextual_score(
                            video, query, sent['text'], concepts, full_script, topic
                        )
                        video['relevance_score'] = relevance
                        all_results.append(video)
            
            if PIXABAY_KEYS and PIXABAY_KEYS[0]:
                pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS, page)
                for video in pixabay_results:
                    if video['url'] not in USED_VIDEO_URLS:
                        relevance = calculate_contextual_score(
                            video, query, sent['text'], concepts, full_script, topic
                        )
                        video['relevance_score'] = relevance
                        all_results.append(video)
            
            # Sort by score
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Select best video
            found_link = None
            selected_video = None
            
            for threshold in [60, 50, 40, 30, 20]:
                for video in all_results:
                    if video['url'] not in USED_VIDEO_URLS and video['relevance_score'] >= threshold:
                        found_link = video['url']
                        selected_video = video
                        break
                if found_link:
                    break
            
            if found_link and selected_video:
                USED_VIDEO_URLS.add(found_link)
                print(f"    ✓ {selected_video['service']} (score: {selected_video['relevance_score']})")
                
                try:
                    raw = TEMP_DIR / f"r_{i}.mp4"
                    response = requests.get(found_link, timeout=40, stream=True)
                    
                    with open(raw, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Apply cinematic effects
                    apply_cinematic_effects(raw, out, dur)
                    return str(out)
                    
                except Exception as e:
                    print(f"    ✗ Download failed: {str(e)[:60]}")
                    continue
            else:
                print(f"    ⚠️ No videos found (attempt {attempt+1})")
        
        # Fallback - cinematic gradient
        print(f"  → Using cinematic fallback for clip {i+1}")
        
        gradient_colors = [
            ("0x0F3460", "0x533483"),  # Blue to Purple
            ("0x1A1A2E", "0x16213E"),  # Dark Blue
            ("0x1E4D2B", "0x2D5016"),  # Green
        ]
        
        color1, color2 = random.choice(gradient_colors)
        
        out = TEMP_DIR / f"s_{i}_fallback.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"gradients=s=1920x1080:c0={color1}:c1={color2}:d={dur}",
            "-vf", f"fade=in:0:30,fade=out:st={dur-1}:d=1,boxblur=2:1",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-t", str(dur),
            str(out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)
    
    # Process all clips
    print(f"\n📥 Downloading {len(sentences)} clips...")
    clips = []
    
    for i, sent in enumerate(sentences):
        update_status(60 + int((i/len(sentences))*30), f"Processing clip {i+1}/{len(sentences)}...")
        clip_path = get_clip_with_context(i, sent)
        clips.append(clip_path)
    
    # Concatenate
    print("🔗 Concatenating video clips...")
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for c in clips:
            if os.path.exists(c):
                f.write(f"file '{c}'\n")
    
    base_video = TEMP_DIR / "visual_base.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "10M",
        "-an",
        str(base_video)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Create two versions
    video_no_subs, video_with_subs = create_two_videos(
        base_video, audio_path, logo_path, ass_file, job_id
    )
    
    return video_no_subs, video_with_subs

# ========================================== 
# 16. SCRIPT GENERATION
# ========================================== 

def generate_script(topic, minutes):
    words = int(minutes * 180)
    print(f"Generating Islamic-compliant Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
CRITICAL RULES:
1. Write ONLY spoken narration text
2. NO stage directions like [Music fades], [Intro], [Outro]
3. NO sound effects descriptions
4. NO [anything in brackets]
5. Start directly with the content
6. End directly with the conclusion
7. Pure voiceover script only
8. 100% Islamic compliant - NO mention of: alcohol, drugs, violence, sexual content, music instruments, idols, magic
9. Educational and professional tone
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"{base_instructions}\nWrite Part {i+1}/{chunks} of an educational video about '{topic}'. Context: {context}. Length: 700 words. Keep it Islamic compliant."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nWrite an educational video script about '{topic}'. {words} words. Keep it 100% Islamic compliant."
        script = call_gemini(prompt)
    
    # Clean up
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'\(.*?music.*?\)', '', script, flags=re.IGNORECASE)
    script = re.sub(r'\*\*.*?\*\*', '', script)
    
    return script.strip()

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()
        except:
            continue
    return f"Educational content about the topic. This video explores important aspects in a professional and Islamic-compliant manner."

# ========================================== 
# 17. MAIN EXECUTION
# ========================================== 

print("--- 🚀 ISLAMIC VIDEO GENERATOR (Enhanced) ---")
update_status(1, "Initializing Islamic-compliant system...")

# Download assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png" if LOGO_PATH and LOGO_PATH != "None" else None

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice asset download failed", "failed")
    exit(1)

print(f"✅ Voice reference downloaded")

if ref_logo and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if os.path.exists(ref_logo):
        print(f"✅ Logo downloaded")
    else:
        ref_logo = None
else:
    ref_logo = None

# Generate script
update_status(10, "Generating Islamic-compliant script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT
    # Ensure script is Islamic compliant
    if not is_islamic_compliant(text):
        print("⚠️ Script contains non-Islamic content, filtering...")
        text = re.sub(r'\b(alcohol|drug|sex|violence|magic)\b', 'education', text, flags=re.IGNORECASE)

if not text or len(text) < 100:
    print("❌ Script too short")
    update_status(0, "Script generation failed", "failed")
    exit(1)

print(f"✅ Script generated ({len(text.split())} words)")

# Generate audio with enhanced controls
update_status(20, f"Audio Synthesis (Speed: {AUDIO_SPEED}, Pitch: {AUDIO_PITCH})...")
audio_out = TEMP_DIR / "out.wav"

if clone_voice_robust(text, ref_voice, audio_out, AUDIO_SPEED, AUDIO_PITCH, AUDIO_EXAGGERATION):
    update_status(50, "Creating enhanced subtitles...")
    
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
            # Fallback timing
            import wave
            with wave.open(str(audio_out), 'rb') as wav_file:
                total_duration = wav_file.getnframes() / float(wav_file.getframerate())
            
            words = text.split()
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
        import wave
        with wave.open(str(audio_out), 'rb') as wav_file:
            total_duration = wav_file.getnframes() / float(wav_file.getframerate())
        
        words = text.split()
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
    
    # Create ASS subtitles with selected style
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file, SUBTITLE_STYLE)
    
    # Process visuals with context awareness
    update_status(60, "Gathering context-aware visuals...")
    
    # Create dual videos
    video_no_subs_path, video_with_subs_path = process_visuals(
        sentences, audio_out, ass_file, ref_logo, JOB_ID, text, TOPIC
    )
    
    if os.path.exists(video_no_subs_path) and os.path.exists(video_with_subs_path):
        file_size_no_subs = os.path.getsize(video_no_subs_path) / (1024 * 1024)
        file_size_with_subs = os.path.getsize(video_with_subs_path) / (1024 * 1024)
        
        print(f"✅ Videos created:")
        print(f"   - Without subtitles: {file_size_no_subs:.1f} MB")
        print(f"   - With subtitles: {file_size_with_subs:.1f} MB")
        
        update_status(99, "Uploading to Google Drive...")
        
        # Upload both videos
        VIDEO_NO_SUBS_URL = upload_to_google_drive(video_no_subs_path, f"{JOB_ID}_no_subs.mp4")
        VIDEO_WITH_SUBS_URL = upload_to_google_drive(video_with_subs_path, f"{JOB_ID}_with_subs.mp4")
        
        if VIDEO_NO_SUBS_URL and VIDEO_WITH_SUBS_URL:
            update_status(
                100, 
                "Success! Both videos ready.", 
                "completed",
                video_no_subs_url=VIDEO_NO_SUBS_URL,
                video_with_subs_url=VIDEO_WITH_SUBS_URL
            )
            print(f"🎉 Video WITHOUT subtitles: {VIDEO_NO_SUBS_URL}")
            print(f"🎉 Video WITH subtitles: {VIDEO_WITH_SUBS_URL}")
        elif VIDEO_NO_SUBS_URL:
            update_status(
                100, 
                "One video ready (No Subtitles)", 
                "completed",
                video_no_subs_url=VIDEO_NO_SUBS_URL
            )
            print(f"🎉 Video WITHOUT subtitles: {VIDEO_NO_SUBS_URL}")
        elif VIDEO_WITH_SUBS_URL:
            update_status(
                100, 
                "One video ready (With Subtitles)", 
                "completed",
                video_with_subs_url=VIDEO_WITH_SUBS_URL
            )
            print(f"🎉 Video WITH subtitles: {VIDEO_WITH_SUBS_URL}")
        else:
            update_status(100, "Videos created locally", "completed")
    else:
        update_status(0, "Video rendering failed", "failed")
else:
    update_status(0, "Audio synthesis failed", "failed")

# Cleanup
print("🧹 Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["concat_list.txt", "visual_base.mp4"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ ISLAMIC-COMPLIANT PROCESS COMPLETE ---")
