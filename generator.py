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
        "chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", 
        "requests", "beautifulsoup4", "pydub", "numpy", "--quiet"
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
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. PROFESSIONAL SUBTITLE STYLES (YOUTUBE-QUALITY)
# ==========================================
SUBTITLE_STYLES = {
    "youtube_white_box": {
        "name": "YouTube White Box",
        "fontname": "Arial",
        "fontsize": 42,
        "primary_colour": "&H00FFFFFF",  # White text
        "back_colour": "&HCC000000",     # Black box background
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 4,  # Box with soft edges
        "outline": 0,
        "shadow": 0,
        "margin_v": 80,  # Distance from bottom
        "alignment": 2,   # Bottom center
        "spacing": 0
    },
    "modern_yellow": {
        "name": "Modern Yellow",
        "fontname": "Arial",
        "fontsize": 44,
        "primary_colour": "&H0000FFFF",  # Yellow text
        "back_colour": "&HC0000000",     # Black background
        "outline_colour": "&H00000000",  # Black outline
        "bold": -1,
        "italic": 0,
        "border_style": 4,
        "outline": 2,
        "shadow": 0,
        "margin_v": 85,
        "alignment": 2,
        "spacing": 0
    },
    "clean_white_outline": {
        "name": "Clean White Outline",
        "fontname": "Arial",
        "fontsize": 46,
        "primary_colour": "&H00FFFFFF",  # White text
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",  # Black outline
        "bold": -1,
        "italic": 0,
        "border_style": 1,  # Outline only
        "outline": 3,
        "shadow": 2,
        "margin_v": 90,
        "alignment": 2,
        "spacing": 0
    },
    "netflix_style": {
        "name": "Netflix Style",
        "fontname": "Arial",
        "fontsize": 40,
        "primary_colour": "&H00FFFFFF",  # White text
        "back_colour": "&HE6000000",     # Very dark box
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 4,
        "outline": 0,
        "shadow": 0,
        "margin_v": 75,
        "alignment": 2,
        "spacing": 0
    },
    "bold_cyan": {
        "name": "Bold Cyan",
        "fontname": "Arial",
        "fontsize": 45,
        "primary_colour": "&H00FFFF00",  # Cyan text
        "back_colour": "&HB0000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 4,
        "outline": 2,
        "shadow": 1,
        "margin_v": 82,
        "alignment": 2,
        "spacing": 0
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with professional YouTube-style formatting"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8") as f:
        # Header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n")  # Smart wrapping
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        # Style definition
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},100,100,{style['margin_v']},1\n\n")
        
        # Events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            
            # Clean and format text with proper line breaks
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            
            # Smart line breaking for readability (max ~50 chars per line)
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > 50 and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    current_line.append(word)
                    current_length += word_length
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Join with line breaks (max 2 lines for readability)
            if len(lines) > 2:
                # Combine to fit in 2 lines
                mid = len(lines) // 2
                formatted_text = ' '.join(lines[:mid]) + '\\N' + ' '.join(lines[mid:])
            else:
                formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp (H:MM:SS.CS)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ==========================================
# 4. ROBUST UPLOAD
# ==========================================
def robust_upload(file_path):
    """Upload with multiple service fallbacks"""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None

    filename = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"🚀 Uploading {filename} ({file_size_mb:.2f} MB)...")

    # Transfer.sh
    print("👉 Attempting Transfer.sh...")
    try:
        cmd = ["curl", "--upload-file", str(file_path), f"https://transfer.sh/{filename}", "--max-time", "120"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        link = result.stdout.strip()
        
        if "transfer.sh" in link and link.startswith("http"):
            print(f"✅ Success! Link: {link}")
            return link
        else:
            print(f"⚠️ Transfer.sh returned invalid response: {link}")
    except Exception as e:
        print(f"❌ Transfer.sh failed: {str(e)}")

    # Catbox.moe
    print("👉 Attempting Catbox.moe...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": f},
                timeout=120
            )
        
        if response.status_code == 200:
            link = response.text.strip()
            if link.startswith("http"):
                print(f"✅ Success! Link: {link}")
                return link
    except Exception as e:
        print(f"❌ Catbox error: {str(e)}")

    # File.io
    print("👉 Attempting File.io...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post("https://file.io", files={"file": f}, timeout=60)
        
        if response.status_code == 200:
            link = response.json().get("link")
            print(f"✅ Success! Link: {link}")
            return link
    except Exception as e:
        print(f"❌ File.io error: {str(e)}")

    print("💀 All upload attempts failed.")
    return None

# ==========================================
# 5. EXPANDED VISUAL DICTIONARY (700+ TOPICS)
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

def get_visual_query(text):
    """Intelligent visual query - extracts most relevant keywords from sentence"""
    text = text.lower()
    
    # Remove common filler words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    # Extract meaningful words (4+ letters)
    words = [w for w in re.findall(r'\b\w+\b', text) if len(w) >= 4 and w not in stop_words]
    
    # Priority 1: Check for exact category matches
    for word in words:
        for category, terms in VISUAL_MAP.items():
            if word == category:
                return random.choice(terms)
    
    # Priority 2: Check for partial matches in categories
    for word in words:
        for category, terms in VISUAL_MAP.items():
            if word in category or category in word:
                return random.choice(terms)
    
    # Priority 3: Use the most significant nouns (6+ letters, not common)
    significant = [w for w in words if len(w) >= 6]
    if significant:
        main_word = significant[0]
        
        # Check if any category contains this word
        for category, terms in VISUAL_MAP.items():
            if main_word in category or any(main_word in term for term in terms):
                return random.choice(terms)
        
        return f"{main_word} cinematic 4k"
    
    # Priority 4: Use any meaningful noun with visual enhancement
    if words:
        return f"{words[0]} nature documentary"
    
    # Final fallback
    fallbacks = [
        "nature documentary 4k",
        "abstract motion graphics",
        "cinematic landscape",
        "time lapse clouds",
        "ocean waves sunset",
        "mountain vista aerial",
        "forest canopy drone"
    ]
# ==========================================
# 6. UTILS: STATUS & DOWNLOAD
# ==========================================
def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return

    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    data = {"progress": progress, "message": message, "status": status, "timestamp": time.time()}
    if file_url: data["file_io_url"] = file_url
    
    import base64
    content = base64.b64encode(json.dumps(data).encode()).decode()
    
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        sha = requests.get(url, headers=headers).json().get("sha")
        payload = {"message": "upd", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# ==========================================
# 7. SCRIPT & AUDIO
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
    
    # Clean any remaining bracketed content
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'\(.*?music.*?\)', '', script, flags=re.IGNORECASE)
    return script.strip()

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except: continue
    return "Script generation failed."

def clone_voice_robust(text, ref_audio, out_path):
    """Synthesize audio with padding to prevent cutoff"""
    print("🎤 Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Clean text and split into sentences
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        print(f"📝 Processing {len(sentences)} sentences...")
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            if i % 10 == 0: 
                update_status(20 + int((i/len(sentences))*30), f"Voice Gen {i}/{len(sentences)}")
            
            try:
                with torch.no_grad():
                    # Clean quotes and special characters
                    chunk_clean = chunk.replace('"', '').replace('"', '').replace('"', '')
                    
                    # Add pause at end of sentence for natural flow
                    if chunk_clean.endswith('.'):
                        chunk_clean = chunk_clean + ' '
                    
                    wav = model.generate(
                        text=chunk_clean, 
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                    
                # Memory management
                if i % 20 == 0 and device == "cuda": 
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                print(f"⚠️ Skipping sentence {i}: {str(e)[:50]}")
                continue
        
        if not all_wavs:
            print("❌ No audio generated")
            return False
        
        # Concatenate all audio
        full_audio = torch.cat(all_wavs, dim=1)
        
        # Add 2 seconds of silence at the end to prevent cutoff
        silence_samples = int(2.0 * 24000)  # 2 seconds at 24kHz
        silence = torch.zeros((full_audio.shape[0], silence_samples))
        full_audio_padded = torch.cat([full_audio, silence], dim=1)
        
        # Save with padding
        torchaudio.save(out_path, full_audio_padded, 24000)
        
        # Verify audio length
        audio_duration = full_audio_padded.shape[1] / 24000
        print(f"✅ Audio generated: {audio_duration:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio synthesis failed: {e}")
        return False

# ==========================================
# 8. VISUALS & RENDER (GPU ACCELERATED)
# ==========================================
USED_VIDEO_URLS = set()

def process_visuals(sentences, audio_path, ass_file, logo_path, final_out):
    print("Visuals & Render...")
    
    def get_clip(args):
        i, sent = args
        dur = max(3.5, sent['end'] - sent['start'])
        query = get_visual_query(sent['text'])
        out = TEMP_DIR / f"s_{i}.mp4"
        
        found_link = None
        attempt = 0
        max_attempts = 5
        
        # Alternate between services
        services = ['pexels', 'pixabay'] if (i % 2 == 0) else ['pixabay', 'pexels']
        
        while attempt < max_attempts and not found_link:
            attempt += 1
            service = services[attempt % 2]
            
            # === PEXELS API ===
            if service == 'pexels' and PEXELS_KEYS:
                try:
                    h = {"Authorization": random.choice(PEXELS_KEYS)}
                    page = random.randint(1, 5)
                    r = requests.get(
                        f"https://api.pexels.com/videos/search?query={query}&size=large&orientation=landscape&per_page=25&page={page}", 
                        headers=h, 
                        timeout=8
                    )
                    videos = r.json().get('videos', [])
                    random.shuffle(videos)
                    
                    for v in videos:
                        video_files = sorted(v.get('video_files', []), key=lambda x: x.get('width', 0), reverse=True)
                        if video_files:
                            link = video_files[0]['link']
                            if link not in USED_VIDEO_URLS:
                                found_link = link
                                USED_VIDEO_URLS.add(link)
                                print(f"  ✓ Clip {i}: {query[:35]}... (Pexels HD)")
                                break
                    
                    if not found_link and attempt < max_attempts:
                        query = get_visual_query(sent['text'] + " cinematic")
                        
                except Exception as e:
                    print(f"  ✗ Pexels error clip {i}: {str(e)[:50]}")
            
            # === PIXABAY API ===
            if service == 'pixabay' and not found_link and PIXABAY_KEYS:
                try:
                    key = random.choice(PIXABAY_KEYS)
                    page = random.randint(1, 5)
                    r = requests.get(
                        f"https://pixabay.com/api/videos/?key={key}&q={query}&per_page=30&page={page}", 
                        timeout=8
                    )
                    videos = r.json().get('hits', [])
                    random.shuffle(videos)
                    
                    for v in videos:
                        video_data = v.get('videos', {})
                        link = video_data.get('large', {}).get('url') or \
                               video_data.get('medium', {}).get('url') or \
                               video_data.get('small', {}).get('url')
                        
                        if link and link not in USED_VIDEO_URLS:
                            found_link = link
                            USED_VIDEO_URLS.add(link)
                            print(f"  ✓ Clip {i}: {query[:35]}... (Pixabay HD)")
                            break
                    
                    if not found_link and attempt < max_attempts:
                        query = get_visual_query(sent['text'] + " stock footage")
                        
                except Exception as e:
                    print(f"  ✗ Pixabay error clip {i}: {str(e)[:50]}")
        
        # Download and process with GPU
        if found_link:
            try:
                raw = TEMP_DIR / f"r_{i}.mp4"
                with open(raw, "wb") as f: 
                    f.write(requests.get(found_link, timeout=40).content)
                
                # GPU-accelerated processing
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda", "-i", str(raw), 
                    "-t", str(dur),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                    "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M",
                    "-an", str(out)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                return str(out)
            except Exception as e:
                print(f"  ✗ Download/process failed clip {i}: {str(e)[:60]}")
        
        # Fallback gradient
        print(f"  → Clip {i}: Fallback gradient")
        colors = ["0x1a1a2e:0x16213e", "0x0f3460:0x533483", "0x2a2d34:0x1e3a5f"]
        gradient = random.choice(colors)
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={dur}",
            "-vf", f"fade=in:0:30,fade=out:st={dur-1}:d=1",
            "-c:v", "h264_nvenc", "-preset", "p1", "-t", str(dur), str(out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    # Parallel processing
    print(f"Downloading {len(sentences)} video clips...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        clips = list(ex.map(get_clip, [(i, s) for i, s in enumerate(sentences)]))

    # Fast concatenation
    print("Concatenating video clips...")
    with open("list.txt", "w") as f:
        for c in clips: 
            if os.path.exists(c):
                f.write(f"file '{c}'\n")
    
    subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i list.txt -c:v h264_nvenc -preset p1 -b:v 10M visual.mp4", 
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Final render with GPU
    print("🎬 Rendering final video with subtitles and audio...")
    ass_path = str(ass_file).replace('\\', '\\\\').replace(':', '\\:')
    
    # Get audio duration to ensure video isn't cut short
    import wave
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_duration = frames / float(rate)
        print(f"🎵 Audio duration: {audio_duration:.1f} seconds")
    except:
        audio_duration = None
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]ass='{ass_path}'[v]"
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M",
            "-c:a", "aac", "-b:a", "256k"
        ]
    else:
        filter_complex = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];[bg]ass='{ass_path}'[v]"
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", "visual.mp4", "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M",
            "-c:a", "aac", "-b:a", "256k"
        ]
    
    # Add audio duration parameter if available (don't use -shortest)
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
# 9. EXECUTION
# ==========================================
print("--- 🚀 START ---")
update_status(1, "Initializing...")

# Download assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice asset download failed", "failed")
    exit(1)

print(f"✅ Voice reference downloaded: {ref_voice}")

if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if os.path.exists(ref_logo):
        print(f"✅ Logo downloaded: {ref_logo}")
    else:
        print("⚠️ Logo not found or download failed")
        ref_logo = None
else:
    ref_logo = None

# Generate or use provided script
update_status(10, "Scripting...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if not text or len(text) < 100:
    print("❌ Script is too short or empty")
    update_status(0, "Script generation failed", "failed")
    exit(1)

print(f"✅ Script generated ({len(text.split())} words)")

# Generate audio
update_status(20, "Audio Synthesis...")
audio_out = TEMP_DIR / "out.wav"

if clone_voice_robust(text, ref_voice, audio_out):
    update_status(50, "Creating Subtitles...")
    
    # Create subtitles
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            
            print("📝 Transcribing audio for subtitles...")
            transcript = transcriber.transcribe(str(audio_out))
            
            if transcript.status == aai.TranscriptStatus.completed:
                sentences = []
                for sentence in transcript.get_sentences():
                    sentences.append({
                        "text": sentence.text,
                        "start": sentence.start / 1000,
                        "end": sentence.end / 1000
                    })
                
                # Extend last subtitle by 1 second to prevent cutoff
                if sentences:
                    sentences[-1]['end'] += 1.0
                
                print(f"✅ Transcription complete: {len(sentences)} sentences")
                
            else:
                print(f"❌ Transcription failed: {transcript.status}")
                raise Exception("Transcription failed")
                
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}. Using fallback timing...")
            # Fallback timing
            words = text.split()
            
            # Get actual audio duration
            import wave
            try:
                with wave.open(str(audio_out), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    total_duration = frames / float(rate)
            except:
                total_duration = DURATION_MINS * 60
            
            words_per_second = len(words) / total_duration
            
            sentences = []
            current_time = 0
            words_per_sentence = 12  # Shorter chunks for better readability
            
            for i in range(0, len(words), words_per_sentence):
                chunk = words[i:i + words_per_sentence]
                sentence_text = ' '.join(chunk)
                sentence_duration = len(chunk) / words_per_second
                
                sentences.append({
                    "text": sentence_text,
                    "start": current_time,
                    "end": current_time + sentence_duration
                })
                current_time += sentence_duration
            
            # Extend last subtitle by 1.5 seconds
            if sentences:
                sentences[-1]['end'] += 1.5
    else:
        print("⚠️ No AssemblyAI key. Using fallback timing...")
        words = text.split()
        
        # Get actual audio duration
        import wave
        try:
            with wave.open(str(audio_out), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                total_duration = frames / float(rate)
        except:
            total_duration = DURATION_MINS * 60
        
        words_per_second = len(words) / total_duration
        
        sentences = []
        current_time = 0
        words_per_sentence = 12
        
        for i in range(0, len(words), words_per_sentence):
            chunk = words[i:i + words_per_sentence]
            sentence_text = ' '.join(chunk)
            sentence_duration = len(chunk) / words_per_second
            
            sentences.append({
                "text": sentence_text,
                "start": current_time,
                "end": current_time + sentence_duration
            })
            current_time += sentence_duration
        
        # Extend last subtitle
        if sentences:
            sentences[-1]['end'] += 1.5
    
    # Create ASS subtitle file
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # Process visuals and render
    update_status(60, "Gathering Visuals...")
    final_output = OUTPUT_DIR / f"final_{JOB_ID}.mp4"
    
    if process_visuals(sentences, audio_out, ass_file, ref_logo, final_output):
        if final_output.exists():
            file_size = os.path.getsize(final_output) / (1024 * 1024)
            print(f"✅ Video created: {final_output} ({file_size:.1f} MB)")
            
            update_status(99, "Uploading Final Video...")
            link = robust_upload(final_output)
            
            if link:
                update_status(100, "Success! Video Complete!", "completed", link)
                print(f"🎉 Final video uploaded: {link}")
            else:
                update_status(100, "Upload Failed - Video Ready Locally", "completed")
                print(f"📁 Video saved locally: {final_output}")
        else:
            update_status(0, "Rendering Failed - No Output File", "failed")
            print("❌ Final video file was not created")
    else:
        update_status(0, "Visual Processing Failed", "failed")
        print("❌ Visual processing failed")
        
else:
    update_status(0, "Audio Synthesis Failed", "failed")
    print("❌ Audio synthesis failed")

# Cleanup
print("Cleaning up temporary files...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
        print("✅ Temporary files cleaned")
    except:
        print("⚠️ Could not clean all temporary files")

# Clean up intermediate files
for temp_file in ["visual.mp4", "list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ PROCESS COMPLETE ---")
