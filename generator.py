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
# 3. SUBTITLE STYLES (5 PROFESSIONAL DESIGNS)
# ==========================================
SUBTITLE_STYLES = {
    "modern_bold": {
        "name": "Modern Bold",
        "fontname": "Arial",
        "fontsize": 32,
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&HD0000000",     # Black with transparency
        "outline_colour": "&H00000000",  # Black outline
        "bold": -1,
        "italic": 0,
        "border_style": 3,  # Opaque box
        "outline": 0,
        "shadow": 0,
        "margin_v": 40,
        "alignment": 2  # Bottom center
    },
    "neon_glow": {
        "name": "Neon Glow",
        "fontname": "Arial",
        "fontsize": 34,
        "primary_colour": "&H0000FFFF",  # Cyan/Yellow
        "back_colour": "&H80000000",     # Semi-transparent black
        "outline_colour": "&H00FFFF00",  # Cyan outline
        "bold": -1,
        "italic": 0,
        "border_style": 1,  # Outline + shadow
        "outline": 3,
        "shadow": 2,
        "margin_v": 45,
        "alignment": 2
    },
    "minimal_elegant": {
        "name": "Minimal Elegant",
        "fontname": "Arial",
        "fontsize": 28,
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&HA0000000",     # Darker transparent
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 3,
        "outline": 0,
        "shadow": 0,
        "margin_v": 35,
        "alignment": 2
    },
    "youtube_pro": {
        "name": "YouTube Pro",
        "fontname": "Arial",
        "fontsize": 30,
        "primary_colour": "&H00FFFFFF",  # White
        "back_colour": "&HC0000000",     # Black box
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 3,
        "outline": 0,
        "shadow": 1,
        "margin_v": 50,
        "alignment": 2
    },
    "cinematic": {
        "name": "Cinematic",
        "fontname": "Arial",
        "fontsize": 36,
        "primary_colour": "&H00F0F0F0",  # Off-white
        "back_colour": "&HE0000000",     # Very opaque black
        "outline_colour": "&H00404040",  # Dark gray outline
        "bold": -1,
        "italic": 0,
        "border_style": 3,
        "outline": 2,
        "shadow": 3,
        "margin_v": 55,
        "alignment": 2
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with random professional style"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,0,0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},20,20,{style['margin_v']},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            # Clean and escape text
            text = s['text'].replace('\n', ' ').replace('\\', '\\\\')
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp (H:MM:SS.CS)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ==========================================
# 4. ROBUST UPLOAD (MULTIPLE SERVICES)
# ==========================================
def robust_upload(file_path):
    """
    Uploads a file to Transfer.sh with a fallback to Catbox.moe.
    Includes aggressive timeouts and error handling to prevent hanging.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None

    filename = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"🚀 Uploading {filename} ({file_size_mb:.2f} MB)...")

    # --- OPTION 1: Transfer.sh (Best for CLI/Scripts) ---
    print("👉 Attempting Transfer.sh...")
    try:
        # We use 'curl' via subprocess because it is often more robust than requests for this specific API
        # The --upload-file flag is critical for transfer.sh
        cmd = [
            "curl", 
            "--upload-file", str(file_path), 
            f"https://transfer.sh/{filename}",
            "--max-time", "120" # 2-minute hard timeout
        ]
        
        # Run command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        link = result.stdout.strip()
        
        # Verify we got a valid URL back
        if "transfer.sh" in link and link.startswith("http"):
            print(f"✅ Success! Link: {link}")
            return link
        else:
            print(f"⚠️ Transfer.sh returned invalid response: {link}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Transfer.sh failed (Exit Code {e.returncode})")
    except Exception as e:
        print(f"❌ Transfer.sh error: {str(e)}")

    # --- OPTION 2: Catbox.moe (Reliable Fallback) ---
    print("👉 Attempting Catbox.moe (Fallback)...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": f},
                timeout=120  # 2-minute timeout
            )
        
        if response.status_code == 200:
            link = response.text.strip()
            if link.startswith("http"):
                print(f"✅ Success! Link: {link}")
                return link
            else:
                print(f"⚠️ Catbox returned invalid response: {link}")
        else:
            print(f"❌ Catbox failed with status: {response.status_code}")

    except Exception as e:
        print(f"❌ Catbox error: {str(e)}")

    # --- OPTION 3: File.io (Last Resort) ---
    print("👉 Attempting File.io (Last Resort)...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://file.io", 
                files={"file": f},
                timeout=60
            )
        
        if response.status_code == 200:
            link = response.json().get("link")
            print(f"✅ Success! Link: {link}")
            return link
    except Exception as e:
        print(f"❌ File.io error: {str(e)}")

    print("💀 All upload attempts failed.")
    return None
# ==========================================
# 5. MASSIVE VISUAL DICTIONARY (500+ TOPICS)
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
    
    # FOOD & COOKING
    "food": ["delicious food", "gourmet meal", "food preparation", "culinary", "restaurant dish"],
    "cooking": ["chef cooking", "kitchen", "recipe", "culinary arts"],
    "chef": ["professional chef", "restaurant kitchen", "chef preparing", "culinary expert"],
    "restaurant": ["fine dining", "restaurant service", "food service", "dining experience"],
    "eat": ["eating food", "meal time", "dining", "food consumption"],
    "drink": ["beverage", "cocktail", "coffee", "pouring drink"],
    "coffee": ["coffee brewing", "coffee shop", "espresso", "coffee beans"],
    "meal": ["family meal", "dinner", "lunch", "breakfast"],
    
    # SPORTS & FITNESS
    "sport": ["sports action", "athletic competition", "stadium", "sports event"],
    "fitness": ["gym workout", "exercise", "fitness training", "healthy lifestyle"],
    "exercise": ["exercising", "cardio", "strength training", "workout"],
    "gym": ["gym equipment", "weight training", "fitness center", "workout space"],
    "run": ["running", "marathon", "jogging", "sprint"],
    "soccer": ["soccer match", "football game", "soccer stadium", "goal"],
    "basketball": ["basketball game", "NBA", "dunk", "basketball court"],
    "football": ["american football", "NFL", "touchdown", "football stadium"],
    
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
    "abstract": ["ink in water", "smoke swirls", "light leaks", "geometric patterns", "abstract art", "fluid motion"],
    "concept": ["conceptual art", "idea visualization", "theoretical", "abstract concept"],
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
    
    # PROBLEM & SOLUTION
    "problem": ["challenge", "obstacle", "difficulty", "issue"],
    "solution": ["problem solving", "answer", "resolution", "fix"],
    "challenge": ["challenging task", "difficult", "test", "hurdle"],
    "crisis": ["emergency", "critical situation", "disaster response", "urgent"],
    "risk": ["dangerous", "hazard", "threat", "risky situation"],
    "danger": ["warning sign", "dangerous situation", "caution", "peril"],
    "safe": ["safety", "secure", "protected", "safeguard"],
    
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
    "ice": ["ice formation", "frozen", "icicles", "ice crystals"],
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
    "health": ["healthcare", "wellness", "healthy living", "medical checkup"],
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
    "hair": ["hair follicles", "hairstyle", "hair growth", "scalp"],
    "tooth": ["teeth", "dental", "dentist", "smile"],
    "smell": ["fragrance", "aroma", "scent", "perfume"],
    "taste": ["tasting food", "flavor", "taste buds", "culinary experience"],
    "touch": ["tactile", "feeling texture", "sense of touch", "contact"],
    "sound": ["audio waves", "sound system", "acoustics", "noise"],
    "hear": ["listening", "hearing", "ear", "audio"],
    "see": ["vision", "seeing", "eyesight", "visual"],
    "feel": ["feeling", "emotion", "sensation", "touch"],
    "sense": ["five senses", "sensory", "perception", "awareness"],
    "memory": ["memories", "remembering", "nostalgia", "brain memory"],
    "forget": ["forgetfulness", "amnesia", "memory loss", "fading"],
    "remember": ["remembrance", "recollection", "recall", "memory"],
    "know": ["knowledge", "knowing", "information", "understanding"],
    "understand": ["comprehension", "understanding", "clarity", "insight"],
    "wisdom": ["wise", "sage", "enlightenment", "knowledge"],
    "intelligence": ["IQ", "smart", "genius", "intellectual"],
    "smart": ["intelligent person", "cleverness", "wit", "bright"],
    "stupid": ["foolish", "mistake", "error", "blunder"],
    "genius": ["brilliant mind", "exceptional talent", "prodigy", "mastermind"],
    "talent": ["talented person", "gifted", "ability", "skill"],
    "ability": ["capability", "competence", "aptitude", "capacity"],
    "create": ["creation", "making", "building", "inventing"],
    "make": ["manufacturing", "producing", "crafting", "making things"],
    "build": ["construction", "building process", "development", "assembly"],
    "destroy": ["destruction", "demolition", "breaking", "ruins"],
    "break": ["breaking", "shatter", "fracture", "smash"],
    "fix": ["repair", "fixing", "maintenance", "mending"],
    "repair": ["repairing", "restoration", "fixing broken", "service"],
    "improve": ["improvement", "enhancement", "upgrade", "better"],
    "better": ["improvement", "superior", "excellence", "quality"],
    "worse": ["deterioration", "decline", "degradation", "failing"],
    "good": ["goodness", "positive", "excellence", "quality"],
    "bad": ["negative", "poor quality", "defective", "wrong"],
    "right": ["correct", "accuracy", "proper", "appropriate"],
    "wrong": ["incorrect", "mistake", "error", "false"],
    "true": ["truth", "reality", "fact", "genuine"],
    "false": ["fake", "counterfeit", "lie", "deception"],
    "real": ["reality", "authentic", "genuine", "actual"],
    "fake": ["counterfeit", "imitation", "fraud", "forgery"],
    "original": ["authentic", "first", "unique", "genuine"],
    "copy": ["duplicate", "replica", "reproduction", "clone"],
    "same": ["identical", "similar", "matching", "uniform"],
    "different": ["diversity", "variety", "contrast", "unique"],
    "similar": ["resemblance", "alike", "comparable", "analogous"],
    "opposite": ["contrast", "reverse", "contrary", "antithesis"],
    "equal": ["equality", "balance", "parity", "equivalence"],
    "more": ["increase", "greater", "additional", "surplus"],
    "less": ["decrease", "reduction", "fewer", "diminished"],
    "high": ["height", "tall", "elevated", "peak"],
    "low": ["lowland", "valley", "depression", "bottom"],
    "up": ["upward", "ascending", "rise", "elevation"],
    "down": ["downward", "descending", "fall", "decline"],
    "top": ["summit", "peak", "apex", "highest point"],
    "bottom": ["base", "foundation", "lowest point", "ground"],
    "center": ["middle", "core", "central", "heart"],
    "side": ["lateral", "edge", "flank", "periphery"],
    "front": ["forward", "facade", "frontline", "forefront"],
    "back": ["rear", "behind", "background", "posterior"],
    "inside": ["interior", "within", "internal", "indoors"],
    "outside": ["exterior", "outdoor", "external", "outdoors"],
    "open": ["opening", "accessible", "unlocked", "revealed"],
    "close": ["closing", "shut", "sealed", "locked"],
    "start": ["beginning", "commencement", "launch", "initiation"],
    "end": ["conclusion", "finish", "termination", "finale"],
    "begin": ["starting point", "origin", "inception", "dawn"],
    "finish": ["completion", "end result", "final", "done"],
    "complete": ["completion", "finished product", "whole", "entire"],
    "continue": ["ongoing", "persistence", "continuation", "proceed"],
    "stop": ["halt", "pause", "cessation", "standstill"],
    "pause": ["break", "intermission", "temporary stop", "rest"],
    "wait": ["waiting", "queue", "patience", "anticipation"],
    "move": ["movement", "motion", "mobility", "relocation"],
    "stay": ["remaining", "stationary", "residence", "dwell"],
    "go": ["going", "departure", "leaving", "travel"],
    "come": ["arrival", "approaching", "incoming", "return"],
    "arrive": ["arrival", "reaching destination", "coming", "landing"],
    "leave": ["departure", "exit", "goodbye", "leaving"],
    "enter": ["entrance", "entry", "coming in", "access"],
    "exit": ["way out", "departure", "leaving", "egress"],
    "push": ["pushing", "force", "pressing", "propulsion"],
    "pull": ["pulling", "tug", "drag", "traction"],
    "lift": ["lifting", "raising", "elevation", "hoist"],
    "drop": ["dropping", "fall", "release", "descent"],
    "throw": ["throwing", "toss", "hurl", "pitch"],
    "catch": ["catching", "grab", "capture", "interception"],
    "hold": ["holding", "grip", "grasp", "possession"],
    "release": ["releasing", "let go", "free", "discharge"],
    "give": ["giving", "donation", "present", "offering"],
    "take": ["taking", "receiving", "acquisition", "grab"],
    "send": ["sending", "dispatch", "delivery", "transmission"],
    "receive": ["receiving", "acceptance", "acquisition", "getting"],
    "buy": ["purchasing", "shopping", "acquisition", "buying"],
    "sell": ["selling", "sale", "commerce", "vending"],
    "pay": ["payment", "transaction", "paying", "purchase"],
    "cost": ["price", "expense", "value", "charge"],
    "price": ["pricing", "cost", "valuation", "rate"],
    "value": ["worth", "importance", "significance", "merit"],
    "cheap": ["affordable", "inexpensive", "budget", "economical"],
    "expensive": ["costly", "premium", "luxury", "high-priced"],
    "free": ["freedom", "complimentary", "no cost", "liberty"],
    "trade": ["trading", "commerce", "exchange", "business deal"],
    "exchange": ["swap", "trade", "barter", "currency exchange"],
    "deal": ["business deal", "agreement", "transaction", "negotiation"],
    "contract": ["legal contract", "agreement", "document", "signing"],
    "agreement": ["accord", "deal", "treaty", "consensus"],
    "promise": ["commitment", "pledge", "vow", "guarantee"],
    "trust": ["confidence", "faith", "reliability", "belief"],
    "lie": ["deception", "untruth", "falsehood", "dishonesty"],
    "honest": ["honesty", "truth", "integrity", "sincere"],
    "truth": ["reality", "fact", "veracity", "true nature"],
    "secret": ["hidden", "mystery", "confidential", "covert"],
    "hide": ["hiding", "concealment", "cover", "camouflage"],
    "show": ["display", "exhibition", "reveal", "presentation"],
    "reveal": ["revelation", "unveiling", "disclosure", "expose"],
    "discover": ["discovery", "finding", "exploration", "uncover"],
    "find": ["finding", "locating", "search success", "uncovering"],
    "lose": ["loss", "losing", "misplaced", "defeat"],
    "search": ["searching", "quest", "investigation", "looking"],
    "look": ["looking", "gaze", "observation", "viewing"],
    "watch": ["watching", "observing", "viewing", "monitoring"],
    "observe": ["observation", "study", "examination", "surveillance"],
    "examine": ["examination", "inspection", "analysis", "scrutiny"],
    "test": ["testing", "trial", "experiment", "assessment"],
    "try": ["attempt", "effort", "trial", "endeavor"],
    "attempt": ["trying", "effort", "endeavor", "venture"],
    "fail": ["failure", "unsuccessful", "defeat", "collapse"],
    "failure": ["failed attempt", "breakdown", "fiasco", "flop"],
    "win": ["victory", "winning", "triumph", "success"],
    "victory": ["celebration", "triumph", "conquest", "win"],
    "lose": ["defeat", "loss", "losing game", "failure"],
    "defeat": ["beaten", "conquered", "overcome", "vanquished"],
    "compete": ["competition", "rivalry", "contest", "race"],
    "competition": ["competitive event", "contest", "rivalry", "tournament"],
    "race": ["racing", "competition", "sprint", "marathon"],
    "battle": ["fight", "combat", "conflict", "warfare"],
    "protect": ["protection", "defense", "shield", "safeguard"],
    "defend": ["defense", "protecting", "guard", "resistance"],
    "attack": ["assault", "offensive", "strike", "aggression"],
    "escape": ["fleeing", "getaway", "evasion", "breakout"],
    "rescue": ["saving", "rescue operation", "liberation", "recovery"],
    "save": ["saving", "preservation", "rescue", "conservation"],
    "help": ["assistance", "helping hand", "aid", "support"],
    "support": ["backing", "assistance", "reinforcement", "aid"],
    "care": ["caring", "compassion", "attention", "nurture"],
    "protect": ["guardian", "protection", "safety", "security"],
    "share": ["sharing", "distribution", "divide", "common"],
    "join": ["joining", "union", "connection", "merge"],
    "separate": ["separation", "division", "split", "apart"],
    "connect": ["connection", "link", "bond", "network"],
    "disconnect": ["disconnection", "separation", "break", "detach"],
    "unite": ["unity", "together", "unification", "alliance"],
    "divide": ["division", "split", "partition", "separate"],
    "mix": ["mixing", "blend", "combination", "merge"],
    "combine": ["combination", "fusion", "merge", "integration"],
    "split": ["splitting", "divide", "crack", "break apart"],
    "whole": ["complete", "entire", "total", "full"],
    "part": ["portion", "piece", "section", "component"],
    "piece": ["fragment", "part", "section", "bit"],
    "all": ["everything", "whole", "total", "complete"],
    "none": ["nothing", "zero", "empty", "void"],
    "some": ["portion", "several", "few", "certain"],
    "many": ["numerous", "multiple", "abundance", "plenty"],
    "few": ["small number", "scarce", "limited", "handful"],
    "several": ["multiple", "various", "diverse", "some"],
    "every": ["each", "all", "universal", "complete"],
    "each": ["individual", "every one", "per", "single"],
    "other": ["alternative", "another", "different", "else"],
    "another": ["additional", "one more", "different one", "alternative"],
    "one": ["single", "unity", "individual", "first"],
    "two": ["pair", "duo", "couple", "double"],
    "three": ["trio", "triple", "three items", "trinity"],
    "multiple": ["many", "several", "numerous", "variety"],
    "single": ["one", "sole", "individual", "alone"],
    "double": ["twice", "dual", "pair", "twofold"],
    "half": ["50 percent", "bisect", "partial", "semi"],
    "quarter": ["one fourth", "25 percent", "fourth", "fraction"],
    "full": ["complete", "filled", "maximum", "total"],
    "empty": ["void", "vacant", "hollow", "blank"],
    "fill": ["filling", "pouring", "loading", "saturate"],
    "pour": ["pouring liquid", "flow", "stream", "cascade"],
    "flow": ["flowing", "stream", "current", "movement"],
    "stream": ["flowing water", "brook", "current", "live stream"],
    "wave": ["ocean wave", "waving", "oscillation", "surge"],
    "splash": ["water splash", "splashing", "spray", "droplets"],
    "drop": ["water drop", "falling drop", "droplet", "drip"],
    "bubble": ["soap bubbles", "air bubble", "fizz", "foam"],
    "foam": ["sea foam", "bubbles", "froth", "lather"]
}

def get_visual_query(text):
    """Enhanced visual query with fallback intelligence"""
    text = text.lower()
    
    # Priority 1: Check exact matches in dictionary
    for category, terms in VISUAL_MAP.items():
        if category in text:
            return random.choice(terms)
    
    # Priority 2: Check partial word matches
    words = re.findall(r'\b\w{4,}\b', text)
    for word in words:
        for category, terms in VISUAL_MAP.items():
            if word in category or category in word:
                return random.choice(terms)
    
    # Priority 3: Extract meaningful nouns (6+ letters)
    meaningful_words = [w for w in words if len(w) >= 6]
    if meaningful_words:
        return random.choice(meaningful_words) + " cinematic 4k"
    
    # Priority 4: Extract any noun (4+ letters)
    if words:
        return random.choice(words) + " abstract art"
    
    # Final fallback
    fallbacks = [
        "abstract motion graphics",
        "cinematic bokeh",
        "light particles",
        "geometric animation",
        "smooth gradient",
        "ink in water 4k",
        "particle effects"
    ]
    return random.choice(fallbacks)

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
    words = int(minutes * 150)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"Write Part {i+1}/{chunks} of a documentary about '{topic}'. Context: {context}. Length: 700 words. Spoken Text ONLY."
            full_script.append(call_gemini(prompt))
        return " ".join(full_script)
    else:
        prompt = f"Write a YouTube script about '{topic}'. {words} words. Spoken Text ONLY. No [Music] tags."
        return call_gemini(prompt)

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except: continue
    return "Script generation failed."

def clone_voice_robust(text, ref_audio, out_path):
    print("Synthesizing Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        clean = re.sub(r'\[.*?\]', '', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 2]
        
        all_wavs = []
        for i, chunk in enumerate(sentences):
            if i%10==0: update_status(20 + int((i/len(sentences))*30), f"Voice Gen {i}/{len(sentences)}")
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk.replace('"',''), 
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                if i%20==0 and device=="cuda": torch.cuda.empty_cache(); gc.collect()
            except: pass
            
        if not all_wavs: return False
        torchaudio.save(out_path, torch.cat(all_wavs, dim=1), 24000)
        return True
    except: return False

# ==========================================
# 8. VISUALS & RENDER (NO DUPLICATES)
# ==========================================
USED_VIDEO_URLS = set()  # Global tracker for used videos

def process_visuals(sentences, audio_path, ass_file, logo_path, final_out):
    print("Visuals & Render...")
    
    def get_clip(args):
        i, sent = args
        dur = max(3.5, sent['end'] - sent['start'])
        query = get_visual_query(sent['text'])
        out = TEMP_DIR / f"s_{i}.mp4"
        
        found_link = None
        attempt = 0
        max_attempts = 3
        
        # Try multiple searches if needed to find unique video
        while attempt < max_attempts and not found_link:
            attempt += 1
            
            if PEXELS_KEYS:
                try:
                    h = {"Authorization": random.choice(PEXELS_KEYS)}
                    # Fetch more results for variety
                    page = random.randint(1, 3)
                    r = requests.get(
                        f"https://api.pexels.com/videos/search?query={query}&size=medium&orientation=landscape&per_page=20&page={page}", 
                        headers=h, 
                        timeout=5
                    )
                    videos = r.json().get('videos', [])
                    random.shuffle(videos)
                    
                    for v in videos:
                        link = v['video_files'][0]['link']
                        # Check if we've used this exact URL before
                        if link not in USED_VIDEO_URLS:
                            found_link = link
                            USED_VIDEO_URLS.add(link)
                            print(f"  ✓ Clip {i}: {query[:30]}... (unique)")
                            break
                    
                    # If no unique video found, try a different search query
                    if not found_link and attempt < max_attempts:
                        query = get_visual_query(sent['text'] + " background")  # Modify query
                        
                except Exception as e:
                    print(f"  ✗ Clip {i} API error: {e}")
            
            if not found_link and PIXABAY_KEYS:
                try:
                    # Try Pixabay as backup
                    key = random.choice(PIXABAY_KEYS)
                    r = requests.get(
                        f"https://pixabay.com/api/videos/?key={key}&q={query}&per_page=20", 
                        timeout=5
                    )
                    videos = r.json().get('hits', [])
                    random.shuffle(videos)
                    
                    for v in videos:
                        link = v['videos']['medium']['url']
                        if link not in USED_VIDEO_URLS:
                            found_link = link
                            USED_VIDEO_URLS.add(link)
                            print(f"  ✓ Clip {i}: {query[:30]}... (Pixabay, unique)")
                            break
                except: pass
        
        # Download and process video
        if found_link:
            try:
                raw = TEMP_DIR / f"r_{i}.mp4"
                with open(raw, "wb") as f: 
                    f.write(requests.get(found_link, timeout=30).content)
                
                cmd = [
                    "ffmpeg", "-y", "-i", str(raw), "-t", str(dur),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(out)
            except Exception as e:
                print(f"  ✗ Download failed for clip {i}: {e}")
        
        # Fallback: Create colored background instead of black
        print(f"  → Clip {i}: Using fallback background")
        colors = ["0x1a1a2e", "0x16213e", "0x0f3460", "0x1e3a5f", "0x2a2d34"]
        color = random.choice(colors)
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", f"color=c={color}:s=1920x1080:d={dur}",
            "-vf", f"geq=random(1)*255:128:128,fps=30",  # Add subtle noise
            "-t", str(dur), str(out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    # Parallel Download with limited workers to avoid rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        clips = list(ex.map(get_clip, [(i, s) for i, s in enumerate(sentences)]))

    # Concatenate clips
    with open("list.txt", "w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Create ASS subtitle file with random style
    create_ass_file(sentences, ass_file)

        # Final Render with Subtitles & Logo
    print("Rendering final video with subtitles...")
    
    # Escape ASS file path for FFmpeg
    ass_path = str(ass_file).replace('\\', '\\\\').replace(':', '\\:')
    
    if os.path.exists(logo_path):
        # With logo: scale video, add logo, then subtitles
        filter_complex = f"""
        [0:v]scale=1920:1080:force_original_aspect_ratio=decrease,
              pad=1920:1080:(ow-iw)/2:(oh-ih)/2,
              setsar=1[bg];
        [1:v]scale=230:-1[logo];
        [bg][logo]overlay=30:30[withlogo];
        [withlogo]ass='{ass_path}'[v]
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", "visual.mp4",
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(final_out)
        ]
    else:
        # Without logo: just scale video and add subtitles
        filter_complex = f"""
        [0:v]scale=1920:1080:force_original_aspect_ratio=decrease,
              pad=1920:1080:(ow-iw)/2:(oh-ih)/2,
              setsar=1[bg];
        [bg]ass='{ass_path}'[v]
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", "visual.mp4",
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(final_out)
        ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Final video rendered: {final_out}")
        return True
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
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
    
    # Create subtitles using AssemblyAI or fallback
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            
            print("Transcribing audio for subtitles...")
            transcript = transcriber.transcribe(str(audio_out))
            
            if transcript.status == aai.TranscriptStatus.completed:
                sentences = []
                for sentence in transcript.get_sentences():
                    sentences.append({
                        "text": sentence.text,
                        "start": sentence.start / 1000,
                        "end": sentence.end / 1000
                    })
                
                print(f"✅ Transcription complete: {len(sentences)} sentences")
                
            else:
                print(f"❌ Transcription failed: {transcript.status}")
                raise Exception("Transcription failed")
                
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}. Using fallback timing...")
            # Fallback: Create approximate timing
            words = text.split()
            total_duration = DURATION_MINS * 60
            words_per_second = len(words) / total_duration
            
            sentences = []
            current_time = 0
            words_per_sentence = 15
            
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
    else:
        print("⚠️ No AssemblyAI key. Using fallback timing...")
        # Fallback timing
        words = text.split()
        total_duration = DURATION_MINS * 60
        words_per_second = len(words) / total_duration
        
        sentences = []
        current_time = 0
        words_per_sentence = 15
        
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
