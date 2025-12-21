"""
AI VIDEO GENERATOR - PRODUCTION READY WITH SUBMAGIC-LEVEL INTELLIGENCE
=====================================================================
FEATURES:
✅ SubMagic-level semantic understanding using NLP
✅ Advanced context extraction with entity recognition
✅ Multi-layer query generation (semantic, contextual, visual)
✅ Robust error handling with intelligent fallbacks
✅ Islamic content filtering at multiple levels
✅ Parallel processing with resource management
✅ Professional dual-output rendering
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
import hashlib

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
        "nltk",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, check=True)
    
    # Download NLTK data
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass  # Continue even if NLTK downloads fail
    
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import google.generativeai as genai
import numpy as np

# Import NLTK with fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available, using simplified analysis")

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

# API Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
PEXELS_KEYS = [k.strip() for k in os.environ.get("PEXELS_KEYS", "").split(",") if k.strip()]
PIXABAY_KEYS = [k.strip() for k in os.environ.get("PIXABAY_KEYS", "").split(",") if k.strip()]

# Islamic Content Filtering - Multi-layer
FORBIDDEN_KEYWORDS = {
    'sexual': ['nude', 'nudity', 'sexy', 'erotic', 'sexual', 'bikini', 'swimsuit', 'lingerie', 'porn', 'adult'],
    'haram': ['alcohol', 'wine', 'beer', 'drunk', 'pork', 'bacon', 'ham', 'gambling', 'casino', 'bet'],
    'inappropriate': ['violence', 'blood', 'gore', 'weapon', 'gun', 'rifle', 'war', 'terror', 'bomb'],
    'immoral': ['drugs', 'marijuana', 'cocaine', 'heroin', 'smoking', 'cigarette', 'weed']
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
STOP_WORDS = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being'])

# Initialize NLTK stopwords if available
try:
    if NLTK_AVAILABLE:
        STOP_WORDS = set(stopwords.words('english'))
except:
    pass

# ========================================== 
# 3. SUBMAGIC-LEVEL NLP ANALYZER
# ========================================== 

class SubMagicNLPAnalyzer:
    """Advanced NLP analysis for SubMagic-level semantic understanding"""
    
    # Emotion indicators with confidence scores
    EMOTION_PATTERNS = {
        "excitement": {
            "words": ["amazing", "incredible", "wow", "boom", "unbelievable", "shocking", "explosive"],
            "punctuation": ["!", "?!"],
            "intensity": "high",
            "visual_style": "dynamic_fast"
        },
        "curiosity": {
            "words": ["discover", "found", "secret", "mystery", "hidden", "reveal", "uncover"],
            "punctuation": ["?"],
            "intensity": "medium",
            "visual_style": "mysterious_zoom"
        },
        "frustration": {
            "words": ["problem", "struggle", "failed", "difficult", "hard", "challenge", "impossible"],
            "punctuation": ["..."],
            "intensity": "medium",
            "visual_style": "dark_slow"
        },
        "success": {
            "words": ["success", "achieved", "won", "victory", "breakthrough", "finally", "accomplished"],
            "punctuation": ["!"],
            "intensity": "high",
            "visual_style": "bright_rising"
        },
        "analytical": {
            "words": ["data", "research", "study", "analysis", "shows", "indicates", "proves"],
            "punctuation": ["."],
            "intensity": "low",
            "visual_style": "stable_professional"
        }
    }
    
    # Visual intent mapping
    INTENT_PATTERNS = {
        "explanation": ["because", "reason", "why", "how", "means", "explains"],
        "demonstration": ["shows", "demonstrates", "reveals", "displays", "presents"],
        "comparison": ["versus", "compared", "difference", "better", "worse", "than"],
        "process": ["first", "then", "next", "finally", "step", "stage"],
        "result": ["result", "outcome", "effect", "impact", "consequence"],
        "question": ["what", "who", "where", "when", "which", "whose"]
    }
    
    @staticmethod
    def extract_entities(text):
        """Extract key entities (nouns, verbs, concepts) - with NLTK fallback"""
        entities = {
            'nouns': [],
            'verbs': [],
            'adjectives': [],
            'concepts': []
        }
        
        if not NLTK_AVAILABLE:
            # Fallback: Simple word extraction
            words = re.findall(r'\b\w{4,}\b', text.lower())
            entities['nouns'] = [w for w in words if w not in STOP_WORDS][:5]
            entities['verbs'] = entities['nouns'][:2]
            entities['adjectives'] = []
            
            # Extract 2-word phrases
            words_list = [w for w in words if w not in STOP_WORDS]
            for i in range(len(words_list) - 1):
                entities['concepts'].append(f"{words_list[i]} {words_list[i+1]}")
            
            return entities
        
        try:
            # Use NLTK if available
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            for word, tag in tagged:
                if len(word) < 3 or word in STOP_WORDS:
                    continue
                    
                if tag.startswith('NN'):  # Noun
                    entities['nouns'].append(word)
                elif tag.startswith('VB'):  # Verb
                    entities['verbs'].append(word)
                elif tag.startswith('JJ'):  # Adjective
                    entities['adjectives'].append(word)
            
            # Extract multi-word concepts
            words = [w for w in tokens if w not in STOP_WORDS and len(w) > 3]
            for i in range(len(words) - 1):
                entities['concepts'].append(f"{words[i]} {words[i+1]}")
        except:
            # Fallback on error
            words = re.findall(r'\b\w{4,}\b', text.lower())
            entities['nouns'] = [w for w in words if w not in STOP_WORDS][:5]
        
        return entities
    
    @staticmethod
    def detect_emotion(text):
        """Detect primary emotion with confidence score"""
        text_lower = text.lower()
        scores = {}
        
        for emotion, data in SubMagicNLPAnalyzer.EMOTION_PATTERNS.items():
            score = 0
            
            # Check words
            for word in data["words"]:
                if word in text_lower:
                    score += 2
            
            # Check punctuation
            for punct in data["punctuation"]:
                if punct in text:
                    score += 1
            
            scores[emotion] = score
        
        if not scores or max(scores.values()) == 0:
            return "analytical", 0.5
        
        best_emotion = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_emotion[1] / 5.0, 1.0)
        
        return best_emotion[0], confidence
    
    @staticmethod
    def detect_intent(text):
        """Detect visual intent"""
        text_lower = text.lower()
        
        for intent, keywords in SubMagicNLPAnalyzer.INTENT_PATTERNS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        return "explanation"
    
    @staticmethod
    def analyze_sentence_deep(text):
        """Deep semantic analysis - SubMagic level"""
        entities = SubMagicNLPAnalyzer.extract_entities(text)
        emotion, confidence = SubMagicNLPAnalyzer.detect_emotion(text)
        intent = SubMagicNLPAnalyzer.detect_intent(text)
        
        # Extract key concepts (top entities)
        key_nouns = entities['nouns'][:3]
        key_verbs = entities['verbs'][:2]
        key_concepts = entities['concepts'][:2]
        
        return {
            'text': text,
            'emotion': emotion,
            'emotion_confidence': confidence,
            'intent': intent,
            'key_nouns': key_nouns,
            'key_verbs': key_verbs,
            'key_concepts': key_concepts,
            'visual_style': SubMagicNLPAnalyzer.EMOTION_PATTERNS[emotion]['visual_style']
        }

# ========================================== 
# 4. SEMANTIC QUERY GENERATOR
# ========================================== 

class SemanticQueryGenerator:
    """Generate SubMagic-level contextual queries"""
    
    CATEGORY_VISUAL_KEYWORDS = {
        "artificial intelligence": {
            "core": ["ai", "neural network", "machine learning", "algorithm"],
            "visual": ["data flow", "digital brain", "tech visualization", "code animation"],
            "mood": ["futuristic", "innovative", "advanced", "intelligent"]
        },
        "technology": {
            "core": ["technology", "digital", "innovation", "computing"],
            "visual": ["circuit board", "data stream", "tech interface", "coding"],
            "mood": ["modern", "cutting-edge", "sleek", "professional"]
        },
        "business": {
            "core": ["business", "corporate", "strategy", "growth"],
            "visual": ["office", "meeting", "collaboration", "presentation"],
            "mood": ["professional", "successful", "dynamic", "strategic"]
        },
        "finance": {
            "core": ["finance", "money", "investment", "wealth"],
            "visual": ["charts", "graphs", "stock market", "growth"],
            "mood": ["prosperous", "analytical", "successful", "strategic"]
        },
        "personal development": {
            "core": ["growth", "improvement", "success", "mindset"],
            "visual": ["journey", "progress", "achievement", "transformation"],
            "mood": ["inspiring", "motivational", "uplifting", "empowering"]
        },
        "health": {
            "core": ["health", "wellness", "fitness", "vitality"],
            "visual": ["exercise", "nutrition", "wellbeing", "balance"],
            "mood": ["energetic", "healthy", "positive", "active"]
        },
        "education": {
            "core": ["learning", "education", "knowledge", "study"],
            "visual": ["classroom", "books", "research", "teaching"],
            "mood": ["educational", "informative", "academic", "scholarly"]
        },
        "science": {
            "core": ["science", "research", "discovery", "experiment"],
            "visual": ["laboratory", "microscope", "data", "analysis"],
            "mood": ["scientific", "analytical", "precise", "innovative"]
        }
    }
    
    @staticmethod
    def generate_queries(analysis, category):
        """Generate multi-layer semantic queries"""
        queries = []
        
        # Get category data
        cat_data = SemanticQueryGenerator.CATEGORY_VISUAL_KEYWORDS.get(
            category, 
            SemanticQueryGenerator.CATEGORY_VISUAL_KEYWORDS["technology"]
        )
        
        emotion = analysis['emotion']
        intent = analysis['intent']
        key_nouns = analysis['key_nouns']
        key_verbs = analysis['key_verbs']
        key_concepts = analysis['key_concepts']
        
        # Layer 1: Exact semantic match (highest priority)
        if key_concepts:
            queries.append(f"{key_concepts[0]} {cat_data['core'][0]} cinematic")
        
        if key_nouns:
            top_noun = key_nouns[0]
            queries.append(f"{top_noun} {emotion} {cat_data['visual'][0]}")
        
        # Layer 2: Contextual combinations
        if key_verbs and key_nouns:
            queries.append(f"{key_verbs[0]} {key_nouns[0]} {cat_data['mood'][0]}")
        
        # Layer 3: Emotion + Intent + Category
        queries.append(f"{cat_data['visual'][0]} {emotion} {intent}")
        queries.append(f"{cat_data['mood'][0]} {cat_data['visual'][1]} cinematic")
        
        # Layer 4: Category-specific visuals
        for visual in cat_data['visual'][:2]:
            queries.append(f"{visual} professional 4k")
        
        # Layer 5: Emotion-driven queries
        if emotion == "excitement":
            queries.append(f"{cat_data['core'][0]} dynamic fast motion")
            queries.append(f"energetic {cat_data['visual'][0]} movement")
        elif emotion == "curiosity":
            queries.append(f"{cat_data['core'][0]} mysterious reveal")
            queries.append(f"discovery {cat_data['visual'][1]} zoom")
        elif emotion == "success":
            queries.append(f"{cat_data['core'][0]} achievement celebration")
            queries.append(f"victory {cat_data['visual'][0]} triumphant")
        
        # Layer 6: Intent-driven queries
        if intent == "explanation":
            queries.append(f"{cat_data['core'][0]} educational clear")
        elif intent == "demonstration":
            queries.append(f"{cat_data['core'][0]} showing process")
        
        # Layer 7: Fallback safe queries
        queries.extend([
            f"{cat_data['core'][0]} cinematic 4k",
            f"professional {cat_data['visual'][0]}",
            f"{cat_data['mood'][0]} background footage"
        ])
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:10]

# ========================================== 
# 5. ISLAMIC CONTENT FILTER
# ========================================== 

class IslamicContentFilter:
    """Multi-layer Islamic compliance checking"""
    
    @staticmethod
    def is_query_safe(query):
        """Check if query is Islamic-compliant"""
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Check forbidden keywords
        for category, keywords in FORBIDDEN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return False
        
        # Check problematic patterns
        problematic = [
            r'bikini|swimsuit|underwear',
            r'sexy|erotic|sensual',
            r'alcohol|drunk|wine|beer|liquor',
            r'violence|blood|gore|brutal',
            r'drugs|marijuana|cocaine',
            r'gambling|casino|poker'
        ]
        
        for pattern in problematic:
            if re.search(pattern, query_lower):
                return False
        
        return True
    
    @staticmethod
    def filter_results(results):
        """Filter video results for compliance"""
        if not results:
            return []
        
        safe_results = []
        for result in results:
            url = result.get('url', '').lower()
            
            # Check URL for problematic content
            is_safe = True
            for category, keywords in FORBIDDEN_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in url:
                        is_safe = False
                        break
                if not is_safe:
                    break
            
            if is_safe:
                safe_results.append(result)
        
        return safe_results

# ========================================== 
# 6. AUDIO GENERATOR (MISSING CLASS)
# ========================================== 

class AudioGenerator:
    """Generate TTS audio with ChatterBox - FIXED VERSION"""
    
    def __init__(self, text, voice_path, output_path):
        self.text = text
        self.voice_path = voice_path
        self.output_path = output_path
        self.success = False
        self.completed = False
        self.error = None
    
    def generate_in_background(self):
        """Generate audio in background thread - Using working ChatterboxTTS method"""
        try:
            print("🎤 Generating audio with ChatterBox TTS...")
            
            # Use the correct ChatterboxTTS class
            from chatterbox.tts import ChatterboxTTS
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Using device: {device}")
            
            # Initialize model
            print("🔧 Loading ChatterboxTTS model...")
            model = ChatterboxTTS.from_pretrained(device=device)
            
            # Clean text - remove any stage directions or brackets
            clean_text = re.sub(r'\[.*?\]', '', self.text)
            clean_text = re.sub(r'\(.*?music.*?\)', '', clean_text, flags=re.IGNORECASE)
            clean_text = clean_text.strip()
            
            # Split into sentences for better processing
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 2]
            
            print(f"📝 Processing {len(sentences)} sentences...")
            all_wavs = []
            
            for i, sentence in enumerate(sentences):
                if i % 10 == 0:
                    print(f"   Processing sentence {i+1}/{len(sentences)}...")
                
                try:
                    with torch.no_grad():
                        # Clean sentence
                        sentence_clean = sentence.replace('"', '').replace('"', '').replace('"', '')
                        
                        # Ensure sentence ends with proper punctuation
                        if not sentence_clean.endswith(('.', '!', '?')):
                            sentence_clean = sentence_clean + '.'
                        
                        # Add slight pause after sentence
                        sentence_clean = sentence_clean + ' '
                        
                        # Generate audio for this sentence
                        wav = model.generate(
                            text=sentence_clean,
                            audio_prompt_path=str(self.voice_path),
                            exaggeration=0.5  # Natural voice variation
                        )
                        
                        all_wavs.append(wav.cpu())
                    
                    # Clear GPU cache periodically
                    if i % 20 == 0 and device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                
                except Exception as e:
                    print(f"⚠️ Skipping sentence {i+1}: {str(e)[:50]}")
                    continue
            
            if not all_wavs:
                self.error = "No audio segments generated"
                print("❌ No audio segments were generated")
                return
            
            # Concatenate all audio segments
            print("🔗 Concatenating audio segments...")
            full_audio = torch.cat(all_wavs, dim=1)
            
            # Add ending silence (2 seconds)
            silence_samples = int(2.0 * 24000)
            silence = torch.zeros((full_audio.shape[0], silence_samples))
            full_audio_padded = torch.cat([full_audio, silence], dim=1)
            
            # Save audio
            print("💾 Saving audio file...")
            torchaudio.save(str(self.output_path), full_audio_padded, 24000)
            
            # Verify file was created
            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 10000:
                audio_duration = full_audio_padded.shape[1] / 24000
                self.success = True
                print(f"✅ Audio generated successfully: {audio_duration:.1f} seconds ({os.path.getsize(self.output_path)/(1024*1024):.1f}MB)")
            else:
                self.error = "Audio file too small or not created"
                print("❌ Audio file validation failed")
        
        except Exception as e:
            self.error = str(e)
            print(f"❌ Audio generation failed: {e}")
            print(f"📋 Full error: {type(e).__name__}: {e}")
        
        finally:
            self.completed = True
            # Final cleanup
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass

# ========================================== 
# 7. SCRIPT GENERATOR
# ========================================== 

def generate_script(topic, duration_mins):
    """Generate script using Gemini"""
    if not GEMINI_KEYS:
        print("❌ No Gemini API keys available")
        return None
    
    try:
        genai.configure(api_key=random.choice(GEMINI_KEYS))
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        word_count = int(duration_mins * 150)  # 150 words per minute
        
        prompt = f"""Write an engaging {duration_mins}-minute video script about: {topic}

Requirements:
- Approximately {word_count} words
- Conversational, engaging tone
- Hook in first 10 seconds
- Clear structure with smooth transitions
- Suitable for voiceover
- No markdown formatting
- Islamic-compliant content (no haram references)

Write ONLY the script text, no titles or labels."""
        
        response = model.generate_content(prompt)
        script = response.text.strip()
        
        # Clean script
        script = re.sub(r'\*\*.*?\*\*', '', script)
        script = re.sub(r'#{1,6}\s', '', script)
        script = re.sub(r'\[.*?\]', '', script)
        script = re.sub(r'\n\s*\n', '\n', script)
        
        print(f"✅ Generated script: {len(script.split())} words")
        return script
        
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        return None

# ========================================== 
# 8. STATUS UPDATER
# ========================================== 

def update_status(progress, message, status="processing", file_url=None):
    """Update status in GitHub repo"""
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
        print(f"⚠️ Status update failed: {e}")

def download_asset(path, local):
    """Download asset from GitHub repo"""
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(local, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Asset download failed: {e}")
    return False

# ========================================== 
# 9. SUBTITLE SYSTEM
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
}
def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def create_subtitles(sentences, ass_file):
    """Create ASS subtitle file"""
    style = random.choice(list(SUBTITLE_STYLES.values()))
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\nPlayResY: 1080\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},2,25,25,45,1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            text = s['text'].strip().upper()
            start = format_ass_time(max(0, s['start'] - 0.1))
            end = format_ass_time(s['end'] - 0.2)
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

# ========================================== 
# 10. VIDEO SEARCH & DOWNLOAD
# ========================================== 

def search_videos(query, service, keys):
    """Search single video service"""
    try:
        if not IslamicContentFilter.is_query_safe(query):
            return []
        
        if service == 'pexels' and keys:
            key = random.choice(keys)
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 15, "orientation": "landscape"},
                timeout=10
            )
            
            if response.status_code == 200:
                results = []
                for video in response.json().get('videos', [])[:8]:
                    files = video.get('video_files', [])
                    if files:
                        best = next((f for f in files if f.get('quality') == 'hd'), files[0])
                        results.append({
                            'url': best['link'],
                            'duration': video.get('duration', 0),
                            'service': 'pexels'
                        })
                return IslamicContentFilter.filter_results(results)
        
        elif service == 'pixabay' and keys:
            key = random.choice(keys)
            response = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": key, "q": query, "per_page": 15},
                timeout=10
            )
            
            if response.status_code == 200:
                results = []
                for video in response.json().get('hits', [])[:8]:
                    videos = video.get('videos', {})
                    if 'large' in videos:
                        results.append({
                            'url': videos['large']['url'],
                            'duration': video.get('duration', 0),
                            'service': 'pixabay'
                        })
                return IslamicContentFilter.filter_results(results)
    
    except Exception as e:
        print(f"Search failed: {e}")
    
    return []

def download_and_process_video(video_info, output_path, duration):
    """Download and process single video"""
    try:
        response = requests.get(video_info['url'], timeout=15, stream=True)
        temp_file = output_path.parent / f"temp_{output_path.name}"
        
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        
        if os.path.getsize(temp_file) < 50000:
            return False
        
        # Process with FFmpeg
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_file),
            "-t", str(duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-b:v", "6M", "-an", str(output_path)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        os.remove(temp_file)
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 10000
    
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# ========================================== 
# 11. PARALLEL VIDEO PROCESSING
# ========================================== 

def process_clip_with_submagic(i, sent, category):
    """Process single clip with SubMagic-level intelligence"""
    try:
        # Deep semantic analysis
        analysis = SubMagicNLPAnalyzer.analyze_sentence_deep(sent['text'])
        
        # Generate semantic queries
        queries = SemanticQueryGenerator.generate_queries(analysis, category)
        
        print(f"    🧠 Clip {i+1} analysis:")
        print(f"       Emotion: {analysis['emotion']} ({analysis['emotion_confidence']:.2f})")
        print(f"       Intent: {analysis['intent']}")
        print(f"       Key concepts: {', '.join(analysis['key_concepts'][:2])}")
        print(f"       Top queries: {queries[:2]}")
        
        duration = max(3.5, sent['end'] - sent['start'])
        
        # Try queries in order of semantic relevance
        for idx, query in enumerate(queries):
            if not IslamicContentFilter.is_query_safe(query):
                continue
            
            # Search both services in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                if PEXELS_KEYS:
                    futures.append(executor.submit(search_videos, query, 'pexels', PEXELS_KEYS))
                if PIXABAY_KEYS:
                    futures.append(executor.submit(search_videos, query, 'pixabay', PIXABAY_KEYS))
                
                all_results = []
                for future in concurrent.futures.as_completed(futures, timeout=15):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except:
                        pass
            
            # Filter unused videos
            with URL_LOCK:
                available = [v for v in all_results if v['url'] not in USED_VIDEO_URLS]
            
            if available:
                video = random.choice(available[:3])
                
                with URL_LOCK:
                    USED_VIDEO_URLS.add(video['url'])
                
                output_path = TEMP_DIR / f"clip_{i}.mp4"
                
                if download_and_process_video(video, output_path, duration):
                    print(f"    ✅ Clip {i+1} processed with query: '{query[:50]}'")
                    return str(output_path)
            
            if idx < 3:
                print(f"    🔄 Clip {i+1} trying query {idx+2}/{len(queries)}")
        
        print(f"    ⚠️ Clip {i+1} - no suitable video found")
        return None
    
    except Exception as e:
        print(f"    ✗ Clip {i+1} failed: {e}")
        return None

def process_all_clips_parallel(sentences, category):
    """Process all clips with parallel execution"""
    processed = []
    
    batch_size = 5
    total = len(sentences)
    
    for batch_idx in range(0, total, batch_size):
        batch = sentences[batch_idx:batch_idx + batch_size]
        progress = 60 + int((batch_idx / total) * 30)
        update_status(progress, f"Processing clips {batch_idx+1}-{min(batch_idx+batch_size, total)}/{total}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(process_clip_with_submagic, batch_idx + i, sent, category): i 
                for i, sent in enumerate(batch)
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    if result:
                        processed.append(result)
                except Exception as e:
                    print(f"Clip processing timeout: {e}")
    
    print(f"✅ Successfully processed {len(processed)}/{total} clips")
    return processed

# ========================================== 
# 12. CATEGORY ANALYZER
# ========================================== 

CATEGORY_MAP = {
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "machine learning": "artificial intelligence",
    "technology": "technology",
    "digital": "technology",
    "business": "business",
    "startup": "business",
    "finance": "finance",
    "money": "finance",
    "investment": "finance",
    "personal development": "personal development",
    "motivation": "personal development",
    "health": "health",
    "fitness": "health",
    "education": "education",
    "learning": "education",
    "science": "science",
    "research": "science"
}

def detect_category(topic, script):
    """Detect content category"""
    topic_lower = topic.lower()
    script_lower = script.lower()
    
    # Check topic
    for keyword, category in CATEGORY_MAP.items():
        if keyword in topic_lower:
            return category
    
    # Check script frequency
    word_freq = defaultdict(int)
    for word in script_lower.split():
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 4:
            word_freq[word] += 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for word, _ in sorted_words:
        for keyword, category in CATEGORY_MAP.items():
            if keyword in word:
                return category
    
    return "technology"

# ========================================== 
# 13. GOOGLE DRIVE UPLOADER
# ========================================== 

def upload_to_google_drive(file_path):
    """Upload video to Google Drive"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print("🔑 Authenticating with Google Drive...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Missing OAuth credentials")
        return None
    
    # Refresh access token
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
        )
        access_token = response.json()['access_token']
        print("✅ Access token obtained")
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        return None
    
    # Upload file
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    metadata = {"name": filename, "mimeType": "video/mp4"}
    if folder_id:
        metadata["parents"] = [folder_id]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size)
    }
    
    # Initialize upload
    response = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers=headers,
        json=metadata
    )
    
    if response.status_code != 200:
        print(f"❌ Upload init failed: {response.text}")
        return None
    
    session_uri = response.headers.get("Location")
    
    # Upload file data
    print(f"☁️ Uploading {filename} ({file_size/(1024*1024):.1f}MB)...")
    
    with open(file_path, "rb") as f:
        upload_response = requests.put(
            session_uri,
            headers={"Content-Length": str(file_size)},
            data=f
        )
    
    if upload_response.status_code in [200, 201]:
        file_id = upload_response.json().get('id')
        print(f"✅ Upload successful! ID: {file_id}")
        
        # Make public
        requests.post(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"🔗 Link: {link}")
        return link
    else:
        print(f"❌ Upload failed: {upload_response.text}")
        return None

# ========================================== 
# 14. VIDEO RENDERER (FIXED - Creates BOTH versions)
# ========================================== 

def render_final_videos(clips, audio_path, ass_file, logo_path):
    """Render two versions: with and without subtitles - FIXED VERSION"""
    
    if not clips:
        print("❌ No clips to render")
        return None, None
    
    valid_clips = [c for c in clips if c and os.path.exists(c)]
    if not valid_clips:
        print("❌ No valid clips")
        return None, None
    
    print(f"🎬 Concatenating {len(valid_clips)} clips...")
    
    # Create concat file
    concat_file = TEMP_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        for clip in valid_clips:
            f.write(f"file '{clip}'\n")
    
    concatenated = TEMP_DIR / "concat.mp4"
    
    # FIX: Proper concatenation command
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", "8M", "-an", str(concatenated)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    
    if not os.path.exists(concatenated) or os.path.getsize(concatenated) < 10000:
        print("❌ Concatenation failed")
        return None, None
    
    print(f"✅ Clips concatenated: {os.path.getsize(concatenated)/(1024*1024):.1f}MB")
    
    # Get audio duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        audio_duration = float(result.stdout.strip())
        print(f"🎵 Audio duration: {audio_duration:.1f}s")
    except:
        print("⚠️ Could not get audio duration, using full video")
        audio_duration = None
    
    # ==================== VERSION 1: NO SUBTITLES ====================
    print("\n🎬 Rendering Version 1 (no subtitles)...")
    final_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_no_subs.mp4"
    
    # Remove existing file if it exists
    if os.path.exists(final_no_subs):
        os.remove(final_no_subs)
    
    if logo_path and os.path.exists(logo_path):
        # With logo
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
            "-shortest"
        ]
        
        if audio_duration:
            cmd.extend(["-t", str(audio_duration)])
        
        cmd.append(str(final_no_subs))
    else:
        # Without logo
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(audio_path),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
            "-shortest"
        ]
        
        if audio_duration:
            cmd.extend(["-t", str(audio_duration)])
        
        cmd.append(str(final_no_subs))
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode == 0 and os.path.exists(final_no_subs):
        file_size = os.path.getsize(final_no_subs) / (1024 * 1024)
        print(f"✅ Version 1 rendered: {file_size:.1f}MB")
    else:
        print(f"❌ Version 1 failed: {result.stderr[:200]}")
        final_no_subs = None
    
    # ==================== VERSION 2: WITH SUBTITLES ====================
    print("\n🎬 Rendering Version 2 (with subtitles)...")
    final_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_with_subs.mp4"
    
    # Remove existing file if it exists
    if os.path.exists(final_with_subs):
        os.remove(final_with_subs)
    
    # CRITICAL FIX: Escape ASS path for FFmpeg
    ass_escaped = str(ass_file).replace('\\', '\\\\').replace(':', '\\:')
    
    if logo_path and os.path.exists(logo_path):
        # With logo AND subtitles
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
            "[1:v]scale=230:-1[logo];"
            f"[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles=filename='{ass_escaped}':force_style='FontSize=60'[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
            "-shortest"
        ]
        
        if audio_duration:
            cmd.extend(["-t", str(audio_duration)])
        
        cmd.append(str(final_with_subs))
    else:
        # Without logo, with subtitles
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"subtitles=filename='{ass_escaped}':force_style='FontSize=60'[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", "10M", "-c:a", "aac", "-b:a", "256k",
            "-shortest"
        ]
        
        if audio_duration:
            cmd.extend(["-t", str(audio_duration)])
        
        cmd.append(str(final_with_subs))
    
    print(f"Running command: {' '.join(cmd[:10])}...")  # Show partial command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode == 0 and os.path.exists(final_with_subs):
        file_size = os.path.getsize(final_with_subs) / (1024 * 1024)
        print(f"✅ Version 2 rendered: {file_size:.1f}MB")
    else:
        print(f"❌ Version 2 failed: {result.stderr[:200]}")
        final_with_subs = None
    
    # ==================== VERIFY BOTH FILES ====================
    print("\n📊 Final Output Verification:")
    if final_no_subs and os.path.exists(final_no_subs):
        size1 = os.path.getsize(final_no_subs) / (1024 * 1024)
        print(f"✅ Version 1 (no subs): {size1:.1f}MB - {final_no_subs}")
    
    if final_with_subs and os.path.exists(final_with_subs):
        size2 = os.path.getsize(final_with_subs) / (1024 * 1024)
        print(f"✅ Version 2 (with subs): {size2:.1f}MB - {final_with_subs}")
    
    return final_no_subs, final_with_subs

# ========================================== 
# 15. MAIN EXECUTION
# ========================================== 

def main():
    """Main execution flow"""
    
    print("\n" + "="*60)
    print("🚀 AI VIDEO GENERATOR - SUBMAGIC INTELLIGENCE")
    print("="*60 + "\n")
    
    update_status(1, "Initializing SubMagic-level engine...")
    
    # Download voice sample
    ref_voice = TEMP_DIR / "voice.mp3"
    if not download_asset(VOICE_PATH, ref_voice):
        update_status(0, "Voice download failed", "failed")
        return
    
    print(f"✅ Voice sample: {os.path.getsize(ref_voice)/(1024*1024):.1f}MB")
    
    # Download logo (optional)
    ref_logo = None
    if LOGO_PATH and LOGO_PATH != "None":
        ref_logo = TEMP_DIR / "logo.png"
        if download_asset(LOGO_PATH, ref_logo):
            print(f"✅ Logo downloaded")
        else:
            ref_logo = None
    
    # Generate or use script
    update_status(10, "Generating script with Gemini...")
    
    if MODE == "topic":
        script = generate_script(TOPIC, DURATION_MINS)
        if not script:
            update_status(0, "Script generation failed", "failed")
            return
    else:
        script = SCRIPT_TEXT
    
    if len(script) < 100:
        update_status(0, "Script too short", "failed")
        return
    
    print(f"✅ Script ready: {len(script.split())} words")
    
    # Detect category
    category = detect_category(TOPIC, script)
    print(f"🎯 Category: {category.upper()}")
    
    # Split into timed sentences
    update_status(15, "Creating timeline with semantic analysis...")
    
    words = script.split()
    total_duration = len(words) / 2.5
    
    sentences = []
    current_time = 0
    chunk_size = 10
    
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        duration = len(chunk) / 2.5
        sentences.append({
            "text": ' '.join(chunk),
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration
        chunk_size = random.randint(8, 12)
    
    if sentences:
        sentences[-1]['end'] += 1.5
    
    print(f"📊 Timeline: {len(sentences)} segments, {current_time:.1f}s")
    
    # Create subtitles
    ass_file = TEMP_DIR / "subtitles.ass"
    create_subtitles(sentences, ass_file)
    print(f"✅ Subtitles created")
    
    # Start audio generation in background
    update_status(20, "Starting parallel audio generation...")
    audio_out = TEMP_DIR / "audio.wav"
    audio_gen = AudioGenerator(script, ref_voice, audio_out)
    audio_thread = Thread(target=audio_gen.generate_in_background)
    audio_thread.start()
    
    # Process videos with SubMagic intelligence
    update_status(25, "Starting SubMagic-level video processing...")
    
    start_time = time.time()
    processed_clips = process_all_clips_parallel(sentences, category)
    elapsed = time.time() - start_time
    
    print(f"⏱️ Video processing: {elapsed:.1f}s")
    
    # Wait for audio
    update_status(85, "Finalizing audio generation...")
    audio_thread.join(timeout=300)
    
    if not audio_gen.success or not os.path.exists(audio_out):
        print("❌ Audio generation failed")
        update_status(0, "Audio failed", "failed")
        return
    
    print(f"✅ Audio ready: {os.path.getsize(audio_out)/(1024*1024):.1f}MB")
    
    # Check clips
    if not processed_clips:
        print("❌ No clips processed")
        update_status(0, "No video content", "failed")
        return
    
    # Render final videos
    update_status(90, "Rendering final videos...")
    final_no_subs, final_with_subs = render_final_videos(
        processed_clips, audio_out, ass_file, ref_logo
    )
    
    # Upload to Google Drive - BOTH VERSIONS
    uploaded_links = []
    
    if final_no_subs and os.path.exists(final_no_subs):
        update_status(93, "Uploading version 1 (no subtitles)...")
        link1 = upload_to_google_drive(final_no_subs)
        if link1:
            uploaded_links.append({"type": "no_subs", "url": link1})
    
    if final_with_subs and os.path.exists(final_with_subs):
        update_status(97, "Uploading version 2 (with subtitles)...")
        link2 = upload_to_google_drive(final_with_subs)
        if link2:
            uploaded_links.append({"type": "with_subs", "url": link2})
    
    # Final status
    if uploaded_links:
        # Show both links
        for link_info in uploaded_links:
            print(f"🔗 {link_info['type']}: {link_info['url']}")
        
        final_url = uploaded_links[-1]["url"]
        update_status(100, "Complete! Videos uploaded.", "completed", final_url)
        print(f"\n🎉 Successfully created {len(uploaded_links)} videos!")
    else:
        update_status(100, "Rendering complete", "completed")
        print("\n⚠️ Videos rendered but not uploaded")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        shutil.rmtree(TEMP_DIR)
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"📊 Final Stats:")
    print(f"   Script: {len(script.split())} words")
    print(f"   Clips: {len(processed_clips)} processed")
    print(f"   Videos: {len(uploaded_links)} uploaded")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
