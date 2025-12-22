"""
AI VIDEO GENERATOR WITH GOOGLE DRIVE UPLOAD - ENHANCED VERSION
==============================================================
FIXED VERSION WITH ENHANCEMENTS:
1. Subtitle Design Implementation (ASS format properly applied)
2. Enhanced 100% Context-Aligned Scoring System
3. Pixabay & Pexels with Visual Map
4. T5 Transformer for intelligent query generation
5. CLIP model for exact visual matching
6. Dual video output (with/without subtitles)
7. Enhanced Islamic content filtering
8. Kaggle-compatible installation
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
import torchaudio
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
import assemblyai as aai
import google.generativeai as genai

# ========================================== 
# 1. KAGGLE-COMPATIBLE INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies (Kaggle Compatible) ---")
try:
    # Kaggle usually has many packages pre-installed
    # We'll only install what's likely missing
    libs = [
        "chatterbox-tts",
        "torchaudio",
        "assemblyai",
        "google-generativeai",
        "pydub",
        "transformers>=4.36.0",  # For T5 and CLIP
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "accelerate",  # For model loading optimization
        "sentencepiece",  # Required for T5 tokenizer
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    
    # Install ffmpeg on Kaggle
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, check=False)
except Exception as e:
    print(f"Install Note: {e}")

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
# 4. T5 TRANSFORMER FOR SMART QUERY GENERATION
# ========================================== 

class T5QueryGenerator:
    def __init__(self):
        """Initialize T5 model for intelligent query generation"""
        print("🧠 Loading T5 Transformer for query generation...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            print("✅ T5 Transformer loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load T5: {e}. Using fallback method.")
            self.model = None
    
    def generate_smart_query(self, script_segment, max_length=50):
        """
        Generate intelligent search queries from script text using T5
        """
        if not self.model:
            # Fallback to keyword extraction
            words = script_segment.split()
            meaningful_words = [w for w in words if len(w) > 4][:3]
            return " ".join(meaningful_words) if meaningful_words else "background"
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                [script_segment], 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate tags
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode output
            decoded_output = self.tokenizer.batch_decode(
                output, 
                skip_special_tokens=True
            )[0]
            
            # Extract first tag as primary query
            tags = list(set(decoded_output.strip().split(", ")))
            
            if tags:
                # Filter for Islamic safety
                primary_tag = tags[0]
                safe_tag = filter_islamic_safe_text(primary_tag)
                
                # Add quality indicator
                final_query = f"{safe_tag} 4k cinematic"
                print(f"    🤖 T5 Generated: '{primary_tag}' → '{final_query}'")
                return final_query
            else:
                return "cinematic background 4k"
                
        except Exception as e:
            print(f"⚠️ T5 generation failed: {e}")
            # Fallback
            words = script_segment.split()
            keywords = [w for w in words if len(w) > 4][:2]
            return f"{' '.join(keywords)} 4k" if keywords else "abstract 4k"

# ========================================== 
# 5. CLIP MODEL FOR EXACT VISUAL MATCHING
# ========================================== 

class CLIPVisualMatcher:
    def __init__(self):
        """Initialize CLIP model for visual-text matching"""
        print("👁️ Loading CLIP Model for visual matching...")
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            print("✅ CLIP Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP: {e}")
            self.clip_model = None
    
    def extract_middle_frame(self, video_path):
        """Extract a single frame from the middle of a video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Get total frames and jump to middle
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                return None
            
            middle_frame = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            
            return None
        except Exception as e:
            print(f"⚠️ Frame extraction failed for {video_path}: {e}")
            return None
    
    def find_best_visual_match(self, script_sentence, video_paths, segment_index):
        """
        Find the best matching video for a script sentence using CLIP
        Returns: (best_video_path, confidence_score)
        """
        if not self.clip_model or not video_paths:
            return video_paths[0] if video_paths else None, 0.0
        
        print(f"    🔍 CLIP Matching for segment {segment_index+1}...")
        
        # Extract frames from all candidate videos
        images = []
        valid_paths = []
        
        for video_path in video_paths:
            frame = self.extract_middle_frame(video_path)
            if frame:
                images.append(frame)
                valid_paths.append(video_path)
        
        if not images:
            print(f"    ⚠️ No valid frames extracted")
            return video_paths[0] if video_paths else None, 0.0
        
        # Process with CLIP
        try:
            inputs = self.clip_processor(
                text=[script_sentence],
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            
            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0)
            
            # Find best match
            best_idx = probs.argmax().item()
            confidence = probs[best_idx].item() * 100
            best_video = valid_paths[best_idx]
            
            print(f"    ✅ CLIP selected: {os.path.basename(best_video)} "
                  f"(Confidence: {confidence:.1f}%)")
            
            return best_video, confidence
            
        except Exception as e:
            print(f"    ⚠️ CLIP matching failed: {e}")
            return video_paths[0] if video_paths else None, 0.0

# ========================================== 
# 6. ENHANCED CONTENT FILTER (ISLAMIC SAFETY)
# ========================================== 

ISLAMIC_FORBIDDEN_TERMS = {
    # Alcohol & Drugs
    "alcohol", "beer", "wine", "whiskey", "vodka", "liquor", "drunk", "intoxicated",
    "drugs", "cocaine", "heroin", "marijuana", "weed", "hashish", "opium", "addiction",
    
    # Nudity & Immodesty
    "nudity", "nude", "naked", "topless", "bikini", "swimsuit", "lingerie", "underwear",
    "sexy", "seductive", "erotic", "porn", "xxx", "adult", "nsfw", "bare", "exposed",
    "cleavage", "braless", "thong", "miniskirt", "short dress", "see-through",
    
    # Violence & War
    "war", "battle", "combat", "soldier", "military", "weapon", "gun", "rifle", "pistol",
    "bullet", "bomb", "explosion", "terrorist", "attack", "murder", "kill", "dead",
    "corpse", "blood", "gore", "violence", "fight", "punch", "hit", "stab", "shoot",
    
    # Haram Animals & Practices
    "pig", "pork", "bacon", "ham", "swine", "dog meat", "cat meat",
    "gambling", "casino", "poker", "bet", "lottery", "slot machine",
    "fortune telling", "witchcraft", "magic", "sorcery", "occult",
    
    # Idolatry & Shirk
    "idol", "statue worship", "false god", "pagan", "satan", "devil", "demon",
    
    # Other Haram Content
    "homosexual", "gay", "lesbian", "lgbt", "transgender", "prostitution",
    "interest", "usury", "riba", "loan shark", "extortion",
    
    # Contextual Haram (when not in religious context)
    "cross", "crucifix", "church altar", "buddha statue", "hindu idol"
}

def filter_islamic_safe_text(text):
    """Remove or replace forbidden Islamic terms from text"""
    text_lower = text.lower()
    
    # Check for forbidden terms
    found_terms = []
    for term in ISLAMIC_FORBIDDEN_TERMS:
        if term in text_lower:
            found_terms.append(term)
    
    if found_terms:
        print(f"⚠️ Found forbidden terms: {found_terms}")
        # Replace with safe alternatives or remove context
        for term in found_terms:
            # Create a regex pattern for the term
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            # Define replacements based on term category
            if term in ["alcohol", "beer", "wine", "whiskey"]:
                text = pattern.sub("beverage", text)
            elif term in ["drugs", "cocaine", "heroin"]:
                text = pattern.sub("substance", text)
            elif term in ["nude", "naked", "topless"]:
                text = pattern.sub("person", text)
            elif term in ["war", "battle", "combat"]:
                text = pattern.sub("conflict", text)
            elif term in ["gun", "rifle", "pistol"]:
                text = pattern.sub("tool", text)
            elif term in ["pig", "pork", "bacon"]:
                text = pattern.sub("animal", text)
            else:
                # Default: remove the word
                text = pattern.sub("", text)
    
    return text.strip()

# ========================================== 
# 7. DUAL VIDEO RENDER PIPELINE
# ========================================== 

def create_dual_video_outputs(video_without_subs, audio_path, ass_file, logo_path, job_id):
    """
    Create two versions of the video:
    1. With burned-in subtitles
    2. Without subtitles (clean version)
    """
    print("🎬 Creating dual video outputs...")
    
    # Output file names
    final_no_subs = OUTPUT_DIR / f"final_{job_id}_no_subs.mp4"
    final_with_subs = OUTPUT_DIR / f"final_{job_id}_with_subs.mp4"
    
    # Ensure ass_file path is properly escaped for FFmpeg
    ass_path_escaped = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    
    # ===== RENDER 1: Video WITHOUT Subtitles =====
    print("   📹 Rendering clean version (no subtitles)...")
    
    if logo_path and os.path.exists(logo_path):
        # With logo overlay
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        cmd_no_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(video_without_subs),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    else:
        # Without logo
        cmd_no_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(video_without_subs),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    
    # Add duration limit if audio duration is known
    if audio_duration:
        cmd_no_subs.extend(["-t", str(audio_duration)])
    
    cmd_no_subs.append(str(final_no_subs))
    
    # Run first render
    try:
        result = subprocess.run(
            cmd_no_subs,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"   ✅ Clean version created: {final_no_subs}")
        else:
            print(f"   ❌ Clean version failed: {result.stderr[:200]}")
            return None, None
    except Exception as e:
        print(f"   ❌ Clean version exception: {e}")
        return None, None
    
    # ===== RENDER 2: Video WITH Subtitles =====
    print("   🎞️ Rendering version with subtitles...")
    
    if logo_path and os.path.exists(logo_path):
        # With logo AND subtitles
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[1:v]scale=230:-1[logo];"
            "[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles='{ass_path_escaped}'[v]"
        )
        cmd_with_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(video_without_subs),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    else:
        # Without logo, just subtitles
        filter_complex = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[bg]subtitles='{ass_path_escaped}'[v]"
        )
        cmd_with_subs = [
            "ffmpeg", "-y", "-hwaccel", "cuda" if torch.cuda.is_available() else "auto",
            "-i", str(video_without_subs),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", "8M",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    
    # Add duration limit if audio duration is known
    if audio_duration:
        cmd_with_subs.extend(["-t", str(audio_duration)])
    
    cmd_with_subs.append(str(final_with_subs))
    
    # Run second render
    try:
        result = subprocess.run(
            cmd_with_subs,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"   ✅ Subtitle version created: {final_with_subs}")
            return final_no_subs, final_with_subs
        else:
            print(f"   ❌ Subtitle version failed: {result.stderr[:200]}")
            # Return at least the clean version
            return final_no_subs, None
    except Exception as e:
        print(f"   ❌ Subtitle version exception: {e}")
        return final_no_subs, None

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        import wave
        with wave.open(str(audio_path), 'rb') as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    except:
        return None

# ========================================== 
# 8. GOOGLE DRIVE UPLOAD (DUAL FILES)
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

def upload_dual_to_google_drive(no_subs_path, with_subs_path):
    """
    Upload both video versions to Google Drive
    Returns: (link_no_subs, link_with_subs)
    """
    print("☁️ Uploading dual videos to Google Drive...")
    
    links = {}
    
    for video_name, video_path in [("no_subs", no_subs_path), ("with_subs", with_subs_path)]:
        if not video_path or not os.path.exists(video_path):
            print(f"   ⚠️ Skipping {video_name} - file not found")
            links[video_name] = None
            continue
        
        print(f"   📤 Uploading {video_name} version...")
        link = upload_to_google_drive(video_path)
        links[video_name] = link
    
    return links.get("no_subs"), links.get("with_subs")

# ========================================== 
# 9. VISUAL DICTIONARY (700+ TOPICS) - SAME AS BEFORE
# ========================================== 

VISUAL_MAP = {
    # TECH & AI (truncated for brevity - keep your original 700+ entries)
    "tech": ["server room", "circuit board", "hologram display", "robot assembly", "coding screen", "data center", "fiber optics", "microchip manufacturing"],
    "technology": ["innovation lab", "tech startup", "silicon valley", "hardware engineering", "semiconductor", "quantum computer"],
    "ai": ["artificial intelligence", "neural network visualization", "machine learning", "deep learning", "robot face", "digital brain", "AI processing"],
    # ... (keep all your original VISUAL_MAP entries)
}

# ========================================== 
# 10. INTELLIGENT CATEGORY-LOCKED SYSTEM - SAME AS BEFORE
# ========================================== 

VIDEO_CATEGORY = None
CATEGORY_KEYWORDS = []

def analyze_script_and_set_category(script, topic):
    """Analyze script to determine primary video category"""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    print("\n🔍 Analyzing script to determine video category...")
    
    full_text = (script + " " + topic).lower()
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has',
                  'this', 'that', 'will', 'can', 'could', 'would', 'should', 'may', 'might'}
    
    words = [w for w in re.findall(r'\b\w+\b', full_text) if len(w) >= 4 and w not in stop_words]
    
    # Count category matches
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
    
    # Sort categories by score
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_categories:
        VIDEO_CATEGORY = sorted_categories[0][0]
        category_score = sorted_categories[0][1]
        
        # Extract all keywords related to this category
        CATEGORY_KEYWORDS = [VIDEO_CATEGORY]
        
        for word in words[:30]:
            if word in VIDEO_CATEGORY or VIDEO_CATEGORY in word:
                CATEGORY_KEYWORDS.append(word)
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
        VIDEO_CATEGORY = "technology"
        CATEGORY_KEYWORDS = ["technology", "tech", "digital", "innovation"]
        print(f"⚠️ No clear category detected. Using default: '{VIDEO_CATEGORY}'")
    
    return VIDEO_CATEGORY, CATEGORY_KEYWORDS

# ========================================== 
# 11. ENHANCED SCORING SYSTEM - SAME AS BEFORE
# ========================================== 

def calculate_enhanced_relevance_score(video, query, sentence_text, context_keywords, full_script="", topic=""):
    """Smart scoring with Islamic content filtering"""
    global VIDEO_CATEGORY, CATEGORY_KEYWORDS
    
    score = 0
    video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
    sentence_lower = sentence_text.lower()
    query_lower = query.lower()
    
    # === ISLAMIC CONTENT FILTERING ===
    for term in ISLAMIC_FORBIDDEN_TERMS:
        if term in video_text:
            score -= 1000  # Instant disqualification
            print(f"      🚫 BLOCKED: Islamic forbidden term '{term}'")
            return score
    
    # === SMART CATEGORY VALIDATION ===
    category_trust_score = 0
    
    if VIDEO_CATEGORY and VIDEO_CATEGORY in video_text:
        category_trust_score += 25
        print(f"      ✓ Category '{VIDEO_CATEGORY}' in video")
    
    keyword_matches = 0
    for keyword in CATEGORY_KEYWORDS[:5]:
        if keyword in video_text:
            keyword_matches += 1
            category_trust_score += 8
    
    if keyword_matches > 0:
        print(f"      ✓ {keyword_matches} category keywords matched")
    
    # Add category trust score
    score += category_trust_score
    
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
    
    # === 3. VIDEO QUALITY (15 points) ===
    quality = video.get('quality', '').lower()
    if '4k' in quality or 'uhd' in quality:
        score += 15
    elif 'hd' in quality or 'high' in quality or 'large' in quality:
        score += 12
    else:
        score += 4
    
    duration = video.get('duration', 0)
    if duration >= 15:
        score += 5
    elif duration >= 10:
        score += 3
    elif duration >= 5:
        score += 1
    
    # === 4. LANDSCAPE VERIFICATION ===
    landscape_indicators = ['landscape', 'horizontal', 'wide', 'panoramic', 'widescreen', '16:9']
    portrait_indicators = ['vertical', 'portrait', '9:16', 'instagram', 'tiktok', 'reel', 'story']
    
    if any(indicator in video_text for indicator in landscape_indicators):
        score += 10
    
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
        if aspect_ratio >= 1.5:
            score += 10
        elif aspect_ratio >= 1.2:
            score += 5
        elif aspect_ratio < 1.0:
            score -= 50
            print(f"      ✗ Portrait aspect ratio detected")
    
    # === 5. PLATFORM & SOURCE QUALITY ===
    service = video.get('service', '')
    if service == 'pexels':
        score += 5
    elif service == 'pixabay':
        score += 3
    
    # === 6. BONUS: Multi-factor Perfect Match ===
    if query_match_count >= 2 and context_match_count >= 1 and quality in ['hd', 'large', '4k', 'uhd']:
        score += 10
        print(f"      ⭐ Perfect match bonus")
    
    # Small random factor for variety
    score += random.randint(0, 2)
    
    # Cap between 0-100
    final_score = min(100, max(0, score))
    
    return final_score

# ========================================== 
# 12. VIDEO SEARCH - SAME AS BEFORE
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
# 13. UTILS (FIXED: Updates GitHub Status)
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Updates status.json in GitHub repo so HTML can read it"""
    
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
# 14. SCRIPT & AUDIO - SAME AS BEFORE
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
    
    # Apply Islamic content filtering
    script = filter_islamic_safe_text(script)
    
    return script.strip()

def call_gemini(prompt):
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            text = response.text.replace("*","").replace("#","").strip()
            
            # Apply Islamic filtering to Gemini output
            text = filter_islamic_safe_text(text)
            
            return text
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
# 15. ENHANCED VISUAL PROCESSING WITH AI MODELS
# ========================================== 

def process_visuals_with_ai(sentences, audio_path, ass_file, logo_path, job_id, full_script="", topic=""):
    """
    Enhanced visual processing with T5 and CLIP integration
    """
    print("🎬 Advanced Visual Processing with AI Models...")
    
    # Initialize AI models
    t5_generator = T5QueryGenerator()
    clip_matcher = CLIPVisualMatcher()
    
    # Analyze script category
    analyze_script_and_set_category(full_script, topic)
    
    USED_VIDEO_URLS = set()
    
    def download_clip_candidates(query, max_candidates=5):
        """Download multiple candidate clips for CLIP evaluation"""
        candidates = []
        
        # Apply Islamic filtering to query
        query = filter_islamic_safe_text(query)
        
        # Try Pexels
        if PEXELS_KEYS and PEXELS_KEYS[0]:
            try:
                key = random.choice([k for k in PEXELS_KEYS if k])
                url = "https://api.pexels.com/videos/search"
                headers = {"Authorization": key}
                params = {
                    "query": query,
                    "per_page": min(max_candidates, 10),
                    "orientation": "landscape",
                    "size": "medium"
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for video in data.get('videos', [])[:max_candidates]:
                        video_files = video.get('video_files', [])
                        if video_files:
                            hd_files = [f for f in video_files if f.get('quality') == 'hd']
                            if hd_files:
                                video_url = hd_files[0]['link']
                                if video_url not in USED_VIDEO_URLS:
                                    # Check video description for Islamic content
                                    video_text = (video.get('user', {}).get('name', '') + ' ' + 
                                                 video.get('url', '')).lower()
                                    has_forbidden_content = False
                                    for term in ISLAMIC_FORBIDDEN_TERMS:
                                        if term in video_text:
                                            has_forbidden_content = True
                                            break
                                    
                                    if not has_forbidden_content:
                                        candidates.append({
                                            'url': video_url,
                                            'duration': video.get('duration', 0),
                                            'service': 'pexels'
                                        })
            except Exception as e:
                print(f"    ⚠️ Pexels error: {e}")
        
        # Try Pixabay if we need more candidates
        if len(candidates) < max_candidates and PIXABAY_KEYS and PIXABAY_KEYS[0]:
            try:
                key = random.choice([k for k in PIXABAY_KEYS if k])
                url = "https://pixabay.com/api/videos/"
                params = {
                    "key": key,
                    "q": query,
                    "per_page": min(max_candidates - len(candidates), 10),
                    "orientation": "horizontal"
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for video in data.get('hits', []):
                        videos_dict = video.get('videos', {})
                        if 'large' in videos_dict:
                            video_url = videos_dict['large']['url']
                            if video_url not in USED_VIDEO_URLS:
                                # Check video tags for Islamic content
                                video_tags = video.get('tags', '').lower()
                                has_forbidden_content = False
                                for term in ISLAMIC_FORBIDDEN_TERMS:
                                    if term in video_tags:
                                        has_forbidden_content = True
                                        break
                                
                                if not has_forbidden_content:
                                    candidates.append({
                                        'url': video_url,
                                        'duration': video.get('duration', 0),
                                        'service': 'pixabay'
                                    })
            except Exception as e:
                print(f"    ⚠️ Pixabay error: {e}")
        
        return candidates
    
    def download_and_save_video(video_info, output_path):
        """Download video to file"""
        try:
            response = requests.get(video_info['url'], timeout=30, stream=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            USED_VIDEO_URLS.add(video_info['url'])
            return True
        except Exception as e:
            print(f"    ❌ Download failed: {e}")
            return False
    
    # Process each sentence
    print(f"📥 Processing {len(sentences)} segments...")
    clip_paths = []
    
    for i, sent in enumerate(sentences):
        update_status(60 + int((i/len(sentences))*20), f"Processing clip {i+1}/{len(sentences)}...")
        
        duration = max(3.5, sent['end'] - sent['start'])
        sentence_text = sent['text']
        
        print(f"  📝 Segment {i+1}: '{sentence_text[:60]}...'")
        
        # Step 1: Generate intelligent query using T5
        query = t5_generator.generate_smart_query(sentence_text)
        print(f"    🔍 T5 Query: '{query}'")
        
        # Step 2: Download candidate clips
        candidates = download_clip_candidates(query, max_candidates=4)
        
        if not candidates:
            print(f"    ⚠️ No clips found, using fallback")
            # Fallback gradient background
            fallback_path = TEMP_DIR / f"fallback_{i}.mp4"
            create_gradient_background(fallback_path, duration)
            clip_paths.append(str(fallback_path))
            continue
        
        # Step 3: Download all candidates locally
        candidate_files = []
        for j, candidate in enumerate(candidates):
            candidate_path = TEMP_DIR / f"candidate_{i}_{j}.mp4"
            if download_and_save_video(candidate, candidate_path):
                # Trim to required duration
                trimmed_path = TEMP_DIR / f"candidate_{i}_{j}_trimmed.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(candidate_path),
                    "-t", str(duration),
                    "-c:v", "copy",
                    str(trimmed_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(trimmed_path):
                    candidate_files.append(str(trimmed_path))
        
        if not candidate_files:
            print(f"    ⚠️ All downloads failed, using fallback")
            fallback_path = TEMP_DIR / f"fallback_{i}.mp4"
            create_gradient_background(fallback_path, duration)
            clip_paths.append(str(fallback_path))
            continue
        
        # Step 4: Use CLIP to find best match
        best_video_path, confidence = clip_matcher.find_best_visual_match(
            sentence_text, candidate_files, i
        )
        
        if best_video_path and confidence > 10:  # Minimum confidence threshold
            print(f"    ✅ Selected: {os.path.basename(best_video_path)} "
                  f"({confidence:.1f}% match)")
            clip_paths.append(best_video_path)
        else:
            print(f"    ⚠️ CLIP low confidence, using first candidate")
            clip_paths.append(candidate_files[0])
        
        # Clean up unused candidates
        for candidate_file in candidate_files:
            if candidate_file != best_video_path:
                try:
                    os.remove(candidate_file)
                except:
                    pass
    
    # Step 5: Concatenate all clips
    print("🔗 Concatenating selected clips...")
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")
    
    concatenated_video = TEMP_DIR / "concatenated.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", "8M",
        str(concatenated_video)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(concatenated_video):
        print(f"✅ Concatenated video created: {concatenated_video}")
        
        # Step 6: Create dual outputs
        no_subs_path, with_subs_path = create_dual_video_outputs(
            concatenated_video, audio_path, ass_file, logo_path, job_id
        )
        
        return no_subs_path, with_subs_path
    else:
        print("❌ Failed to concatenate videos")
        return None, None

def create_gradient_background(output_path, duration):
    """Create a gradient background as fallback"""
    gradients = [
        ("0x0f3460:0x533483", "tech"),
        ("0x1a1a2e:0x16213e", "dark"),
        ("0x1e3a5f:0x2a2d34", "business"),
        ("0x1e4d2b:0x2d5016", "nature")
    ]
    
    gradient, _ = random.choice(gradients)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={duration}",
        "-vf", f"fade=in:0:30,fade=out:st={duration-1}:d=1",
        "-c:v", "libx264",
        "-preset", "fast",
        "-t", str(duration),
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ========================================== 
# 16. MAIN EXECUTION (UPDATED)
# ========================================== 

print("--- 🚀 START (ENHANCED VERSION: T5 + CLIP + Dual Output) ---")
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
    sentences = []
    
    # Try AssemblyAI first
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            print("📝 Transcribing audio...")
            transcript = transcriber.transcribe(str(audio_out))
            
            if transcript.status == aai.TranscriptStatus.completed:
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
    
    # Process visuals with AI
    update_status(60, "Gathering Visuals with AI Matching...")
    no_subs_path, with_subs_path = process_visuals_with_ai(
        sentences, audio_out, ass_file, ref_logo, JOB_ID, text, TOPIC
    )
    
    if no_subs_path or with_subs_path:
        if no_subs_path and os.path.exists(no_subs_path):
            file_size_no_subs = os.path.getsize(no_subs_path) / (1024 * 1024)
            print(f"✅ Clean version created: {file_size_no_subs:.1f} MB")
        
        if with_subs_path and os.path.exists(with_subs_path):
            file_size_with_subs = os.path.getsize(with_subs_path) / (1024 * 1024)
            print(f"✅ Subtitle version created: {file_size_with_subs:.1f} MB")
        
        # Upload to Google Drive
        update_status(90, "Uploading to Google Drive...")
        drive_link_no_subs, drive_link_with_subs = upload_dual_to_google_drive(
            no_subs_path, with_subs_path
        )
        
        # Final status update
        if drive_link_no_subs or drive_link_with_subs:
            links_msg = "Uploaded: "
            if drive_link_no_subs:
                links_msg += "Clean Version | "
            if drive_link_with_subs:
                links_msg += "With Subtitles"
            
            update_status(100, links_msg, "completed", 
                         drive_link_with_subs or drive_link_no_subs)
            
            print(f"\n🎉 DUAL VIDEO GENERATION COMPLETE!")
            if drive_link_no_subs:
                print(f"🔗 Clean Version (No Subtitles): {drive_link_no_subs}")
            if drive_link_with_subs:
                print(f"🔗 With Subtitles: {drive_link_with_subs}")
        else:
            update_status(100, "Videos Ready Locally", "completed")
            print(f"\n📁 Videos saved locally:")
            if no_subs_path:
                print(f"   Clean: {no_subs_path}")
            if with_subs_path:
                print(f"   With Subtitles: {with_subs_path}")
    else:
        update_status(0, "Visual Processing Failed", "failed")
else:
    update_status(0, "Audio Synthesis Failed", "failed")

# Cleanup
print("🧹 Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["visual.mp4", "list.txt", "concat_list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ PROCESS COMPLETE ---")
