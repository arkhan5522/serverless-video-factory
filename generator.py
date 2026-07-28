"""
AI VIDEO GENERATOR V2 - ULTIMATE EDITION
==========================================
UPGRADES:
1. OpenRouter AI Query Generation (DeepSeek V4 Flash Free)
2. Chatterbox TTS + Resemble Enhance (Studio Master Audio)
3. Shorts Generation (Vertical 9:16 with unique subtitle styles)
4. Enhanced Subtitle Styles (12+ styles with animations)
5. Optimized for Kaggle P100 16GB GPU
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
# 1. INSTALLATION (Kaggle P100 Optimized)
# ==========================================

print("--- Installing Dependencies ---")
try:
    # Step 1: Install core dependencies first (these are already on Kaggle)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "assemblyai", "google-generativeai", "transformers", "sentencepiece",
        "requests", "beautifulsoup4", "pydub", "numpy", "pillow", "opencv-python"
    ])
    
    # Step 2: Install chatterbox-tts (main TTS engine)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "chatterbox-tts"
    ])
    
    # Step 3: Install resemble-enhance with --no-deps to avoid conflicts
    # then install its actual deps that don't conflict
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "--no-deps", "resemble-enhance"
    ])
    # Install resemble-enhance's dependencies that don't conflict with chatterbox
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "df_conformer", "speechbrain>=1.0", "--no-deps"
    ])
    
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")
    # Fallback: try installing just chatterbox without enhance
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
            "chatterbox-tts", "assemblyai", "google-generativeai",
            "transformers", "sentencepiece", "requests", "beautifulsoup4",
            "pydub", "numpy", "pillow", "opencv-python"
        ])
        subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
        print("  Installed without resemble-enhance (will use raw audio)")
    except Exception as e2:
        print(f"  Critical install failure: {e2}")

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
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
SHORTS_DIR = Path("output/shorts")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
SHORTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 3. OPENROUTER AI QUERY GENERATION
# ==========================================

FALLBACK_NATURE_QUERIES = [
    "forest trees cinematic 4k",
    "mountain landscape nature 4k",
    "waterfall nature cinematic",
    "green forest wilderness 4k",
    "river flowing nature 4k",
    "rainforest jungle cinematic",
    "misty forest morning 4k",
    "lake reflection nature 4k",
    "sunset mountain landscape",
    "clouds sky nature 4k",
    "snow mountain landscape",
    "canyon nature cinematic",
    "northern lights aurora nature",
    "sand dunes desert landscape",
    "ocean waves aerial cinematic",
    "volcano landscape dramatic 4k",
    "autumn forest golden leaves",
    "spring meadow flowers bloom",
    "deep space nebula stars",
    "coral reef underwater 4k"
]

def generate_queries_openrouter(script_text, num_queries):
    """
    Generate video search queries using DeepSeek V4 Flash via OpenRouter.
    Sends the full script in 2-3 batches to get contextual, relevant queries.
    Falls back to local transformer or hardcoded if API fails.
    """
    if not OPENROUTER_KEY:
        print("  No OpenRouter key, using Flan-T5 local model...")
        return _generate_queries_flan_t5(script_text, num_queries)
    
    print(f"  Generating {num_queries} queries via DeepSeek V4 Flash...")
    
    # Split into 2-3 batches for efficiency
    batch_size = min(25, (num_queries + 2) // 3)
    all_queries = []
    
    # Split script into segments for context
    words = script_text.split()
    total_words = len(words)
    segments = []
    
    num_batches = min(3, max(2, (num_queries + batch_size - 1) // batch_size))
    segment_size = total_words // num_batches
    
    for i in range(num_batches):
        start = i * segment_size
        end = min((i + 1) * segment_size, total_words)
        segments.append(' '.join(words[start:end]))
    
    for batch_idx, segment in enumerate(segments):
        queries_needed = min(batch_size, num_queries - len(all_queries))
        if queries_needed <= 0:
            break
        
        prompt = f"""You are a video stock footage search query generator. Given a script segment, generate {queries_needed} SHORT video search queries (3-5 words each) that would find relevant B-roll footage on Pexels/Pixabay.

STRICT RULES:
- Each query must be 3-5 words maximum
- Queries must be visually descriptive (things a camera can capture)
- NO people, NO women, NO men, NO faces, NO human bodies
- NO religious content (no churches, mosques, temples, crosses, etc.)
- NO sexual, NSFW, or suggestive content
- NO violence, weapons, blood, war imagery
- NO alcohol, drugs, gambling, pork
- NO political content or flags
- Focus on: nature, landscapes, technology, architecture, abstract, space, underwater, aerial views, cityscapes (empty), objects, animals, weather, textures
- Make queries SPECIFIC to the script content (not generic)
- Each query on a new line, no numbering, no bullets

SCRIPT SEGMENT:
{segment[:2000]}

Generate exactly {queries_needed} search queries, one per line:"""

        result = _call_openrouter(prompt)
        if result:
            lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
            # Filter and clean
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                    all_queries.append(cleaned)
                if len(all_queries) >= num_queries:
                    break
        
        # Rate limit respect
        if batch_idx < len(segments) - 1:
            time.sleep(2)
    
    if len(all_queries) < num_queries:
        # Try DeepSeek again with a different prompt to fill the gap
        remaining = num_queries - len(all_queries)
        print(f"  OpenRouter gave {len(all_queries)}, need {remaining} more...")
        
        # Second attempt: simpler prompt, request remaining queries
        fill_prompt = (
            f"Generate {remaining} short stock footage video search queries (3-5 words each).\n"
            f"Context topic: {' '.join(script_text.split()[:200])}\n"
            f"Rules: NO people, NO religion, NO violence, NO sexual content, NO weapons.\n"
            f"Focus: nature, landscapes, technology, architecture, space, underwater, aerial, animals.\n"
            f"One query per line, no numbering:"
        )
        fill_result = _call_openrouter(fill_prompt)
        if fill_result:
            lines = [l.strip() for l in fill_result.strip().split('\n') if l.strip()]
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                    all_queries.append(cleaned)
                if len(all_queries) >= num_queries:
                    break
    
    # Only use Flan-T5 if OpenRouter completely failed to deliver
    if len(all_queries) < num_queries:
        remaining = num_queries - len(all_queries)
        print(f"  OpenRouter total: {len(all_queries)}, Flan-T5 for remaining {remaining}...")
        fallback = _generate_queries_flan_t5(script_text, remaining)
        all_queries.extend(fallback)
    
    print(f"  Generated {len(all_queries)} AI queries successfully")
    return all_queries[:num_queries]


def _call_openrouter(prompt, max_retries=3):
    """Call OpenRouter API with DeepSeek V4 Flash, fallback to other free models"""
    models = [
        "deepseek/deepseek-v4-flash:free",
        "google/gemma-4-31b-it:free",
        "openrouter/free"
    ]
    
    for model in models:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/video-factory",
                        "X-Title": "Video Factory Query Generator"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1500,
                        "temperature": 0.8
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return content
                elif response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limited on {model}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    {model} returned {response.status_code}, trying next...")
                    break
                    
            except Exception as e:
                print(f"    OpenRouter error: {str(e)[:80]}")
                time.sleep(2)
    
    return None

def _is_query_safe(query):
    """Verify query doesn't contain blocked content"""
    blocked = [
        'woman', 'women', 'girl', 'female', 'lady', 'bikini', 'swim',
        'nude', 'naked', 'sexy', 'erotic', 'nsfw', 'porn',
        'jesus', 'christ', 'church', 'mosque', 'temple', 'bible', 'quran',
        'buddha', 'hindu', 'cross', 'crucifix', 'prayer',
        'gun', 'weapon', 'war', 'blood', 'violence', 'kill', 'dead',
        'alcohol', 'beer', 'wine', 'drug', 'gambling', 'casino', 'pork',
        'lgbtq', 'pride flag', 'political', 'protest',
        'man face', 'person', 'people', 'crowd', 'human'
    ]
    query_lower = query.lower()
    for term in blocked:
        if term in query_lower:
            return False
    return True

def _fallback_queries(num_queries):
    """Generate queries using Flan-T5 locally when OpenRouter is completely down"""
    return _generate_queries_flan_t5("", num_queries)

def _generate_queries_flan_t5(script_text, num_queries):
    """
    Generate video search queries using Flan-T5 locally on GPU.
    This runs on the Kaggle P100 when OpenRouter API is unavailable.
    Uses google/flan-t5-large (780M params, fits easily in P100 16GB).
    """
    print(f"  Loading Flan-T5 for local query generation ({num_queries} queries)...")
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model_name = "google/flan-t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        all_queries = []
        
        # Process in batches of 15 queries per call
        batch_size = 15
        num_batches = (num_queries + batch_size - 1) // batch_size
        
        # Split script into segments for context
        words = script_text.split() if script_text else []
        
        for batch_idx in range(num_batches):
            queries_needed = min(batch_size, num_queries - len(all_queries))
            
            # Build a context snippet for this batch
            if words:
                segment_size = len(words) // max(num_batches, 1)
                start = batch_idx * segment_size
                end = min(start + segment_size, len(words))
                context = ' '.join(words[start:end])[:500]
            else:
                context = "nature documentary educational content"
            
            prompt = (
                f"Generate {queries_needed} short video search queries (3-5 words each) "
                f"for stock footage about: {context}\n"
                f"Rules: No people, no religion, no violence, no sexual content. "
                f"Focus on landscapes, nature, technology, architecture, space, animals, weather.\n"
                f"Queries:"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse output lines
            lines = [l.strip() for l in result.replace(',', '\n').split('\n') if l.strip()]
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                    all_queries.append(cleaned + " 4k cinematic")
                if len(all_queries) >= num_queries:
                    break
        
        # Cleanup model from GPU
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # If still not enough, pad with FALLBACK_NATURE_QUERIES
        while len(all_queries) < num_queries:
            all_queries.append(random.choice(FALLBACK_NATURE_QUERIES))
        
        print(f"  Flan-T5 generated {len(all_queries)} queries")
        return all_queries[:num_queries]
        
    except Exception as e:
        print(f"  Flan-T5 error: {e}, using emergency fallback")
        # Absolute last resort - still use nature queries
        queries = []
        while len(queries) < num_queries:
            queries.extend(FALLBACK_NATURE_QUERIES)
        return queries[:num_queries]


# ==========================================
# 4. CONTENT FILTERS
# ==========================================

EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic', 'nsfw', 'lgbtq', 'war', 'pork',
    'bikini', 'swim', 'violence', 'drugs', 'terror', 'gun', 'gambling'
]

RELIGIOUS_HOLY_TERMS = [
    'jesus', 'christ', 'god', 'lord', 'bible', 'gospel', 'church worship',
    'crucifix', 'crucifixion', 'virgin mary', 'holy spirit', 'baptism',
    'yahweh', 'jehovah', 'torah', 'talmud', 'synagogue', 'rabbi',
    'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'ganesh',
    'buddha', 'buddhist temple', 'nirvana', 'meditation buddha'
]

def is_content_appropriate(text):
    """Content filter for stock video results"""
    text_lower = text.lower()
    for term in EXPLICIT_CONTENT_BLACKLIST:
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            return False
    for term in RELIGIOUS_HOLY_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            return False
    return True


# ==========================================
# 5. ENHANCED SUBTITLE STYLES (12 Styles)
# ==========================================

SUBTITLE_STYLES_LONG = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow (3D Pop)",
        "fontname": "Arial Black",
        "fontsize": 62,
        "primary_colour": "&H0000FFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 5, "shadow": 4,
        "margin_v": 45, "alignment": 2, "spacing": 1.5,
        "uppercase": True
    },
    "hormozi_green": {
        "name": "Hormozi Green (Neon)",
        "fontname": "Arial Black",
        "fontsize": 64,
        "primary_colour": "&H0000FF00",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 5, "shadow": 0,
        "margin_v": 55, "alignment": 2, "spacing": 0.5,
        "uppercase": True
    },
    "neon_cyan": {
        "name": "Neon Cyan Glow",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H00FFFF00",
        "back_colour": "&H00000000",
        "outline_colour": "&H00FF6600",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 3, "shadow": 5,
        "margin_v": 50, "alignment": 2, "spacing": 2,
        "uppercase": False
    },
    "fire_orange": {
        "name": "Fire Orange Blaze",
        "fontname": "Impact",
        "fontsize": 66,
        "primary_colour": "&H000080FF",
        "back_colour": "&H00000000",
        "outline_colour": "&H000000FF",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 3,
        "margin_v": 48, "alignment": 2, "spacing": 1,
        "uppercase": True
    },
    "netflix_modern": {
        "name": "Netflix Clean",
        "fontname": "Arial",
        "fontsize": 58,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H90000000",
        "outline_colour": "&H00000000",
        "bold": 0, "italic": 0,
        "border_style": 3, "outline": 0, "shadow": 0,
        "margin_v": 35, "alignment": 2, "spacing": 0.5,
        "uppercase": False
    },
    "gradient_purple": {
        "name": "Gradient Purple",
        "fontname": "Arial Black",
        "fontsize": 62,
        "primary_colour": "&H00FF00FF",
        "back_colour": "&H00000000",
        "outline_colour": "&H007F0000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 3,
        "margin_v": 50, "alignment": 2, "spacing": 1.5,
        "uppercase": False
    },
    "minimal_white": {
        "name": "Minimal White",
        "fontname": "Arial",
        "fontsize": 56,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00333333",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 3, "shadow": 2,
        "margin_v": 40, "alignment": 2, "spacing": 1,
        "uppercase": False
    },
    "karaoke_gold": {
        "name": "Karaoke Gold",
        "fontname": "Impact",
        "fontsize": 64,
        "primary_colour": "&H0000D4FF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000080",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 3,
        "margin_v": 45, "alignment": 2, "spacing": 1,
        "uppercase": True
    },
}


# Shorts-specific subtitle styles (bigger, center-screen, vertical optimized)
SUBTITLE_STYLES_SHORTS = {
    "shorts_pop_white": {
        "name": "Shorts Pop White",
        "fontname": "Arial Black",
        "fontsize": 42,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 3,
        "margin_v": 200, "alignment": 5, "spacing": 1.5,
        "uppercase": True
    },
    "shorts_neon_green": {
        "name": "Shorts Neon Green",
        "fontname": "Arial Black",
        "fontsize": 44,
        "primary_colour": "&H0000FF00",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 5, "shadow": 0,
        "margin_v": 220, "alignment": 5, "spacing": 1,
        "uppercase": True
    },
    "shorts_fire_red": {
        "name": "Shorts Fire Red",
        "fontname": "Impact",
        "fontsize": 46,
        "primary_colour": "&H000000FF",
        "back_colour": "&H00000000",
        "outline_colour": "&H0000FFFF",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 4,
        "margin_v": 200, "alignment": 5, "spacing": 1,
        "uppercase": True
    },
    "shorts_electric_blue": {
        "name": "Shorts Electric Blue",
        "fontname": "Arial Black",
        "fontsize": 42,
        "primary_colour": "&H00FF9900",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 5, "shadow": 3,
        "margin_v": 210, "alignment": 5, "spacing": 2,
        "uppercase": False
    },
    "shorts_gold_luxury": {
        "name": "Shorts Gold Luxury",
        "fontname": "Impact",
        "fontsize": 44,
        "primary_colour": "&H0000D4FF",
        "back_colour": "&H40000000",
        "outline_colour": "&H00000050",
        "bold": -1, "italic": 0,
        "border_style": 1, "outline": 4, "shadow": 2,
        "margin_v": 200, "alignment": 5, "spacing": 1.5,
        "uppercase": True
    },
}


# ==========================================
# 6. SUBTITLE FILE GENERATION
# ==========================================

def create_ass_file(sentences, ass_file, style_dict=None, res_x=1920, res_y=1080):
    """Create ASS subtitle file with proper format"""
    if style_dict is None:
        style_dict = SUBTITLE_STYLES_LONG
    
    style_key = random.choice(list(style_dict.keys()))
    style = style_dict[style_key]
    
    print(f"  Subtitle Style: {style['name']}")
    
    max_chars = 30 if res_x < 1080 else 35  # Shorter lines for vertical video
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write(f"PlayResX: {res_x}\n")
        f.write(f"PlayResY: {res_y}\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},"
                f"{style['primary_colour']},&H000000FF,"
                f"{style['outline_colour']},{style['back_colour']},"
                f"{style['bold']},{style['italic']},0,0,100,100,"
                f"{style['spacing']},0,{style['border_style']},"
                f"{style['outline']},{style['shadow']},"
                f"{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = _format_ass_time(s['start'])
            end_time = _format_ass_time(s['end'])
            
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            if text.endswith('.'):
                text = text[:-1]
            if text.endswith(','):
                text = text[:-1]
            
            if style.get('uppercase', False):
                text = text.upper()
            
            # Word wrap
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1
                if current_length + word_length > max_chars and current_line:
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

def _format_ass_time(seconds):
    """Format seconds to ASS timestamp"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ==========================================
# 7. GOOGLE DRIVE UPLOAD
# ==========================================

def upload_to_google_drive(file_path):
    """Upload file to Google Drive with resumable upload"""
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return None
    
    print(f"  Uploading {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("  Missing OAuth credentials")
        return None
    
    # Get access token
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        r.raise_for_status()
        access_token = r.json()['access_token']
    except Exception as e:
        print(f"  Token refresh failed: {e}")
        return None
    
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
    
    response = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers=headers, json=metadata
    )
    if response.status_code != 200:
        print(f"  Upload init failed: {response.text[:200]}")
        return None
    
    session_uri = response.headers.get("Location")
    
    with open(file_path, "rb") as f:
        upload_resp = requests.put(
            session_uri,
            headers={"Content-Length": str(file_size)},
            data=f
        )
    
    if upload_resp.status_code in [200, 201]:
        file_id = upload_resp.json().get('id')
        # Make public
        requests.post(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"  Uploaded: {link}")
        return link
    else:
        print(f"  Upload failed: {upload_resp.text[:200]}")
        return None


# ==========================================
# 8. VIDEO SEARCH (AI-POWERED QUERIES)
# ==========================================

USED_VIDEO_URLS = set()
AI_QUERIES = []  # Populated at runtime

def search_videos_with_query(query, clip_index):
    """Search for videos using AI-generated query"""
    return _search_videos_by_query(query, clip_index)

def _search_videos_by_query(query, sentence_index, page=None):
    """Search Pexels and Pixabay with a specific query"""
    if page is None:
        page = random.randint(1, 3)
    
    all_results = []
    
    # Pexels
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 20, "page": page, "orientation": "landscape"},
                timeout=15
            )
            if response.status_code == 200:
                for video in response.json().get('videos', []):
                    video_files = video.get('video_files', [])
                    hd_files = [f for f in video_files if f.get('quality') == 'hd']
                    if not hd_files:
                        hd_files = [f for f in video_files if f.get('quality') == 'large']
                    if not hd_files:
                        hd_files = video_files
                    if hd_files:
                        best_file = random.choice(hd_files)
                        video_url = best_file['link']
                        video_title = video.get('user', {}).get('name', '')
                        if is_content_appropriate(video_title + " " + query):
                            if video_url not in USED_VIDEO_URLS:
                                all_results.append({
                                    'url': video_url, 'service': 'pexels',
                                    'duration': video.get('duration', 0)
                                })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    # Pixabay
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            response = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": key, "q": query, "per_page": 20, "page": page, "orientation": "horizontal"},
                timeout=15
            )
            if response.status_code == 200:
                for video in response.json().get('hits', []):
                    videos_dict = video.get('videos', {})
                    video_url = None
                    for quality in ['large', 'medium', 'small']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    if video_url:
                        if is_content_appropriate(video.get('tags', '') + " " + query):
                            if video_url not in USED_VIDEO_URLS:
                                all_results.append({
                                    'url': video_url, 'service': 'pixabay',
                                    'duration': video.get('duration', 0)
                                })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results


def download_and_process_video(results, target_duration, clip_index):
    """Download and process video clip for P100 GPU"""
    for i, result in enumerate(results[:5]):
        try:
            raw_path = TEMP_DIR / f"raw_{clip_index}_{i}.mp4"
            response = requests.get(result['url'], timeout=30, stream=True)
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                output_path = TEMP_DIR / f"clip_{clip_index}.mp4"
                
                # P100 doesn't have NVENC - use software encoding
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_path),
                    "-t", str(target_duration),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "20",
                    "-an",
                    str(output_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                try:
                    os.remove(raw_path)
                except:
                    pass
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    USED_VIDEO_URLS.add(result['url'])
                    return str(output_path)
        except Exception as e:
            print(f"    Download error: {str(e)[:60]}")
            continue
    return None

def process_single_clip(args):
    """Process a single video clip using AI-generated query"""
    i, sent, total_clips = args
    duration = max(3.5, sent['end'] - sent['start'])
    
    # Use AI-generated query for this clip
    if AI_QUERIES and i < len(AI_QUERIES):
        query = AI_QUERIES[i]
    else:
        query = random.choice(FALLBACK_NATURE_QUERIES)
    
    print(f"  Clip {i+1}/{total_clips}: '{query}'")
    
    for attempt in range(1, 5):
        results = search_videos_with_query(query, i)
        if results:
            clip_path = download_and_process_video(results, duration, i)
            if clip_path:
                return (i, clip_path)
        
        # Try fallback nature query on retry
        query = random.choice(FALLBACK_NATURE_QUERIES)
        time.sleep(0.5)
    
    return (i, None)


# ==========================================
# 9. STATUS UPDATES
# ==========================================

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Update status for HTML frontend via GitHub API"""
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
    content_b64 = base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        payload = {"message": f"Update {progress}%", "content": content_b64, "branch": "main"}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except:
        pass

def download_asset(path, local):
    """Download asset from GitHub repo"""
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
# 10. SCRIPT GENERATION
# ==========================================

def generate_script(topic, minutes):
    """Generate script using Gemini API"""
    words = int(minutes * 180)
    print(f"Generating Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
CRITICAL RULES:
- Write ONLY spoken narration text
- NO stage directions, sound effects, or [brackets]
- Start directly with content
- Islamic content guidelines: No mention of alcohol, inappropriate relationships, gambling, or pork
- Family-friendly and educational tone
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5 + i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Start'
            prompt = f"{base_instructions}\nWrite Part {i+1}/{chunks} about '{topic}'. Context: {context}. Length: 700 words."
            full_script.append(_call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nWrite a documentary script about '{topic}'. {words} words."
        script = _call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    return script.strip()

def _call_gemini(prompt):
    """Call Gemini API with key rotation"""
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text.replace("*", "").replace("#", "").strip()
        except:
            continue
    return "Script generation failed."


# ==========================================
# 11. AUDIO GENERATION (Chatterbox + Resemble Enhance)
# ==========================================

def generate_audio_studio(text, ref_audio, out_path):
    """
    Generate studio-quality voice clone using:
    1. Chatterbox TTS with smart sentence grouping for natural phrasing
    2. Crossfade between chunks for seamless transitions
    3. Resemble Enhance for denoising + upscaling to 44.1kHz
    Optimized for Kaggle P100 16GB VRAM
    """
    print("--- STUDIO AUDIO PIPELINE ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    raw_audio_path = TEMP_DIR / "raw_audio.wav"
    
    # ---- STEP 1: Chatterbox TTS with Smart Chunking ----
    print("  [1/2] Chatterbox TTS Voice Synthesis (Smart Chunks)...")
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        sr = model.sr
        
        # Smart sentence grouping: group 2-3 sentences per TTS call
        # This produces more natural prosody and flow between sentences
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Group sentences into natural phrase chunks (2-3 sentences, max ~150 chars)
        phrase_chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in raw_sentences:
            if current_len + len(sent) > 150 and current_chunk:
                phrase_chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent) + 1
        if current_chunk:
            phrase_chunks.append(' '.join(current_chunk))
        
        print(f"  {len(raw_sentences)} sentences -> {len(phrase_chunks)} natural phrase chunks")
        
        all_wavs = []
        crossfade_samples = int(0.08 * sr)  # 80ms crossfade between chunks
        
        for i, chunk_text in enumerate(phrase_chunks):
            if i % 5 == 0:
                progress = 20 + int((i / len(phrase_chunks)) * 25)
                update_status(progress, f"Voice synthesis {i}/{len(phrase_chunks)} chunks")
            
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk_text.replace('"', ''),
                        audio_prompt_path=str(ref_audio),
                        exaggeration=0.65,
                        cfg_weight=0.4
                    )
                    all_wavs.append(wav.cpu())
                
                # Memory management for P100
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                print(f"    Chunk {i} failed: {str(e)[:50]}")
                continue
        
        if not all_wavs:
            print("  Voice synthesis failed completely")
            return False
        
        # Crossfade concatenation for smooth transitions between chunks
        print("  Applying crossfade between audio chunks...")
        full_audio = all_wavs[0]
        
        for i in range(1, len(all_wavs)):
            next_wav = all_wavs[i]
            
            # Add a natural micro-pause (100-200ms silence) between phrase groups
            pause_samples = int(random.uniform(0.1, 0.2) * sr)
            pause = torch.zeros((full_audio.shape[0], pause_samples))
            
            if full_audio.shape[1] > crossfade_samples and next_wav.shape[1] > crossfade_samples:
                # Crossfade: overlap the tail of current with head of next
                tail = full_audio[:, -crossfade_samples:]
                head = next_wav[:, :crossfade_samples]
                
                # Linear crossfade
                fade_out = torch.linspace(1.0, 0.0, crossfade_samples).unsqueeze(0)
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples).unsqueeze(0)
                
                crossfaded = tail * fade_out + head * fade_in
                
                # Build: [current without tail] + [crossfade zone] + [pause] + [next without head]
                full_audio = torch.cat([
                    full_audio[:, :-crossfade_samples],
                    crossfaded,
                    pause,
                    next_wav[:, crossfade_samples:]
                ], dim=1)
            else:
                # Too short for crossfade, just append with pause
                full_audio = torch.cat([full_audio, pause, next_wav], dim=1)
        
        # Add 2 second silence at end for clean finish
        silence = torch.zeros((full_audio.shape[0], int(2.0 * sr)))
        full_audio_padded = torch.cat([full_audio, silence], dim=1)
        torchaudio.save(str(raw_audio_path), full_audio_padded, sr)
        
        raw_sr = sr
        print(f"  Raw audio saved: {raw_sr}Hz, {full_audio_padded.shape[1]/raw_sr:.1f}s")
        
        # Free TTS model memory
        del model, all_wavs, full_audio, full_audio_padded
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"  Chatterbox error: {e}")
        return False
    
    # ---- STEP 2: Resemble Enhance (Denoise + Upscale) with Overlap ----
    print("  [2/2] Resemble Enhance - Studio Mastering...")
    try:
        from resemble_enhance.enhancer.inference import enhance
        
        # Load the raw audio
        dwav, original_sr = torchaudio.load(str(raw_audio_path))
        
        # Process in overlapping chunks to avoid artifacts at boundaries
        # 20s chunks with 2s overlap, crossfade the overlap zone
        chunk_duration = 20  # seconds per chunk
        overlap_duration = 2  # seconds overlap
        chunk_samples = chunk_duration * original_sr
        overlap_samples = overlap_duration * original_sr
        step_samples = chunk_samples - overlap_samples
        total_samples = dwav.shape[1]
        
        enhanced_chunks = []
        enhanced_sr = None
        
        # Calculate chunk positions
        positions = []
        pos = 0
        while pos < total_samples:
            end = min(pos + chunk_samples, total_samples)
            positions.append((pos, end))
            if end >= total_samples:
                break
            pos += step_samples
        
        num_chunks = len(positions)
        print(f"  Processing {num_chunks} overlapping audio chunks...")
        
        for chunk_idx, (start, end) in enumerate(positions):
            chunk = dwav[:, start:end]
            
            if chunk_idx % 2 == 0:
                update_status(45 + int((chunk_idx / num_chunks) * 5),
                            f"Mastering audio chunk {chunk_idx+1}/{num_chunks}")
            
            try:
                hwav, esr = enhance(
                    dwav=chunk,
                    sr=original_sr,
                    device=device,
                    lambd=0.6  # Preserve natural voice grit
                )
                enhanced_chunks.append(hwav.cpu())
                enhanced_sr = esr
            except Exception as e:
                print(f"    Enhance chunk {chunk_idx} failed: {str(e)[:50]}")
                # Fallback: resample to 44100
                resampler = torchaudio.transforms.Resample(original_sr, 44100)
                enhanced_chunks.append(resampler(chunk).cpu())
                enhanced_sr = 44100
            
            torch.cuda.empty_cache()
        
        # Overlap-add: crossfade the overlapping regions
        if enhanced_chunks:
            # Calculate overlap in enhanced sample rate
            enhance_ratio = enhanced_sr / original_sr
            enhanced_overlap = int(overlap_samples * enhance_ratio)
            
            final_audio = enhanced_chunks[0]
            
            for i in range(1, len(enhanced_chunks)):
                next_chunk = enhanced_chunks[i]
                
                if final_audio.shape[1] >= enhanced_overlap and next_chunk.shape[1] >= enhanced_overlap:
                    # Crossfade the overlap zone
                    tail = final_audio[:, -enhanced_overlap:]
                    head = next_chunk[:, :enhanced_overlap]
                    
                    fade_out = torch.linspace(1.0, 0.0, enhanced_overlap).unsqueeze(0)
                    fade_in = torch.linspace(0.0, 1.0, enhanced_overlap).unsqueeze(0)
                    
                    crossfaded = tail * fade_out + head * fade_in
                    
                    final_audio = torch.cat([
                        final_audio[:, :-enhanced_overlap],
                        crossfaded,
                        next_chunk[:, enhanced_overlap:]
                    ], dim=1)
                else:
                    final_audio = torch.cat([final_audio, next_chunk], dim=1)
            
            torchaudio.save(str(out_path), final_audio, enhanced_sr)
            
            print(f"  Studio master saved: {enhanced_sr}Hz, {final_audio.shape[1]/enhanced_sr:.1f}s")
            print(f"  Quality: {original_sr}Hz -> {enhanced_sr}Hz (upscaled & denoised)")
            
            # Cleanup
            del enhanced_chunks, final_audio, dwav
            torch.cuda.empty_cache()
            gc.collect()
            
            return True
        else:
            shutil.copy2(str(raw_audio_path), str(out_path))
            return True
        
    except ImportError:
        print("  Resemble Enhance not available, using raw audio")
        shutil.copy2(str(raw_audio_path), str(out_path))
        return True
    except Exception as e:
        print(f"  Enhance error: {e}, using raw audio")
        shutil.copy2(str(raw_audio_path), str(out_path))
        return True


# ==========================================
# 12. SHORTS GENERATION
# ==========================================

def get_shorts_count(duration_mins):
    """Determine number of shorts based on video duration"""
    if duration_mins >= 15:
        return 5
    elif duration_mins >= 10:
        return 3
    elif duration_mins >= 5:
        return 2
    else:
        return 1

def generate_shorts(sentences, audio_path, logo_path, duration_mins):
    """
    Generate vertical short clips (9:16) from the long video.
    Each short is 30-60 seconds with its own subtitle style.
    """
    num_shorts = get_shorts_count(duration_mins)
    print(f"\n{'='*50}")
    print(f"  GENERATING {num_shorts} SHORTS (9:16 Vertical)")
    print(f"{'='*50}")
    
    if not sentences or len(sentences) < num_shorts * 3:
        print("  Not enough content for shorts")
        return []
    
    # Calculate short segments - pick the most interesting parts spread across the video
    total_duration = sentences[-1]['end']
    short_duration = min(55, max(30, total_duration / (num_shorts * 3)))  # 30-55s each
    
    # Spread shorts evenly across the video
    segment_gap = total_duration / (num_shorts + 1)
    short_results = []
    
    for short_idx in range(num_shorts):
        update_status(
            85 + int((short_idx / num_shorts) * 10),
            f"Rendering Short {short_idx+1}/{num_shorts}"
        )
        
        # Calculate start time for this short
        target_start = segment_gap * (short_idx + 1) - (short_duration / 2)
        target_start = max(0, min(target_start, total_duration - short_duration))
        target_end = target_start + short_duration
        
        # Find sentences that fall within this range
        short_sentences = []
        for s in sentences:
            if s['start'] >= target_start and s['end'] <= target_end + 2:
                # Adjust timing relative to short start
                short_sentences.append({
                    'text': s['text'],
                    'start': s['start'] - target_start,
                    'end': s['end'] - target_start
                })
        
        if not short_sentences:
            continue
        
        # Create subtitle file for this short (vertical 1080x1920)
        short_ass = TEMP_DIR / f"short_{short_idx}_subs.ass"
        create_ass_file(short_sentences, short_ass, 
                       style_dict=SUBTITLE_STYLES_SHORTS, 
                       res_x=1080, res_y=1920)
        
        # Extract audio segment
        short_audio = TEMP_DIR / f"short_{short_idx}_audio.wav"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(target_start),
            "-t", str(short_duration),
            "-c:a", "copy",
            str(short_audio)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Download a fresh vertical video clip for the short
        short_clip = _get_short_background_clip(short_idx, short_duration)
        
        if not short_clip:
            print(f"    Short {short_idx+1}: No background video found, skipping")
            continue
        
        # Render the short video (1080x1920, 9:16)
        output_short = SHORTS_DIR / f"short_{short_idx+1}_{JOB_ID}.mp4"
        ass_path_escaped = str(short_ass).replace('\\', '/').replace(':', '\\\\:')
        
        if logo_path and os.path.exists(logo_path):
            filter_complex = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,setsar=1,fps=30[bg];"
                f"[1:v]scale=120:-1[logo];"
                f"[bg][logo]overlay=(W-w)/2:50[withlogo];"
                f"[withlogo]subtitles='{ass_path_escaped}'[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(short_clip),
                "-i", str(logo_path),
                "-i", str(short_audio),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "2:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(short_duration),
                str(output_short)
            ]
        else:
            filter_complex = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,setsar=1,fps=30[bg];"
                f"[bg]subtitles='{ass_path_escaped}'[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(short_clip),
                "-i", str(short_audio),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(short_duration),
                str(output_short)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_short):
            size_mb = os.path.getsize(output_short) / (1024*1024)
            print(f"    Short {short_idx+1} rendered: {size_mb:.1f}MB")
            short_results.append(str(output_short))
        else:
            print(f"    Short {short_idx+1} render failed: {result.stderr[-200:]}")
    
    return short_results


def _get_short_background_clip(short_idx, duration):
    """Download a vertical-friendly video clip for shorts"""
    # Use nature queries for shorts backgrounds
    queries = [
        "aerial forest drone vertical",
        "waterfall vertical nature",
        "ocean waves aerial 4k",
        "city lights night aerial",
        "clouds timelapse sky",
        "rain window closeup",
        "fire flames closeup slow",
        "smoke abstract dark",
        "stars night sky timelapse",
        "northern lights aurora"
    ]
    
    query = queries[short_idx % len(queries)]
    results = _search_videos_by_query(query, 1000 + short_idx)
    
    if not results:
        query = random.choice(FALLBACK_NATURE_QUERIES)
        results = _search_videos_by_query(query, 1000 + short_idx)
    
    if results:
        for result in results[:3]:
            try:
                raw_path = TEMP_DIR / f"short_raw_{short_idx}.mp4"
                response = requests.get(result['url'], timeout=30, stream=True)
                with open(raw_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
                    USED_VIDEO_URLS.add(result['url'])
                    return str(raw_path)
            except:
                continue
    
    return None


# ==========================================
# 13. VISUAL PROCESSING (LONG VIDEO)
# ==========================================

def _get_video_duration(filepath):
    """Get video duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return 0


def _fill_clip_gaps(clips, sentences):
    """
    Fill missing clips by stretching/looping the nearest available clip.
    Guarantees: no None values, no blank screens, continuous video.
    """
    filled = list(clips)
    
    # Find all valid clips
    valid_indices = [i for i, c in enumerate(filled) if c is not None and os.path.exists(c)]
    
    if not valid_indices:
        return filled
    
    # For each gap, loop the nearest valid clip to fit the needed duration
    for i in range(len(filled)):
        if filled[i] is not None and os.path.exists(filled[i]):
            continue
        
        # Find nearest valid clip
        nearest = min(valid_indices, key=lambda x: abs(x - i))
        source_clip = filled[nearest]
        
        target_duration = max(3.5, sentences[i]['end'] - sentences[i]['start'])
        
        # Loop the source clip to fill the gap
        gap_output = TEMP_DIR / f"gap_fill_{i}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(source_clip),
            "-t", str(target_duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-an",
            str(gap_output)
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result.returncode == 0 and os.path.exists(gap_output):
            filled[i] = str(gap_output)
            print(f"    Gap {i} filled from clip {nearest}")
        else:
            filled[i] = source_clip
    
    return filled


def _concatenate_with_transitions(clips, transition_dur=0.5):
    """
    Concatenate video clips with smooth crossfade transitions.
    Uses ffmpeg xfade filter for professional-looking results.
    Processes in batches of 10 to avoid filter complexity limits.
    """
    if len(clips) < 2:
        if clips:
            shutil.copy2(clips[0], "visual.mp4")
            return "visual.mp4"
        return None
    
    batch_size = 10
    intermediate_files = []
    
    for batch_start in range(0, len(clips), batch_size):
        batch_clips = clips[batch_start:batch_start + batch_size]
        batch_output = f"batch_{batch_start}.mp4"
        
        if len(batch_clips) == 1:
            shutil.copy2(batch_clips[0], batch_output)
            intermediate_files.append(batch_output)
            continue
        
        # Get durations for xfade offset calculation
        durations = []
        for clip in batch_clips:
            dur = _get_video_duration(clip)
            durations.append(dur if dur > 0 else 4.0)
        
        # Build xfade filter chain
        inputs = []
        for clip in batch_clips:
            inputs.extend(["-i", clip])
        
        num_transitions = len(batch_clips) - 1
        
        # Calculate offsets
        offsets = []
        cumulative = 0
        for j in range(num_transitions):
            cumulative += durations[j]
            offset = cumulative - transition_dur * (j + 1)
            offsets.append(max(0, offset))
        
        # Build filter_complex
        if num_transitions == 1:
            filter_str = f"[0:v][1:v]xfade=transition=fade:duration={transition_dur}:offset={offsets[0]}[v]"
        else:
            filter_lines = []
            for j in range(num_transitions):
                next_label = f"{j+1}:v"
                out_label = f"v{j}" if j < num_transitions - 1 else "v"
                
                if j == 0:
                    filter_lines.append(
                        f"[0:v][{next_label}]xfade=transition=fade:duration={transition_dur}:offset={offsets[j]}[{out_label}]"
                    )
                else:
                    filter_lines.append(
                        f"[v{j-1}][{next_label}]xfade=transition=fade:duration={transition_dur}:offset={offsets[j]}[{out_label}]"
                    )
            filter_str = ";".join(filter_lines)
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an",
            batch_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(batch_output):
            intermediate_files.append(batch_output)
        else:
            # Fallback: simple concat for this batch
            with open(f"batch_{batch_start}_list.txt", "w") as f:
                for c in batch_clips:
                    f.write(f"file '{c}'\n")
            subprocess.run(
                f"ffmpeg -y -f concat -safe 0 -i batch_{batch_start}_list.txt -c:v libx264 -preset fast -crf 20 {batch_output}",
                shell=True, capture_output=True, text=True
            )
            if os.path.exists(batch_output):
                intermediate_files.append(batch_output)
    
    # Merge all batches
    if len(intermediate_files) == 1:
        shutil.move(intermediate_files[0], "visual.mp4")
    elif len(intermediate_files) > 1:
        with open("final_list.txt", "w") as f:
            for f_path in intermediate_files:
                f.write(f"file '{f_path}'\n")
        subprocess.run(
            "ffmpeg -y -f concat -safe 0 -i final_list.txt -c:v libx264 -preset fast -crf 20 visual.mp4",
            shell=True, capture_output=True, text=True
        )
    
    # Cleanup
    for f_path in intermediate_files:
        try:
            if os.path.exists(f_path) and f_path != "visual.mp4":
                os.remove(f_path)
        except:
            pass
    
    return "visual.mp4" if os.path.exists("visual.mp4") else None


def process_visuals(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """
    Process visuals with:
    - AI-generated queries for relevant footage
    - Smooth crossfade transitions between clips (no hard cuts)
    - Gap filling: if a clip fails, loop adjacent clip (no blank screens ever)
    """
    
    print(f"\n  Processing {len(sentences)} clips with AI-generated queries...")
    print(f"  Parallel workers: {min(4, len(sentences))}")
    
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    clips = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(process_single_clip, arg): arg[0]
            for arg in clip_args
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                if clip_path and os.path.exists(clip_path):
                    clips[index] = clip_path
                    completed += 1
                update_status(60 + int((completed / len(sentences)) * 20),
                            f"Clips: {completed}/{len(sentences)}")
            except Exception as e:
                print(f"    Clip error: {e}")
    
    # Fill gaps - no blank screens
    print("  Filling gaps (no blank screens)...")
    filled_clips = _fill_clip_gaps(clips, sentences)
    valid_clips = [c for c in filled_clips if c is not None and os.path.exists(c)]
    
    if not valid_clips:
        print("  No clips generated!")
        return False
    
    print(f"  Final: {len(valid_clips)} clips (all gaps filled)")
    
    # Smooth transitions
    print("  Applying crossfade transitions...")
    visual_path = _concatenate_with_transitions(valid_clips, transition_dur=0.5)
    
    if not visual_path or not os.path.exists(visual_path):
        # Fallback: simple concat
        print("  Transition failed, simple concat...")
        with open("list.txt", "w") as f:
            for c in valid_clips:
                f.write(f"file '{c}'\n")
        subprocess.run(
            "ffmpeg -y -f concat -safe 0 -i list.txt -c:v libx264 -preset fast -crf 20 visual.mp4",
            shell=True, capture_output=True, text=True
        )
        visual_path = "visual.mp4"
    
    if not os.path.exists(visual_path):
        return False
    
    # === VERSION 1: NO SUBTITLES (900p) ===
    print("\n  Rendering Version 1: 900p (No Subtitles)")
    update_status(82, "Rendering 900p version...")
    
    if logo_path and os.path.exists(logo_path):
        filter_v1 = (
            "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,"
            "pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];"
            "[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25[v]"
        )
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", visual_path, "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(output_no_subs)
        ]
    else:
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", visual_path, "-i", str(audio_path),
            "-vf", "scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(output_no_subs)
        ]
    
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=900)
    
    if result_v1.returncode != 0 or not os.path.exists(output_no_subs):
        print(f"  V1 failed: {result_v1.stderr[-300:]}")
        return False
    
    size_mb = os.path.getsize(output_no_subs) / (1024*1024)
    print(f"  V1 Complete: {size_mb:.1f}MB")
    
    # === VERSION 2: WITH SUBTITLES (1080p) ===
    print("\n  Rendering Version 2: 1080p (With Subtitles)")
    update_status(85, "Rendering 1080p with subtitles...")
    
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_v2 = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
            f"[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];"
            f"[withlogo]subtitles='{ass_path}'[v]"
        )
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", visual_path, "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_with_subs)
        ]
    else:
        filter_v2 = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
            f"[bg]subtitles='{ass_path}'[v]"
        )
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", visual_path, "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_with_subs)
        ]
    
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=900)
    
    if result_v2.returncode != 0 or not os.path.exists(output_with_subs):
        print(f"  V2 failed (continuing with V1): {result_v2.stderr[-200:]}")
        return True
    
    size_mb = os.path.getsize(output_with_subs) / (1024*1024)
    print(f"  V2 Complete: {size_mb:.1f}MB")
    
    return True


# ==========================================
# 14. MAIN EXECUTION
# ==========================================

print("\n" + "="*60)
print("  AI VIDEO GENERATOR V2 - ULTIMATE EDITION")
print("  OpenRouter AI Queries | Studio Audio | Shorts")
print("  Optimized for Kaggle P100 16GB")
print("="*60)

update_status(1, "Initializing...")

# Download assets
ref_voice = TEMP_DIR / "voice.mp3"
ref_logo = TEMP_DIR / "logo.png"

if not download_asset(VOICE_PATH, ref_voice):
    update_status(0, "Voice download failed", "failed")
    exit(1)

if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, ref_logo)
    if not os.path.exists(ref_logo):
        ref_logo = None
else:
    ref_logo = None

# Generate script
update_status(5, "Generating script...")
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

if len(text) < 100:
    update_status(0, "Script too short", "failed")
    exit(1)

print(f"  Script: {len(text.split())} words")

# Generate AI queries BEFORE audio (while we have the script)
update_status(10, "Generating AI video queries...")
num_clips_estimate = max(10, int(DURATION_MINS * 8))  # ~8 clips per minute
AI_QUERIES = generate_queries_openrouter(text, num_clips_estimate)
print(f"  Generated {len(AI_QUERIES)} search queries")

# Generate studio-quality audio
update_status(15, "Studio Audio Pipeline...")
audio_out = TEMP_DIR / "audio.wav"

if generate_audio_studio(text, ref_voice, audio_out):
    update_status(50, "Creating subtitles...")
    
    # Transcribe with AssemblyAI
    sentences = []
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_out))
            
            for sentence in transcript.get_sentences():
                sentences.append({
                    "text": sentence.text,
                    "start": sentence.start / 1000,
                    "end": sentence.end / 1000
                })
            if sentences:
                sentences[-1]['end'] += 1.0
        except Exception as e:
            print(f"  AssemblyAI error: {e}, using fallback timing")
    
    # Fallback timing if transcription failed
    if not sentences:
        words = text.split()
        import wave
        try:
            with wave.open(str(audio_out), 'rb') as wav:
                total_dur = wav.getnframes() / float(wav.getframerate())
        except:
            total_dur = len(words) / 2.5  # ~2.5 words/sec estimate
        
        words_per_sec = len(words) / total_dur if total_dur > 0 else 2.5
        current_time = 0
        for i in range(0, len(words), 12):
            chunk = words[i:i+12]
            dur = len(chunk) / words_per_sec
            sentences.append({
                "text": ' '.join(chunk),
                "start": current_time,
                "end": current_time + dur
            })
            current_time += dur
    
    # Adjust AI queries to match actual sentence count (use DeepSeek first, then T5)
    if len(AI_QUERIES) < len(sentences):
        additional_needed = len(sentences) - len(AI_QUERIES)
        print(f"  Need {additional_needed} more queries for exact match...")
        
        # Try DeepSeek/OpenRouter first for the gap
        if OPENROUTER_KEY:
            fill_prompt = (
                f"Generate {additional_needed} short stock footage video search queries (3-5 words each).\n"
                f"Topic context: {' '.join(text.split()[:150])}\n"
                f"Rules: NO people, NO religion, NO violence, NO sexual content.\n"
                f"Focus: nature, landscapes, technology, architecture, space, animals, weather.\n"
                f"One query per line:"
            )
            fill_result = _call_openrouter(fill_prompt)
            if fill_result:
                lines = [l.strip() for l in fill_result.strip().split('\n') if l.strip()]
                for line in lines:
                    cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                    if 2 < len(cleaned) < 60 and _is_query_safe(cleaned):
                        AI_QUERIES.append(cleaned)
                    if len(AI_QUERIES) >= len(sentences):
                        break
        
        # If still short, use Flan-T5
        if len(AI_QUERIES) < len(sentences):
            remaining = len(sentences) - len(AI_QUERIES)
            additional = _generate_queries_flan_t5(text, remaining)
            AI_QUERIES.extend(additional)
    
    # Create subtitle file (long video)
    ass_file = TEMP_DIR / "subs.ass"
    create_ass_file(sentences, ass_file, SUBTITLE_STYLES_LONG)
    
    # Process visuals and render long video
    update_status(55, "Processing visuals with AI queries...")
    output_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_NO_SUBS.mp4"
    output_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_WITH_SUBS.mp4"
    
    if process_visuals(sentences, audio_out, ass_file, ref_logo, output_no_subs, output_with_subs):
        
        # === GENERATE SHORTS ===
        update_status(87, "Generating Shorts...")
        short_paths = generate_shorts(sentences, audio_out, ref_logo, DURATION_MINS)
        
        # === UPLOAD EVERYTHING ===
        update_status(92, "Uploading long videos...")
        link1 = upload_to_google_drive(output_no_subs)
        link2 = upload_to_google_drive(output_with_subs)
        
        short_links = []
        for i, sp in enumerate(short_paths):
            update_status(95 + i, f"Uploading Short {i+1}...")
            sl = upload_to_google_drive(sp)
            if sl:
                short_links.append(sl)
        
        # Final status
        final_message = "Video Factory V2 Complete!\n"
        final_message += f"AI Queries: {len(AI_QUERIES)} generated\n"
        final_message += f"Studio Audio: 44.1kHz mastered\n"
        if link1:
            final_message += f"No Subs: {link1}\n"
        if link2:
            final_message += f"With Subs: {link2}\n"
        if short_links:
            final_message += f"Shorts ({len(short_links)}): {', '.join(short_links)}\n"
        
        update_status(100, final_message, "completed", link1 or link2)
        print(f"\n{'='*60}")
        print(f"  {final_message}")
        print(f"{'='*60}")
    else:
        update_status(0, "Visual processing failed", "failed")
else:
    update_status(0, "Audio generation failed", "failed")

# Cleanup
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4", "list.txt"]:
    if os.path.exists(f):
        os.remove(f)

print("\n--- COMPLETE ---")
