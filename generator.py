"""
AI VIDEO GENERATOR - SPANISH VERSION (FIXED)
============================================
✅ Chatterbox Multilingual TTS for Spanish audio (language_id="es")
✅ Helsinki-NLP translation model for live Spanish->English queries
✅ No hardcoded dictionaries - all live AI translation
✅ Proper error handling and installation
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
from pathlib import Path

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "torch",
        "torchaudio",
        "assemblyai",  # For accurate subtitle timing
        "google-generativeai",
        "requests",
        "numpy",
        "transformers",
        "pillow",
        "sentencepiece",
        "chatterbox-tts"
    ]
    
    for lib in libs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
            print(f"✅ {lib}")
        except Exception as e:
            print(f"⚠️  {lib}: {str(e)[:50]}")
    
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, check=False)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio as ta
import google.generativeai as genai
from transformers import MarianMTModel, MarianTokenizer

# Import Chatterbox Multilingual for Spanish
TTS_MODEL = None
TTS_AVAILABLE = False
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    TTS_AVAILABLE = True
    print("✅ Chatterbox Multilingual TTS imported successfully")
except ImportError as e:
    print(f"⚠️ Chatterbox import failed: {e}")
    print("Will use fallback TTS")

# Import AssemblyAI
import assemblyai as aai

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
# 3. LOAD AI MODELS
# ========================================== 

print("--- 🤖 Loading AI Models ---")

# Load Chatterbox Multilingual TTS for Spanish
if TTS_AVAILABLE:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chatterbox Multilingual on {device}...")
        TTS_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("✅ Chatterbox Multilingual TTS loaded successfully")
    except Exception as e:
        print(f"⚠️ Chatterbox loading failed: {e}")
        TTS_AVAILABLE = False

# Translation Model (Spanish -> English)
print("Loading Spanish->English Translation Model...")
TRANSLATOR = None
TRANSLATOR_TOKENIZER = None
TRANSLATOR_AVAILABLE = False
try:
    translation_model_name = "Helsinki-NLP/opus-mt-es-en"
    TRANSLATOR_TOKENIZER = MarianTokenizer.from_pretrained(translation_model_name)
    TRANSLATOR = MarianMTModel.from_pretrained(translation_model_name)
    TRANSLATOR_AVAILABLE = True
    print("✅ Translation Model loaded (Helsinki-NLP opus-mt-es-en)")
except Exception as e:
    print(f"⚠️ Translation Model failed: {e}")

# ========================================== 
# 4. CONTENT FILTERS
# ========================================== 

EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic xxx', 'nsfw','lgbtq','LGBTQ','war','pork',
    'bikini','swim','violence','drugs','terror','gun','gambling'
]

RELIGIOUS_HOLY_TERMS = [
    'jesus', 'christ', 'god', 'lord', 'bible', 'gospel', 'church worship',
    'crucifix', 'crucifixion', 'virgin mary', 'holy spirit', 'baptism',
    'yahweh', 'jehovah', 'torah', 'talmud', 'synagogue', 'rabbi', 'kosher',
    'hanukkah', 'yom kippur', 'passover',
    'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'ganesh', 'hindu temple',
    'vedas', 'bhagavad gita', 'diwali',
    'buddha', 'buddhist temple', 'nirvana', 'dharma', 'meditation buddha',
    'tibetan monk', 'dalai lama',
    'holy book', 'scripture', 'religious ceremony', 'worship service',
    'religious ritual', 'sacred text', 'divine revelation'
]

def is_content_appropriate(text):
    """Content filter with word boundary matching"""
    text_lower = text.lower()
    
    for term in EXPLICIT_CONTENT_BLACKLIST:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Inappropriate - '{term}'")
            return False
    
    for term in RELIGIOUS_HOLY_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious - '{term}'")
            return False
    
    return True

# ========================================== 
# 5. LIVE TRANSLATION (NO DICTIONARIES)
# ========================================== 

def translate_spanish_to_english(spanish_text):
    """Live translation using Helsinki-NLP model"""
    if not TRANSLATOR_AVAILABLE:
        print("    ⚠️ Translator not available")
        return "cinematic background"
    
    try:
        text = spanish_text.strip()[:500]
        
        inputs = TRANSLATOR_TOKENIZER(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        translated_tokens = TRANSLATOR.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        
        english_text = TRANSLATOR_TOKENIZER.decode(
            translated_tokens[0], 
            skip_special_tokens=True
        )
        
        print(f"    🌐 Translation:")
        print(f"       ES: '{spanish_text[:60]}...'")
        print(f"       EN: '{english_text}'")
        
        return english_text.strip()
        
    except Exception as e:
        print(f"    ⚠️ Translation error: {e}")
        words = re.findall(r'\b\w{4,}\b', spanish_text)
        return words[0] if words else "background"

def generate_search_query_from_spanish(spanish_text):
    """Generate English search query from Spanish text"""
    english_text = translate_spanish_to_english(spanish_text)
    
    if not english_text or len(english_text) < 3:
        return "cinematic 4k"
    
    words = re.findall(r'\b\w{4,}\b', english_text.lower())
    keywords = [w for w in words if w not in ['this', 'that', 'with', 'from', 'have', 'been', 'were', 'will']][:2]
    
    if not keywords:
        return "background cinematic"
    
    query = " ".join(keywords) + " cinematic 4k"
    
    if not is_content_appropriate(query):
        print(f"      ⚠️ Query filtered, using generic")
        return "nature cinematic"
    
    print(f"    🎯 Search Query: '{query}'")
    return query

# ========================================== 
# 6. STATUS UPDATES
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Update status for frontend"""
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
            "message": f"Update {progress}%",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
        
        requests.put(url, headers=headers, json=payload)
    except:
        pass

def download_asset(path, local):
    """Download asset from GitHub"""
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
    
    alt_paths = [
        f"static/{path}",
        f"uploads/{path}",
        path.replace("uploads/", "static/"),
        path.replace("static/", "uploads/")
    ]
    
    for alt_path in alt_paths:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{alt_path}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                with open(local, "wb") as f:
                    f.write(r.content)
                return True
        except:
            continue
    
    return False

# ========================================== 
# 7. SPANISH SCRIPT GENERATION
# ========================================== 

def generate_spanish_script(topic, minutes):
    """Generate Spanish script using Gemini"""
    words = int(minutes * 180)
    print(f"Generating Spanish Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
INSTRUCCIONES CRÍTICAS:
- Escribe SOLO texto de narración hablada en ESPAÑOL
- NO incluyas direcciones de escenario, efectos de sonido o [corchetes]
- Comienza directamente con el contenido
- Directrices de contenido islámico: No menciones alcohol, relaciones inapropiadas, apuestas o cerdo
- Tono educativo y apropiado para toda la familia
- Escribe en un estilo documental profesional
- Mantén los párrafos cohesivos y fluidos
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Comenzar'
            prompt = f"{base_instructions}\nEscribe la Parte {i+1}/{chunks} sobre '{topic}'. Contexto: {context}. Longitud: 700 palabras. Mantén la coherencia con la parte anterior."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nEscribe un guión documental en español sobre '{topic}'. {words} palabras. Sé informativo y atractivo."
        script = call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'\n+', ' ', script)
    return script.strip()

def call_gemini(prompt):
    """Call Gemini API"""
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            return response.text.replace("*","").replace("#","").strip()
        except Exception as e:
            print(f"Gemini error: {str(e)[:50]}")
            continue
    return "Error en la generación del guión."

# ========================================== 
# 8. CHATTERBOX MULTILINGUAL TTS FOR SPANISH
# ========================================== 

def generate_tts_from_script(script_text, output_path, audio_prompt_path=None):
    """
    Generate Spanish TTS audio from full script using Chatterbox Multilingual
    Uses language_id="es" for Spanish (23 language support)
    """
    if not TTS_AVAILABLE or TTS_MODEL is None:
        print("⚠️ Chatterbox TTS not available, creating silent audio")
        return create_silent_audio_fallback(output_path, 180)
    
    print("🎙️ Generating Spanish Audio with Chatterbox Multilingual (language_id='es')...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Split script into sentences for processing
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text) if len(s.strip()) > 2]
        
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            if i % 10 == 0:
                print(f"    ✅ Generated {i}/{len(sentences)} audio segments")
            
            try:
                with torch.no_grad():
                    # Clean text
                    text = chunk.replace('"', '').strip()
                    
                    # Generate Spanish audio with language_id="es"
                    # Optional: can pass audio_prompt_path for voice cloning
                    if audio_prompt_path and os.path.exists(audio_prompt_path):
                        wav = TTS_MODEL.generate(
                            text,
                            language_id="es",
                            audio_prompt_path=str(audio_prompt_path)
                        )
                    else:
                        wav = TTS_MODEL.generate(
                            text,
                            language_id="es"
                        )
                    
                    all_wavs.append(wav.cpu())
                
                # Memory cleanup
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    ⚠️ Error on segment {i}: {str(e)[:50]}")
                continue
        
        if all_wavs:
            # Concatenate all audio
            full_audio = torch.cat(all_wavs, dim=1)
            
            # Add 2 seconds of silence at the end
            sample_rate = TTS_MODEL.sr
            silence = torch.zeros((full_audio.shape[0], int(2.0 * sample_rate)))
            full_audio_padded = torch.cat([full_audio, silence], dim=1)
            
            # Save audio
            ta.save(str(output_path), full_audio_padded, sample_rate)
            
            duration = full_audio_padded.shape[-1] / sample_rate
            print(f"✅ Chatterbox Spanish TTS audio saved: {output_path}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Language: Spanish (es)")
            return True
        else:
            print("❌ No audio segments generated")
            return create_silent_audio_fallback(output_path, 180)
        
    except Exception as e:
        print(f"❌ Chatterbox TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return create_silent_audio_fallback(output_path, 180)

def create_silent_audio_fallback(output_path, duration_seconds):
    """Create silent audio as fallback"""
    import wave
    with wave.open(str(output_path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\x00\x00' * int(duration_seconds * 24000))
    
    print(f"⚠️ Silent audio created: {duration_seconds}s")
    return True

# ========================================== 
# 9. ASSEMBLYAI TRANSCRIPTION (FROM GLOBAL VERSION)
# ========================================== 

def transcribe_audio_assemblyai(audio_path, script_text):
    """
    Transcribe audio using AssemblyAI to get accurate subtitle timing
    Matches global version's transcription logic
    """
    sentences = []
    
    if ASSEMBLY_KEY:
        try:
            print("🎯 Transcribing with AssemblyAI for accurate timing...")
            aai.settings.api_key = ASSEMBLY_KEY
            
            config = aai.TranscriptionConfig(
                language_code="es"  # Spanish language
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_path), config=config)
            
            if transcript.status == aai.TranscriptStatus.error:
                print(f"⚠️ AssemblyAI error: {transcript.error}")
                raise Exception("Transcription failed")
            
            # Get sentences with timing
            for sentence in transcript.get_sentences():
                sentences.append({
                    "text": sentence.text,
                    "start": sentence.start / 1000,  # Convert ms to seconds
                    "end": sentence.end / 1000
                })
            
            # Add 1 second to last sentence
            if sentences:
                sentences[-1]['end'] += 1.0
            
            print(f"✅ AssemblyAI transcription complete: {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            print(f"⚠️ AssemblyAI failed: {e}")
            print("Falling back to manual timing...")
    
    # Fallback: Manual timing based on word count
    print("⚠️ Using fallback timing (no AssemblyAI)")
    words = script_text.split()
    
    import wave
    with wave.open(str(audio_path), 'rb') as wav:
        total_duration = wav.getnframes() / float(wav.getframerate())
    
    words_per_second = len(words) / total_duration
    sentences = []
    current_time = 0
    
    # Split into ~12 word chunks
    for i in range(0, len(words), 12):
        chunk = words[i:i+12]
        duration = len(chunk) / words_per_second
        sentences.append({
            "text": ' '.join(chunk),
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration
    
    print(f"✅ Fallback timing complete: {len(sentences)} segments")
    return sentences

# ========================================== 
# 9. SUBTITLE SYSTEM
# ========================================== 

SUBTITLE_STYLES = {
    "modern_white": {
        "name": "Modern White",
        "fontname": "Arial",
        "fontsize": 56,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 1,
        "margin_v": 60,
        "alignment": 2,
        "spacing": 1.0
    },
    "spanish_yellow": {
        "name": "Spanish Yellow",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 3,
        "shadow": 2,
        "margin_v": 50,
        "alignment": 2,
        "spacing": 1.5
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            text = s['text'].strip().replace('\\', '\\\\')
            
            # Split into lines (max 35 chars)
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 35 and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
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
# 10. VIDEO SEARCH WITH LIVE TRANSLATION
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_with_translation(spanish_text, sentence_index):
    """Search videos using live AI translation"""
    english_query = generate_search_query_from_spanish(spanish_text)
    return search_videos_by_query(english_query, sentence_index)

def search_videos_by_query(query, sentence_index, page=None):
    """Search Pexels and Pixabay"""
    if page is None:
        page = random.randint(1, 3)
    
    all_results = []
    
    # Pexels
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 15,
                "page": page,
                "orientation": "landscape"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    video_files = video.get('video_files', [])
                    if video_files:
                        hd_files = [f for f in video_files if f.get('quality') == 'hd']
                        if not hd_files:
                            hd_files = video_files
                        
                        if hd_files:
                            best_file = random.choice(hd_files)
                            video_url = best_file['link']
                            
                            if video_url not in USED_VIDEO_URLS:
                                all_results.append({
                                    'url': video_url,
                                    'service': 'pexels',
                                    'duration': video.get('duration', 0)
                                })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    # Pixabay
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 15,
                "page": page
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    video_url = None
                    for quality in ['large', 'medium', 'small']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    
                    if video_url and video_url not in USED_VIDEO_URLS:
                        all_results.append({
                            'url': video_url,
                            'service': 'pixabay',
                            'duration': video.get('duration', 0)
                        })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

def download_and_process_video(results, target_duration, clip_index):
    """Download and process video"""
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
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_path),
                    "-t", str(target_duration),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-an",
                    str(output_path)
                ]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                try:
                    os.remove(raw_path)
                except:
                    pass
                
                if os.path.exists(output_path):
                    USED_VIDEO_URLS.add(result['url'])
                    print(f"    ✓ {result['service']} video")
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:60]}")
            continue
    
    return None

def process_single_clip(args):
    """Process single clip with live translation"""
    i, sent, sentences_count = args
    
    duration = max(3.5, sent['end'] - sent['start'])
    
    print(f"  🔍 Clip {i+1}/{sentences_count}: '{sent['text'][:50]}...'")
    
    for attempt in range(1, 7):
        print(f"    Attempt {attempt}")
        
        if attempt == 1:
            results = search_videos_with_translation(sent['text'], i)
        elif attempt == 2:
            alt_text = " ".join(sent['text'].split()[:10])
            results = search_videos_with_translation(alt_text, i)
        elif attempt == 3:
            results = search_videos_by_query("nature cinematic", i)
        elif attempt == 4:
            results = search_videos_by_query("abstract motion", i)
        elif attempt == 5:
            results = search_videos_by_query("documentary footage", i)
        else:
            results = search_videos_by_query("stock video", i, page=random.randint(1, 5))
        
        if results:
            clip_path = download_and_process_video(results, duration, i)
            if clip_path:
                print(f"    ✅ Success")
                return (i, clip_path)
        
        time.sleep(0.5)
    
    print(f"    ❌ Failed")
    return (i, None)

def process_visuals(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """Process visuals with parallel processing"""
    print("🎬 Processing Visuals...")
    
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    clips = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(process_single_clip, arg): arg[0] 
            for arg in clip_args
        }
        
        completed = 0
        failed_clips = []
        
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                
                if clip_path:
                    clips[index] = clip_path
                    completed += 1
                else:
                    failed_clips.append(index)
                
                update_status(60 + int((completed/len(sentences))*25), f"Completed {completed}/{len(sentences)}")
                
            except Exception as e:
                index = future_to_index[future]
                failed_clips.append(index)
    
    # Create color backgrounds for failed clips
    if failed_clips:
        print(f"⚠️ Creating backgrounds for {len(failed_clips)} clips")
        for idx in failed_clips:
            if idx < len(sentences):
                duration = max(3.5, sentences[idx]['end'] - sentences[idx]['start'])
                color_path = TEMP_DIR / f"color_{idx}.mp4"
                colors = ["0x2E86C1", "0x27AE60", "0x8E44AD"]
                
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"color=c={colors[idx % 3]}:s=1920x1080:d={duration}",
                    "-c:v", "libx264", str(color_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                clips[idx] = str(color_path)
    
    valid_clips = [c for c in clips if c and os.path.exists(c)]
    
    if not valid_clips:
        return False
    
    # Concatenate clips
    with open("list.txt", "w") as f:
        for c in valid_clips:
            f.write(f"file '{c}'\n")
    
    subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i list.txt -c:v libx264 visual.mp4",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    if not os.path.exists("visual.mp4"):
        return False
    
    # Create versions
    print("📹 Creating final videos...")
    
    # Version 1: 900p no subs
    if logo_path and os.path.exists(logo_path):
        filter_v1 = "[0:v]scale=1600:900[bg];[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25[v]"
        cmd_v1 = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_no_subs)
        ]
    else:
        cmd_v1 = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio_path),
            "-vf", "scale=1600:900",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_no_subs)
        ]
    
    subprocess.run(cmd_v1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Version 2: 1080p with subs
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_v2 = f"[0:v]scale=1920:1080[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_with_subs)
        ]
    else:
        filter_v2 = f"[0:v]scale=1920:1080[bg];[bg]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y", "-i", "visual.mp4", "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_with_subs)
        ]
    
    subprocess.run(cmd_v2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return True

def upload_to_google_drive(file_path):
    """Upload to Google Drive"""
    if not os.path.exists(file_path):
        return None
    
    print(f"☁️ Uploading {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    
    if not all([client_id, client_secret, refresh_token]):
        return None
    
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        access_token = r.json()['access_token']
        
        metadata = {"name": os.path.basename(file_path), "mimeType": "video/mp4"}
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(os.path.getsize(file_path))
        }
        
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
            headers=headers,
            json=metadata
        )
        
        session_uri = r.headers.get("Location")
        
        with open(file_path, "rb") as f:
            upload_resp = requests.put(session_uri, data=f)
        
        if upload_resp.status_code in [200, 201]:
            file_id = upload_resp.json().get('id')
            
            requests.post(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
                headers={"Authorization": f"Bearer {access_token}"},
                json={'role': 'reader', 'type': 'anyone'}
            )
            
            link = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"✅ Uploaded: {link}")
            return link
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
    
    return None

# ========================================== 
# MAIN EXECUTION
# ========================================== 

print("\n" + "="*60)
print("🎬 SPANISH VIDEO GENERATOR")
print("✅ Chatterbox Multilingual TTS (language_id='es')")
print("✅ AssemblyAI for Accurate Spanish Subtitles")
print("✅ Helsinki-NLP Live Translation for Video Search")
print("="*60)

try:
    update_status(1, "Initializing...")
    
    # Download voice reference
    ref_voice = TEMP_DIR / "voice_ref.mp3"
    if not download_asset(VOICE_PATH, ref_voice):
        print("⚠️ Voice download failed, will use default voice")
        ref_voice = None
    else:
        print(f"✅ Voice: {os.path.getsize(ref_voice)} bytes")
    
    # Download logo
    ref_logo = None
    if LOGO_PATH and LOGO_PATH != "None":
        ref_logo = TEMP_DIR / "logo.png"
        if not download_asset(LOGO_PATH, ref_logo):
            ref_logo = None
    
    # Generate script
    update_status(10, "Generating Spanish script...")
    
    if MODE == "topic":
        script_text = generate_spanish_script(TOPIC, DURATION_MINS)
    else:
        script_text = SCRIPT_TEXT
    
    print(f"📝 Script: {len(script_text)} chars")
    
    # Split into sentences
    update_status(15, "Processing sentences...")
    sentences_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text) if len(s.strip()) > 2]
    
    if not sentences_list:
        raise Exception("No valid sentences")
    
    # Create timing
    total_duration = DURATION_MINS * 60
    sentences = []
    current_time = 0
    sentence_duration = total_duration / len(sentences_list)
    
    for text in sentences_list:
        duration = sentence_duration * (0.8 + random.random() * 0.4)
        sentences.append({
            "text": text,
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration
    
    if sentences:
        sentences[-1]['end'] = total_duration
    
    print(f"📊 Sentences: {len(sentences)}")
    
    # Generate Chatterbox TTS audio FIRST (without predefined timing)
    update_status(20, "🎙️ Generating Spanish TTS with Chatterbox (language_id='es')...")
    audio_file = TEMP_DIR / "audio.wav"
    
    # Generate audio from script text (all at once)
    if not generate_tts_from_script(script_text, audio_file, ref_voice):
        raise Exception("Audio generation failed")
    
    if not os.path.exists(audio_file):
        raise Exception("Audio generation failed")
    
    print(f"✅ Audio: {os.path.getsize(audio_file)} bytes")
    
    # Now transcribe with AssemblyAI to get accurate timing
    update_status(50, "Creating Spanish subtitles with AssemblyAI...")
    sentences = transcribe_audio_assemblyai(audio_file, script_text)
    
    if not sentences:
        raise Exception("Subtitle generation failed")
    
    print(f"📊 Sentences: {len(sentences)}")
    
    # Create subtitles
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # Process visuals with live translation
    update_status(60, "🌐 Processing visuals with AI translation...")
    output_no_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_no_subs.mp4"
    output_with_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_with_subs.mp4"
    
    if process_visuals(sentences, audio_file, ass_file, ref_logo, output_no_subs, output_with_subs):
        
        # Upload to Google Drive
        update_status(90, "☁️ Uploading to Google Drive...")
        
        links = {}
        if os.path.exists(output_no_subs):
            links['no_subs'] = upload_to_google_drive(output_no_subs)
        if os.path.exists(output_with_subs):
            links['with_subs'] = upload_to_google_drive(output_with_subs)
        
        # Final status
        final_msg = "✅ Spanish Video Complete!\n"
        final_msg += "🎙️ Chatterbox Multilingual Spanish TTS (language_id='es')\n"
        final_msg += "🎯 AssemblyAI Spanish Subtitles\n"
        final_msg += "🌐 AI Translation for Video Search\n"
        if links.get('no_subs'):
            final_msg += f"📹 No Subs: {links['no_subs']}\n"
        if links.get('with_subs'):
            final_msg += f"📹 With Subs: {links['with_subs']}"
        
        update_status(100, final_msg, "completed", links.get('no_subs') or links.get('with_subs'))
        
        print("\n🎉 SUCCESS!")
        print(f"Script: {len(script_text)} chars")
        print(f"Sentences: {len(sentences)}")
        print(f"Duration: {total_duration:.1f}s")
        if links:
            print("Links:", links)
        
    else:
        raise Exception("Visual processing failed")

except Exception as e:
    error_msg = f"❌ Error: {str(e)}"
    print(error_msg)
    import traceback
    traceback.print_exc()
    update_status(0, error_msg, "failed")
    raise

finally:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    for f in ["visual.mp4", "list.txt"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

print("\n✅ COMPLETE")
