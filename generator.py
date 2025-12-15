import os
import subprocess
import sys
import re
import time
import random
import shutil
import json
import concurrent.futures
from pathlib import Path

# ==========================================
# 1. AUTO-INSTALLER & SETUP
# ==========================================
print("Installing Dependencies...")
try:
    libs = [
        "chatterbox-tts", "torchaudio", "assemblyai", "google-generativeai", 
        "requests", "beautifulsoup4", "pydub", "nltk", "numpy", "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
import requests
import nltk
from pydub import AudioSegment
from chatterbox.tts import ChatterboxTTS

# Download NLTK data for sentence splitting
nltk.download('punkt', quiet=True)

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Template Placeholders (Do not change these manually)
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_URL = """{{VOICE_URL_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# API Keys
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

# Aggressive Cleanup Start
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. VISUAL DICTIONARY
# ==========================================
VISUAL_MAP = {
    "tech": ["futuristic technology", "server room", "coding screen", "robot arm", "circuit board", "hologram"],
    "business": ["corporate meeting", "handshake slow motion", "office skyscraper", "man in suit walking"],
    "finance": ["stock market graph", "money falling", "gold coins", "bitcoin", "wallet"],
    "nature": ["forest aerial", "ocean waves", "mountain sunset", "flowers blooming", "waterfall"],
    "history": ["ancient ruins", "museum statue", "old map", "vintage library", "pyramids"],
    "abstract": ["ink in water", "light leaks", "particles dust", "geometric shapes", "smoke swirling", "bokeh lights"]
}

def get_visual_queries(text):
    text = text.lower()
    queries = []
    # 1. Match categories
    for cat, terms in VISUAL_MAP.items():
        if cat in text:
            queries.extend(random.sample(terms, min(2, len(terms))))
    
    # 2. Extract significant words
    words = [w for w in re.findall(r'\w+', text) if len(w) > 6]
    if words: 
        queries.append(random.choice(words) + " cinematic")
    
    # 3. Fallback
    if not queries: 
        queries = random.sample(VISUAL_MAP["abstract"], 3)
    
    random.shuffle(queries)
    return list(set(queries))[:3]

# ==========================================
# 4. SCRIPT GENERATION (OPTIMIZED)
# ==========================================
def generate_script(topic, minutes):
    # Word count estimation (approx 145 words/min for smooth reading)
    total_words = int(minutes * 145)
    genai.configure(api_key=GEMINI_KEY)
    
    # Use the requested high-performance model
    # Note: 'gemini-2.0-flash-exp' is the current technical name for next-gen flash
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')

    print(f"--- 📝 Generating Script: {topic} ({minutes} mins / ~{total_words} words) ---")

    # CASE A: Short/Medium Video (<= 15 mins) -> Single Shot (Better Flow)
    if minutes <= 15:
        print("   Mode: Single-Shot Generation")
        prompt = f"""
        Write a complete spoken-word script for a video about '{topic}'.
        Target Length: {total_words} words.
        Style: Engaging, storytelling, documentary flow.
        Format: RAW TEXT ONLY. No headers, no 'Scene 1', no [Music]. Just the narration.
        """
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            print(f"Generation Error: {e}")
            return f"Welcome to our deep dive into {topic}."

    # CASE B: Long Video (> 15 mins) -> Chunked Generation (To avoid limits)
    else:
        print("   Mode: Chunked Generation (Marathon)")
        
        # 1. Create a Hidden Flow (Outline)
        num_sections = int(minutes / 5) # One chunk every 5 mins
        outline_prompt = f"""
        Plan a {minutes}-minute documentary on '{topic}'.
        Break it down into {num_sections} key narrative sections.
        Output just the list of sections.
        """
        try:
            outline_res = model.generate_content(outline_prompt).text
            sections = [s for s in outline_res.split('\n') if s.strip() and not s.startswith('*')]
        except:
            sections = [f"Aspect {i} of {topic}" for i in range(num_sections)]
            
        full_script = []
        
        # 2. Write Chunks
        for i, section in enumerate(sections):
            print(f"   Writing Section {i+1}/{len(sections)}: {section}...")
            
            chunk_words = int(total_words / len(sections))
            
            context_prompt = ""
            if i > 0:
                prev_text = full_script[-1][-200:] # Last 200 chars for context
                context_prompt = f"Context: The previous section ended with: '...{prev_text}'."

            prompt = f"""
            Write the next part of the script focusing on: "{section}".
            {context_prompt}
            Length: ~{chunk_words} words.
            Constraint: Write as a CONTINUOUS FLOW. Do NOT use headings like 'Part 1'. Do NOT say 'Welcome back'.
            Make it sound like one seamless story.
            """
            
            try:
                res = model.generate_content(prompt).text
                # Clean formatting
                clean_text = res.replace("##", "").replace("**", "").replace("Section:", "")
                full_script.append(clean_text.strip())
            except Exception as e:
                print(f"     ⚠️ Skip section {i}: {e}")
                time.sleep(1)

        print(f"   ✓ Script Assembled: {len(' '.join(full_script).split())} words")
        return " ".join(full_script)

# ==========================================
# 5. AUDIO ENGINE (ROBUST)
# ==========================================
def clone_voice_robust(text, ref_audio_path, out_path):
    print("\n--- 🎙️ Starting Voice Cloning Engine ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Intelligent Splitting
        sentences = nltk.sent_tokenize(text)
        print(f"   Queue: {len(sentences)} sentences")
        
        all_tensors = []
        
        for i, sent in enumerate(sentences):
            clean = sent.strip()
            if len(clean) < 2: continue
            
            if i % 10 == 0: print(f"   Rendering Audio {i}/{len(sentences)}...")
            
            try:
                with torch.no_grad():
                    # Generate
                    wav = model.generate(
                        text=clean, 
                        audio_prompt_path=str(ref_audio_path), 
                        exaggeration=0.5, 
                        cfg_weight=0.5
                    )
                    all_tensors.append(wav.cpu())
                    
                    # Micro-pause for natural flow (0.25s)
                    silence = torch.zeros(1, int(24000 * 0.25))
                    all_tensors.append(silence)
            except: 
                continue

        if not all_tensors: return False

        # Merge
        print("   Merging Audio Streams...")
        final_wav = torch.cat(all_tensors, dim=1)
        torchaudio.save(out_path, final_wav, 24000)
        return True

    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ==========================================
# 6. VISUAL ENGINE (BATCHED)
# ==========================================
def download_voice_sample(url, path):
    print(f"Downloading voice: {url}")
    try:
        token = os.environ.get('GH_PAT', '')
        headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3.raw'} if token else {}
        r = requests.get(url, headers=headers, allow_redirects=True)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def generate_subtitles(audio_path):
    print("Generating Subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error: return [], None
    
    ass_path = TEMP_DIR / "style.ass"
    # High Quality Subtitle Style
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w") as f:
        f.write(header)
        for s in t.get_sentences():
            start = format_time_ass(s.start)
            end = format_time_ass(s.end)
            clean = s.text.replace("'", "").replace('"', '')
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{clean}\n")
            
    sentences = [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()]
    return sentences, ass_path

def format_time_ass(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms//10:02d}"

def process_clip(data):
    i, sent, used_ids = data
    duration = max(3.5, sent['end'] - sent['start'])
    queries = get_visual_queries(sent['text'])
    out_p = TEMP_DIR / f"seg_{i}.mp4"
    
    # Try Pexels first, then Pixabay
    found_url = None
    if PEXELS_KEYS:
        try:
            h = {"Authorization": random.choice(PEXELS_KEYS)}
            r = requests.get(f"https://api.pexels.com/videos/search?query={queries[0]}&orientation=landscape&size=medium", headers=h, timeout=4)
            videos = r.json().get('videos', [])
            for v in videos:
                if v['id'] not in used_ids:
                    found_url = v['video_files'][0]['link']; used_ids.add(v['id']); break
        except: pass
        
    if not found_url and PIXABAY_KEYS:
        try:
            k = random.choice(PIXABAY_KEYS)
            r = requests.get(f"https://pixabay.com/api/videos/?key={k}&q={queries[0]}&video_type=film", timeout=4)
            hits = r.json().get('hits', [])
            if hits: found_url = hits[0]['videos']['medium']['url']
        except: pass

    # FFmpeg Processing
    if found_url:
        try:
            raw_p = TEMP_DIR / f"raw_{i}.mp4"
            with open(raw_p, "wb") as f: f.write(requests.get(found_url).content)
            
            subprocess.run([
                "ffmpeg", "-y", "-i", str(raw_p), "-t", str(duration),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an", str(out_p)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(raw_p): os.remove(raw_p)
            return str(out_p)
        except: pass

    # Fallback
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}", "-t", str(duration), "-vf", "fps=30", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_p)

def render_final_video(sentences, audio_path, ass_file, final_output):
    BATCH_SIZE = 50
    total = len(sentences)
    parts = []
    used_ids = set()

    print(f"\n--- 🎬 Visual Processing ({total} clips) ---")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        print(f"   Batch {start}-{end}...")
        
        tasks = [(start + i, sent, used_ids) for i, sent in enumerate(sentences[start:end])]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
            clips = list(exe.map(process_clip, tasks))
            
        # Concat Batch
        batch_txt = TEMP_DIR / f"list_{start}.txt"
        batch_vid = TEMP_DIR / f"part_{start}.mp4"
        with open(batch_txt, "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(batch_txt), "-c", "copy", str(batch_vid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup clips
        for c in clips: 
            if os.path.exists(c): os.remove(c)
        parts.append(str(batch_vid))

    # Merge All
    print("   Merging Batches...")
    full_vis = TEMP_DIR / "full_visual.mp4"
    final_list = TEMP_DIR / "final_list.txt"
    with open(final_list, "w") as f:
        for p in parts: f.write(f"file '{p}'\n")
        
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(final_list), "-c", "copy", str(full_vis)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Final Burn
    print("   Final Render (Burning Subtitles)...")
    ass_fix = str(Path(ass_file).resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", str(full_vis), "-i", str(audio_path),
        "-vf", f"ass='{ass_fix}'",
        "-c:v", "libx264", "-preset", "medium", "-b:v", "4500k",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(final_output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
print("--- INITIALIZING PIPELINE ---")
ref_voice = TEMP_DIR / "ref_voice.mp3"

if not download_voice_sample(VOICE_URL, ref_voice):
    print("Critical: Failed to download voice sample.")
    exit(1)

# 1. Script
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

# 2. Audio
audio_out = TEMP_DIR / "tts_out.wav"
if clone_voice_robust(text, ref_voice, audio_out):
    
    # 3. Subtitles & Video
    sentences, ass_file = generate_subtitles(audio_out)
    if sentences:
        final_file = OUTPUT_DIR / f"final_video_{JOB_ID}.mp4"
        render_final_video(sentences, audio_out, ass_file, final_file)
        print(f"--- SUCCESS: {final_file} ---")
    else:
        print("Subtitle Generation Failed.")
else:
    print("Audio Generation Failed.")
