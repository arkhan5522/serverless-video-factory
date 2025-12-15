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
# 1. AUTO-INSTALLER
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

nltk.download('punkt', quiet=True)

# ==========================================
# 2. CONFIGURATION
# ==========================================
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_URL = """{{VOICE_URL_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_visual_queries(text):
    text = text.lower()
    queries = []
    
    # Simple keyword mapping
    map_ = {
        "tech": ["technology", "AI", "computer", "future"],
        "war": ["soldiers", "battle", "army", "smoke"],
        "history": ["ancient", "museum", "ruins", "old map"],
        "nature": ["forest", "river", "sky", "mountains"],
        "city": ["skyscraper", "traffic", "people walking", "neon"],
    }
    
    for k, v in map_.items():
        if k in text: queries.extend(v)
            
    words = [w for w in re.findall(r'\w+', text) if len(w) > 6]
    if words: queries.append(random.choice(words))
    
    if not queries: queries = ["abstract background", "particles", "cinematic light"]
    
    random.shuffle(queries)
    return list(set(queries))[:3]

def generate_with_retry(model, prompt):
    """Retries generation if API quota is hit"""
    for attempt in range(3):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                print(f"     ⚠️ Quota Hit. Sleeping 40s (Attempt {attempt+1}/3)...")
                time.sleep(40)
            else:
                print(f"     ⚠️ Gen Error: {e}")
                break
    return ""

# ==========================================
# 4. SCRIPT GENERATION (FIXED)
# ==========================================
def generate_script(topic, minutes):
    words_per_min = 145
    total_words = int(minutes * words_per_min)
    
    genai.configure(api_key=GEMINI_KEY)
    # Using 1.5 Flash because it has much higher limits than 2.0/experimental
    model = genai.GenerativeModel('gemini-2.5-flash')

    print(f"--- 📝 Script Gen: {topic} ({minutes}m / ~{total_words}w) ---")

    # MODE A: Short Video (Single Shot)
    if minutes <= 15:
        print("   Mode: Single Pass")
        prompt = f"""
        Write a YouTube script for a {minutes}-minute video about '{topic}'.
        Target: ~{total_words} words.
        Format: Just the narrated text. No visual cues.
        Style: Engaging documentary.
        """
        script = generate_with_retry(model, prompt)
        return script if script else f"Welcome to our video about {topic}."

    # MODE B: Long Video (Max 5 Chunks)
    else:
        print("   Mode: Multi-Chunk (Max 5)")
        # Limit to 5 chunks maximum to save API calls
        num_chunks = 5
        words_per_chunk = int(total_words / num_chunks)
        
        full_script = []
        
        for i in range(num_chunks):
            print(f"   Writing Chunk {i+1}/{num_chunks}...")
            
            # Context Management (Fixes IndexError)
            context = ""
            if i > 0 and full_script:
                context = f"Context: The previous part ended with: '...{full_script[-1][-300:]}'."

            prompt = f"""
            Write Part {i+1} of 5 for a documentary script about '{topic}'.
            {context}
            Target Length: ~{words_per_chunk} words.
            IMPORTANT: Write ONLY the narration. Do not repeat the intro. Continue the story seamlessly.
            """
            
            chunk_text = generate_with_retry(model, prompt)
            
            if chunk_text:
                # Clean up markdown
                chunk_text = chunk_text.replace("##", "").replace("**", "")
                full_script.append(chunk_text)
            else:
                print("     Skipping failed chunk.")

        final_text = " ".join(full_script)
        if len(final_text) < 50: return f"Deep dive into {topic}."
        print(f"   ✓ Generated {len(final_text.split())} words.")
        return final_text

# ==========================================
# 5. AUDIO & VISUAL ENGINES
# ==========================================
def clone_voice_safe(text, ref_audio, out_path):
    print("\n--- 🎙️ Audio Engine ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        sentences = nltk.sent_tokenize(text)
        print(f"   Sentences: {len(sentences)}")
        
        tensors = []
        for i, sent in enumerate(sentences):
            if len(sent.strip()) < 2: continue
            if i % 10 == 0: print(f"   Processing {i}/{len(sentences)}...")
            
            try:
                with torch.no_grad():
                    wav = model.generate(sent, str(ref_audio), exaggeration=0.5)
                    tensors.append(wav.cpu())
                    tensors.append(torch.zeros(1, int(24000 * 0.2))) # Pause
            except: pass
            
        if not tensors: return False
        
        print("   Merging Audio...")
        final = torch.cat(tensors, dim=1)
        torchaudio.save(out_path, final, 24000)
        return True
    except Exception as e:
        print(f"TTS Failed: {e}")
        return False

def get_subtitles(audio_path):
    aai.settings.api_key = ASSEMBLY_KEY
    t = aai.Transcriber().transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error: return [], None
    
    ass_path = TEMP_DIR / "style.ass"
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
            start = format_time(s.start)
            end = format_time(s.end)
            clean = s.text.replace("'", "").replace('"', '')
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{clean}\n")
            
    return [{"text": s.text, "start": s.start/1000, "end": s.end/1000} for s in t.get_sentences()], ass_path

def format_time(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms//10:02d}"

def process_visuals(sentences, audio_path, ass_file, final_out):
    print(f"\n--- 🎬 Visual Engine ({len(sentences)} clips) ---")
    BATCH_SIZE = 50
    used_ids = set()
    parts = []

    def download_clip(args):
        i, sent, used = args
        dur = max(3.5, sent['end'] - sent['start'])
        q = get_visual_queries(sent['text'])[0]
        out = TEMP_DIR / f"seg_{i}.mp4"
        
        # Try Pexels
        found_url = None
        if PEXELS_KEYS:
            try:
                h = {"Authorization": random.choice(PEXELS_KEYS)}
                r = requests.get(f"https://api.pexels.com/videos/search?query={q}&orientation=landscape&size=medium", headers=h, timeout=4)
                vids = r.json().get('videos', [])
                for v in vids:
                    if v['id'] not in used:
                        found_url = v['video_files'][0]['link']; used.add(v['id']); break
            except: pass
            
        # Download or Black Screen
        if found_url:
            try:
                raw = TEMP_DIR / f"raw_{i}.mp4"
                with open(raw, "wb") as f: f.write(requests.get(found_url).content)
                subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(dur), "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30", "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if raw.exists(): os.remove(raw)
                return str(out)
            except: pass
            
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur}", "-t", str(dur), "-vf", "fps=30", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)

    for start in range(0, len(sentences), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(sentences))
        print(f"   Batch {start}-{end}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            clips = list(ex.map(download_clip, [(k, sentences[k], used_ids) for k in range(start, end)]))
            
        batch_txt = TEMP_DIR / f"list_{start}.txt"
        batch_vid = TEMP_DIR / f"part_{start}.mp4"
        with open(batch_txt, "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(batch_txt), "-c", "copy", str(batch_vid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for c in clips: 
            if os.path.exists(c): os.remove(c)
        parts.append(str(batch_vid))
        
    # Final Merge
    print("   Merging & Burning...")
    full_vis = TEMP_DIR / "full.mp4"
    list_txt = TEMP_DIR / "full_list.txt"
    with open(list_txt, "w") as f:
        for p in parts: f.write(f"file '{p}'\n")
        
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(full_vis)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    ass_fix = str(Path(ass_file).resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", str(full_vis), "-i", str(audio_path),
        "-vf", f"ass='{ass_fix}'", "-c:v", "libx264", "-preset", "medium", "-b:v", "4500k",
        "-c:a", "aac", "-shortest", str(final_out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 6. MAIN
# ==========================================
print("--- STARTING ---")
ref_voice = TEMP_DIR / "voice.mp3"

# Download Voice
headers = {'Authorization': f'token {os.environ.get("GH_PAT","")}', 'Accept': 'application/vnd.github.v3.raw'}
try:
    r = requests.get(VOICE_URL, headers=headers, allow_redirects=True)
    with open(ref_voice, "wb") as f: f.write(r.content)
except:
    print("Voice DL failed"); exit(1)

# Script
if MODE == "topic":
    text = generate_script(TOPIC, DURATION_MINS)
else:
    text = SCRIPT_TEXT

# Audio
audio_out = TEMP_DIR / "out.wav"
if clone_voice_safe(text, ref_voice, audio_out):
    sents, ass = get_subtitles(audio_out)
    if sents:
        final = OUTPUT_DIR / f"final_video_{JOB_ID}.mp4"
        process_visuals(sents, audio_out, ass, final)
        print(f"--- DONE: {final.name} ---")
    else: print("Subs failed")
else: print("Audio failed")
