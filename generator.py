"""
VIDEO FACTORY V4 - PRODUCTION ENGINE
======================================
- Groq GPT-OSS-120B: sentence-level visual matching (JSON approach)
- Chatterbox Multilingual TTS + Resemble Enhance (44.1kHz studio)
- GPU-accelerated encoding (h264_nvenc on Kaggle T4/P100)
- Ken Burns cinematic effects on clips
- Multiple subtitle presets
- Optimized for Kaggle GPU
"""

import os, subprocess, sys, re, time, random, shutil, json
import concurrent.futures, requests, gc
from pathlib import Path

# ==========================================
# 1. INSTALLATION
# ==========================================
print("--- Installing Dependencies ---")

# Core deps (fast, safe)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "groq", "assemblyai", "google-generativeai", "requests",
    "pydub", "numpy", "pillow", "librosa", "scipy"
])

# TTS engine (needs torch>=2.6 which is already on Kaggle GPU image)
# NOTE: PyPI's chatterbox-tts (0.1.7 as of this writing) does NOT have the
# t3_model= kwarg needed for the V3 multilingual checkpoint - that only
# exists on the GitHub master branch (confirmed by inspecting the actual
# wheel contents). Install from source to get real V3 support.
print("  Installing chatterbox-tts from GitHub (master) for V3 support...")
_cb_installed = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "git+https://github.com/resemble-ai/chatterbox.git"],
    capture_output=True, text=True
)
if _cb_installed.returncode != 0:
    print("  git install failed, falling back to PyPI chatterbox-tts (no V3 t3_model support)")
    print(f"  (reason: {_cb_installed.stderr.strip()[-300:]})")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "chatterbox-tts"
    ])
else:
    print("  chatterbox-tts installed from source (V3-capable)")

# Resemble Enhance - install WITHOUT deps to avoid torch version conflicts
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--no-deps", "resemble-enhance"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "librosa", "scipy", "soundfile"], capture_output=True)

subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, capture_output=True)

import torch, torchaudio
import assemblyai as aai
import google.generativeai as genai

# Check if NVENC is available
def _has_nvenc():
    r = subprocess.run("ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc",
        shell=True, capture_output=True, text=True)
    return "h264_nvenc" in r.stdout

USE_GPU = _has_nvenc()
print(f"  GPU Encoding: {'h264_nvenc' if USE_GPU else 'libx264 (CPU)'}")

def _enc_args():
    """Return encoder args based on GPU availability"""
    if USE_GPU:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]



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
LANGUAGE = """{{LANGUAGE_PLACEHOLDER}}"""

GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEY","").split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS","").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS","").split(",")
GROQ_KEY = os.environ.get("GROQ_API_KEY","")

OUTPUT_DIR = Path("output"); TEMP_DIR = Path("temp")
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True); TEMP_DIR.mkdir(exist_ok=True)

IS_SPANISH = LANGUAGE.lower().strip() in ["spanish","es","espanol"]
USED_URLS = set()
AI_QUERIES = []

# Shorts count by long-video duration: 15min->5, 10min->3, 5min->2
def get_shorts_count(mins):
    if mins >= 13: return 5
    if mins >= 8: return 3
    return 2
SHORTS_COUNT = get_shorts_count(DURATION_MINS)
SHORT_DUR_TARGET = 60  # seconds, target length per short



# ==========================================
# 3. GROQ QUERY ENGINE (Sentence-Matched JSON)
# ==========================================
def generate_queries_for_sentences(sentences):
    """
    Send ALL sentences to Groq, get back a mapping of each sentence to its
    ideal visual search query. Batches into chunks of ~40 sentences per call
    so long scripts (especially Spanish, which tends to run more characters
    per sentence than English) never get silently truncated and dropped -
    that was previously causing queries to desync from sentences.
    """
    if not GROQ_KEY or not sentences:
        return [random.choice(FALLBACK) for _ in sentences]

    n = len(sentences)
    print(f"  Groq: matching {n} sentences to visuals...")

    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    BATCH_SIZE = 40
    all_queries = [None] * n

    for batch_start in range(0, n, BATCH_SIZE):
        batch = sentences[batch_start:batch_start + BATCH_SIZE]
        # Numbered locally within the batch (1..len(batch)) - offset back to
        # global index when parsing, so per-batch parsing stays simple.
        numbered = "\n".join([f"{i+1}. {s['text'][:100]}" for i, s in enumerate(batch)])

        try:
            r = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a professional video editor. For each numbered sentence from a script, you must provide the PERFECT stock footage search query that visually represents what is being said.

RULES:
- Each query must be in ENGLISH (even if script is in another language)
- Each query is 3-6 words describing what a CAMERA would film
- The visual MUST match the sentence content:
  * "Technology is advancing rapidly" -> "futuristic circuit board closeup"
  * "The ocean is vast and mysterious" -> "deep ocean underwater darkness"
  * "Cities are growing faster" -> "aerial cityscape construction cranes"
  * "Ancient civilizations built pyramids" -> "egyptian pyramids aerial sunset"
- NEVER default to generic nature unless the sentence is ABOUT nature
- NO people/faces/bodies, NO religion, NO violence, NO NSFW
- Return ONLY the queries, one per line, numbered to match (same numbers you were given)"""},
                    {"role": "user", "content": f"Match each sentence to a video search query:\n\n{numbered}"}
                ],
                model="openai/gpt-oss-120b",
                max_tokens=2000,
                temperature=0.5
            )

            result = r.choices[0].message.content
            # Parse by EXPLICIT leading index (e.g. "1. some query" -> index 0),
            # not by line order. This prevents misalignment: if Groq skips a
            # number, returns a blank/rejected line, or the model merges two
            # lines, sequential-order parsing would silently shift every
            # subsequent query onto the WRONG sentence. That was the root
            # cause of Spanish (and other long-sentence scripts) queries
            # ending up mismatched with their actual sentence content.
            for line in result.strip().split('\n'):
                m = re.match(r'^\s*(\d+)[\.\)\-]\s*(.+)$', line.strip())
                if not m:
                    continue
                local_idx = int(m.group(1)) - 1
                cleaned = m.group(2).strip().strip('"\'')
                if 3 < len(cleaned) < 60 and _safe(cleaned) and 0 <= local_idx < len(batch):
                    all_queries[batch_start + local_idx] = cleaned

        except Exception as e:
            print(f"  Groq error on batch {batch_start}-{batch_start+len(batch)}: {e}")
            # leave this batch's entries as None -> filled by fallback below

    queries = [q if q else random.choice(FALLBACK) for q in all_queries]

    # Show matching for debug
    for i in range(min(3, len(queries), len(sentences))):
        print(f"    [{i+1}] '{sentences[i]['text'][:35]}...' -> '{queries[i]}'")

    return queries

FALLBACK = [
    "technology data center servers", "futuristic city aerial night",
    "abstract digital particles", "space nebula stars 4k",
    "ocean waves aerial cinematic", "mountain landscape dramatic",
    "circuit board macro electronics", "laboratory scientific research",
    "architecture modern building glass", "sunrise golden hour landscape",
    "underwater coral reef fish", "clouds timelapse dramatic sky",
    "desert sand dunes aerial", "forest aerial cinematic fog",
    "lightning storm dramatic clouds", "volcano lava flow night"
]

def _safe(q):
    bad = ['woman','women','girl','female','bikini','nude','naked','sexy',
           'jesus','christ','church','mosque','temple','bible','buddha',
           'gun','weapon','war','blood','violence','kill','alcohol',
           'drug','gambling','pork','lgbtq','person','people','crowd']
    return not any(t in q.lower() for t in bad)



# ==========================================
# 3B. AI SHORT-CUTTING (Groq oss-120b picks best 1-min hook moments)
# ==========================================
def select_short_segments(sentences, n_shorts, target_seconds=60):
    """
    Ask oss-120b to pick the N most engaging/hook-worthy ~60s windows from
    anywhere in the full sentence list (not sequential chunks - best moments,
    can skip filler and overlap thematically). Returns a list of dicts:
    {"start_idx":int, "end_idx":int, "reason":str} (inclusive sentence indices).
    Falls back to evenly-spaced windows if Groq is unavailable or output is malformed.
    """
    def _fallback_windows():
        # Evenly space fallback windows across the script, sized by duration
        windows = []
        total_dur = sentences[-1]['end'] if sentences else 0
        if total_dur <= 0:
            return windows
        step = total_dur / (n_shorts + 1)
        for k in range(n_shorts):
            center = step * (k + 1)
            lo, hi = center - target_seconds/2, center + target_seconds/2
            s_idx = next((i for i,s in enumerate(sentences) if s['end'] >= lo), 0)
            e_idx = next((i for i,s in enumerate(sentences) if s['start'] >= hi), len(sentences)-1)
            e_idx = max(e_idx, s_idx)
            windows.append({"start_idx": s_idx, "end_idx": e_idx, "reason": "evenly-spaced fallback"})
        return windows

    if not GROQ_KEY or not sentences:
        return _fallback_windows()

    print(f"  Groq: selecting {n_shorts} best ~{target_seconds}s hook moments from {len(sentences)} sentences...")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)

        numbered = "\n".join([
            f"{i} [{s['start']:.1f}s-{s['end']:.1f}s]: {s['text'][:140]}"
            for i, s in enumerate(sentences)
        ])

        r = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""You are an expert short-form video editor (TikTok/Reels/Shorts).
You will be given a numbered list of sentences from a long-form narration script, each with its timestamp.

Your job: pick the {n_shorts} BEST standalone moments to turn into ~{target_seconds}-second short videos.

RULES:
- Pick moments ANYWHERE in the script - do NOT just chunk it sequentially. Prioritize hooks, surprising facts, emotional peaks, cliffhangers, or strong claims - whatever would make someone stop scrolling.
- Each selected window's total duration (end timestamp - start timestamp) should be close to {target_seconds} seconds (between {int(target_seconds*0.7)}s and {int(target_seconds*1.3)}s).
- Each window must be a CONTINUOUS range of sentence indices (start_idx to end_idx inclusive) - do not skip sentences within a single short.
- Windows for different shorts MAY overlap or reuse similar sentences if that content is strong enough to work twice, but prefer variety across the {n_shorts} shorts when possible.
- Each window must make sense on its own without needing earlier context (avoid starting mid-thought on a pronoun like "it" or "this" referring to something far earlier).
- Return ONLY a JSON array, nothing else. No markdown, no explanation outside the JSON.

Format exactly like this:
[{{"start_idx": 4, "end_idx": 9, "reason": "strong opening claim + payoff"}}, {{"start_idx": 22, "end_idx": 27, "reason": "surprising statistic"}}]"""},
                {"role": "user", "content": f"Sentences:\n\n{numbered[:12000]}\n\nPick the {n_shorts} best windows as JSON."}
            ],
            model="openai/gpt-oss-120b",
            max_tokens=1500,
            temperature=0.6
        )

        raw = r.choices[0].message.content.strip()
        # Strip potential markdown fences
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            print("  Groq short-selection: no JSON array found, using fallback windows")
            return _fallback_windows()

        picks = json.loads(match.group(0))
        windows = []
        for p in picks:
            s_idx = int(p.get("start_idx", -1))
            e_idx = int(p.get("end_idx", -1))
            if 0 <= s_idx <= e_idx < len(sentences):
                windows.append({"start_idx": s_idx, "end_idx": e_idx, "reason": p.get("reason","")})

        if not windows:
            print("  Groq short-selection: parsed JSON had no valid windows, using fallback")
            return _fallback_windows()

        for w in windows[:n_shorts]:
            dur = sentences[w['end_idx']]['end'] - sentences[w['start_idx']]['start']
            print(f"    Short: sentences {w['start_idx']}-{w['end_idx']} (~{dur:.0f}s) - {w['reason'][:50]}")

        while len(windows) < n_shorts:
            windows.append(random.choice(_fallback_windows() or [{"start_idx":0,"end_idx":min(5,len(sentences)-1),"reason":"padding fallback"}]))

        return windows[:n_shorts]

    except Exception as e:
        print(f"  Groq short-selection error: {e}")
        return _fallback_windows()



# ==========================================
# 4. SUBTITLE PRESETS (Variety)
# ==========================================
SUBTITLE_STYLES = {
    "cinema": {"name":"Cinema","font":"Arial","size":62,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&H80000000",
        "border":3,"outline":0,"shadow":0,"margin":45,"spacing":0.5},
    "modern_bold": {"name":"Modern Bold","font":"Arial Black","size":68,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00111111","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":50,"spacing":1.2},
    "neon_yellow": {"name":"Neon Yellow","font":"Arial Black","size":70,"bold":-1,
        "primary":"&H0000FFFF","outline_c":"&H00000044","back":"&H00000000",
        "border":1,"outline":5,"shadow":3,"margin":52,"spacing":1.5},
    "soft_white": {"name":"Soft White","font":"Arial","size":60,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00333333","back":"&H00000000",
        "border":1,"outline":3,"shadow":4,"margin":45,"spacing":0.8},
    "electric_cyan": {"name":"Electric Cyan","font":"Arial Black","size":66,"bold":-1,
        "primary":"&H00FFFF00","outline_c":"&H00663300","back":"&H00000000",
        "border":1,"outline":4,"shadow":3,"margin":48,"spacing":1},
}

# Short-specific vertical styles (1080x1920 canvas). Bigger fonts than
# landscape presets since viewers are closer to phone screens, and a much
# larger MarginV to clear TikTok/Reels/YouTube Shorts UI (caption, like/
# share buttons, progress bar) which occupies the bottom ~20-25% of frame.
SHORT_SUBTITLE_STYLES = {
    "short_punch": {"name":"Short Punch","font":"Arial Black","size":78,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&H00000000",
        "border":1,"outline":6,"shadow":3,"margin":480,"spacing":1},
    "short_neon": {"name":"Short Neon","font":"Arial Black","size":82,"bold":-1,
        "primary":"&H0000FFFF","outline_c":"&H00330033","back":"&H00000000",
        "border":1,"outline":6,"shadow":4,"margin":480,"spacing":1.2},
    "short_clean": {"name":"Short Clean","font":"Arial","size":74,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00111111","back":"&H80000000",
        "border":3,"outline":0,"shadow":0,"margin":460,"spacing":0.6},
}

def create_subtitles(sentences, ass_path, word_data=None, style_key=None,
                      style_set=None, play_res=(1920,1080), max_chars=46):
    """
    Word-level highlighted subtitles (like Submagic/Captions.ai).
    If word_data is provided, each word lights up as it's spoken.
    Falls back to sentence-level if no word data.

    style_set: dict of style presets to choose from (defaults to landscape SUBTITLE_STYLES)
    play_res: (width, height) of the ASS canvas - must match final video resolution
    max_chars: character budget per 2-line chunk - tune per play_res/font size
    """
    style_set = style_set or SUBTITLE_STYLES
    key = style_key or random.choice(list(style_set.keys()))
    s = style_set[key]
    print(f"  Subtitle: {s['name']} {'(word-highlight)' if word_data else '(sentence)'} @ {play_res[0]}x{play_res[1]}")
    
    # Highlight color (the word currently being spoken)
    highlight = "&H0000FFFF"  # Yellow highlight
    if "cyan" in key: highlight = "&H0000FF00"  # Green for cyan style
    if "yellow" in key or "neon" in key: highlight = "&H00FFFFFF"  # White for yellow/neon styles
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {play_res[0]}\nPlayResY: {play_res[1]}\n")
        f.write("WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{s['font']},{s['size']},{s['primary']},&H00FFFFFF,"
                f"{s['outline_c']},{s['back']},{s['bold']},0,0,0,100,100,"
                f"{s['spacing']},0,{s['border']},{s['outline']},{s['shadow']},"
                f"2,50,50,{s['margin']},1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        if word_data:
            # Word-level highlighting: group words into display chunks by
            # character budget (not raw word count), so a chunk of short
            # words can hold more words than a chunk of long words.
            MAX_CHARS_PER_CHUNK = max_chars   # total chars across both lines, tune per font/size/canvas
            MAX_WORDS_PER_CHUNK = 14   # hard ceiling so very short words can't run away

            chunks = []
            buf, blen = [], 0
            for word in word_data:
                wlen = len(word['text']) + 1  # +1 for the joining space
                if buf and (blen + wlen > MAX_CHARS_PER_CHUNK or len(buf) >= MAX_WORDS_PER_CHUNK):
                    chunks.append(buf)
                    buf, blen = [], 0
                buf.append(word)
                blen += wlen
            if buf:
                chunks.append(buf)

            for c_idx, chunk_words in enumerate(chunks):
                if not chunk_words: continue

                # Balance the 2-line split by character length, not word count,
                # so line 1 and line 2 come out visually similar in width.
                total_chars = sum(len(w['text']) for w in chunk_words)
                running, split_idx = 0, len(chunk_words) - 1
                for j, cw in enumerate(chunk_words):
                    running += len(cw['text'])
                    if running >= total_chars / 2:
                        split_idx = j
                        break

                # For each word in chunk, create a dialogue line where THAT word is highlighted.
                # IMPORTANT: extend each word's display End to the START of the next word
                # (or, for the last word in the chunk, to the first word of the NEXT chunk
                # if one exists - otherwise its own end). Word-level ASR timestamps often
                # have small gaps between word['end'] and the next word['start']
                # (pauses/breaths) - using word['end'] directly as the Dialogue End causes
                # the subtitle to go blank during those gaps, both within and between chunks.
                is_last_chunk = (c_idx == len(chunks) - 1)
                next_chunk_start = chunks[c_idx+1][0]['start'] if not is_last_chunk else chunk_words[-1]['end']
                for w_idx, word in enumerate(chunk_words):
                    w_start = _fmt(word['start'])
                    if w_idx + 1 < len(chunk_words):
                        next_start = chunk_words[w_idx+1]['start']
                    else:
                        next_start = next_chunk_start
                    w_end = _fmt(next_start)

                    p1, p2 = [], []
                    for j, cw in enumerate(chunk_words):
                        txt = f"{{\\c{highlight}\\fscx115\\fscy115}}{cw['text']}{{\\r}}" if j == w_idx else cw['text']
                        if j <= split_idx: p1.append(txt)
                        else: p2.append(txt)

                    line = ' '.join(p1) + "\\N" + ' '.join(p2) if p2 else ' '.join(p1)
                    f.write(f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n")
        else:
            # Sentence-level fallback - split into 2 lines balanced by character length.
            # Extend each sentence's End to the next sentence's Start to avoid blank gaps.
            for idx, sent in enumerate(sentences):
                t1 = _fmt(sent['start'])
                next_start = sentences[idx+1]['start'] if idx+1 < len(sentences) else sent['end']
                t2 = _fmt(next_start)
                txt = sent['text'].strip().rstrip('.,;:')
                w = txt.split()
                if len(w) > 3:
                    total_chars = sum(len(x) for x in w)
                    running, split_idx = 0, len(w) - 1
                    for j, word in enumerate(w):
                        running += len(word)
                        if running >= total_chars / 2:
                            split_idx = j
                            break
                    txt = ' '.join(w[:split_idx+1]) + "\\N" + ' '.join(w[split_idx+1:])
                f.write(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{txt}\n")

def search_and_download_vertical(query, idx, duration, tag=""):
    """
    Same as search_and_download but requests portrait/vertical source video
    where possible and always crops/scales to 1080x1920 (9:16) for Shorts.
    Uses a distinct USED_URLS-safe idx namespace via `tag` so long-video and
    shorts clip fetching never collide on temp filenames.
    """
    urls = []
    page = random.randint(1,3)

    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization":key},
                params={"query":query,"per_page":15,"page":page,"orientation":"portrait"}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('videos',[]):
                    files = v.get('video_files',[])
                    # Prefer genuinely portrait files (height > width), else take any hd/large
                    portrait = [f for f in files if f.get('height',0) > f.get('width',0) and f.get('height',0)>=1280]
                    hd = portrait or [f for f in files if f.get('quality') in ['hd','large']]
                    if hd:
                        url = random.choice(hd)['link']
                        if url not in USED_URLS: urls.append(url)
        except: pass

    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key":key,"q":query,"per_page":15,"page":page}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('hits',[]):
                    vd = v.get('videos',{})
                    url = vd.get('large',vd.get('medium',{})).get('url')
                    if url and url not in USED_URLS: urls.append(url)
        except: pass

    for url in urls[:3]:
        try:
            raw = TEMP_DIR / f"raw_s{tag}_{idx}.mp4"
            out = TEMP_DIR / f"clip_s{tag}_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(raw) < 5000: continue

            # Force crop/scale to 1080x1920 regardless of source orientation
            vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30"
            cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",str(raw),"-t",str(duration),
                   "-vf",vf] + _enc_args() + ["-an",str(out)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)

            try: os.remove(raw)
            except: pass
            if os.path.exists(out) and os.path.getsize(out) > 2000:
                USED_URLS.add(url); return str(out)
        except: continue
    return None


def process_short_clip(args):
    i, sent, tag = args
    dur = max(2.5, sent['end'] - sent['start'])
    query = AI_QUERIES[sent.get('orig_idx', i)] if sent.get('orig_idx', i) < len(AI_QUERIES) else random.choice(FALLBACK)

    for _ in range(3):
        clip = search_and_download_vertical(query, i, dur, tag=tag)
        if clip: return (i, clip)
        query = random.choice(FALLBACK)
        time.sleep(0.3)
    return (i, None)


def render_short(short_idx, sentences_slice, audio_path, ass_path, logo_path, out_path):
    """
    Render a single 1080x1920 short: fetch fresh vertical stock clips for
    this segment's sentences, concat, overlay a LARGE left-side logo (shorts
    need bigger branding since screen real estate is smaller/closer-viewed),
    and burn vertical-tuned subtitles.
    """
    tag = f"sh{short_idx}"
    n = len(sentences_slice)
    print(f"\n  Short {short_idx+1}: fetching {n} vertical clips...")

    clips = [None]*n
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(process_short_clip, (i,s,tag)): i for i,s in enumerate(sentences_slice)}
        for f in concurrent.futures.as_completed(futs):
            try:
                idx, path = f.result()
                if path: clips[idx]=path
            except: pass

    valid = [i for i,c in enumerate(clips) if c and os.path.exists(c)]
    if not valid:
        print(f"  Short {short_idx+1}: no clips found, skipping")
        return False
    for i in range(n):
        if clips[i] and os.path.exists(clips[i]): continue
        nearest = min(valid, key=lambda x:abs(x-i))
        dur = max(2.5, sentences_slice[i]['end']-sentences_slice[i]['start'])
        gap = TEMP_DIR / f"gap_{tag}_{i}.mp4"
        subprocess.run(["ffmpeg","-y","-stream_loop","-1","-i",clips[nearest],
            "-t",str(dur)] + _enc_args() + ["-an",str(gap)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips[i] = str(gap) if os.path.exists(gap) else clips[nearest]

    list_path = f"list_{tag}.txt"
    visual_path = f"visual_{tag}.mp4"
    with open(list_path,"w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {visual_path}",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists(visual_path):
        subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {list_path}"
            f" -c:v libx264 -preset ultrafast -crf 18 {visual_path}",
            shell=True, capture_output=True)
    if not os.path.exists(visual_path): return False

    enc = _enc_args()
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        # Logo on left, LARGE (shorts are watched up close on phones - a
        # small landscape-style logo reads as invisible on a 1080x1920 frame).
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];"
                f"[1:v]scale=280:-1[l];[bg][l]overlay=30:40[wl];"
                f"[wl]subtitles='{ass_esc}'[v]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",visual_path,"-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    else:
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",visual_path,"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","1:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    subprocess.run(cmd, capture_output=True, timeout=300)

    for p in [list_path, visual_path]:
        if os.path.exists(p): os.remove(p)

    if os.path.exists(out_path):
        print(f"  Short {short_idx+1}: {os.path.getsize(out_path)/(1024**2):.0f}MB")
        return True
    return False


def extract_short_audio(full_audio_path, start_sec, end_sec, out_path, fade=0.25):
    """
    Cut a clean segment directly from the full enhanced audio (fast, exact
    same voice/quality - no re-TTS). Applies short fade in/out so the cut
    doesn't click/pop at the boundaries.
    """
    dur = max(0.5, end_sec - start_sec)
    af = f"afade=t=in:st=0:d={fade},afade=t=out:st={max(0,dur-fade):.2f}:d={fade}"
    cmd = ["ffmpeg","-y","-i",str(full_audio_path),
           "-ss",f"{start_sec:.3f}","-t",f"{dur:.3f}",
           "-af",af,"-ar","44100",str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1000



# ==========================================
# 5. AUDIO ENGINE (TTS + Resemble Enhance)
# ==========================================
def generate_audio(text, ref_audio, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_path = TEMP_DIR / "raw_tts.wav"
    
    print(f"  TTS: {'Spanish (V3)' if IS_SPANISH else 'English'} on {device}")
    try:
        if IS_SPANISH:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            try:
                # Newer chatterbox versions support t3_model="v3" (better quality).
                model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            except TypeError:
                # Installed pip version (e.g. 0.1.7) doesn't have this kwarg yet
                # (only on GitHub main / HF docs as of this writing). Fall back.
                print("  chatterbox-tts: t3_model kwarg unsupported by installed version, using default checkpoint")
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        
        sr = model.sr
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Group 2-3 sentences per chunk
        chunks, buf, blen = [], [], 0
        for s in sents:
            if blen + len(s) > 160 and buf:
                chunks.append(' '.join(buf)); buf, blen = [s], len(s)
            else: buf.append(s); blen += len(s)+1
        if buf: chunks.append(' '.join(buf))
        
        print(f"  {len(sents)} sentences -> {len(chunks)} chunks")
        wavs = []
        for i, c in enumerate(chunks):
            if i%5==0: update_status(18+int((i/len(chunks))*27), f"TTS {i+1}/{len(chunks)}")
            try:
                with torch.no_grad():
                    if IS_SPANISH:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio),
                                            language_id="es", exaggeration=0.4, cfg_weight=0.65)
                    else:
                        w = model.generate(c.replace('"',''), audio_prompt_path=str(ref_audio), exaggeration=0.4, cfg_weight=0.65)
                    wavs.append(w.cpu())
                if i%8==0: torch.cuda.empty_cache()
            except: continue
        
        if not wavs: return False
        full = wavs[0]
        for w in wavs[1:]:
            full = torch.cat([full, torch.zeros((full.shape[0], int(0.15*sr))), w], dim=1)
        full = torch.cat([full, torch.zeros((full.shape[0], int(1.5*sr)))], dim=1)
        torchaudio.save(str(raw_path), full, sr)
        print(f"  TTS: {full.shape[1]/sr:.1f}s at {sr}Hz")
        del model, wavs, full; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"  TTS error: {e}"); return False
    
    # --- RESEMBLE ENHANCE ---
    print("  Enhancing audio...")
    try:
        from unittest.mock import MagicMock
        # Mock every deepspeed submodule that resemble_enhance imports
        mock_names = [
            'deepspeed', 'deepspeed.accelerator', 'deepspeed.runtime',
            'deepspeed.runtime.engine', 'deepspeed.runtime.config',
            'deepspeed.runtime.utils', 'deepspeed.utils',
            'deepspeed.ops', 'deepspeed.ops.adam', 'deepspeed.comm',
        ]
        for name in mock_names:
            sys.modules[name] = MagicMock()
        sys.modules['deepspeed.accelerator'].get_accelerator = MagicMock()
        sys.modules['deepspeed.runtime.engine'].DeepSpeedEngine = MagicMock()
        sys.modules['deepspeed.runtime.utils'].clip_grad_norm_ = MagicMock()
        
        from resemble_enhance.enhancer.inference import enhance as re_enhance
        dwav, osr = torchaudio.load(str(raw_path))
        if dwav.shape[0] > 1:
            dwav = dwav.mean(dim=0, keepdim=True)
        
        chunk_s = 20 * osr; parts = []; esr = 44100
        total = dwav.shape[1]
        n_chunks = (total + chunk_s - 1) // chunk_s
        print(f"  Processing {n_chunks} chunks...")
        
        for i in range(0, total, chunk_s):
            chunk = dwav[:, i:i+chunk_s]
            try:
                hw, esr = re_enhance(dwav=chunk.squeeze(0), sr=osr, device=device, lambd=0.6)
                parts.append(hw.cpu().unsqueeze(0))
                print(f"    Chunk {i//chunk_s+1}/{n_chunks}: OK ({esr}Hz)")
            except Exception as e:
                print(f"    Chunk {i//chunk_s+1}/{n_chunks}: fallback ({str(e)[:40]})")
                parts.append(torchaudio.transforms.Resample(osr, 44100)(chunk).cpu())
                esr = 44100
            torch.cuda.empty_cache()
        
        final = torch.cat(parts, dim=1)
        torchaudio.save(str(out_path), final, esr)
        print(f"  Enhanced: {esr}Hz, {final.shape[1]/esr:.1f}s")
        del parts, final, dwav; torch.cuda.empty_cache(); gc.collect()
        return True
    except Exception as e:
        print(f"  Enhance failed: {e}, using raw audio")
        shutil.copy2(str(raw_path), str(out_path))
        return True



# ==========================================
# 6. VIDEO ENGINE (GPU-Accelerated)
# ==========================================
def search_and_download(query, idx, duration):
    """Search + download + encode with GPU"""
    urls = []
    page = random.randint(1,3)
    
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization":key},
                params={"query":query,"per_page":15,"page":page,"orientation":"landscape"}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('videos',[]):
                    files = v.get('video_files',[])
                    hd = [f for f in files if f.get('quality')=='hd' and f.get('width',0)>=1280]
                    if not hd: hd = [f for f in files if f.get('quality') in ['hd','large']]
                    if hd:
                        url = random.choice(hd)['link']
                        if url not in USED_URLS: urls.append(url)
        except: pass
    
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key":key,"q":query,"per_page":15,"page":page}, timeout=12)
            if r.status_code == 200:
                for v in r.json().get('hits',[]):
                    vd = v.get('videos',{})
                    url = vd.get('large',vd.get('medium',{})).get('url')
                    if url and url not in USED_URLS: urls.append(url)
        except: pass
    
    for url in urls[:3]:
        try:
            raw = TEMP_DIR / f"raw_{idx}.mp4"
            out = TEMP_DIR / f"clip_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(raw) < 5000: continue
            
            # Encode with GPU (h264_nvenc) or CPU fallback
            vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30"
            cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",str(raw),"-t",str(duration),
                   "-vf",vf] + _enc_args() + ["-an",str(out)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            
            try: os.remove(raw)
            except: pass
            if os.path.exists(out) and os.path.getsize(out) > 2000:
                USED_URLS.add(url); return str(out)
        except: continue
    return None

def process_clip(args):
    i, sent, total = args
    dur = max(3.5, sent['end'] - sent['start'])
    query = AI_QUERIES[i] if i < len(AI_QUERIES) else random.choice(FALLBACK)
    
    for _ in range(3):
        clip = search_and_download(query, i, dur)
        if clip: return (i, clip)
        query = random.choice(FALLBACK)
        time.sleep(0.3)
    return (i, None)



# ==========================================
# 7. RENDER ENGINE (GPU-Accelerated)
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_nosub, out_sub):
    n = len(sentences)
    print(f"\n  Rendering {n} clips (5 workers)...")
    
    clips = [None]*n
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(process_clip, (i,s,n)): i for i,s in enumerate(sentences)}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            try:
                idx, path = f.result()
                if path: clips[idx]=path; done+=1
                update_status(55+int((done/n)*22), f"Clips {done}/{n}")
            except: pass
    
    # Fill gaps
    valid = [i for i,c in enumerate(clips) if c and os.path.exists(c)]
    if not valid: return False
    for i in range(n):
        if clips[i] and os.path.exists(clips[i]): continue
        nearest = min(valid, key=lambda x:abs(x-i))
        dur = max(3.5, sentences[i]['end']-sentences[i]['start'])
        gap = TEMP_DIR / f"gap_{i}.mp4"
        subprocess.run(["ffmpeg","-y","-stream_loop","-1","-i",clips[nearest],
            "-t",str(dur)] + _enc_args() + ["-an",str(gap)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips[i] = str(gap) if os.path.exists(gap) else clips[nearest]
    
    # Concat (stream copy)
    print("  Concatenating...")
    with open("list.txt","w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists("visual.mp4"):
        subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt" + 
            " -c:v libx264 -preset ultrafast -crf 18 visual.mp4",
            shell=True, capture_output=True)
    if not os.path.exists("visual.mp4"): return False
    
    # V1: No subs (900p, GPU)
    update_status(80, "Rendering V1 (900p)...")
    enc = _enc_args()
    if logo_path and os.path.exists(logo_path):
        filt = "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=160:-1[l];[bg][l]overlay=20:20[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a"] + enc + ["-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    else:
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(audio_path),
            "-vf","scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2"
            ] + enc + ["-c:a","aac","-b:a","128k","-shortest",str(out_nosub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if not os.path.exists(out_nosub): return False
    print(f"  V1: {os.path.getsize(out_nosub)/(1024**2):.0f}MB")
    
    # V2: With subs (1080p, GPU)
    update_status(88, "Rendering V2 (1080p + subs)...")
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];[wl]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","2:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    else:
        filt = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v]"
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","1:a"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if os.path.exists(out_sub): print(f"  V2: {os.path.getsize(out_sub)/(1024**2):.0f}MB")
    return True



# ==========================================
# 8. UTILITIES
# ==========================================
LOG_BUF = []
def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")
    LOG_BUF.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(LOG_BUF) > 30: LOG_BUF.pop(0)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return
    import base64
    data = {"progress":progress,"message":message,"status":status,
            "logs":"\n".join(LOG_BUF),"timestamp":time.time()}
    if file_url: data["file_io_url"] = file_url
    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    headers = {"Authorization":f"token {token}","Accept":"application/vnd.github.v3+json"}
    try:
        gr = requests.get(url, headers=headers)
        sha = gr.json().get("sha") if gr.status_code==200 else None
        payload = {"message":f"s{progress}","content":base64.b64encode(json.dumps(data).encode()).decode(),"branch":"main"}
        if sha: payload["sha"]=sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization":f"token {token}","Accept":"application/vnd.github.v3.raw"})
        if r.status_code==200:
            with open(local,"wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def upload_drive(fp):
    if not os.path.exists(fp): return None
    print(f"  Uploading {os.path.basename(fp)}...")
    cid,cs,rt,fid = (os.environ.get(k) for k in ["OAUTH_CLIENT_ID","OAUTH_CLIENT_SECRET","OAUTH_REFRESH_TOKEN","GOOGLE_DRIVE_FOLDER_ID"])
    if not all([cid,cs,rt]): return None
    try: tok=requests.post("https://oauth2.googleapis.com/token",data={"client_id":cid,"client_secret":cs,"refresh_token":rt,"grant_type":"refresh_token"}).json()['access_token']
    except: return None
    sz=os.path.getsize(fp); meta={"name":os.path.basename(fp),"mimeType":"video/mp4"}
    if fid: meta["parents"]=[fid]
    resp=requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json","X-Upload-Content-Type":"video/mp4","X-Upload-Content-Length":str(sz)},json=meta)
    if resp.status_code!=200: return None
    with open(fp,"rb") as f: ur=requests.put(resp.headers["Location"],headers={"Content-Length":str(sz)},data=f)
    if ur.status_code in [200,201]:
        fid2=ur.json().get('id')
        requests.post(f"https://www.googleapis.com/drive/v3/files/{fid2}/permissions",
            headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},json={'role':'reader','type':'anyone'})
        link=f"https://drive.google.com/file/d/{fid2}/view?usp=sharing"; print(f"  -> {link}"); return link
    return None

def generate_script(topic, mins):
    words=int(mins*180); lang="Write in Spanish." if IS_SPANISH else "Write in English."
    prompt=f"Write a documentary narration about '{topic}'. {words} words.\n{lang}\nRules: Only narration. No brackets. Islamic guidelines. Family-friendly."
    random.shuffle(GEMINI_KEYS)
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            r=genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt)
            return re.sub(r'\[.*?\]','',r.text.replace("*","").replace("#","").strip())
        except: continue
    return ""



# ==========================================
# 9. MAIN EXECUTION
# ==========================================
print(f"\n{'='*50}")
print(f"  VIDEO FACTORY V4")
print(f"  Lang: {'ES' if IS_SPANISH else 'EN'} | GPU: {USE_GPU}")
print(f"{'='*50}\n")

update_status(1, "Starting...")
voice = TEMP_DIR/"voice.mp3"; logo = TEMP_DIR/"logo.png"
if not download_asset(VOICE_PATH, voice):
    update_status(0,"Voice failed","failed"); exit(1)
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, logo)
    logo = str(logo) if os.path.exists(logo) else None
else: logo = None

# Script
update_status(5, "Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE=="topic" else SCRIPT_TEXT
if len(text)<50: update_status(0,"Script too short","failed"); exit(1)
print(f"  {len(text.split())} words")

# Audio
update_status(12, "Audio...")
audio = TEMP_DIR/"audio.wav"
if not generate_audio(text, voice, audio):
    update_status(0,"Audio failed","failed"); exit(1)

# Transcribe
update_status(48, "Transcribing...")
sentences = []
word_data = []  # Word-level timestamps for highlighting
if ASSEMBLY_KEY:
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        tx = aai.Transcriber().transcribe(str(audio))
        for s in tx.get_sentences():
            sentences.append({"text":s.text,"start":s.start/1000,"end":s.end/1000})
        if sentences: sentences[-1]['end']+=0.5
        
        # Get word-level timestamps for subtitle highlighting
        for word in tx.words:
            word_data.append({"text": word.text, "start": word.start/1000, "end": word.end/1000})
        print(f"  Got {len(word_data)} word timestamps for highlighting")
    except Exception as e: print(f"  Transcribe err: {e}")

if not sentences:
    import wave
    try:
        with wave.open(str(audio),'rb') as w: dur=w.getnframes()/float(w.getframerate())
    except: dur=len(text.split())/2.5
    wps=len(text.split())/dur if dur>0 else 2.5; t=0
    for i in range(0,len(text.split()),8):
        chunk=text.split()[i:i+8]; d=len(chunk)/wps
        sentences.append({"text":' '.join(chunk),"start":t,"end":t+d}); t+=d

# Queries (sentence-matched via Groq JSON approach)
update_status(50, "Matching visuals to sentences...")
AI_QUERIES = generate_queries_for_sentences(sentences)

# Subtitles (word-level highlighting if available)
update_status(52, "Subtitles...")
ass = TEMP_DIR/"subs.ass"
create_subtitles(sentences, ass, word_data=word_data if word_data else None)

# Render
update_status(54, "Processing video...")
o1 = OUTPUT_DIR/f"final_{JOB_ID}_NO_SUBS.mp4"
o2 = OUTPUT_DIR/f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o1, o2):
    update_status(93, "Uploading...")
    l1 = upload_drive(o1); l2 = upload_drive(o2)
    msg = "Done!\n"
    if l1: msg += f"No Subs: {l1}\n"
    if l2: msg += f"With Subs: {l2}\n"

    # ==========================================
    # SHORTS PIPELINE
    # ==========================================
    # Only worth generating shorts from audio that's actually long enough
    # for the requested count (avoid generating garbage 60s shorts from a
    # 90s total-runtime video, etc.)
    if sentences and sentences[-1]['end'] >= SHORT_DUR_TARGET * 0.6:
        update_status(95, f"Generating {SHORTS_COUNT} shorts...")
        try:
            windows = select_short_segments(sentences, SHORTS_COUNT, SHORT_DUR_TARGET)
            short_links = []
            for si, win in enumerate(windows):
                update_status(95, f"Short {si+1}/{len(windows)}...")
                s_idx, e_idx = win['start_idx'], win['end_idx']
                seg_sentences = sentences[s_idx:e_idx+1]
                if not seg_sentences:
                    continue

                t_start = seg_sentences[0]['start']
                t_end = seg_sentences[-1]['end']

                # Re-base timestamps to 0 for this short, keep original index
                # for AI_QUERIES lookups (queries were generated per full-script sentence).
                rebased = []
                for orig_i, s in enumerate(seg_sentences, start=s_idx):
                    rebased.append({
                        "text": s['text'],
                        "start": s['start'] - t_start,
                        "end": s['end'] - t_start,
                        "orig_idx": orig_i,
                    })

                short_audio = TEMP_DIR / f"short_{si}_audio.wav"
                if not extract_short_audio(audio, t_start, t_end, short_audio):
                    print(f"  Short {si+1}: audio extraction failed, skipping")
                    continue

                # Vertical-tuned word data (re-based) for this short's subtitle window
                short_word_data = None
                if word_data:
                    ww = [w for w in word_data if t_start <= w['start'] < t_end]
                    if ww:
                        short_word_data = [
                            {"text": w['text'], "start": w['start']-t_start, "end": w['end']-t_start}
                            for w in ww
                        ]

                short_ass = TEMP_DIR / f"short_{si}_subs.ass"
                create_subtitles(
                    rebased, short_ass,
                    word_data=short_word_data,
                    style_set=SHORT_SUBTITLE_STYLES,
                    play_res=(1080, 1920),
                    max_chars=26,   # narrower canvas than landscape -> tighter budget
                )

                short_out = OUTPUT_DIR / f"short_{JOB_ID}_{si+1}.mp4"
                if render_short(si, rebased, short_audio, short_ass, logo, short_out):
                    link = upload_drive(short_out)
                    if link:
                        short_links.append(link)
                        msg += f"Short {si+1}: {link}\n"
        except Exception as e:
            print(f"  Shorts pipeline error: {e}")

    update_status(100, msg, "completed", l1 or l2)
    print(f"\n  {msg}")
else:
    update_status(0, "Render failed", "failed")

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4","list.txt"]:
    if os.path.exists(f): os.remove(f)
import glob
for f in glob.glob("list_sh*.txt") + glob.glob("visual_sh*.mp4"):
    try: os.remove(f)
    except: pass
print("--- DONE ---")
