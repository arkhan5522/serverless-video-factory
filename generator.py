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
# Force explicit reseed with high-entropy source. Kaggle kernels can
# sometimes start with a low-entropy or reused random state between runs,
# which was causing subtitle style selection to always pick the same
# option instead of shuffling. os.urandom pulls from the OS entropy pool
# directly, bypassing whatever default seeding Python did on interpreter start.
random.seed(int.from_bytes(os.urandom(8), "big") ^ int(time.time() * 1000))
import concurrent.futures, requests, gc, threading
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
# SmolVLM2 - local vision-language model for clip visual verification.
# Replaces the earlier Groq-based vision check, which hit constant rate
# limits at scale (100+ clips per video, each needing a verification call).
# This runs locally on the Kaggle GPU with zero API limits. Query
# GENERATION still uses Groq (openai/gpt-oss-120b) unchanged - that part
# works well and isn't being replaced, only the per-clip visual check.
# NOTE: "av" (PyAV) is REQUIRED for SmolVLM2's video-loading path - without
# it, every single verification call fails at frame-decode time and
# silently falls back to accepting the clip unverified. This previously
# went undetected for an entire run since the failure message looked like
# a generic skip rather than a missing-dependency error.
_vision_deps = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "transformers>=4.50.0", "einops", "pyvips-binary", "num2words", "av"],
    capture_output=True, text=True)
if _vision_deps.returncode != 0:
    print(f"  WARNING: vision verification dependencies failed to install - "
          f"clip verification will not work this run: {_vision_deps.stderr[-300:]}")
else:
    try:
        import av as _av_check
        print(f"  Vision verification deps OK (PyAV {_av_check.__version__})")
    except ImportError as e:
        print(f"  WARNING: PyAV still not importable after install ({e}) - "
              f"verification will silently fail open for every clip this run")

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
AI_BACKUPS = []

# Shorts count by long-video duration:
#   5 min  -> 2 shorts
#   10 min -> 3 shorts
#   15 min -> 5 shorts
#   15min+ -> capped at 5 (confirmed intentional - does not keep scaling up)
def get_shorts_count(mins):
    if mins >= 13: return 5   # covers 15min and anything above, capped here
    if mins >= 8: return 3    # covers 10min
    return 2                  # covers 5min (and anything shorter)
SHORTS_COUNT = get_shorts_count(DURATION_MINS)
SHORT_DUR_TARGET = 60  # seconds, target length per short



# ==========================================
# 3. GROQ QUERY ENGINE (Sentence-Matched JSON)
# ==========================================
def generate_queries_for_sentences(sentences):
    """
    Send ALL sentences to Groq, get back a mapping of each sentence to its
    ideal visual search query PLUS a topic-relevant backup query. Batches
    into chunks of ~40 sentences per call so long scripts (especially
    Spanish, which tends to run more characters per sentence than English)
    never get silently truncated and dropped - that was previously causing
    queries to desync from sentences.

    Returns (queries, backups) - two parallel lists. `backups` gives a
    still-on-topic alternate search term to try if the primary query
    returns no usable stock footage, instead of falling back to the
    generic FALLBACK list (which was the direct cause of unrelated
    footage like ocean/waterfall clips appearing in e.g. medical topics).
    """
    if not GROQ_KEY or not sentences:
        qs = [random.choice(FALLBACK) for _ in sentences]
        return qs, list(qs)

    n = len(sentences)
    print(f"  Groq: matching {n} sentences to visuals...")

    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    BATCH_SIZE = 40
    all_queries = [None] * n
    all_backups = [None] * n

    for batch_start in range(0, n, BATCH_SIZE):
        batch = sentences[batch_start:batch_start + BATCH_SIZE]
        # Numbered locally within the batch (1..len(batch)) - offset back to
        # global index when parsing, so per-batch parsing stays simple.
        numbered = "\n".join([f"{i+1}. {s['text'][:100]}" for i, s in enumerate(batch)])

        try:
            r = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a professional video editor working with STOCK FOOTAGE ONLY (Pexels/Pixabay libraries) - you cannot commission custom footage, so you must pick queries that actually EXIST in stock libraries.

For each numbered sentence from a script, provide the best STOCK-FINDABLE search query that visually represents what is being said.

RULES:
- Each query must be in ENGLISH (even if script is in another language)
- Each query is 3-6 words describing what a CAMERA would film
- The visual MUST match the sentence's MEANING/TOPIC, not just literal keywords:
  * "Technology is advancing rapidly" -> "futuristic circuit board closeup"
  * "The ocean is vast and mysterious" -> "deep ocean underwater darkness"
  * "Cities are growing faster" -> "aerial cityscape construction cranes"
  * "Ancient civilizations built pyramids" -> "egyptian pyramids aerial sunset"
- CRITICAL - STOCK SCARCITY AWARENESS: many topics (medical conditions, internal body processes, abstract science, specific diseases, niche technical concepts) have ZERO or near-zero literal stock footage. For these, do NOT search the literal term (e.g. "kidney stone" returns almost nothing usable and forces a random unrelated fallback). Instead pick the closest AVAILABLE stock category that still evokes the right idea:
  * "Kidney stones form when minerals crystallize" -> "crystal formation macro closeup" (visually evokes crystallization, actually findable)
  * "The kidney filters toxins from blood" -> "medical illustration human anatomy" or "doctor reviewing x-ray scan" (findable medical-adjacent stock)
  * "A rare genetic disorder affects..." -> "dna helix medical research lab" (findable, thematically correct)
  * "The economy is collapsing" -> "stock market crash graph red" (findable, matches meaning)
- NEVER pick a generic/unrelated query (like nature, ocean, space) just because the literal topic has no footage - always find the CLOSEST THEMATICALLY RELEVANT stock-findable alternative instead. A query is only acceptable if it is both findable in stock libraries AND still clearly connects to the sentence's subject.
- NO people/faces/bodies as the main subject, NO religion, NO violence, NO NSFW
- For EACH sentence, also provide ONE backup query - a different but still thematically-relevant angle on the same sentence, in case the primary query returns no results. The backup must NEVER be a generic unrelated filler (no random nature/space/ocean unless the sentence is actually about that) - it must still connect to the sentence's actual subject.
- Return in this EXACT format, one sentence per two lines:
1. primary query here
1b. backup query here
2. primary query here
2b. backup query here
(continue for all sentences, no other text)"""},
                    {"role": "user", "content": f"Match each sentence to a video search query:\n\n{numbered}"}
                ],
                model="openai/gpt-oss-120b",
                max_tokens=4000,
                temperature=0.5
            )

            result = r.choices[0].message.content
            # Parse by EXPLICIT leading index (e.g. "1. some query" -> index 0,
            # "1b. backup query" -> backup for index 0), not by line order.
            # This prevents misalignment: if Groq skips a number, returns a
            # blank/rejected line, or merges lines, sequential-order parsing
            # would silently shift every subsequent query onto the WRONG
            # sentence. That was the root cause of Spanish (and other
            # long-sentence scripts) queries ending up mismatched with
            # their actual sentence content.
            parsed_count = 0
            for line in result.strip().split('\n'):
                line = line.strip()
                mb = re.match(r'^\s*(\d+)b[\.\)\-]\s*(.+)$', line, re.IGNORECASE)
                if mb:
                    local_idx = int(mb.group(1)) - 1
                    cleaned = mb.group(2).strip().strip('"\'')
                    if 3 < len(cleaned) < 60 and _safe(cleaned) and 0 <= local_idx < len(batch):
                        all_backups[batch_start + local_idx] = cleaned
                    continue
                m = re.match(r'^\s*(\d+)[\.\)\-]\s*(.+)$', line)
                if not m:
                    continue
                local_idx = int(m.group(1)) - 1
                cleaned = m.group(2).strip().strip('"\'')
                if 3 < len(cleaned) < 60 and _safe(cleaned) and 0 <= local_idx < len(batch):
                    all_queries[batch_start + local_idx] = cleaned
                    parsed_count += 1

            # Diagnostic: if we parsed far fewer queries than sentences in
            # this batch, something is wrong with the model's output format
            # (e.g. it ignored the numbering instruction, wrapped in
            # markdown, or refused part of the request) - previously this
            # failed completely silently, with sentences just quietly
            # defaulting to random FALLBACK queries with zero explanation.
            if parsed_count < len(batch) * 0.5:
                print(f"  WARNING: batch {batch_start}-{batch_start+len(batch)} only parsed "
                      f"{parsed_count}/{len(batch)} queries. Raw model output (first 300 chars):")
                print(f"    {result[:300]!r}")

        except Exception as e:
            print(f"  Groq error on batch {batch_start}-{batch_start+len(batch)}: {e}")
            # leave this batch's entries as None -> filled by fallback below

    # Backups fall back to the primary query itself (still topic-relevant)
    # rather than the generic FALLBACK list, if Groq didn't provide one -
    # this ensures retries NEVER silently jump to unrelated filler (ocean/
    # nature/space) just because a specific query found no stock results.
    queries = [q if q else random.choice(FALLBACK) for q in all_queries]
    backups = [all_backups[i] if all_backups[i] else queries[i] for i in range(n)]

    # Show matching for debug
    for i in range(min(3, len(queries), len(sentences))):
        print(f"    [{i+1}] '{sentences[i]['text'][:35]}...' -> '{queries[i]}' (backup: '{backups[i]}')")

    return queries, backups

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
# 3B. AI SHORT SCRIPT WRITER (Groq oss-120b writes standalone short scripts)
# ==========================================
def generate_short_scripts(sentences, topic, n_shorts, target_seconds=60):
    """
    Two-step process using Groq oss-120b:
      1. Identify the N best hook-worthy themes/moments from the full
         long-form script (grounds the shorts in the actual video content).
      2. For each theme, WRITE a fresh, standalone, hook-first short-form
         script (~130-160 words for ~60s of narration at normal pace) -
         NOT a cut/excerpt of the original audio. This gets its own TTS
         pass, so it never depends on the main audio's timing/quality.

    Returns a list of dicts: {"script": str, "theme": str}
    """
    words_target = int(target_seconds / 60 * 150)  # ~150 wpm normal pace
    full_text = " ".join(s['text'] for s in sentences)[:15000]

    def _fallback_scripts():
        # If Groq is unavailable, fall back to lightly-summarized chunks of
        # the original script text, evenly spaced, rewritten as a short
        # standalone paragraph (still not a literal audio cut - this is
        # text, which gets its own fresh TTS regardless of this path).
        out = []
        step = max(1, len(sentences) // (n_shorts + 1))
        for k in range(n_shorts):
            start = min(k * step, max(0, len(sentences)-3))
            chunk = sentences[start:start+4]
            text = " ".join(s['text'] for s in chunk)
            out.append({"script": text, "theme": "fallback excerpt"})
        return out

    if not GROQ_KEY or not sentences:
        return _fallback_scripts()

    print(f"  Groq: writing {n_shorts} standalone short scripts (~{words_target} words each)...")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        lang_instruction = "Write in Spanish." if IS_SPANISH else "Write in English."

        r = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""You are an expert short-form (TikTok/Reels/Shorts) scriptwriter.

You will be given the full text of a long-form documentary/narration script. Your job: write {n_shorts} COMPLETELY STANDALONE short-form scripts inspired by the best hooks, surprising facts, emotional peaks, or claims in the source material.

RULES:
- Each script must be a SELF-CONTAINED mini-narration, NOT a literal excerpt or copy-paste of the source text. Rewrite/condense the idea into a tight, punchy standalone script.
- Each script should be approximately {words_target} words (for ~{target_seconds} seconds of narration at normal pace).
- Start with a strong HOOK in the first sentence - something that stops someone scrolling (a surprising claim, a question, a cliffhanger).
- {lang_instruction}
- Each script must make complete sense on its own with zero external context needed.
- Family-friendly, no NSFW, no violence, no religion-baiting content.
- Return ONLY a JSON array, nothing else. No markdown, no explanation.

Format exactly like this:
[{{"script": "full standalone narration text here...", "theme": "short label like 'the vanishing lake'"}}]"""},
                {"role": "user", "content": f"Source script:\n\n{full_text}\n\nWrite {n_shorts} standalone short scripts as JSON."}
            ],
            model="openai/gpt-oss-120b",
            max_tokens=4000,
            temperature=0.75
        )

        raw = r.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            print("  Groq short-script: no JSON array found, using fallback")
            return _fallback_scripts()

        picks = json.loads(match.group(0))
        scripts = []
        for p in picks:
            txt = str(p.get("script","")).strip()
            if len(txt) > 20:
                scripts.append({"script": txt, "theme": p.get("theme","")})

        if not scripts:
            print("  Groq short-script: parsed JSON had no valid scripts, using fallback")
            return _fallback_scripts()

        for s in scripts[:n_shorts]:
            wc = len(s['script'].split())
            print(f"    Short script: '{s['theme'][:40]}' ({wc} words)")

        fb = _fallback_scripts()
        while len(scripts) < n_shorts:
            scripts.append(fb[len(scripts) % len(fb)] if fb else {"script": topic, "theme": "padding"})

        return scripts[:n_shorts]

    except Exception as e:
        print(f"  Groq short-script error: {e}")
        return _fallback_scripts()



# ==========================================
# 4. SUBTITLE PRESETS (Variety)
# ==========================================
SUBTITLE_STYLES = {
    "classic": {"name":"Classic","font":"Arial Black","size":64,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":48,"spacing":0.5,
        "highlight":"&H0059C7FF"},   # white text, warm gold highlight
    "cyan_pop": {"name":"Cyan Pop","font":"Arial Black","size":64,"bold":-1,
        "primary":"&H00FFDC78","outline_c":"&H002D140F","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":48,"spacing":0.5,
        "highlight":"&H00FFFFFF"},   # soft cyan text, white highlight
    "boxed": {"name":"Boxed","font":"Arial","size":58,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&HB0000000",
        "border":3,"outline":0,"shadow":0,"margin":48,"spacing":0.3,
        "highlight":"&H0059C7FF"},   # white on translucent black box, gold highlight
}

# Short-specific vertical styles (1080x1920 canvas). Bigger fonts than
# landscape presets since viewers are closer to phone screens, and a much
# larger MarginV to clear TikTok/Reels/YouTube Shorts UI (caption, like/
# share buttons, progress bar) which occupies the bottom ~20-25% of frame.
# Every color pair below was computed programmatically (RGB -> ASS BGR),
# not hand-written, and verified for real contrast before use.
SHORT_SUBTITLE_STYLES = {
    "short_classic": {"name":"Short Classic","font":"Arial Black","size":90,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":8,"shadow":3,"margin":480,"spacing":0.5,
        "highlight":"&H003BEBFF"},   # white text, electric yellow highlight
    "short_pink": {"name":"Short Pink","font":"Arial Black","size":90,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":8,"shadow":3,"margin":480,"spacing":0.5,
        "highlight":"&H008140FF"},   # white text, hot pink highlight
    "short_boxed": {"name":"Short Boxed","font":"Arial","size":82,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&HB0000000",
        "border":3,"outline":0,"shadow":0,"margin":460,"spacing":0.3,
        "highlight":"&H0040FFAE"},   # white on translucent black box, lime highlight
}

def _fmt(sec):
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60); cs=int((sec%1)*100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


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
    if style_key:
        key = style_key
    else:
        # Derive selection from a hash of JOB_ID + wall-clock time + OS
        # entropy, rather than trusting random.choice() alone. JOB_ID is
        # guaranteed unique per run (templated in externally per job), so
        # this guarantees different style selection across separate runs
        # even if Python's RNG state behaves unexpectedly in the Kaggle
        # kernel environment (e.g. container/process state carrying over
        # between runs in a way that defeats a simple reseed).
        import hashlib
        seed_material = f"{JOB_ID}-{time.time()}-{os.urandom(4).hex()}"
        h = int(hashlib.sha256(seed_material.encode()).hexdigest(), 16)
        keys = list(style_set.keys())
        key = keys[h % len(keys)]
    s = style_set[key]
    print(f"  Subtitle: {s['name']} {'(word-highlight)' if word_data else '(sentence)'} @ {play_res[0]}x{play_res[1]}")
    
    # Highlight color (the word currently being spoken) - explicit per-style,
    # chosen for contrast against that style's own primary text color.
    highlight = s.get("highlight", "&H0000FFFF")
    
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

                MIN_HIGHLIGHT_DUR = 0.12  # seconds - guarantees every word gets
                # a real, renderable display window. Fast speech or ASR
                # quirks can produce near-zero-duration words (start ~= end,
                # or the next word starting almost immediately). libass can
                # skip or misrender these near-zero-duration events, which
                # made the highlight appear to jump past the correct word
                # onto the next one while the audio was still on the first.

                prev_end_sec = None  # tracks previous word's (possibly-extended) end,
                                      # so extending one word's duration can never
                                      # cause it to overlap the next word's event
                for w_idx, word in enumerate(chunk_words):
                    w_start_sec = word['start']
                    if prev_end_sec is not None and w_start_sec < prev_end_sec:
                        w_start_sec = prev_end_sec  # never start before prior word ended

                    if w_idx + 1 < len(chunk_words):
                        next_start_sec = chunk_words[w_idx+1]['start']
                    else:
                        next_start_sec = next_chunk_start

                    # Enforce minimum duration - if the natural gap to the
                    # next word is too small, extend this word's END forward
                    # rather than shrinking its START, so timing still lines
                    # up with when the word actually begins being spoken.
                    if next_start_sec - w_start_sec < MIN_HIGHLIGHT_DUR:
                        next_start_sec = w_start_sec + MIN_HIGHLIGHT_DUR
                    prev_end_sec = next_start_sec

                    w_start = _fmt(w_start_sec)
                    w_end = _fmt(next_start_sec)

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

def search_and_download_vertical(query, idx, duration, tag="", verify=True):
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

    # Try more candidates than before (was 3) since some will now be
    # rejected on VISUAL grounds, not just download failure.
    for url in urls[:6]:
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
            if not (os.path.exists(out) and os.path.getsize(out) > 2000):
                continue

            if verify:
                matches = verify_clip_matches_query(out, query)
                if not matches:
                    print(f"    Rejected short clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(out)
                    except: pass
                    continue

            USED_URLS.add(url); return str(out)
        except: continue
    return None


def process_short_clip(args):
    i, sent, tag = args
    dur = max(2.5, sent['end'] - sent['start'])
    orig_idx = sent.get('orig_idx', i)
    primary = AI_QUERIES[orig_idx] if orig_idx < len(AI_QUERIES) else random.choice(FALLBACK)
    backup = AI_BACKUPS[orig_idx] if orig_idx < len(AI_BACKUPS) else primary

    attempts = [primary, backup, random.choice(FALLBACK)]
    for query in attempts:
        clip = search_and_download_vertical(query, i, dur, tag=tag)
        if clip: return (i, clip)
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
                f"[wl]subtitles='{ass_esc}'[v];"
                f"[2:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",visual_path,"-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    else:
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i",visual_path,"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=300)

    for p in [list_path, visual_path]:
        if os.path.exists(p): os.remove(p)

    if not os.path.exists(out_path):
        print(f"  Short {short_idx+1}: mux failed - {r.stderr.decode(errors='ignore')[-400:]}")
        return False

    # Verify the output actually has a real audio stream - a bad stream
    # mapping or corrupt upstream audio could still produce a "successful"
    # (file exists, exit 0) mp4 with silent/missing audio.
    #
    # NOTE: ffprobe's per-stream `duration` field is unreliable for MP4/AAC
    # (frequently reports N/A even on perfectly valid, audible streams).
    # That was causing a previous version of this check to report 0.0s and
    # incorrectly discard EVERY short's audio, even when it was fine.
    # Use frame/packet count instead, which reliably reflects whether
    # actual audio data is present.
    try:
        rp = subprocess.run(["ffprobe","-v","error","-select_streams","a:0",
            "-count_packets","-show_entries","stream=nb_read_packets",
            "-of","default=noprint_wrappers=1:nokey=1",
            str(out_path)], capture_output=True, text=True, timeout=20)
        out = rp.stdout.strip()
        n_packets = int(out) if out.isdigit() else -1
        if n_packets == 0:
            print(f"  Short {short_idx+1}: output audio stream has 0 packets - silent, treating as failed")
            return False
        elif n_packets < 0:
            print(f"  Short {short_idx+1}: could not verify audio packet count, proceeding (assuming OK)")
    except Exception as e:
        print(f"  Short {short_idx+1}: audio-stream verification failed ({e}), proceeding cautiously")

    print(f"  Short {short_idx+1}: {os.path.getsize(out_path)/(1024**2):.0f}MB")
    return True


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
           "-af",af,"-ar","44100","-ac","2",str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        print(f"  extract_short_audio failed: {r.stderr.decode(errors='ignore')[-300:]}")
        return False

    # Validate ACTUAL audio duration, not just file existence/size. A
    # corrupt or truncated extraction (e.g. seek landing past end of
    # source, or a near-silent slice) can still produce a small-but-
    # nonzero WAV file that passes the size check while being
    # functionally silent or far too short - this was causing "voice in
    # one short, no voice in another" with no visible error.
    try:
        rp = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",str(out_path)],
            capture_output=True, text=True, timeout=15)
        actual_dur = float(rp.stdout.strip())
        if actual_dur < dur * 0.5:
            print(f"  extract_short_audio: got {actual_dur:.2f}s, expected ~{dur:.2f}s - extraction likely broken")
            return False
    except Exception as e:
        print(f"  extract_short_audio: duration probe failed ({e}), proceeding cautiously")

    return True



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
# Local video-chunk verification model (SmolVLM2-500M-Video-Instruct).
# Replaces the earlier Groq vision-based single-frame check, which hit
# constant rate limits at scale (100+ verification calls per video) and
# only ever looked at ONE static frame per clip, missing motion/scene
# changes across the chunk. This model takes the actual video file
# directly and reasons about the whole chunk. Runs 100% locally on the
# Kaggle GPU - zero API calls, zero rate limits. Query GENERATION still
# uses Groq (openai/gpt-oss-120b) unchanged, since that part works well.
_smolvlm_model = None
_smolvlm_processor = None
_smolvlm_lock = threading.Lock()  # model isn't thread-safe for concurrent .generate() calls

def _load_smolvlm():
    global _smolvlm_model, _smolvlm_processor
    if _smolvlm_model is not None:
        return
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print("  Loading local video verification model (SmolVLM2-500M)...")
    _smolvlm_processor = AutoProcessor.from_pretrained(model_path)
    _smolvlm_model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")


def verify_clip_matches_query(clip_path, query, filter_women=True):
    """
    Ask the local SmolVLM2 model whether the given video CLIP (not just a
    single frame) actually matches the intended search query, AND whether
    it shows a woman (if filter_women is True) - combined into ONE model
    call for efficiency rather than two separate passes.

    This is the real fix for stock footage that "downloads fine" but is
    visually unrelated to the query - there was previously ZERO check
    that a downloaded clip actually looked like what it was searched for,
    and no check on content restrictions beyond the query TEXT (a neutral
    query like "person walking city" could still return a clip showing a
    woman, since the restriction was only ever applied to search terms,
    not actual visual content).

    Returns True if the clip should be USED (topic matches AND, if
    filter_women, no woman detected), False if it should be rejected.
    Fails open (returns True) on any error, so a model hiccup never blocks
    the whole pipeline - it's a quality filter, not a hard gate. No rate
    limiting needed since this runs 100% locally.
    """
    try:
        _load_smolvlm()
    except Exception as e:
        print(f"    Video verification model unavailable ({str(e)[:60]}), accepting clip")
        return True

    try:
        with _smolvlm_lock:  # serialize GPU access across worker threads
            if filter_women:
                prompt = f"""Look at this video clip and answer two questions, each on its own line, in this EXACT format:
1. <YES or NO>
2. <YES or NO>

1. Does this video clip visually match the concept: "{query}"? Answer YES if it reasonably represents the concept (even loosely/thematically - stock footage rarely is a perfect literal match). Answer NO only if it's clearly unrelated.
2. Does the clip show any woman or women as a visible person in frame (not just implied)? Answer YES or NO."""
            else:
                prompt = f"""Look at this video clip and answer, in this EXACT format:
1. <YES or NO>
2. NO

1. Does this video clip visually match the concept: "{query}"? Answer YES if it reasonably represents the concept (even loosely/thematically - stock footage rarely is a perfect literal match). Answer NO only if it's clearly unrelated."""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "path": str(clip_path)},
                    {"type": "text", "text": prompt}
                ]
            }]
            inputs = _smolvlm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(_smolvlm_model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated_ids = _smolvlm_model.generate(**inputs, do_sample=False, max_new_tokens=20)
            # Slice off the input tokens so we only decode the model's NEW
            # output, not the full prompt+answer text. This is the correct,
            # robust way to isolate the answer.
            new_tokens = generated_ids[:, input_len:]
            answer = _smolvlm_processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip().upper()

            # Parse line 1 (topic match) and line 2 (woman detection)
            # independently - don't just search the whole blob for "YES",
            # since that would conflate the two answers if the model
            # returns e.g. "1. NO / 2. YES" (topic mismatch AND a woman -
            # searching the whole string for "YES" would wrongly pass it).
            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            line1 = lines[0] if len(lines) > 0 else ""
            line2 = lines[1] if len(lines) > 1 else ""

            topic_match = "YES" in line1 or "NO" not in line1
            has_woman = filter_women and "YES" in line2

            if has_woman:
                print(f"    Rejected clip for '{query[:40]}' (woman detected in frame)")
                return False
            return topic_match
    except Exception as e:
        print(f"    Visual verification skipped ({str(e)[:60]}), accepting clip")
        return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def search_and_download(query, idx, duration, verify=True):
    """Search + download + encode with GPU, with optional visual verification"""
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
    
    # Try more candidates than before (was 3) since some will now be
    # rejected on VISUAL grounds, not just download failure - need more
    # attempts to still find a genuinely matching clip.
    for url in urls[:6]:
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
            if not (os.path.exists(out) and os.path.getsize(out) > 2000):
                continue

            # --- Real visual verification (the actual "no relevance check" fix) ---
            if verify:
                matches = verify_clip_matches_query(out, query)
                if not matches:
                    print(f"    Rejected clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(out)
                    except: pass
                    continue

            USED_URLS.add(url); return str(out)
        except: continue
    return None

def process_clip(args):
    i, sent, total = args
    dur = max(3.5, sent['end'] - sent['start'])
    primary = AI_QUERIES[i] if i < len(AI_QUERIES) else random.choice(FALLBACK)
    backup = AI_BACKUPS[i] if i < len(AI_BACKUPS) else primary

    # Try order: primary query, then topic-aware backup query, then ONLY as
    # a last resort the generic FALLBACK list. Previously this jumped
    # straight to a fixed generic list (ocean/nature/space/etc.) on the
    # very first retry, which is why scarce-footage topics (medical,
    # niche technical) ended up with visually unrelated clips - the
    # backup is chosen by the AI to still connect to the actual sentence.
    attempts = [primary, backup, random.choice(FALLBACK)]
    for query in attempts:
        clip = search_and_download(query, i, dur)
        if clip: return (i, clip)
        time.sleep(0.3)
    return (i, None)



# ==========================================
# 7. RENDER ENGINE (GPU-Accelerated)
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_sub):
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

    # Safety: stream-copy concat of many clips can lose fractions of a
    # second per clip boundary (keyframe/GOP alignment), and clip fetches
    # can also come back shorter than requested. Either way, if the
    # resulting visual track ends up shorter than the audio, -shortest
    # below would truncate the LONGER stream (the audio) to match -
    # silently cutting off the last spoken sentence(s).
    #
    # FIX: instead of freeze-padding (which produces a long dead/frozen
    # frame - unacceptable for anything more than ~1s), fetch REAL
    # additional stock clips to cover the deficit, using fresh queries
    # from AI_QUERIES/FALLBACK so the extra footage still looks intentional.
    def _probe_dur(path):
        try:
            r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                "-of","default=noprint_wrappers=1:nokey=1",str(path)],
                capture_output=True, text=True, timeout=15)
            return float(r.stdout.strip())
        except: return 0.0

    vdur = _probe_dur("visual.mp4")
    adur = _probe_dur(audio_path)
    if vdur > 0 and adur > 0 and vdur < adur - 0.5:
        deficit = (adur - vdur) + 0.5  # small extra buffer
        print(f"  Visual track {vdur:.2f}s shorter than audio {adur:.2f}s, fetching {deficit:.1f}s of extra footage...")

        extra_clips = []
        remaining = deficit
        attempt = 0
        # Pull from the tail of AI_QUERIES first (thematically closest to
        # the video's ending), falling back to FALLBACK queries if needed.
        query_pool = list(reversed(AI_QUERIES)) if AI_QUERIES else []
        while remaining > 0.5 and attempt < 15:
            q = query_pool[attempt % len(query_pool)] if query_pool else random.choice(FALLBACK)
            chunk_dur = min(remaining, 8.0)  # fetch in ~8s chunks
            clip = search_and_download(q, f"pad{attempt}", chunk_dur)
            if clip:
                extra_clips.append(clip)
                remaining -= chunk_dur
            attempt += 1

        if extra_clips:
            with open("list.txt", "a") as f:
                for c in extra_clips:
                    f.write(f"file '{c}'\n")
            subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual_ext.mp4",
                shell=True, capture_output=True, timeout=60)
            if os.path.exists("visual_ext.mp4"):
                os.replace("visual_ext.mp4", "visual.mp4")
                vdur = _probe_dur("visual.mp4")

        # If real footage still didn't fully close the gap (fetch failures,
        # rate limits, etc.), loop the last clip rather than freeze on a
        # single static frame - motion is far less noticeable/jarring than
        # a dead frame, and this only covers whatever small remainder is left.
        vdur = _probe_dur("visual.mp4")
        if vdur > 0 and vdur < adur - 0.5:
            still_needed = (adur - vdur) + 0.3
            print(f"  Still {still_needed:.1f}s short after extra fetch, looping last clip (not freezing)")
            last_source = extra_clips[-1] if extra_clips else clips[-1]
            loop_fill = "visual_loopfill.mp4"
            subprocess.run(["ffmpeg","-y","-stream_loop","-1","-i",last_source,
                "-t",f"{still_needed:.2f}"] + _enc_args() + ["-an","loopfill_clip.mp4"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            if os.path.exists("loopfill_clip.mp4"):
                with open("list.txt", "a") as f:
                    f.write(f"file 'loopfill_clip.mp4'\n")
                subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual_final.mp4",
                    shell=True, capture_output=True, timeout=60)
                if os.path.exists("visual_final.mp4"):
                    os.replace("visual_final.mp4", "visual.mp4")
            for tmp in ["loopfill_clip.mp4"]:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass
    
    # Render final video WITH subtitles only (no-subs version removed - not needed)
    update_status(85, "Rendering final video (1080p + subs)...")
    enc = _enc_args()
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
                f"[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];"
                f"[wl]subtitles='{ass_esc}'[v];"
                f"[2:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    else:
        filt = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y","-hwaccel","cuda","-i","visual.mp4","-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_sub)]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if not os.path.exists(out_sub):
        print(f"  Final render failed - {r.stderr.decode(errors='ignore')[-400:]}")
        return False
    print(f"  Final: {os.path.getsize(out_sub)/(1024**2):.0f}MB")
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
        # CRITICAL: language must be explicitly set. Without this,
        # AssemblyAI defaults to English and on Spanish audio produces
        # heavily garbled/dropped transcription (most words treated as
        # unintelligible noise) - this was the root cause of only getting
        # ~163 words out of an 8:42 Spanish narration.
        tx_config = aai.TranscriptionConfig(
            language_code="es" if IS_SPANISH else "en",
            punctuate=True,
            format_text=True,
        )
        tx = aai.Transcriber(config=tx_config).transcribe(str(audio))
        if tx.status == aai.TranscriptStatus.error:
            print(f"  Transcribe failed: {tx.error}")
        else:
            for s in tx.get_sentences():
                sentences.append({"text":s.text,"start":s.start/1000,"end":s.end/1000})
            if sentences: sentences[-1]['end']+=0.5

            # Get word-level timestamps for subtitle highlighting
            for word in tx.words:
                word_data.append({"text": word.text, "start": word.start/1000, "end": word.end/1000})
            print(f"  Got {len(word_data)} word timestamps for highlighting")

            # Sanity check: normal speech is ~2-3 words/sec. If AssemblyAI
            # returned far fewer words than the audio duration implies, the
            # transcription is almost certainly garbled/wrong-language
            # (words dropped as "unintelligible") rather than genuinely
            # sparse audio. Discard it and fall through to the estimated-
            # timing path below instead of silently building subtitles and
            # visual-matching off broken data.
            audio_dur = sentences[-1]['end'] if sentences else 0
            if audio_dur > 10:
                wpm = len(word_data) / (audio_dur/60)
                if wpm < 60:  # normal narration is ~120-180 wpm; below 60 is a red flag
                    print(f"  WARNING: only {wpm:.0f} words/min detected (expected 100+) - "
                          f"transcription looks broken, discarding and using estimated timing instead")
                    sentences = []
                    word_data = []
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
AI_QUERIES, AI_BACKUPS = generate_queries_for_sentences(sentences)

# Subtitles (word-level highlighting if available)
update_status(52, "Subtitles...")
ass = TEMP_DIR/"subs.ass"
create_subtitles(sentences, ass, word_data=word_data if word_data else None)

# Render
update_status(54, "Processing video...")
o2 = OUTPUT_DIR/f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o2):
    update_status(93, "Uploading...")
    l2 = upload_drive(o2)
    msg = "Done!\n"
    if l2: msg += f"Video: {l2}\n"

    # ==========================================
    # SHORTS PIPELINE
    # ==========================================
    # Only worth generating shorts from audio that's actually long enough
    # for the requested count (avoid generating garbage 60s shorts from a
    # 90s total-runtime video, etc.)
    if sentences and sentences[-1]['end'] >= SHORT_DUR_TARGET * 0.6:
        update_status(95, f"Generating {SHORTS_COUNT} shorts...")
        try:
            short_scripts = generate_short_scripts(sentences, TOPIC if MODE=="topic" else text[:100], SHORTS_COUNT, SHORT_DUR_TARGET)
            print(f"  Shorts: {len(short_scripts)} scripts generated (requested {SHORTS_COUNT})")
            short_links = []
            short_failures = []  # (short_num, reason) for end-of-run summary

            for si, sc in enumerate(short_scripts):
                update_status(95, f"Short {si+1}/{len(short_scripts)}...")
                script_text = sc['script'].strip()
                if len(script_text) < 20:
                    short_failures.append((si+1, "empty/too-short script"))
                    continue

                # --- Independent TTS for this short (same voice, own audio file) ---
                short_audio = TEMP_DIR / f"short_{si}_audio.wav"
                if not generate_audio(script_text, voice, short_audio):
                    print(f"  Short {si+1}: TTS failed, skipping")
                    short_failures.append((si+1, "TTS failed"))
                    continue

                # --- Independent transcription for accurate word-level timing ---
                # Re-transcribing THIS short's own audio (instead of reusing/cutting
                # from the main video's word_data) guarantees the subtitle timing is
                # always perfectly matched to what's actually in this short's audio -
                # no cross-file drift, no boundary mismatches.
                short_sentences, short_word_data = [], []
                if ASSEMBLY_KEY:
                    try:
                        tx_config = aai.TranscriptionConfig(
                            language_code="es" if IS_SPANISH else "en",
                            punctuate=True, format_text=True,
                        )
                        tx = aai.Transcriber(config=tx_config).transcribe(str(short_audio))
                        if tx.status != aai.TranscriptStatus.error:
                            for s in tx.get_sentences():
                                short_sentences.append({"text": s.text, "start": s.start/1000, "end": s.end/1000})
                            if short_sentences: short_sentences[-1]['end'] += 0.3
                            for w in tx.words:
                                short_word_data.append({"text": w.text, "start": w.start/1000, "end": w.end/1000})
                    except Exception as e:
                        print(f"  Short {si+1}: transcription error ({e}), using estimated timing")

                # Fallback: if transcription failed/unavailable, estimate timing
                # by splitting the script into sentences and spacing them evenly
                # across the actual audio duration (probed via ffprobe).
                if not short_sentences:
                    try:
                        rp = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                            "-of","default=noprint_wrappers=1:nokey=1", str(short_audio)],
                            capture_output=True, text=True, timeout=15)
                        total_dur = float(rp.stdout.strip())
                    except: total_dur = SHORT_DUR_TARGET
                    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', script_text) if len(p.strip()) > 2]
                    if not parts: parts = [script_text]
                    per = total_dur / len(parts)
                    for i, p in enumerate(parts):
                        short_sentences.append({"text": p, "start": i*per, "end": (i+1)*per})

                # --- Vertical-tuned subtitles for this short ---
                for i, s in enumerate(short_sentences):
                    s['orig_idx'] = i
                short_ass = TEMP_DIR / f"short_{si}_subs.ass"
                create_subtitles(
                    short_sentences, short_ass,
                    word_data=short_word_data if short_word_data else None,
                    style_set=SHORT_SUBTITLE_STYLES,
                    play_res=(1080, 1920),
                    max_chars=20,   # narrower canvas + bigger font (86-96px) -> tighter budget
                )

                # --- AI visual-matching queries for THIS short's own sentences ---
                # (independent from the main video's AI_QUERIES, since this is
                # different, freshly-written content)
                short_queries, short_backups = generate_queries_for_sentences(short_sentences)
                # Temporarily point AI_QUERIES/AI_BACKUPS at this short's
                # queries so process_short_clip (which reads the globals)
                # picks the right query/backup per sentence via orig_idx.
                saved_queries = AI_QUERIES
                saved_backups = AI_BACKUPS
                AI_QUERIES = short_queries
                AI_BACKUPS = short_backups

                short_out = OUTPUT_DIR / f"short_{JOB_ID}_{si+1}.mp4"
                ok = render_short(si, short_sentences, short_audio, short_ass, logo, short_out)
                if not ok:
                    print(f"  Short {si+1}: retrying once...")
                    ok = render_short(si, short_sentences, short_audio, short_ass, logo, short_out)

                AI_QUERIES = saved_queries  # restore main video's queries
                AI_BACKUPS = saved_backups  # restore main video's backups

                if ok:
                    link = upload_drive(short_out)
                    if link:
                        short_links.append(link)
                        msg += f"Short {si+1}: {link}\n"
                    else:
                        short_failures.append((si+1, "upload failed (render succeeded)"))
                else:
                    print(f"  Short {si+1}: failed after retry, skipping")
                    short_failures.append((si+1, "render failed after retry"))

            print(f"  Shorts summary: {len(short_links)}/{len(short_scripts)} succeeded")
            if short_failures:
                print(f"  Shorts failures: {short_failures}")
                msg += f"({len(short_failures)} short(s) failed - check logs)\n"
        except Exception as e:
            print(f"  Shorts pipeline error: {e}")

    update_status(100, msg, "completed", l2)
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
