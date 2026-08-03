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
from importlib.metadata import version as _package_version, PackageNotFoundError
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

# Kaggle images often already contain most of these packages. Reinstalling
# them unconditionally cost roughly five minutes on every job, so only invoke
# pip when the import is genuinely missing. This keeps fresh kernels correct
# without paying the install cost on warm/prebuilt images.
def _ensure_package(module_name, requirement, extra_args=None):
    try:
        __import__(module_name)
        return True
    except Exception:
        args = [sys.executable, "-m", "pip", "install", "--quiet"]
        if extra_args:
            args.extend(extra_args)
        args.append(requirement)
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: could not install {requirement}: {(result.stderr or '')[-300:]}")
            return False
        return True

# A Chatterbox install must not replace Kaggle's preinstalled GPU stack.
# Its official metadata pins torch/torchaudio 2.6.0, and a normal dependency
# install can leave cuDNN component libraries from different releases in the
# same process. That is the cause of the native cudnnGetLibConfig abort.
_CUDNN_FORCE_DISABLED = False
_CUDNN_HANDLES = []


def _prepare_cuda_runtime():
    """Prefer one coherent pip NVIDIA runtime before any torch-bearing import."""
    global _CUDNN_FORCE_DISABLED, _CUDNN_HANDLES
    if not sys.platform.startswith("linux"):
        return

    import ctypes
    import site

    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except (AttributeError, TypeError):
        pass
    try:
        site_roots.append(site.getusersitepackages())
    except (AttributeError, TypeError):
        pass

    nvidia_lib_dirs = []
    cudnn_lib_dirs = []
    for root in site_roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        cudnn_dir = nvidia_root / "cudnn" / "lib"
        if cudnn_dir.is_dir():
            cudnn_lib_dirs.append(str(cudnn_dir))
        try:
            components = sorted(nvidia_root.iterdir(), key=lambda path: path.name)
        except OSError:
            components = []
        for component in components:
            lib_dir = component / "lib"
            if lib_dir.is_dir():
                nvidia_lib_dirs.append(str(lib_dir))

    # Put the cuDNN wheel directory first, then the rest of the matching pip
    # NVIDIA runtime, before any system /usr/local/cuda entry can win.
    priority_dirs = cudnn_lib_dirs + nvidia_lib_dirs
    old_path = os.environ.get("LD_LIBRARY_PATH", "")
    ordered_dirs = []
    for directory in priority_dirs + old_path.split(os.pathsep):
        if directory and directory not in ordered_dirs:
            ordered_dirs.append(directory)
    if ordered_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ordered_dirs)

    if not cudnn_lib_dirs:
        print("  CUDA runtime: packaged cuDNN directory not found; keeping Kaggle loader path")
        return

    def _find_library(directory, stem):
        exact = Path(directory) / f"{stem}.so.9"
        if exact.exists():
            return exact
        matches = sorted(Path(directory).glob(f"{stem}.so.9.*"))
        return matches[0] if matches else None

    cudnn_dir = cudnn_lib_dirs[0]
    core_path = _find_library(cudnn_dir, "libcudnn")
    graph_path = _find_library(cudnn_dir, "libcudnn_graph")
    if not core_path or not graph_path:
        print(f"  CUDA runtime: incomplete cuDNN bundle in {cudnn_dir}; using normal torch startup")
        return

    try:
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        core_handle = ctypes.CDLL(str(core_path), mode=global_mode)
        if getattr(core_handle, "cudnnGetLibConfig", None) is None:
            raise OSError(f"{core_path} does not export cudnnGetLibConfig")
        graph_handle = ctypes.CDLL(str(graph_path), mode=global_mode)
        _CUDNN_HANDLES = [core_handle, graph_handle]
        print(f"  CUDA runtime: coherent cuDNN core/graph loaded from {cudnn_dir}")
    except (OSError, AttributeError) as cudnn_error:
        _CUDNN_FORCE_DISABLED = True
        print(f"  WARNING: cuDNN component mismatch detected ({str(cudnn_error)[:180]})")
        print("  WARNING: cuDNN will be disabled for neural inference to prevent a native abort")


_prepare_cuda_runtime()
# Import torch before accelerate/torchvision/bitsandbytes. This makes every
# later torch-bearing import reuse the runtime selected above.
import torch
if _CUDNN_FORCE_DISABLED:
    torch.backends.cudnn.enabled = False
    print("  CUDA runtime: torch.backends.cudnn.enabled=False")

for _module, _requirement in [
    ("groq", "groq"), ("assemblyai", "assemblyai"),
    ("google.generativeai", "google-generativeai"), ("requests", "requests"),
    ("pydub", "pydub"), ("numpy", "numpy"), ("PIL", "pillow"),
    ("librosa", "librosa"), ("scipy", "scipy"), ("soundfile", "soundfile"),
    ("accelerate", "accelerate"),
    ("av", "av"), ("decord", "decord==0.6.0"),
    ("torchvision", "torchvision"), ("sentencepiece", "sentencepiece"),
    ("bitsandbytes", "bitsandbytes>=0.46.1"),
]:
    _ensure_package(_module, _requirement)

# Install Chatterbox's Python-only support packages without dependency
# resolution. These imports are all safe after torch has been initialized and
# none is allowed to pull a replacement CUDA/PyTorch wheel.
for _module, _requirement in [
    ("s3tokenizer", "s3tokenizer"),
    ("diffusers", "diffusers==0.29.0"),
    ("conformer", "conformer==0.3.2"),
    ("safetensors", "safetensors==0.5.3"),
    ("perth", "resemble-perth==1.0.1"),
    ("spacy_pkuseg", "spacy-pkuseg"),
    ("pykakasi", "pykakasi==2.3.0"),
    ("pyloudnorm", "pyloudnorm"),
    ("omegaconf", "omegaconf"),
]:
    _ensure_package(_module, _requirement, ["--no-deps"])

try:
    _package_version("chatterbox-tts")
    print("  chatterbox-tts already available")
except PackageNotFoundError:
    print("  Installing chatterbox-tts from GitHub (master) for V3 support...")
    _cb_installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
         "git+https://github.com/resemble-ai/chatterbox.git"],
        capture_output=True, text=True
    )
    if _cb_installed.returncode != 0:
        print("  git install failed, falling back to PyPI chatterbox-tts")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "chatterbox-tts"],
                       check=False)

# Chatterbox's current source can leave a newer Transformers build installed.
# Keep one exact version that avoids the `mistral-common`
# BACKENDS_MAPPING regression reported with 5.14.x and remains compatible with
# MiniCPM-V 4.5's custom Qwen3-based model code. MiniCPM's published config was
# authored against 4.51.0, but its custom AutoModel code is compatible with the
# 5.5.0 build already validated with Chatterbox in this pipeline.
_TRANSFORMERS_REQUIRED = "5.5.0"
# Transformers 5.5.0 performs this check during its top-level import. Kaggle's
# image currently carries tokenizers 0.21.x, which makes the import fail after
# the expensive dependency/bootstrap phase. Keep this exact and independent of
# pip's resolver so no CUDA package can be changed as a side effect.
_TOKENIZERS_REQUIRED = "0.22.1"
_HUB_REQUIRED = "1.5.0"


def _purge_module_tree(*roots):
    """Remove already-imported package trees after an in-process pip repair."""
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(root + ".") for root in roots):
            del sys.modules[module_name]


print(f"  Ensuring huggingface_hub=={_HUB_REQUIRED}...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
     "--no-deps", "--force-reinstall", f"huggingface_hub=={_HUB_REQUIRED}"],
    check=True,
)
# accelerate is checked above and may already have imported an older Hub
# package. Purge both package trees so Python cannot combine old submodules
# with the freshly installed files (the source of the alternating missing
# Hub-symbol failures).
_purge_module_tree("huggingface_hub", "accelerate", "transformers")
import importlib
importlib.invalidate_caches()
try:
    import huggingface_hub as _hub_check
    from huggingface_hub.errors import RemoteEntryNotFoundError as _RemoteEntryNotFoundError
    importlib.import_module("huggingface_hub.file_download")
    del _RemoteEntryNotFoundError, _hub_check
except Exception as _hub_error:
    raise RuntimeError(
        f"huggingface_hub=={_HUB_REQUIRED} is still inconsistent after clean reinstall: {_hub_error}"
    ) from _hub_error
print(f"  huggingface_hub=={_HUB_REQUIRED} import validated")

try:
    _transformers_version = _package_version("transformers")
except PackageNotFoundError:
    _transformers_version = None
if _transformers_version != _TRANSFORMERS_REQUIRED:
    print(f"  Pinning Transformers {_transformers_version or 'missing'} -> {_TRANSFORMERS_REQUIRED}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "--no-deps", "--force-reinstall", f"transformers=={_TRANSFORMERS_REQUIRED}"],
        check=True,
    )

try:
    _tokenizers_version = _package_version("tokenizers")
except PackageNotFoundError:
    _tokenizers_version = None
if _tokenizers_version != _TOKENIZERS_REQUIRED:
    print(f"  Pinning tokenizers {_tokenizers_version or 'missing'} -> {_TOKENIZERS_REQUIRED}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "--no-deps", "--force-reinstall", f"tokenizers=={_TOKENIZERS_REQUIRED}"],
        check=True,
    )

# Both packages may have been imported by an earlier warm-kernel probe. Purge
# their module trees so the import below cannot combine old Python modules with
# freshly installed package files.
_purge_module_tree("transformers", "tokenizers")
importlib.invalidate_caches()
from transformers import AutoModel, AutoProcessor, AutoTokenizer  # noqa: F401

_ensure_package("resemble_enhance", "resemble-enhance", ["--no-deps"])
if shutil.which("ffmpeg") is None:
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg",
                   shell=True, capture_output=True)

import torchaudio
import assemblyai as aai
import google.generativeai as genai


def _print_gpu_inventory():
    """Log every visible CUDA device so remote accelerator selection is verifiable."""
    if not torch.cuda.is_available():
        print("  CUDA devices visible: 0")
        return
    count = torch.cuda.device_count()
    print(f"  CUDA devices visible: {count}")
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"    GPU {index}: {props.name} ({memory_gb:.1f} GB VRAM)")
        except Exception as e:
            print(f"    GPU {index}: unavailable ({type(e).__name__}: {str(e)[:100]})")


_print_gpu_inventory()

# Check if NVENC is available
def _has_nvenc():
    r = subprocess.run("ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc",
        shell=True, capture_output=True, text=True)
    return "h264_nvenc" in r.stdout

USE_GPU = _has_nvenc()
_nvenc_runtime_failed = False
print(f"  GPU Encoding: {'h264_nvenc' if USE_GPU else 'libx264 (CPU)'}")

def _enc_args():
    """Return encoder args based on currently usable GPU availability."""
    if USE_GPU and not _nvenc_runtime_failed:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]

def _hwaccel_args():
    """Use CUDA input only while the runtime NVENC path is healthy."""
    return ["-hwaccel", "cuda"] if USE_GPU and not _nvenc_runtime_failed else []



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
_USED_URLS_LOCK = threading.Lock()
# Stock API lookup is deliberately shorter than media download. Running the
# two independent providers in parallel avoids serial 12-second stalls while
# still giving each provider enough time for normal network conditions.
_STOCK_API_TIMEOUT = (3, 5)
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
_LOCAL_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with", "about", "after",
    "also", "como", "con", "del", "desde", "el", "ella", "ellos", "en",
    "es", "esta", "este", "la", "las", "los", "para", "por", "que", "se",
    "su", "sus", "una", "uno", "y", "al", "más", "más", "son", "un",
}

# These are deliberately broad, stock-findable visual categories. They are
# only used for a sentence whose Groq response was missing or rejected after
# all targeted retries; they never replace a valid Groq query.
_LOCAL_QUERY_RULES = [
    (("kidney", "renal", "blood", "toxin", "disease", "medical", "hospital",
      "doctor", "anatomy", "health", "organ", "stone", "riñón", "sangre",
      "enfermedad", "medicina", "salud"),
     ("medical research laboratory", "medical anatomy illustration")),
    (("technology", "software", "computer", "digital", "data", "internet",
      "algorithm", "artificial", "intelligence", "device", "technology",
      "tecnología", "computadora", "datos", "algoritmo", "inteligencia"),
     ("technology server room", "digital computer screens")),
    (("science", "scientific", "research", "laboratory", "molecule", "atom",
      "experiment", "microscope", "ciencia", "investigación", "laboratorio"),
     ("scientific laboratory research", "microscope laboratory closeup")),
    (("economy", "economic", "market", "money", "finance", "financial", "bank",
      "stock", "inflation", "economía", "mercado", "dinero", "finanzas"),
     ("stock market trading screens", "financial data charts")),
    (("climate", "carbon", "pollution", "environment", "warming", "emissions",
      "clima", "contaminación", "ambiente"),
     ("climate change satellite earth", "environmental research laboratory")),
    (("city", "cities", "urban", "building", "construction", "architecture",
      "ciudad", "ciudades", "urbano", "construcción", "arquitectura"),
     ("aerial city construction", "urban buildings skyline")),
    (("school", "education", "university", "study", "learning", "classroom",
      "escuela", "educación", "universidad", "estudio"),
     ("education books classroom", "school science laboratory")),
    (("music", "song", "sound", "audio", "instrument", "música", "sonido"),
     ("music studio instruments", "audio recording equipment")),
    (("food", "cooking", "kitchen", "recipe", "agriculture", "farm", "comida",
      "cocina", "agricultura", "granja"),
     ("food preparation closeup", "agriculture farm footage")),
    (("ocean", "sea", "water", "marine", "beach", "ocean", "océano", "mar",
      "agua", "playa"),
     ("ocean waves aerial", "underwater marine life")),
    (("forest", "tree", "植物", "nature", "forest", "bosque", "naturaleza"),
     ("forest aerial landscape", "nature closeup foliage")),
    (("travel", "tourism", "journey", "airport", "hotel", "viaje", "turismo"),
     ("travel destination landscape", "airport travel terminal")),
    (("history", "ancient", "civilization", "museum", "historical", "historia",
      "antigua", "civilización"),
     ("historical architecture museum", "ancient ruins landscape")),
]


def _local_sentence_query_pair(sentence_text):
    """Return aligned stock queries when Groq omits one sentence."""
    text = str(sentence_text or "")
    lowered = text.lower()
    for triggers, pair in _LOCAL_QUERY_RULES:
        if any(trigger in lowered for trigger in triggers):
            return pair

    # Last-resort queries still contain words from this exact sentence, rather
    # than using a neighboring sentence or the old unrelated FALLBACK list.
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", lowered)
    content = []
    for word in words:
        if (len(word) > 2 and word not in _LOCAL_QUERY_STOPWORDS
                and word not in content and _safe(word)):
            content.append(word)
    if len(content) >= 2:
        core = " ".join(content[:4])
        return f"{core} documentary", f"{core} educational illustration"
    return "educational concept illustration", "documentary concept visualization"


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
        raise RuntimeError(
            "Groq is required for sentence-specific visual queries; "
            "refusing to use generic footage"
        )

    n = len(sentences)
    print(f"  Groq: matching {n} sentences to visuals...")

    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    BATCH_SIZE = 50
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

    # Recover incomplete batches without changing index alignment. The first
    # request can occasionally omit a few numbered lines even when Groq is
    # otherwise healthy; re-request only those exact global sentence numbers
    # instead of aborting immediately or inserting generic footage.
    missing_indices = [i for i, query in enumerate(all_queries) if not query]
    for recovery_round in range(1, 3):
        if not missing_indices:
            break
        print(f"  Groq: recovering {len(missing_indices)} missing sentence queries "
              f"(attempt {recovery_round}/2)...")
        requested = set(missing_indices)
        numbered_missing = "\n".join(
            f"{i + 1}. {sentences[i]['text'][:140]}" for i in missing_indices
        )
        try:
            recovery = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """Return one primary and one backup STOCK FOOTAGE query for every numbered sentence. The numbers are global sentence numbers and must not be changed or skipped.

Rules:
- Queries must be in English, 3-6 words, and describe something a camera can film.
- They must clearly represent the exact sentence meaning and remain thematically related.
- For abstract or medical ideas, choose the closest findable visual category, not random nature, ocean, space, or other filler.
- No people, faces, bodies, women, religion, violence, or NSFW.
- Output ONLY these two lines per sentence:
14. primary query
14b. backup query"""},
                    {"role": "user", "content":
                     f"Recover queries for these missing global sentence numbers:\n\n{numbered_missing}"}
                ],
                model="openai/gpt-oss-120b",
                max_tokens=max(600, len(missing_indices) * 120),
                temperature=0.2,
            )
            result = recovery.choices[0].message.content
            recovered = 0
            for line in result.strip().split("\n"):
                line = line.strip()
                mb = re.match(r'^\s*(\d+)b[\.\)\-]\s*(.+)$', line, re.IGNORECASE)
                if mb:
                    global_idx = int(mb.group(1)) - 1
                    cleaned = mb.group(2).strip().strip('"\'')
                    if (global_idx in requested and 3 < len(cleaned) < 60
                            and _safe(cleaned)):
                        all_backups[global_idx] = cleaned
                    continue
                m = re.match(r'^\s*(\d+)[\.\)\-]\s*(.+)$', line)
                if not m:
                    continue
                global_idx = int(m.group(1)) - 1
                cleaned = m.group(2).strip().strip('"\'')
                if (global_idx in requested and 3 < len(cleaned) < 60
                        and _safe(cleaned)):
                    if not all_queries[global_idx]:
                        recovered += 1
                    all_queries[global_idx] = cleaned
            print(f"  Groq: recovered {recovered}/{len(missing_indices)} primary queries")
        except Exception as e:
            print(f"  Groq recovery attempt {recovery_round} failed: {e}")
        missing_indices = [i for i, query in enumerate(all_queries) if not query]

    # Final precision recovery: batch recovery can still omit one stubborn
    # sentence. Ask for each remaining sentence individually, which removes
    # numbering ambiguity without ever inserting a generic substitute.
    missing_indices = [i for i, query in enumerate(all_queries) if not query]
    for precision_round in range(1, 3):
        if not missing_indices:
            break
        print(f"  Groq: precision recovery for {len(missing_indices)} sentence(s) "
              f"(attempt {precision_round}/2)...")
        for global_idx in list(missing_indices):
            try:
                precision = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": """You are selecting one exact stock-footage query for one narration sentence.
Return exactly two lines and nothing else:
1. primary query
1b. backup query

Both must be English, 3-6 words, camera-filmable, thematically tied to the exact sentence, and safe for stock footage. Do not use generic or unrelated nature, ocean, space, landscape, people, women, religion, violence, or NSFW filler."""},
                        {"role": "user", "content":
                         f"Sentence {global_idx + 1}: {sentences[global_idx]['text'][:220]}"},
                    ],
                    model="openai/gpt-oss-120b",
                    max_tokens=120,
                    temperature=0.1,
                )
                text = precision.choices[0].message.content or ""
                primary = None
                backup = None
                for line in text.strip().split("\n"):
                    line = line.strip()
                    mb = re.match(r'^\s*1b[\.\)\-]\s*(.+)$', line, re.IGNORECASE)
                    if mb:
                        candidate = mb.group(1).strip().strip('"\'')
                        if 3 < len(candidate) < 60 and _safe(candidate):
                            backup = candidate
                        continue
                    m = re.match(r'^\s*1[\.\)\-]\s*(.+)$', line)
                    if m:
                        candidate = m.group(1).strip().strip('"\'')
                        if 3 < len(candidate) < 60 and _safe(candidate):
                            primary = candidate
                if primary:
                    all_queries[global_idx] = primary
                    all_backups[global_idx] = backup or primary
                    print(f"  Groq: precision recovered sentence {global_idx + 1}")
            except Exception as e:
                print(f"  Groq: precision recovery failed for sentence {global_idx + 1}: "
                      f"{type(e).__name__}: {str(e)[:100]}")
        missing_indices = [i for i, query in enumerate(all_queries) if not query]

    if missing_indices:
        print(
            "  WARNING: Groq still omitted sentence queries for positions "
            f"{[i + 1 for i in missing_indices]}; using aligned local stock-query fallbacks"
        )
        for missing_idx in missing_indices:
            primary, backup = _local_sentence_query_pair(
                sentences[missing_idx]["text"]
            )
            all_queries[missing_idx] = primary
            all_backups[missing_idx] = backup
            print(
                f"    Local fallback [{missing_idx + 1}]: "
                f"'{primary}' / '{backup}'"
            )

    # This should be unreachable because the local fallback always returns two
    # non-empty strings, but keep the invariant explicit if that helper is
    # changed later.
    still_missing = [i for i, query in enumerate(all_queries) if not query]
    if still_missing:
        raise RuntimeError(
            "Unable to create aligned visual queries for sentence positions "
            f"{[i + 1 for i in still_missing]}"
        )
    queries = list(all_queries)
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


def _query_attempts(index, orig_index=None):
    """Return only sentence-specific queries; never add generic footage."""
    query_index = orig_index if orig_index is not None else index
    if query_index < 0 or query_index >= len(AI_QUERIES):
        raise RuntimeError(f"No AI visual query exists for sentence {query_index + 1}")
    primary = str(AI_QUERIES[query_index] or '').strip()
    if not primary:
        raise RuntimeError(f"Primary visual query is empty for sentence {query_index + 1}")
    backup = str(AI_BACKUPS[query_index] or '').strip() if query_index < len(AI_BACKUPS) else ''
    return [primary] + ([backup] if backup and backup != primary else [])


def _sentence_query_variants(attempts):
    """Generate only semantically tied variants for persistent stock retries."""
    suffixes = ["", " cinematic", " close up", " wide shot", " aerial view", " 4k"]
    seen = set()
    for base in attempts:
        for suffix in suffixes:
            variant = f"{base}{suffix}".strip()
            if variant and variant not in seen:
                seen.add(variant)
                yield variant


# A bounded search is essential: stock providers expose finite pages, and
# cycling already-used URLs forever can otherwise keep a Kaggle job alive for
# hours. After each failed round, Groq supplies fresh sentence-specific terms.
_CLIP_QUERY_ROUNDS = 4  # initial queries plus up to three targeted Groq refreshes
_CLIP_CANDIDATES_PER_QUERY = 5


def _request_fresh_sentence_queries(sentence, previous_queries, orientation):
    """Ask Groq for new queries for one unmatched sentence only."""
    if not GROQ_KEY:
        return []
    from groq import Groq
    previous = "; ".join(previous_queries[-8:]) or "none"
    orientation_hint = "portrait/vertical" if orientation == "portrait" else "landscape"
    try:
        client = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""You are recovering stock-footage queries for one unmatched narration sentence.
Return exactly two new, different, sentence-specific {orientation_hint} stock-footage queries:
1. primary query
1b. backup query

Rules:
- English, 3-6 words, describing something a camera can film.
- Clearly represent the sentence meaning, not merely a broad topic.
- For abstract or medical ideas, choose the closest findable thematic visual.
- Do not use generic nature, ocean, space, landscape, or unrelated filler.
- No people, faces, bodies, women, religion, violence, or NSFW.
- Do not repeat any previous query.
- Output only the two numbered lines."""},
                {"role": "user", "content":
                 f"Sentence:\n{sentence}\n\nPrevious failed queries:\n{previous}"}
            ],
            model="openai/gpt-oss-120b",
            max_tokens=180,
            temperature=0.65,
        )
        text = response.choices[0].message.content or ""
        fresh = []
        for line in text.strip().split("\n"):
            match = re.match(r'^\s*(?:1b|1)[\.\)\-]\s*(.+)$', line, re.IGNORECASE)
            if not match:
                continue
            query = match.group(1).strip().strip('"\'')
            if 3 < len(query) < 60 and _safe(query) and query not in previous_queries and query not in fresh:
                fresh.append(query)
        if fresh:
            print(f"    Groq supplied {len(fresh)} fresh {orientation_hint} queries for unmatched sentence")
        return fresh[:2]
    except Exception as e:
        print(f"    Groq fresh-query recovery failed: {type(e).__name__}: {str(e)[:120]}")
        return []



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

def _mark_url_used(url):
    with _USED_URLS_LOCK:
        USED_URLS.add(url)


def _claim_url(url):
    """Atomically reserve a URL so parallel workers cannot download it twice."""
    with _USED_URLS_LOCK:
        if url in USED_URLS:
            return False
        USED_URLS.add(url)
        return True


def _search_stock_provider(provider, query, page, orientation):
    """Search one stock provider and return candidate video URLs."""
    try:
        if provider == "pexels":
            keys = [key for key in PEXELS_KEYS if key]
            if not keys:
                return []
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": random.choice(keys)},
                params={
                    "query": query,
                    "per_page": 15,
                    "page": page,
                    "orientation": orientation,
                },
                timeout=_STOCK_API_TIMEOUT,
            )
            if response.status_code != 200:
                return []

            urls = []
            for video in response.json().get("videos", []):
                files = video.get("video_files", [])
                if orientation == "portrait":
                    preferred = [
                        file for file in files
                        if file.get("height", 0) > file.get("width", 0)
                        and file.get("height", 0) >= 1280
                    ]
                    candidates = preferred or [
                        file for file in files
                        if file.get("quality") in ["hd", "large"]
                    ]
                else:
                    candidates = [
                        file for file in files
                        if file.get("quality") == "hd"
                        and file.get("width", 0) >= 1280
                    ]
                    if not candidates:
                        candidates = [
                            file for file in files
                            if file.get("quality") in ["hd", "large"]
                        ]

                if candidates:
                    url = random.choice(candidates).get("link")
                    if url:
                        urls.append(url)
            return urls

        if provider == "pixabay":
            keys = [key for key in PIXABAY_KEYS if key]
            if not keys:
                return []
            response = requests.get(
                "https://pixabay.com/api/videos/",
                params={
                    "key": random.choice(keys),
                    "q": query,
                    "per_page": 15,
                    "page": page,
                },
                timeout=_STOCK_API_TIMEOUT,
            )
            if response.status_code != 200:
                return []

            urls = []
            for video in response.json().get("hits", []):
                video_files = video.get("videos", {})
                selected = video_files.get("large", video_files.get("medium", {}))
                url = selected.get("url") if selected else None
                if url:
                    urls.append(url)
            return urls
    except Exception:
        # A provider outage/timeout should only remove that provider's
        # candidates; the other provider and the caller's query retries remain.
        return []

    return []


def _search_stock_urls(query, page, orientation, limit=None):
    """Search providers concurrently and atomically claim only needed URLs."""
    providers = []
    if any(PEXELS_KEYS):
        providers.append("pexels")
    if any(PIXABAY_KEYS):
        providers.append("pixabay")
    if not providers:
        return []

    results = {provider: [] for provider in providers}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(_search_stock_provider, provider, query, page, orientation): provider
            for provider in providers
        }
        for future in concurrent.futures.as_completed(futures):
            provider = futures[future]
            try:
                results[provider] = future.result()
            except Exception:
                results[provider] = []

    urls = []
    for provider in providers:
        for url in results[provider]:
            if limit is not None and len(urls) >= limit:
                return urls
            if url and url not in urls and _claim_url(url):
                urls.append(url)
    return urls


def search_and_download_vertical(query, idx, duration, tag="", verify=True, normalize=True, page=1):
    """
    Same as search_and_download but requests portrait/vertical source video
    where possible and always crops/scales to 1080x1920 (9:16) for Shorts.
    Uses a distinct USED_URLS-safe idx namespace via `tag` so long-video and
    shorts clip fetching never collide on temp filenames.
    """
    urls = _search_stock_urls(query, page, "portrait", _CLIP_CANDIDATES_PER_QUERY)

    # Each search call is bounded; the sentence worker keeps retrying with
    # fresh pages and semantically tied variants until MiniCPM accepts one.
    for url in urls[:_CLIP_CANDIDATES_PER_QUERY]:
        try:
            raw = TEMP_DIR / f"raw_s{tag}_{idx}.mp4"
            out = TEMP_DIR / f"clip_s{tag}_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(raw) < 5000:
                try: os.remove(raw)
                except OSError: pass
                continue

            # Verify the raw vertical video before paying the normalization
            # cost. Shorts use the same authoritative MiniCPM checks as the
            # landscape pipeline.
            if verify:
                matches = verify_clip_matches_query(raw, query)
                if not matches:
                    print(f"    Rejected short clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(raw)
                    except OSError: pass
                    continue

            if not normalize:
                _mark_url_used(url)
                return str(raw)

            normalized = _normalize_vertical_clip(raw, out, duration)
            try: os.remove(raw)
            except OSError: pass
            if not normalized:
                continue

            _mark_url_used(url)
            return normalized
        except Exception:
            for stale in (raw, out):
                try:
                    if stale.exists(): stale.unlink()
                except OSError:
                    pass
            continue
    return None


def _find_verified_normalized_clip(sent, index, orientation, tag=""):
    """Find an exact sentence match using bounded stock/Groq retry rounds."""
    duration = max(2.5 if orientation == "portrait" else 3.5,
                   sent['end'] - sent['start'])
    query_index = sent.get('orig_idx', index) if orientation == "portrait" else index
    queries = _query_attempts(index, query_index)
    previous_queries = list(queries)

    for round_no in range(_CLIP_QUERY_ROUNDS):
        page = round_no + 1
        print(f"    {orientation.title()} clip {index}: query round {round_no + 1}/{_CLIP_QUERY_ROUNDS}")
        for query in queries[:2]:
            try:
                if orientation == "portrait":
                    raw = search_and_download_vertical(
                        query, index, duration, tag=tag, verify=False,
                        normalize=False, page=page,
                    )
                else:
                    raw = search_and_download(
                        query, index, duration, verify=False, page=page,
                    )
                if not raw:
                    continue

                if not verify_clip_matches_query(raw, query):
                    try: os.remove(raw)
                    except OSError: pass
                    continue

                output_name = (f"clip_s{tag}_{index}.mp4" if orientation == "portrait"
                               else f"clip_{index}.mp4")
                normalized = (_normalize_vertical_clip if orientation == "portrait"
                              else _normalize_landscape_clip)(
                    raw, TEMP_DIR / output_name, duration
                )
                try: os.remove(raw)
                except OSError: pass
                if normalized and _normalized_duration_is_usable(normalized, duration):
                    print(f"    {orientation.title()} clip {index}: verified and normalized "
                          f"in round {round_no + 1}")
                    return index, normalized
                if normalized:
                    try: os.remove(normalized)
                    except OSError: pass
            except Exception as e:
                print(f"    {orientation.title()} clip {index}: candidate error "
                      f"({type(e).__name__}: {str(e)[:100]})")

        if round_no + 1 >= _CLIP_QUERY_ROUNDS:
            break
        fresh = _request_fresh_sentence_queries(
            sent['text'], previous_queries, orientation
        )
        if not fresh:
            print(f"    {orientation.title()} clip {index}: Groq returned no new queries; stopping bounded search")
            break
        previous_queries.extend(fresh)
        queries = fresh

    raise RuntimeError(
        f"No verified {orientation} clip found for sentence position {index + 1} "
        f"after {_CLIP_QUERY_ROUNDS} query rounds; no substitute permitted"
    )


def process_short_clip(args):
    i, sent, tag = args
    return _find_verified_normalized_clip(sent, i, "portrait", tag)


def render_short(short_idx, sentences_slice, audio_path, ass_path, logo_path, out_path,
                 release_verifier=True):
    """
    Render a single 1080x1920 short: fetch fresh vertical stock clips for
    this segment's sentences, concat, overlay a LARGE left-side logo (shorts
    need bigger branding since screen real estate is smaller/closer-viewed),
    and burn vertical-tuned subtitles.
    """
    global _nvenc_runtime_failed
    tag = f"sh{short_idx}"
    n = len(sentences_slice)
    print(f"\n  Short {short_idx+1}: fetching {n} vertical clips...")

    clips = [None] * n

    # Each worker searches, verifies, and immediately normalizes its own
    # sentence. This prevents raw clips with inconsistent source durations
    # from accumulating and later producing a long concatenated chunk.
    print(f"  Short {short_idx+1}: streaming verified clips directly into normalized output...")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(process_short_clip, (i, sent, tag)): i
            for i, sent in enumerate(sentences_slice)
        }
        for future in concurrent.futures.as_completed(futures):
            i, clip = future.result()
            clips[i] = clip
            completed += 1
            print(f"    Short {short_idx+1}: completed {completed}/{n} exact clips")

    if release_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Short {short_idx+1}: verifier workers "
              f"{'released before final encoding' if release_verifier else 'retained for the next short'}")

    missing = [i for i, clip in enumerate(clips)
               if not clip or not os.path.exists(clip)]
    if missing:
        print(f"  Short {short_idx+1}: missing normalized clips at positions {missing}; refusing substitution")
        return False

    list_path = f"list_{tag}.txt"
    visual_path = f"visual_{tag}.mp4"
    with open(list_path,"w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {visual_path}",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists(visual_path):
        fallback_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path] + _enc_args() + [visual_path]
        subprocess.run(fallback_cmd, capture_output=True, timeout=60)
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
        cmd = ["ffmpeg","-y"] + _hwaccel_args() + ["-i",visual_path,"-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    else:
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y"] + _hwaccel_args() + ["-i",visual_path,"-i",str(audio_path),
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
        
        # Keep chunks sentence-aligned, but reduce the number of expensive
        # autoregressive model invocations. 240 characters is still short
        # enough for Chatterbox's text context while avoiding the large
        # per-call startup cost of the old 160-character setting.
        try:
            tts_chunk_chars = max(160, int(os.environ.get("TTS_CHUNK_CHARS", "240")))
        except (TypeError, ValueError):
            tts_chunk_chars = 240
        chunks, buf, blen = [], [], 0
        for s in sents:
            if blen + len(s) > tts_chunk_chars and buf:
                chunks.append(' '.join(buf)); buf, blen = [s], len(s)
            else: buf.append(s); blen += len(s)+1
        if buf: chunks.append(' '.join(buf))
        print(f"  {len(sents)} sentences -> {len(chunks)} chunks (target <= {tts_chunk_chars} chars)")

        def _generate_tts_piece(piece):
            with torch.inference_mode():
                if IS_SPANISH:
                    waveform = model.generate(
                        piece.replace('"',''), audio_prompt_path=str(ref_audio),
                        language_id="es", exaggeration=0.4, cfg_weight=0.65
                    )
                else:
                    waveform = model.generate(
                        piece.replace('"',''), audio_prompt_path=str(ref_audio),
                        exaggeration=0.4, cfg_weight=0.65
                    )
            return waveform.cpu()

        wavs = []
        for i, c in enumerate(chunks):
            if i % 5 == 0:
                update_status(18 + int((i / max(1, len(chunks))) * 27), f"TTS {i+1}/{len(chunks)}")
            try:
                wavs.append(_generate_tts_piece(c))
            except Exception as e:
                # A longer chunk can fail on an older Chatterbox build or
                # unusual text. Retry it as smaller sentence/word pieces so
                # increasing the target never silently loses narration.
                print(f"  TTS chunk {i+1} retrying smaller pieces ({str(e)[:100]})")
                fallback_parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', c) if p.strip()]
                if len(fallback_parts) <= 1:
                    words = c.split()
                    midpoint = max(1, len(words) // 2)
                    fallback_parts = [' '.join(words[:midpoint]), ' '.join(words[midpoint:])]
                recovered = 0
                for part in fallback_parts:
                    if not part:
                        continue
                    try:
                        wavs.append(_generate_tts_piece(part))
                        recovered += 1
                    except Exception as sub_error:
                        print(f"  TTS sub-piece skipped ({str(sub_error)[:80]})")
                if not recovered:
                    print(f"  TTS chunk {i+1} could not be recovered")

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
        
        try:
            enhance_chunk_seconds = max(20, int(os.environ.get("ENHANCE_CHUNK_SECONDS", "40")))
        except (TypeError, ValueError):
            enhance_chunk_seconds = 40
        chunk_s = enhance_chunk_seconds * osr
        parts = []; esr = 44100
        total = dwav.shape[1]
        n_chunks = (total + chunk_s - 1) // chunk_s
        print(f"  Processing {n_chunks} enhancement chunks ({enhance_chunk_seconds}s target)...")

        def _enhance_piece(chunk, label):
            try:
                hw, piece_sr = re_enhance(
                    dwav=chunk.squeeze(0), sr=osr, device=device, lambd=0.6
                )
                piece_sr = int(piece_sr)
                hw = hw.detach().cpu()
                if piece_sr != 44100:
                    hw = torchaudio.transforms.Resample(piece_sr, 44100)(hw.unsqueeze(0)).squeeze(0)
                print(f"    Chunk {label}: OK (44100Hz)")
                return hw.unsqueeze(0), 44100
            except Exception as e:
                # If a larger chunk exceeds available VRAM, retry it as two
                # original-size pieces before falling back to resampling.
                # This preserves enhancement whenever possible and avoids
                # turning a memory optimization into a silent quality loss.
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if chunk.shape[1] > 20 * osr + 1:
                    midpoint = chunk.shape[1] // 2
                    left, left_sr = _enhance_piece(chunk[:, :midpoint], f"{label}a")
                    right, right_sr = _enhance_piece(chunk[:, midpoint:], f"{label}b")
                    if left_sr == right_sr:
                        return torch.cat([left, right], dim=1), left_sr
                print(f"    Chunk {label}: fallback ({str(e)[:80]})")
                fallback = torchaudio.transforms.Resample(osr, 44100)(chunk).cpu()
                return fallback, 44100

        for chunk_index, i in enumerate(range(0, total, chunk_s), 1):
            piece, piece_sr = _enhance_piece(dwav[:, i:i+chunk_s], chunk_index)
            parts.append(piece)
            esr = piece_sr

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
# MiniCPM-V 4.5 is the production local video verifier. Use the official
# pre-quantized NF4 checkpoint so one independent worker fits on each Kaggle
# T4. The compatibility names are retained because the rest of the pipeline
# already uses _llava_* worker/release functions.
_llava_workers = []
_llava_next_worker = 0
_llava_worker_select_lock = threading.Lock()
_llava_load_lock = threading.Lock()
_gpu_lock = threading.Lock()
_MINICPM_NUM_FRAMES = 4
_MINICPM_MAX_NEW_TOKENS = 24
_MINICPM_MODEL_PATH = "openbmb/MiniCPM-V-4_5-int4"
# T4 uses float16 reliably; the checkpoint's embedded BNB config also uses
# float16 compute. Do not request bfloat16 on this hardware.
_MINICPM_DTYPE = torch.float16
_MINICPM_TIME_SCALE = 0.1
_MINICPM_PACKING = 4

def _load_llava_worker(gpu_index):
    """Load one official MiniCPM-V 4.5 int4 verifier on one CUDA device."""
    from transformers import AutoModel, AutoProcessor

    if gpu_index is None:
        raise RuntimeError("MiniCPM verification requires a CUDA GPU")
    device = f"cuda:{gpu_index}"
    print(f"  Loading MiniCPM-V 4.5 int4 verifier on {device}...")
    processor = AutoProcessor.from_pretrained(
        _MINICPM_MODEL_PATH,
        trust_remote_code=True,
    )
    # The checkpoint config contains the official bitsandbytes NF4 settings.
    # Passing device_map pins this independent model to exactly one GPU.
    model = AutoModel.from_pretrained(
        _MINICPM_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=_MINICPM_DTYPE,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    return {
        "gpu_index": gpu_index,
        "device": torch.device(device),
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "lock": threading.Lock(),
        "prepare_lock": threading.Lock(),
    }

def _load_llava():
    global _llava_workers
    if _llava_workers:
        return
    with _llava_load_lock:
        if _llava_workers:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM verifier requires CUDA")
        loaded = []
        for gpu_index in range(torch.cuda.device_count()):
            try:
                loaded.append(_load_llava_worker(gpu_index))
            except Exception as e:
                print(f"  WARNING: MiniCPM worker on cuda:{gpu_index} failed: "
                      f"{type(e).__name__}: {str(e)[:180]}")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        if not loaded:
            raise RuntimeError("No MiniCPM verifier worker could be loaded")
        _llava_workers = loaded
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        devices = ", ".join(str(worker["device"]) for worker in loaded)
        print(f"  MiniCPM verifier workers ready: {len(loaded)} ({devices}); "
              f"{_MINICPM_NUM_FRAMES} Decord frames, "
              f"{_MINICPM_PACKING}-frame temporal packing, "
              f"{_MINICPM_MAX_NEW_TOKENS} output tokens")

def _next_llava_worker():
    global _llava_next_worker
    with _llava_worker_select_lock:
        if not _llava_workers:
            raise RuntimeError("MiniCPM verifier workers are not loaded")
        worker = _llava_workers[_llava_next_worker % len(_llava_workers)]
        _llava_next_worker += 1
        return worker

def _prepare_llava_inputs(clip_path, prompt, processor):
    """Sample four Decord frames and build MiniCPM temporal metadata."""
    import numpy as np
    from PIL import Image
    from decord import VideoReader, cpu

    reader = VideoReader(str(clip_path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count == 0:
        raise RuntimeError("Decord returned an empty video")
    indices = np.linspace(
        0, frame_count - 1, _MINICPM_NUM_FRAMES, dtype=np.int64
    )
    frames = reader.get_batch(indices).asnumpy()
    fps = float(reader.get_avg_fps() or 0.0)
    del reader
    if fps <= 0:
        fps = 1.0
    frame_images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    del frames

    # MiniCPM-V 4.5 requires temporal_ids grouped in the same packing layout
    # as the frame list. Four frames in one group lets its 3D resampler jointly
    # reason over the whole candidate clip instead of treating frames as four
    # unrelated still images.
    timestamps = indices.astype(np.float32) / fps
    temporal_ids = np.rint(timestamps / _MINICPM_TIME_SCALE).astype(np.int32)
    temporal_ids = [[int(value) for value in temporal_ids.tolist()]]
    return {"frames": frame_images, "temporal_ids": temporal_ids}



def _verify_clip_matches_query_legacy(clip_path, query, filter_women=True):
    """Compatibility alias for callers that used the previous verifier name."""
    return verify_clip_matches_query(clip_path, query, filter_women=filter_women)

    # Historical implementation retained below only as unreachable reference.
    """
    Legacy MiniCPM verifier retained only as historical code; production uses MiniCPM below.
    just a single frame) actually matches the intended search query, AND
    whether it shows a woman (if filter_women is True) - combined into ONE
    model call for efficiency rather than two separate passes.

    This is the real fix for stock footage that "downloads fine" but is
    visually unrelated to the query - there was previously ZERO check
    that a downloaded clip actually looked like what it was searched for,
    and no check on content restrictions beyond the query TEXT (a neutral
    query like "person walking city" could still return a clip showing a
    woman, since the restriction was only ever applied to search terms,
    not actual visual content).

    Returns True if the clip should be USED (topic matches AND, if
    filter_women, no woman detected), False if it should be rejected.

    Verification is fail-closed; any model-load or per-clip inference error
    rejects the candidate and the sentence worker keeps searching.
    - If the MODEL ITSELF fails to load (happens once, affects every
      clip), fails OPEN (returns True) - otherwise the whole pipeline
      would produce zero usable clips if the model can't load at all,
      which is worse than proceeding unverified.
    - If a PER-CLIP generation call errors out (transient, specific to
      one clip), fails CLOSED (returns False) - the caller will try the
      next candidate URL/query instead of silently accepting an
      unverified clip. This matches the decision that a clip should never
      slip through unverified when verification itself is working but
      this particular attempt failed.
    """
    verify_started = time.perf_counter()
    try:
        _load_llava()
        worker = _next_llava_worker()
    except Exception as e:
        print(f"    Video verification model unavailable ({str(e)[:100]}), rejecting clip (fail-closed)")
        return False

    try:
        with worker["lock"]:
            processor = worker["processor"]
            model = worker["model"]
            device = worker["device"]
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

            inputs_started = time.perf_counter()
            inputs = _prepare_llava_inputs(clip_path, prompt, processor)
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    dtype = model.dtype if value.is_floating_point() else value.dtype
                    inputs[key] = value.to(device=device, dtype=dtype)
            inputs_seconds = time.perf_counter() - inputs_started

            # Keep generation synchronous, exactly as in the tested notebook.
            # CUDA generate() cannot be safely cancelled from a watchdog
            # thread; leaving such a thread alive can continue using the
            # shared model and GPU after this call has returned, preventing
            # clip workers from completing.
            generation_started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=_MINICPM_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            generation_seconds = time.perf_counter() - generation_started
            print(f"    MiniCPM {device} timing: prepare={inputs_seconds:.1f}s generate={generation_seconds:.1f}s total={time.perf_counter() - verify_started:.1f}s")

            full_text = processor.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # Response follows "assistant\n" in the decoded text (chat
            # template format) - slice to isolate just the model's answer.
            answer = full_text.split("assistant")[-1].strip("\n: ").upper()

            del inputs, out

            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            line1 = lines[0] if len(lines) > 0 else ""
            line2 = lines[1] if len(lines) > 1 else ""

            # Parse line 1 (topic match) and line 2 (woman detection)
            # independently - don't just search the whole blob for "YES",
            # since that would conflate the two answers if the model
            # returns e.g. "1. NO / 2. YES" (topic mismatch AND a woman -
            # searching the whole string for "YES" would wrongly pass it).
            topic_match = "YES" in line1 and "NO" not in line1
            has_woman = filter_women and "YES" in line2

            if has_woman:
                print(f"    Rejected clip for '{query[:40]}' (woman detected in frame)")
                return False
            return topic_match
    except Exception as e:
        # Per-call error (not a model-load failure) - fail CLOSED per the
        # accuracy-first decision: reject this clip so the caller tries
        # the next candidate rather than silently accepting an unverified one.
        print(f"    Visual verification error for '{query[:40]}' ({str(e)[:60]}), rejecting clip (fail-closed)")
        return False


def verify_clip_matches_query(clip_path, query, filter_women=True):
    """Verify one clip with MiniCPM-V 4.5; reject every uncertain result."""
    verify_started = time.perf_counter()
    try:
        _load_llava()
        worker = _next_llava_worker()
    except Exception as e:
        print(f"    MiniCPM verification unavailable ({str(e)[:100]}), "
              "rejecting clip (fail-closed)")
        return False

    if filter_women:
        prompt = (
            'Return exactly one JSON object and no other text: '
            '{"match":"YES" or "NO","woman":"YES" or "NO"}. '
            f'Does this video visually match the concept "{query}"? '
            'Set match to YES for a reasonable thematic match and NO only when clearly unrelated. '
            'Set woman to YES only when a visible woman or women appear in any frame.'
        )
    else:
        prompt = (
            'Return exactly one JSON object and no other text: '
            '{"match":"YES" or "NO","woman":"NO"}. '
            f'Does this video visually match the concept "{query}"? '
            'Set match to YES for a reasonable thematic match and NO only when clearly unrelated.'
        )

    try:
        # The processor is shared by no other task, but its preprocessing state
        # is still protected because a worker may be selected by multiple clip
        # threads over the lifetime of the run.
        with worker["prepare_lock"]:
            inputs_started = time.perf_counter()
            prepared = _prepare_llava_inputs(clip_path, prompt, worker["processor"])
            inputs_seconds = time.perf_counter() - inputs_started

        frames = prepared["frames"]
        temporal_ids = prepared["temporal_ids"]
        msgs = [{"role": "user", "content": frames + [prompt]}]

        # MiniCPM-V's custom chat API performs the processor conversion and
        # generation together. Serialize that call per model replica, while
        # allowing different GPU workers to run concurrently.
        with worker["lock"]:
            generation_started = time.perf_counter()
            with torch.inference_mode():
                answer = worker["model"].chat(
                    msgs=msgs,
                    tokenizer=worker["tokenizer"],
                    processor=worker["processor"],
                    max_new_tokens=_MINICPM_MAX_NEW_TOKENS,
                    sampling=False,
                    max_slice_nums=1,
                    use_image_id=False,
                    temporal_ids=temporal_ids,
                    enable_thinking=False,
                )
            generation_seconds = time.perf_counter() - generation_started

        total_seconds = time.perf_counter() - verify_started
        print(f"    MiniCPM {worker['device']} timing: prepare={inputs_seconds:.2f}s "
              f"generate={generation_seconds:.2f}s total={total_seconds:.2f}s")

        answer = answer if isinstance(answer, str) else str(answer or "")
        candidate = re.search(r"\{.*\}", answer, re.DOTALL)
        try:
            data = json.loads(candidate.group(0)) if candidate else None
            match_value = str(data.get("match", "")).upper() if data else ""
            woman_value = str(data.get("woman", "")).upper() if data else ""
        except (ValueError, TypeError, AttributeError):
            data = None
            match_value = ""
            woman_value = ""

        if match_value not in {"YES", "NO"} or woman_value not in {"YES", "NO"}:
            print(f"    MiniCPM returned malformed verification JSON for "
                  f"'{query[:40]}'; rejecting")
            return False
        if filter_women and woman_value == "YES":
            print(f"    Rejected clip for '{query[:40]}' (woman detected in frame)")
            return False
        return match_value == "YES"
    except Exception as e:
        print(f"    MiniCPM verification error for '{query[:40]}' "
              f"({type(e).__name__}: {str(e)[:100]}), rejecting clip (fail-closed)")
        return False


def _normalize_clip_with_recovery(raw_path, output_path, duration, vf, label):
    """Encode one accepted clip, retrying with CPU only if NVENC fails."""
    global _nvenc_runtime_failed
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    partial_path = output_path.with_name(output_path.stem + ".part" + output_path.suffix)

    if not raw_path.exists() or raw_path.stat().st_size < 5000:
        print(f"    {label} input is missing or too small: {raw_path}")
        return None

    try:
        if partial_path.exists(): partial_path.unlink()
        if output_path.exists(): output_path.unlink()
    except OSError:
        pass

    encoders = []
    if USE_GPU and not _nvenc_runtime_failed:
        encoders.append(("NVENC", _enc_args()))
    # This is an automatic recovery path, not the normal path. It prevents a
    # transient/unsupported NVENC initialization from discarding every clip
    # that MiniCPM already accepted.
    encoders.append(("CPU fallback", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]))

    for encoder_name, encoder in encoders:
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", str(raw_path), "-t", str(duration),
            "-vf", vf,
        ] + encoder + ["-pix_fmt", "yuv420p", "-an", str(partial_path)]
        started = time.perf_counter()
        try:
            with _gpu_lock if encoder_name == "NVENC" else _nullcontext():
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=120
                )
        except Exception as e:
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"    {label} {encoder_name} exception: {type(e).__name__}: {str(e)[:160]}")
            result = None

        if result is not None and result.returncode == 0 and partial_path.exists() and partial_path.stat().st_size > 2000:
            try:
                os.replace(partial_path, output_path)
                elapsed = time.perf_counter() - started
                print(f"    {label}: {output_path.name} in {elapsed:.1f}s ({encoder_name})")
                return str(output_path)
            except OSError as e:
                print(f"    {label} publish failed ({encoder_name}): {type(e).__name__}: {str(e)[:120]}")

        if result is not None:
            stderr = (result.stderr or "").strip()
            detail = " | ".join(stderr.splitlines()[-4:])
            if encoder_name == "NVENC":
                # Stop retrying a broken runtime encoder for every remaining
                # clip. The current clip is immediately retried with CPU,
                # while later clips use the same reliable fallback directly.
                _nvenc_runtime_failed = True
            print(f"    {label} {encoder_name} failed ({result.returncode}): {detail[-700:] or 'no FFmpeg stderr'}")
        try: partial_path.unlink()
        except OSError: pass

    return None


def _normalize_landscape_clip(raw_path, output_path, duration):
    """Normalize an accepted landscape clip with validated GPU encoding."""
    vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30"
    return _normalize_clip_with_recovery(
        raw_path, output_path, duration, vf, "Landscape normalization"
    )


def _normalize_vertical_clip(raw_path, output_path, duration):
    """Normalize an accepted vertical clip with the same recovery policy."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30"
    return _normalize_clip_with_recovery(
        raw_path, output_path, duration, vf, "Vertical normalization"
    )


def _normalized_duration_is_usable(path, target_duration):
    """Reject clips whose encoded duration could create concat timing drift."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        actual = float(probe.stdout.strip())
        minimum = max(0.1, target_duration - 0.15)
        maximum = target_duration + 0.15
        if actual < minimum or actual > maximum:
            print(f"    Normalized duration {actual:.2f}s is outside target {target_duration:.2f}s; rejecting and retrying")
            return False
        return True
    except Exception as e:
        print(f"    Could not validate normalized duration ({type(e).__name__}); rejecting and retrying")
        return False


def _nullcontext():
    """Tiny local context manager to avoid importing contextlib in hot code."""
    class _Context:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    return _Context()


def _release_llava_for_encoding():
    """Release every verifier model before final/long encoding stages."""
    global _llava_workers
    with _llava_load_lock:
        workers = _llava_workers
        _llava_workers = []
    for worker in workers:
        try:
            del worker["model"]
            del worker["processor"]
            del worker["tokenizer"]
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def search_and_download(query, idx, duration, verify=True, page=1):
    """Download one bounded page of candidates; caller controls retry rounds."""
    urls = _search_stock_urls(query, page, "landscape", _CLIP_CANDIDATES_PER_QUERY)
    
    # Only a small bounded candidate set is downloaded in this round; the
    # outer finder requests fresh Groq terms if these candidates fail.
    for candidate_no, url in enumerate(urls[:_CLIP_CANDIDATES_PER_QUERY], 1):
        try:
            candidate_started = time.perf_counter()
            raw = TEMP_DIR / f"raw_{idx}.mp4"
            out = TEMP_DIR / f"clip_{idx}.mp4"
            download_started = time.perf_counter()
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            download_seconds = time.perf_counter() - download_started
            if os.path.getsize(raw) < 5000:
                try: os.remove(raw)
                except OSError: pass
                continue

            if not verify:
                # Main streaming path: return the raw download immediately.
                # MiniCPM will inspect it first; rejected candidates never pay
                # the expensive normalization cost.
                print(f"    Clip {idx} candidate {candidate_no}: download={download_seconds:.1f}s raw-ready={time.perf_counter() - candidate_started:.1f}s")
                _mark_url_used(url)
                return str(raw)

            # Non-streaming callers (for example final audio-gap padding) still
            # receive a normalized, verified clip as before.
            normalized = _normalize_landscape_clip(raw, out, duration)
            if not normalized:
                try: os.remove(raw)
                except OSError: pass
                continue
            try: os.remove(raw)
            except OSError: pass

            if verify:
                matches = verify_clip_matches_query(normalized, query)
                if not matches:
                    print(f"    Rejected clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(normalized)
                    except OSError: pass
                    continue

            _mark_url_used(url)
            return normalized
        except Exception as e:
            for stale in (raw, out):
                try:
                    if stale.exists(): stale.unlink()
                except OSError:
                    pass
            print(f"    Clip {idx} query '{query[:40]}' failed: {type(e).__name__}: {str(e)[:100]}")
            continue
    # The outer sentence worker retries this query on fresh pages/variants.
    print(f"    Clip {idx} query '{query[:40]}' produced no usable candidate on this page")
    return None

def process_landscape_clip(args):
    i, sent, _attempts = args
    return _find_verified_normalized_clip(sent, i, "landscape")


def prepare_clip_candidate(args):
    """Compatibility wrapper retained for external callers."""
    i, sent, attempt, query = args
    clip = search_and_download(query, i, max(3.5, sent['end'] - sent['start']), verify=False)
    return i, attempt, query, clip


# ==========================================
# 7. RENDER ENGINE (GPU-Accelerated)
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_sub, keep_verifier=False):
    global _nvenc_runtime_failed
    n = len(sentences)
    clips = [None] * n
    print(f"\n  Rendering {n} clips with bounded Groq re-query rounds and streaming verification/normalization...")

    completed = 0
    update_status(55, f"Finding exact verified clips (0/{n})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(process_landscape_clip, (i, sent, None)): i
            for i, sent in enumerate(sentences)
        }
        for future in concurrent.futures.as_completed(futures):
            i, clip = future.result()
            clips[i] = clip
            completed += 1
            update_status(
                55 + int((completed / max(1, n)) * 25),
                f"Exact clips verified and normalized ({completed}/{n})...",
            )

    if not keep_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Verifier workers "
              f"{'retained through final encoding' if keep_verifier else 'released before final encoding'}")

    missing = [i for i, clip in enumerate(clips)
               if not clip or not os.path.exists(clip)]
    if missing:
        print(f"  Missing normalized clips at positions {missing}; refusing substitution")
        return False

    # Every entry is a verified, duration-trimmed, normalized clip. No nearest
    # clip, gap clip, or generic footage is allowed into the concat list.
    
    # Concat (stream copy)
    print("  Concatenating...")
    with open("list.txt","w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists("visual.mp4"):
        # Stream-copy concat normally needs no GPU and is nearly instant. If
        # source timestamps/codecs prevent it, fall back to GPU encoding when
        # NVENC is available rather than silently returning to libx264 CPU.
        fallback_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt"] + _enc_args() + ["visual.mp4"]
        subprocess.run(fallback_cmd, capture_output=True)
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
        print(f"  Concatenated visual is {adur - vdur:.2f}s short; re-encoding the already-verified trimmed clips")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", "visual_reencoded.mp4"],
            capture_output=True, timeout=300,
        )
        if os.path.exists("visual_reencoded.mp4"):
            os.replace("visual_reencoded.mp4", "visual.mp4")
            vdur = _probe_dur("visual.mp4")
        if vdur < adur - 0.5:
            print(f"  Verified clips still produce a {adur - vdur:.2f}s duration deficit; refusing unmatched padding")
            return False

    # Final render: visual.mp4 is already normalized to 1920x1080, so do not
    # rescale the entire nine-minute stream again. Burn the logo/subtitles and
    # encode with NVENC; Qwen remains resident when Shorts will reuse it.
    if not keep_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Final render: verifier workers "
              f"{'retained for Shorts' if keep_verifier else 'released;'} attempting NVENC")

    update_status(85, "Rendering final video (1080p + subs)...")
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = (f"[0:v]setsar=1[bg];"
                f"[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];"
                f"[wl]subtitles='{ass_esc}'[v];"
                f"[2:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        input_args = ["-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path)]
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        filt = (f"[0:v]setsar=1[bg];"
                f"[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        input_args = ["-i", "visual.mp4", "-i", str(audio_path)]
        maps = ["-map", "[v]", "-map", "[a]"]

    encoders = []
    if USE_GPU:
        # Do not request CUDA decode here. Subtitle rendering is CPU-side and
        # the already-normalized input needs no GPU scaling; keeping decode
        # on CPU leaves more VRAM for NVENC and makes the P100 path reliable.
        encoders.append(("NVENC", [
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M",
        ], 300))
    encoders.append(("CPU fallback", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"], 1800))

    partial_out = Path(str(out_sub) + ".part.mp4")
    for encoder_name, encoder_args, timeout_seconds in encoders:
        for stale in (partial_out, out_sub):
            try:
                if Path(stale).exists(): Path(stale).unlink()
            except OSError:
                pass

        cmd = (['ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error']
               + input_args + ['-filter_complex', filt] + maps + encoder_args
               + ['-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                  '-shortest', str(partial_out)])
        started = time.perf_counter()
        try:
            # Serialize NVENC with any other possible GPU work. CPU fallback
            # does not need the lock and can continue without GPU contention.
            with _gpu_lock if encoder_name == "NVENC" else _nullcontext():
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_seconds
                )
        except Exception as e:
            result = None
            detail = f"{type(e).__name__}: {str(e)[:240]}"
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"  Final render {encoder_name} exception after {time.perf_counter() - started:.1f}s: {detail}")
        else:
            detail = " | ".join((result.stderr or "").splitlines()[-4:])[-900:]
            if result.returncode == 0 and partial_out.exists() and partial_out.stat().st_size > 10000:
                try:
                    os.replace(partial_out, out_sub)
                    elapsed = time.perf_counter() - started
                    print(f"  Final: {os.path.getsize(out_sub)/(1024**2):.0f}MB in {elapsed:.1f}s ({encoder_name})")
                    return True
                except OSError as e:
                    detail = f"output publish failed: {type(e).__name__}: {e}"
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"  Final render {encoder_name} failed after {time.perf_counter() - started:.1f}s: {detail or 'no FFmpeg stderr'}")

        try:
            if partial_out.exists(): partial_out.unlink()
        except OSError:
            pass

    print("  Final render failed with both NVENC and CPU fallback")
    return False



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


def _prepare_short_assets(short_index, script_text, short_audio):
    """Transcribe and prepare one short without mutating global query state."""
    short_sentences, short_word_data = [], []
    if ASSEMBLY_KEY:
        try:
            tx_config = aai.TranscriptionConfig(
                language_code="es" if IS_SPANISH else "en",
                punctuate=True,
                format_text=True,
            )
            tx = aai.Transcriber(config=tx_config).transcribe(str(short_audio))
            if tx.status != aai.TranscriptStatus.error:
                for sentence in tx.get_sentences():
                    short_sentences.append({
                        "text": sentence.text,
                        "start": sentence.start / 1000,
                        "end": sentence.end / 1000,
                    })
                if short_sentences:
                    short_sentences[-1]["end"] += 0.3
                for word in tx.words:
                    short_word_data.append({
                        "text": word.text,
                        "start": word.start / 1000,
                        "end": word.end / 1000,
                    })
        except Exception as e:
            print(f"  Short {short_index+1}: transcription error ({e}), using estimated timing")

    if not short_sentences:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(short_audio)],
                capture_output=True, text=True, timeout=15,
            )
            total_duration = float(probe.stdout.strip())
        except Exception:
            total_duration = SHORT_DUR_TARGET
        parts = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", script_text)
            if len(part.strip()) > 2
        ] or [script_text]
        per_sentence = total_duration / len(parts)
        short_sentences = [
            {"text": part, "start": i * per_sentence, "end": (i + 1) * per_sentence}
            for i, part in enumerate(parts)
        ]

    for sentence_index, sentence in enumerate(short_sentences):
        sentence["orig_idx"] = sentence_index

    short_ass = TEMP_DIR / f"short_{short_index}_subs.ass"
    create_subtitles(
        short_sentences,
        short_ass,
        word_data=short_word_data if short_word_data else None,
        style_set=SHORT_SUBTITLE_STYLES,
        play_res=(1080, 1920),
        max_chars=20,
    )
    short_queries, short_backups = generate_queries_for_sentences(short_sentences)
    return {
        "sentences": short_sentences,
        "word_data": short_word_data,
        "ass": short_ass,
        "queries": short_queries,
        "backups": short_backups,
    }


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

# Start the independent Shorts script request while the main video pipeline
# prepares its own queries and subtitles. This is network/Groq work and does
# not touch the GPU or shared audio files.
shorts_eligible = bool(
    sentences and sentences[-1]["end"] >= SHORT_DUR_TARGET * 0.6
)
short_script_executor = None
short_script_future = None
if shorts_eligible:
    short_script_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    short_script_future = short_script_executor.submit(
        generate_short_scripts,
        sentences,
        TOPIC if MODE == "topic" else text[:100],
        SHORTS_COUNT,
        SHORT_DUR_TARGET,
    )

# Queries and subtitle generation are independent after transcription.
update_status(50, "Matching visuals to sentences...")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep_executor:
    query_future = prep_executor.submit(generate_queries_for_sentences, sentences)
    subtitle_future = prep_executor.submit(
        create_subtitles,
        sentences,
        TEMP_DIR / "subs.ass",
        word_data=word_data if word_data else None,
    )
    AI_QUERIES, AI_BACKUPS = query_future.result()
    subtitle_future.result()

# Subtitles (word-level highlighting if available)
ass = TEMP_DIR / "subs.ass"

# Render
update_status(54, "Processing video...")

# Load the clip-verification model ONCE here, deliberately, on the main
# thread - BEFORE the render step spins up its 5 parallel worker threads.
# The old lazy-load-on-first-use pattern let all 5 workers race to call
# _load_llava() at nearly the same instant (the lock only covered the
# generate() call, not the load itself), so all 5 threads tried to load
# a full 7B model onto the GPU simultaneously - 5x duplicate memory
# allocation, causing OOM even on tiny 20-130MB allocations afterward.
# By this point in the pipeline, Chatterbox TTS and resemble-enhance have
# already freed their GPU memory (see generate_audio()'s explicit
# del+empty_cache+gc.collect calls), so this is the right moment to load
# the verification model into that freed VRAM, once, before any threads exist.
try:
    _load_llava()
    print(f"  Qwen verifier workers loaded before rendering: {len(_llava_workers)}")
except Exception as e:
    print(f"  ERROR: video verification workers failed to load ({str(e)[:120]}); refusing unverified output")
    update_status(0, "Video verification unavailable; render refused", "failed")
    raise SystemExit(1)

o2 = OUTPUT_DIR/f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o2, keep_verifier=True):
    update_status(93, "Uploading main video while preparing Shorts...")
    main_upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    main_upload_future = main_upload_executor.submit(upload_drive, o2)
    msg = "Done!\n"

    # ==========================================
    # SHORTS PIPELINE
    # ==========================================
    # Only worth generating shorts from audio that's actually long enough
    # for the requested count (avoid generating garbage 60s shorts from a
    # 90s total-runtime video, etc.)
    if sentences and sentences[-1]['end'] >= SHORT_DUR_TARGET * 0.6:
        update_status(95, f"Generating {SHORTS_COUNT} shorts...")
        try:
            if short_script_future is not None:
                short_scripts = short_script_future.result()
                short_script_executor.shutdown(wait=True)
                short_script_executor = None
            else:
                short_scripts = generate_short_scripts(
                    sentences,
                    TOPIC if MODE == "topic" else text[:100],
                    SHORTS_COUNT,
                    SHORT_DUR_TARGET,
                )
            print(f"  Shorts: {len(short_scripts)} scripts generated (requested {SHORTS_COUNT})")
            short_links = []
            short_failures = []  # (short_num, reason) for end-of-run summary

            # Generate short audio serially because generate_audio loads and
            # uses shared CUDA/TTS state. While the GPU generates the next
            # short, the previous short's transcription, subtitles, and Groq
            # visual queries run concurrently in the background.
            short_asset_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(3, len(short_scripts)))
            )
            prepared_shorts = []
            for si, sc in enumerate(short_scripts):
                script_text = sc["script"].strip()
                update_status(95, f"Preparing short audio {si+1}/{len(short_scripts)}...")
                if len(script_text) < 20:
                    short_failures.append((si + 1, "empty/too-short script"))
                    continue

                short_audio = TEMP_DIR / f"short_{si}_audio.wav"
                if not generate_audio(script_text, voice, short_audio):
                    print(f"  Short {si+1}: TTS failed, skipping")
                    short_failures.append((si + 1, "TTS failed"))
                    continue

                asset_future = short_asset_executor.submit(
                    _prepare_short_assets, si, script_text, short_audio
                )
                prepared_shorts.append((si, short_audio, asset_future))
            short_asset_executor.shutdown(wait=False)

            # Rendering remains sequential: each short uses the shared Qwen
            # workers and NVENC lock. The verifier models stay loaded between
            # Shorts so the next short does not pay the model-load cost again.
            short_upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            short_uploads = []
            for si, short_audio, asset_future in prepared_shorts:
                try:
                    assets = asset_future.result()
                    short_sentences = assets["sentences"]
                    short_ass = assets["ass"]
                    short_queries = assets["queries"]
                    short_backups = assets["backups"]
                except Exception as e:
                    print(f"  Short {si+1}: preparation failed ({type(e).__name__}: {str(e)[:160]})")
                    short_failures.append((si + 1, "transcription/query preparation failed"))
                    continue

                saved_queries = AI_QUERIES
                saved_backups = AI_BACKUPS
                AI_QUERIES = short_queries
                AI_BACKUPS = short_backups
                try:
                    short_out = OUTPUT_DIR / f"short_{JOB_ID}_{si+1}.mp4"
                    ok = render_short(
                        si, short_sentences, short_audio, short_ass, logo, short_out,
                        release_verifier=False,
                    )
                    if not ok:
                        print(f"  Short {si+1}: retrying once...")
                        ok = render_short(
                            si, short_sentences, short_audio, short_ass, logo, short_out,
                            release_verifier=False,
                        )
                finally:
                    AI_QUERIES = saved_queries
                    AI_BACKUPS = saved_backups

                if ok:
                    short_uploads.append((
                        si,
                        short_upload_executor.submit(upload_drive, short_out),
                    ))
                else:
                    print(f"  Short {si+1}: failed after retry, skipping")
                    short_failures.append((si + 1, "render failed after retry"))

            # Release verifier memory only after all Shorts have been rendered.
            _release_llava_for_encoding()
            for si, upload_future in short_uploads:
                try:
                    link = upload_future.result()
                except Exception:
                    link = None
                if link:
                    short_links.append(link)
                    msg += f"Short {si+1}: {link}\n"
                else:
                    short_failures.append((si + 1, "upload failed (render succeeded)"))
            short_upload_executor.shutdown(wait=True)

            print(f"  Shorts summary: {len(short_links)}/{len(short_scripts)} succeeded")
            if short_failures:
                print(f"  Shorts failures: {short_failures}")
                msg += f"({len(short_failures)} short(s) failed - check logs)\n"
        except Exception as e:
            print(f"  Shorts pipeline error: {e}")

    if _llava_workers:
        _release_llava_for_encoding()
    l2 = main_upload_future.result()
    main_upload_executor.shutdown(wait=True)
    if l2:
        msg += f"Video: {l2}\n"
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
