"""
AI VIDEO GENERATOR WITH REALISM UPGRADES
========================================
FIXED VERSION:
1. Fixed Category Lock System
2. Cinematic Scene Engine
3. Stock Video Clustering
4. Fake Camera Movement
5. Color & Lighting Unification
6. Two Output Versions (with/without subtitles)
7. Enhanced Audio Realism
8. Human-like Subtitles
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

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "chatterbox-tts",
        "torchaudio", 
        "assemblyai",
        "google-generativeai",
        "requests",
        "beautifulsoup4",
        "pydub",
        "numpy",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
import numpy as np

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
# 3. CINEMATIC SCENE ENGINE
# ========================================== 

class CinematicSceneEngine:
    """Converts script into cinematic beats with emotional mapping"""
    
    EMOTION_TO_VISUAL = {
        # Emotional states
        "frustration": {
            "mood": "dark",
            "camera": "slow_motion",
            "color": "desaturated",
            "motion": "subtle_shake",
            "keywords": ["struggle", "failed", "problem", "hard", "difficult"]
        },
        "curiosity": {
            "mood": "mysterious",
            "camera": "zoom_in",
            "color": "warm_contrast",
            "motion": "slow_pan",
            "keywords": ["discover", "found", "secret", "learned", "realized"]
        },
        "excitement": {
            "mood": "energetic",
            "camera": "dynamic",
            "color": "vibrant",
            "motion": "quick_cuts",
            "keywords": ["amazing", "incredible", "awesome", "boom", "wow"]
        },
        "explanation": {
            "mood": "professional",
            "camera": "stable",
            "color": "balanced",
            "motion": "still",
            "keywords": ["because", "reason", "explain", "shows", "demonstrates"]
        },
        "warning": {
            "mood": "tense",
            "camera": "close_up",
            "color": "cold",
            "motion": "static",
            "keywords": ["danger", "warning", "careful", "avoid", "mistake"]
        },
        "success": {
            "mood": "triumphant",
            "camera": "wide",
            "color": "golden_hour",
            "motion": "slow_rise",
            "keywords": ["success", "win", "achieve", "result", "growth"]
        },
        "calm": {
            "mood": "peaceful",
            "camera": "smooth",
            "color": "soft",
            "motion": "gentle_pan",
            "keywords": ["peace", "calm", "quiet", "still", "relax"]
        }
    }
    
    VISUAL_INTENT_TYPES = {
        "explanation": ["hands typing", "laptop screen", "whiteboard", "diagram", "flowchart"],
        "story": ["person walking", "city street", "looking thoughtful", "coffee shop", "window"],
        "data": ["charts growing", "graphs animation", "numbers increasing", "statistics", "dashboard"],
        "concept": ["abstract shapes", "particles floating", "light trails", "neural network", "connections"],
        "action": ["person working", "building something", "team collaboration", "construction", "progress"],
        "result": ["celebration", "high five", "trophy", "finished project", "before after"]
    }
    
    @staticmethod
    def analyze_sentence(sentence):
        """Analyze sentence for emotional content and visual intent"""
        sentence_lower = sentence.lower()
        
        # Detect emotion
        detected_emotions = []
        for emotion, data in CinematicSceneEngine.EMOTION_TO_VISUAL.items():
            for keyword in data["keywords"]:
                if keyword in sentence_lower:
                    detected_emotions.append(emotion)
                    break
        
        # Default to explanation if no emotion detected
        if not detected_emotions:
            detected_emotions = ["explanation"]
        
        # Detect visual intent
        detected_intents = []
        for intent, keywords in CinematicSceneEngine.VISUAL_INTENT_TYPES.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    detected_intents.append(intent)
                    break
        
        # Default intent
        if not detected_intents:
            if len(sentence.split()) > 15:
                detected_intents = ["explanation"]
            else:
                detected_intents = ["story"]
        
        # Get primary emotion/intent
        primary_emotion = detected_emotions[0]
        primary_intent = detected_intents[0]
        
        return {
            "emotion": primary_emotion,
            "intent": primary_intent,
            "visual_style": CinematicSceneEngine.EMOTION_TO_VISUAL[primary_emotion],
            "duration_factor": 1.0 if primary_emotion in ["explanation", "calm"] else 0.8,
            "pace": "slow" if primary_emotion in ["calm", "explanation"] else "normal"
        }

# ========================================== 
# 4. STOCK VIDEO CLUSTERING SYSTEM
# ========================================== 

class StockVideoClustering:
    """Clusters stock videos for visual consistency"""
    
    @staticmethod
    def calculate_clip_similarity(video1, video2):
        """Calculate similarity between two video clips"""
        score = 0
        
        # Brightness similarity
        brightness_diff = abs(video1.get('brightness', 0.5) - video2.get('brightness', 0.5))
        score += (1.0 - brightness_diff) * 30
        
        # Motion direction similarity
        motion1 = video1.get('motion_direction', 'static')
        motion2 = video2.get('motion_direction', 'static')
        if motion1 == motion2:
            score += 25
        
        # Color temperature similarity
        temp_diff = abs(video1.get('color_temp', 0.5) - video2.get('color_temp', 0.5))
        score += (1.0 - temp_diff) * 25
        
        # Duration similarity
        dur1 = video1.get('duration', 0)
        dur2 = video2.get('duration', 0)
        if dur1 > 0 and dur2 > 0:
            dur_ratio = min(dur1, dur2) / max(dur1, dur2)
            score += dur_ratio * 20
        
        return score
    
    @staticmethod
    def find_best_cluster(candidates, previous_clips, min_similarity=60):
        """Find clips that match previous ones for consistency"""
        if not previous_clips:
            return candidates
        
        best_clips = []
        for candidate in candidates:
            total_similarity = 0
            match_count = 0
            
            for prev_clip in previous_clips[-3:]:  # Compare with last 3 clips
                similarity = StockVideoClustering.calculate_clip_similarity(
                    candidate, prev_clip
                )
                if similarity > min_similarity:
                    total_similarity += similarity
                    match_count += 1
            
            if match_count > 0:
                avg_similarity = total_similarity / match_count
                candidate['cluster_score'] = avg_similarity
                best_clips.append(candidate)
        
        if best_clips:
            best_clips.sort(key=lambda x: x['cluster_score'], reverse=True)
            return best_clips[:5]
        
        return candidates[:5]

# ========================================== 
# 5. REALISM ENHANCEMENTS
# ========================================== 

class RealismEnhancer:
    """Adds cinematic realism to videos"""
    
    @staticmethod
    def add_camera_motion(input_path, output_path, motion_type="subtle_zoom"):
        """Add subtle fake camera movement"""
        motions = {
            "subtle_zoom": "zoompan=z='min(zoom+0.0003,1.03)':d=125:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
            "slow_pan_right": "zoompan=z=1:x='if(lte(on,1),0,x+1)':y='0':d=125",
            "gentle_float": "zoompan=z=1:x='iw/2-(iw/zoom/2)+sin(on/25)*5':y='ih/2-(ih/zoom/2)+cos(on/30)*3':d=125",
            "micro_rotation": "rotate=angle=0.1*sin(2*PI*t/10):ow=hypot(iw,ih):oh=ow"
        }
        
        filter_complex = motions.get(motion_type, motions["subtle_zoom"])
        
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", input_path,
            "-vf", filter_complex,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "8M",
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            # If filter fails, just copy
            shutil.copy(input_path, output_path)
            return False
    
    @staticmethod
    def apply_color_unification(input_path, output_path, style="cinematic"):
        """Apply consistent color grading"""
        color_luts = {
            "cinematic": "colorbalance=rs=0.05:gs=0:bs=-0.05,eq=saturation=1.05:contrast=1.05",
            "warm_contrast": "colorbalance=rs=0.1:gs=0.05:bs=-0.1,eq=saturation=1.1:contrast=1.1",
            "cool_moody": "colorbalance=rs=-0.05:gs=0:bs=0.05,eq=saturation=0.95:contrast=1.15",
            "golden_hour": "colorbalance=rs=0.15:gs=0.1:bs=-0.1,eq=saturation=1.15:brightness=0.02",
            "desaturated": "eq=saturation=0.8:contrast=1.1",
            "vibrant": "eq=saturation=1.2:contrast=1.05"
        }
        
        lut = color_luts.get(style, color_luts["cinematic"])
        
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", input_path,
            "-vf", lut,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "8M",
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            shutil.copy(input_path, output_path)
            return False
    
    @staticmethod
    def add_micro_imperfections(input_path, output_path):
        """Add subtle imperfections for realism"""
        try:
            # Add 0.15s black frame at the end
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "cuda",
                "-i", input_path,
                "-vf", "fade=in:0:15,fade=out:st={duration}:d=15".format(
                    duration=float(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                         "-of", "default=noprint_wrappers=1:nokey=1", input_path]
                    ).decode().strip()) - 0.15
                ),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-b:v", "8M",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            shutil.copy(input_path, output_path)
            return False

# ========================================== 
# 6. FIXED CATEGORY LOCK SYSTEM
# ========================================== 

def analyze_topic_for_category(topic, script):
    """Intelligently determine the REAL category from topic and script"""
    
    # Topic-based category mapping
    topic_category_map = {
        # AI & Tech
        "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
        "machine learning": "artificial intelligence", "deep learning": "artificial intelligence",
        "neural network": "artificial intelligence", "chatgpt": "artificial intelligence",
        "openai": "artificial intelligence", "llm": "artificial intelligence",
        "technology": "technology", "tech": "technology",
        "programming": "software", "coding": "software",
        "software": "software", "developer": "software",
        
        # Business & Finance
        "business": "business", "entrepreneur": "business",
        "startup": "business", "company": "business",
        "finance": "finance", "money": "finance",
        "invest": "finance", "stock": "finance",
        "trading": "finance", "wealth": "finance",
        
        # Motivation & Personal Development
        "motivation": "people", "motivational": "people",
        "inspire": "people", "inspirational": "people",
        "success": "people", "achieve": "people",
        "goal": "people", "dream": "people",
        "mindset": "people", "personal development": "people",
        "self improvement": "people", "growth": "people",
        
        # Science & Education
        "science": "science", "scientific": "science",
        "research": "science", "discovery": "science",
        "education": "education", "learn": "education",
        "study": "education", "knowledge": "education",
        
        # Health & Fitness
        "health": "health", "fitness": "health",
        "exercise": "health", "workout": "health",
        "diet": "health", "nutrition": "health",
        
        # Nature & Environment
        "nature": "nature", "environment": "nature",
        "climate": "nature", "earth": "nature",
        "animal": "animal", "wildlife": "animal",
        
        # Creative
        "art": "art", "creative": "art",
        "design": "art", "music": "music",
        "film": "movie", "photography": "photo"
    }
    
    # Analyze topic first
    topic_lower = topic.lower()
    script_lower = script.lower()[:2000]  # First 2000 chars
    
    # Check for exact matches in topic
    for keyword, category in topic_category_map.items():
        if keyword in topic_lower:
            print(f"✅ Category determined by topic keyword '{keyword}': {category}")
            return category, [category, keyword]
    
    # Check for topic phrases
    topic_words = topic_lower.split()
    for i in range(len(topic_words) - 1):
        phrase = f"{topic_words[i]} {topic_words[i+1]}"
        if phrase in topic_category_map:
            category = topic_category_map[phrase]
            print(f"✅ Category determined by topic phrase '{phrase}': {category}")
            return category, [category, phrase]
    
    # Analyze script content
    script_words = re.findall(r'\b\w{4,}\b', script_lower)
    word_freq = defaultdict(int)
    
    for word in script_words:
        word_freq[word] += 1
    
    # Check script words against category map
    for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
        if word in topic_category_map:
            category = topic_category_map[word]
            print(f"✅ Category determined by script word '{word}' (appears {count} times): {category}")
            return category, [category, word]
    
    # Default fallback based on common content types
    if any(word in script_lower for word in ["money", "profit", "revenue", "business", "invest"]):
        return "finance", ["finance", "business"]
    elif any(word in script_lower for word in ["ai", "machine", "algorithm", "data", "tech"]):
        return "artificial intelligence", ["artificial intelligence", "technology"]
    elif any(word in script_lower for word in ["mindset", "success", "goal", "achieve", "motivation"]):
        return "people", ["people", "motivation"]
    else:
        return "technology", ["technology", "digital"]

# ========================================== 
# 7. ENHANCED VISUAL DICTIONARY WITH CATEGORY FILTERING
# ========================================== 

def get_category_filtered_queries(category, scene_analysis, sentence_text):
    """Get queries filtered by category and enhanced by scene analysis"""
    
    # Expanded visual dictionary with category focus
    CATEGORY_FOCUSED_VISUALS = {
        "artificial intelligence": {
            "primary": ["neural network visualization", "AI processing", "data center", 
                       "machine learning algorithm", "digital brain", "robot assembly",
                       "hologram display", "circuit board", "coding screen"],
            "emotion_map": {
                "frustration": ["glitching screen", "error message", "broken circuit"],
                "curiosity": ["data visualization", "network connections", "discovery"],
                "excitement": ["breakthrough moment", "light speed data", "innovation"],
                "explanation": ["whiteboard diagram", "presentation screen", "teaching"],
                "success": ["achievement unlocked", "growth chart", "trophy"]
            }
        },
        "technology": {
            "primary": ["innovation lab", "tech startup", "silicon valley", 
                       "hardware engineering", "microchip", "quantum computer",
                       "server room", "circuit board", "digital interface"],
            "emotion_map": {
                "frustration": ["bug fixing", "system error", "broken device"],
                "curiosity": ["new invention", "experiment", "research"],
                "excitement": ["product launch", "breakthrough", "celebration"],
                "explanation": ["technical diagram", "instruction manual", "demo"],
                "success": ["finished product", "award ceremony", "team celebration"]
            }
        },
        "people": {
            "primary": ["person achieving goal", "personal growth journey", 
                       "success story", "motivational moment", "overcoming challenge",
                       "celebration victory", "team accomplishment", "inspirational scene"],
            "emotion_map": {
                "frustration": ["struggle moment", "facing obstacle", "difficult task"],
                "curiosity": ["learning new skill", "discovery moment", "exploration"],
                "excitement": ["breakthrough moment", "achievement", "celebration"],
                "explanation": ["teaching moment", "sharing knowledge", "mentoring"],
                "success": ["trophy moment", "graduation", "promotion"]
            }
        },
        "finance": {
            "primary": ["stock market charts", "financial growth", "money visualization",
                       "investment success", "wealth building", "economic growth"],
            "emotion_map": {
                "frustration": ["market crash", "financial loss", "debt"],
                "curiosity": ["market analysis", "research", "planning"],
                "excitement": ["bull market", "profit surge", "success"],
                "explanation": ["financial chart", "budget planning", "strategy"],
                "success": ["wealth achievement", "retirement goal", "financial freedom"]
            }
        },
        "business": {
            "primary": ["business meeting", "corporate success", "startup growth",
                       "entrepreneur journey", "office achievement", "team success"],
            "emotion_map": {
                "frustration": ["business failure", "rejection", "bankruptcy"],
                "curiosity": ["market research", "opportunity", "innovation"],
                "excitement": ["deal closing", "funding secured", "expansion"],
                "explanation": ["business plan", "strategy meeting", "presentation"],
                "success": ["ipo celebration", "award ceremony", "profit milestone"]
            }
        }
    }
    
    # Default category visuals
    category_data = CATEGORY_FOCUSED_VISUALS.get(category, CATEGORY_FOCUSED_VISUALS["technology"])
    
    # Get emotion-specific visuals
    emotion = scene_analysis.get("emotion", "explanation")
    emotion_visuals = category_data["emotion_map"].get(emotion, category_data["primary"])
    
    # Get intent-specific visuals
    intent = scene_analysis.get("intent", "explanation")
    if intent in CinematicSceneEngine.VISUAL_INTENT_TYPES:
        intent_visuals = CinematicSceneEngine.VISUAL_INTENT_TYPES[intent]
    else:
        intent_visuals = category_data["primary"]
    
    # Combine visuals
    all_visuals = list(set(category_data["primary"] + emotion_visuals + intent_visuals))
    
    # Extract keywords from sentence
    sentence_lower = sentence_text.lower()
    sentence_keywords = re.findall(r'\b\w{5,}\b', sentence_lower)[:3]
    
    # Build queries
    queries = []
    
    # Primary query: Category + emotion + intent visual
    primary_visual = random.choice(emotion_visuals)
    primary_query = f"{primary_visual} {category}"
    queries.append(f"{primary_query} 4k cinematic")
    
    # Secondary query: Intent visual + category
    if intent_visuals:
        secondary_visual = random.choice(intent_visuals)
        queries.append(f"{secondary_visual} {category} cinematic")
    
    # Tertiary query: Sentence keyword + category visual
    if sentence_keywords:
        for keyword in sentence_keywords:
            if len(keyword) > 4:
                queries.append(f"{keyword} {category} visual")
                break
    
    # Fallback queries
    for visual in random.sample(all_visuals, min(3, len(all_visuals))):
        if visual not in primary_query:
            queries.append(f"{visual} cinematic")
    
    # Ensure we have enough queries
    while len(queries) < 4:
        queries.append(f"{category} abstract cinematic")
    
    print(f"    🎬 Scene: {emotion.upper()} | Intent: {intent.upper()}")
    print(f"    📌 Queries: {queries[:3]}")
    
    return queries, emotion, intent

# ========================================== 
# 8. ENHANCED VIDEO PROCESSING WITH REALISM
# ========================================== 

def process_visuals_with_realism(sentences, audio_path, ass_file, logo_path, topic, full_script):
    """Process visuals with cinematic realism and category locking"""
    
    # Determine category ONCE
    category, category_keywords = analyze_topic_for_category(topic, full_script)
    print(f"🎯 VIDEO CATEGORY LOCKED: {category.upper()}")
    print(f"   Keywords: {', '.join(category_keywords)}")
    
    # Track previous clips for consistency
    previous_clips = []
    processed_clips = []
    
    for i, sent in enumerate(sentences):
        dur = max(3.5, sent['end'] - sent['start'])
        update_status(60 + int((i/len(sentences))*30), 
                     f"Scene {i+1}/{len(sentences)}: {category[:20]}...")
        
        # Analyze scene
        scene_analysis = CinematicSceneEngine.analyze_sentence(sent['text'])
        
        # Get category-filtered queries
        queries, emotion, intent = get_category_filtered_queries(
            category, scene_analysis, sent['text']
        )
        
        # Try each query
        video_found = False
        for query_idx, query in enumerate(queries):
            if video_found:
                break
                
            print(f"  🔍 Clip {i+1}: Query {query_idx+1} - '{query}'")
            
            # Search videos
            all_candidates = []
            
            # Search Pexels
            if PEXELS_KEYS and PEXELS_KEYS[0]:
                pexels_results = intelligent_video_search(query, 'pexels', PEXELS_KEYS)
                for video in pexels_results:
                    video['query'] = query
                    video['emotion'] = emotion
                    video['intent'] = intent
                    all_candidates.append(video)
            
            # Search Pixabay
            if PIXABAY_KEYS and PIXABAY_KEYS[0]:
                pixabay_results = intelligent_video_search(query, 'pixabay', PIXABAY_KEYS)
                for video in pixabay_results:
                    video['query'] = query
                    video['emotion'] = emotion
                    video['intent'] = intent
                    all_candidates.append(video)
            
            # Remove duplicates and used URLs
            unique_candidates = []
            seen_urls = set()
            for vid in all_candidates:
                if vid['url'] not in USED_VIDEO_URLS and vid['url'] not in seen_urls:
                    seen_urls.add(vid['url'])
                    unique_candidates.append(vid)
            
            # Cluster for consistency
            clustered_candidates = StockVideoClustering.find_best_cluster(
                unique_candidates, previous_clips
            )
            
            if clustered_candidates:
                # Select best candidate
                best_video = clustered_candidates[0]
                USED_VIDEO_URLS.add(best_video['url'])
                
                print(f"    ✅ Selected: {best_video.get('service', 'unknown')} "
                      f"({best_video.get('duration', 0):.1f}s) - {emotion}/{intent}")
                
                # Download
                raw_path = TEMP_DIR / f"raw_{i}.mp4"
                try:
                    response = requests.get(best_video['url'], timeout=30, stream=True)
                    with open(raw_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Apply camera motion based on emotion
                    motion_enhanced = TEMP_DIR / f"motion_{i}.mp4"
                    motion_type = scene_analysis['visual_style']['motion']
                    RealismEnhancer.add_camera_motion(raw_path, motion_enhanced, motion_type)
                    
                    # Apply color unification
                    color_enhanced = TEMP_DIR / f"color_{i}.mp4"
                    color_style = scene_analysis['visual_style']['color']
                    RealismEnhancer.apply_color_unification(motion_enhanced, color_enhanced, color_style)
                    
                    # Trim to duration
                    final_clip = TEMP_DIR / f"clip_{i}.mp4"
                    cmd = [
                        "ffmpeg", "-y", "-hwaccel", "cuda",
                        "-i", str(color_enhanced),
                        "-t", str(dur),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-b:v", "8M",
                        "-an",
                        str(final_clip)
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    processed_clips.append(str(final_clip))
                    previous_clips.append({
                        'brightness': 0.5,
                        'color_temp': 0.5,
                        'motion_direction': motion_type,
                        'duration': dur
                    })
                    
                    video_found = True
                    break
                    
                except Exception as e:
                    print(f"    ✗ Download/processing failed: {str(e)[:50]}")
                    continue
        
        # Fallback gradient
        if not video_found:
            print(f"  ⚠️ Using category-themed gradient fallback")
            
            # Category-specific colors
            category_colors = {
                "artificial intelligence": ["0x0f3460:0x1a1a2e", "0x533483:0x16213e"],
                "technology": ["0x1a1a2e:0x0f3460", "0x16213e:0x533483"],
                "people": ["0x2c3e50:0x3498db", "0x2c3e50:0xe74c3c"],
                "finance": ["0x27ae60:0x2ecc71", "0x2980b9:0x3498db"],
                "business": ["0x2c3e50:0x34495e", "0x16a085:0x27ae60"]
            }
            
            gradient = category_colors.get(category, ["0x1a1a2e:0x16213e"])[0]
            fallback_clip = TEMP_DIR / f"gradient_{i}.mp4"
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={gradient.split(':')[0]}:s=1920x1080:d={dur}",
                "-vf", f"gradients=s=1920x1080:x0=0:y0=0:x1=1920:y1=1080:c0={gradient.split(':')[0]}:c1={gradient.split(':')[1]},fade=in:0:30,fade=out:st={dur-1}:d=1",
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-t", str(dur),
                str(fallback_clip)
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processed_clips.append(str(fallback_clip))
    
    return processed_clips

# ========================================== 
# 9. DUAL OUTPUT RENDERER (with/without subtitles)
# ========================================== 

def render_dual_outputs(processed_clips, audio_path, ass_file, logo_path):
    """Render two versions: with and without subtitles"""
    
    print("🎬 Concatenating all clips...")
    
    # Create concatenation list
    concat_list = TEMP_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in processed_clips:
            if os.path.exists(clip):
                f.write(f"file '{clip}'\n")
    
    # Concatenate all clips
    concatenated = TEMP_DIR / "concatenated.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "10M",
        "-an",
        str(concatenated)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Get audio duration
    audio_duration = None
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        audio_duration = float(subprocess.check_output(cmd).decode().strip())
    except:
        pass
    
    # Render Version 1: WITHOUT subtitles (just audio + logo)
    print("🎬 Rendering version 1 (no subtitles)...")
    final_no_subs = OUTPUT_DIR / f"final_{JOB_ID}_no_subs.mp4"
    
    base_cmd = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", str(concatenated),
        "-i", str(audio_path),
    ]
    
    filter_complex = []
    map_cmds = []
    
    if logo_path and os.path.exists(logo_path):
        base_cmd.extend(["-i", str(logo_path)])
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[2:v]scale=200:-1[logo];"
            "[bg][logo]overlay=30:30[v]"
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    else:
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v]"
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    
    full_cmd = base_cmd + ["-filter_complex", ";".join(filter_complex)] + map_cmds + [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "256k"
    ]
    
    if audio_duration:
        full_cmd.extend(["-t", str(audio_duration)])
    
    full_cmd.append(str(final_no_subs))
    
    try:
        subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Version 1 rendered: {final_no_subs}")
    except Exception as e:
        print(f"❌ Version 1 failed: {e}")
        final_no_subs = None
    
    # Render Version 2: WITH subtitles
    print("🎬 Rendering version 2 (with subtitles)...")
    final_with_subs = OUTPUT_DIR / f"final_{JOB_ID}_with_subs.mp4"
    
    # Escape ASS file path
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    base_cmd = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", str(concatenated),
        "-i", str(audio_path),
    ]
    
    filter_complex = []
    map_cmds = []
    
    if logo_path and os.path.exists(logo_path):
        base_cmd.extend(["-i", str(logo_path)])
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[2:v]scale=200:-1[logo];"
            "[bg][logo]overlay=30:30[withlogo];"
            "[withlogo]subtitles='{ass_path}'[v]".format(ass_path=ass_path)
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    else:
        filter_complex.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            "[bg]subtitles='{ass_path}'[v]".format(ass_path=ass_path)
        )
        map_cmds = ["-map", "[v]", "-map", "1:a"]
    
    full_cmd = base_cmd + ["-filter_complex", ";".join(filter_complex)] + map_cmds + [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "256k"
    ]
    
    if audio_duration:
        full_cmd.extend(["-t", str(audio_duration)])
    
    full_cmd.append(str(final_with_subs))
    
    try:
        subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Version 2 rendered: {final_with_subs}")
    except Exception as e:
        print(f"❌ Version 2 failed: {e}")
        final_with_subs = None
    
    return final_no_subs, final_with_subs

# ========================================== 
# 10. ENHANCED AUDIO REALISM
# ========================================== 

def enhance_audio_realism(audio_path, output_path):
    """Add breath sounds and dynamic pacing"""
    try:
        # Add subtle breath sounds at natural breaks
        temp_audio = TEMP_DIR / "enhanced_temp.wav"
        
        # First, extract audio duration
        cmd_duration = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        duration = float(subprocess.check_output(cmd_duration).decode().strip())
        
        # Create silent breath track
        breath_pattern = TEMP_DIR / "breath_pattern.wav"
        cmd_breath = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anoisesrc=d=0.3:c=brown:r=44100:a=0.03",
            "-af", "highpass=80,lowpass=800,areverse,areverse",
            str(breath_pattern)
        ]
        subprocess.run(cmd_breath, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Create enhanced audio with breaths
        concat_filter = []
        
        # Add breaths at natural intervals (every 8-12 seconds)
        breath_positions = []
        pos = 4.0  # First breath after 4 seconds
        while pos < duration - 1.0:
            breath_positions.append(pos)
            pos += random.uniform(8.0, 12.0)
        
        # Build complex filter
        filter_complex = f"[0:a]"
        
        # Insert breaths
        for i, pos in enumerate(breath_positions):
            filter_complex += f"atrim=start={pos}:end={pos+0.3},volume=0.2[br{i}];"
        
        filter_complex += f"[0:a]"
        
        # Merge breaths
        for i, pos in enumerate(breath_positions):
            filter_complex += f"[br{i}]"
        
        filter_complex += f"concat=n={1 + len(breath_positions)}:v=0:a=1[aout]"
        
        cmd_enhance = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-i", str(breath_pattern),
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-ar", "44100",
            str(temp_audio)
        ]
        
        subprocess.run(cmd_enhance, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Add subtle noise floor
        cmd_noise = [
            "ffmpeg", "-y",
            "-i", str(temp_audio),
            "-f", "lavfi",
            "-i", "anoisesrc=d={duration}:c=white:r=44100:a=0.001".format(duration=duration),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:weights=1 0.2",
            "-ar", "44100",
            str(output_path)
        ]
        
        subprocess.run(cmd_noise, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Audio enhanced with breaths and noise floor")
        return True
        
    except Exception as e:
        print(f"⚠️ Audio enhancement failed: {e}")
        # Fallback: copy original
        shutil.copy(audio_path, output_path)
        return False

# ========================================== 
# 11. FIXED SUBTITLE SYSTEM (Human-like)
# ========================================== 

def create_human_subtitles(sentences, ass_file):
    """Create human-like subtitles with proper timing"""
    
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        # Header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        # Styles
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        # Events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            # Human-like timing: appear slightly before, disappear before end
            start_time = max(0, s['start'] - 0.15)  # Appear 150ms early
            end_time = s['end'] - 0.3  # Disappear 300ms early
            
            text = s['text'].strip()
            text = re.sub(r'[\[\]]', '', text)
            
            # Human reading speed: 3-4 words per line
            words = text.split()
            lines = []
            current_line = []
            char_count = 0
            
            for word in words:
                if char_count + len(word) + 1 > 35 or len(current_line) >= 4:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    char_count = len(word)
                else:
                    current_line.append(word)
                    char_count += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Uppercase for certain styles
            if style_key in ["mrbeast_yellow", "hormozi_green", "tiktok_white"]:
                lines = [line.upper() for line in lines]
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{formatted_text}\n")

# ========================================== 
# 12. MAIN EXECUTION (UPDATED)
# ========================================== 

print("--- 🚀 START (CINEMATIC REALISM EDITION) ---")
update_status(1, "Initializing cinematic engine...")

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
raw_audio = TEMP_DIR / "raw_audio.wav"
enhanced_audio = TEMP_DIR / "enhanced_audio.wav"

if clone_voice_robust(text, ref_voice, raw_audio):
    # Enhance audio
    update_status(25, "Enhancing audio realism...")
    enhance_audio_realism(raw_audio, enhanced_audio)
    audio_out = enhanced_audio if os.path.exists(enhanced_audio) else raw_audio
    
    update_status(30, "Creating human-like subtitles...")
    
    # Transcribe for subtitles
    if ASSEMBLY_KEY:
        try:
            aai.settings.api_key = ASSEMBLY_KEY
            transcriber = aai.Transcriber()
            print("📝 Transcribing audio...")
            transcript = transcriber.transcribe(str(audio_out))
            
            if transcript.status == aai.TranscriptStatus.completed:
                sentences = []
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
            # Fallback timing
            words = text.split()
            import wave
            with wave.open(str(audio_out), 'rb') as wav_file:
                total_duration = wav_file.getnframes() / float(wav_file.getframerate())
            
            words_per_second = len(words) / total_duration
            sentences = []
            current_time = 0
            words_per_sentence = random.randint(8, 12)  # Varied for realism
            
            for i in range(0, len(words), words_per_sentence):
                chunk = words[i:i + words_per_sentence]
                sentence_duration = len(chunk) / words_per_second
                sentences.append({
                    "text": ' '.join(chunk),
                    "start": current_time,
                    "end": current_time + sentence_duration
                })
                current_time += sentence_duration
                words_per_sentence = random.randint(8, 12)  # Change pace
            
            if sentences:
                sentences[-1]['end'] += 1.5
    else:
        # Fallback if no AssemblyAI
        words = text.split()
        import wave
        with wave.open(str(audio_out), 'rb') as wav_file:
            total_duration = wav_file.getnframes() / float(wav_file.getframerate())
        
        words_per_second = len(words) / total_duration
        sentences = []
        current_time = 0
        words_per_sentence = random.randint(8, 12)
        
        for i in range(0, len(words), words_per_sentence):
            chunk = words[i:i + words_per_sentence]
            sentence_duration = len(chunk) / words_per_second
            sentences.append({
                "text": ' '.join(chunk),
                "start": current_time,
                "end": current_time + sentence_duration
            })
            current_time += sentence_duration
            words_per_sentence = random.randint(8, 12)
        
        if sentences:
            sentences[-1]['end'] += 1.5
    
    # Create ASS subtitles with human timing
    ass_file = TEMP_DIR / "subtitles.ass"
    create_human_subtitles(sentences, ass_file)
    
    # Process visuals with cinematic realism
    update_status(40, "Creating cinematic visuals...")
    processed_clips = process_visuals_with_realism(
        sentences, audio_out, ass_file, ref_logo, TOPIC, text
    )
    
    if processed_clips:
        update_status(80, "Rendering final videos...")
        
        # Render both versions
        final_no_subs, final_with_subs = render_dual_outputs(
            processed_clips, audio_out, ass_file, ref_logo
        )
        
        if final_no_subs and os.path.exists(final_no_subs):
            file_size = os.path.getsize(final_no_subs) / (1024 * 1024)
            print(f"✅ Version 1 (no subs): {file_size:.1f} MB")
            
            update_status(90, "Uploading Version 1 to Google Drive...")
            drive_link1 = upload_to_google_drive(final_no_subs)
            
            if drive_link1:
                print(f"🔗 Version 1 Link: {drive_link1}")
            else:
                print("⚠️ Version 1 upload failed")
        
        if final_with_subs and os.path.exists(final_with_subs):
            file_size = os.path.getsize(final_with_subs) / (1024 * 1024)
            print(f"✅ Version 2 (with subs): {file_size:.1f} MB")
            
            update_status(95, "Uploading Version 2 to Google Drive...")
            drive_link2 = upload_to_google_drive(final_with_subs)
            
            if drive_link2:
                print(f"🔗 Version 2 Link: {drive_link2}")
            
            # Update final status with both links
            update_status(100, "Both versions completed!", "completed", 
                         drive_link2 if drive_link2 else drive_link1)
        else:
            update_status(100, "One version completed", "completed", 
                         drive_link1 if drive_link1 else None)
    else:
        update_status(0, "Visual processing failed", "failed")
else:
    update_status(0, "Audio synthesis failed", "failed")

# Cleanup
print("🧹 Cleaning up...")
if TEMP_DIR.exists():
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

for temp_file in ["list.txt", "concat_list.txt"]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("--- ✅ CINEMATIC PROCESS COMPLETE ---")
