"""
Silent Voice - FastAPI Backend (COMPLETE VERSION WITH TTS FIX)
Handles gesture recognition, dataset management, and multilingual TTS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import os
import shutil
import stat
from collections import Counter
from googletrans import Translator
from gtts import gTTS
import base64
import io

# ----------------------------
# Configuration
# ----------------------------
DATASET_PATH = "gesture_dataset"
LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "te": "Telugu"
}

# Gesture to multilingual text mapping
GESTURE_SPEECH = {
    "Hello": {
        "en": "Hello",
        "kn": "ನಮಸ್ಕಾರ",
        "hi": "नमस्ते",
        "te": "నమస్కారం"
    },
    "Thaggede_Le": {
        "en": "Bring it on",
        "kn": "ತಗ್ಗದೇ ಲೇ",
        "hi": "ठग्गडे ले",
        "te": "తగ్గేదేలే"
    }
}

# Initialize translator
translator = Translator()

# Create dataset directory if not exists
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Silent Voice API", version="1.0.0")

# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Pydantic Models
# ----------------------------


class LandmarkData(BaseModel):
    landmarks: List[float]
    language: str = "en"


class GestureCreate(BaseModel):
    label: str
    landmarks: List[float]


class GestureEdit(BaseModel):
    old_label: str
    new_label: Optional[str] = None
    landmarks: Optional[List[float]] = None


class GestureDelete(BaseModel):
    label: str


class TextTranslate(BaseModel):
    text: str
    target_lang: str


class SpeakRequest(BaseModel):
    text: str
    language: str

# ----------------------------
# Utility Functions
# ----------------------------


def safe_rmtree(path):
    """Safely remove directory tree (Windows-compatible)"""
    if not os.path.exists(path):
        return True

    def on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    try:
        shutil.rmtree(path, onerror=on_rm_error)
        return True
    except Exception as e:
        print(f"[ERROR] Could not delete {path}: {e}")
        return False


def translate_text(text: str, target_lang: str) -> str:
    """Translate text using googletrans"""
    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"[TRANSLATE ERROR] {e}")
        return text


def load_dataset():
    """Load all gesture datasets from disk"""
    X, y, labels = [], [], []

    if not os.path.exists(DATASET_PATH):
        return None, None, None

    for idx, label in enumerate(sorted(os.listdir(DATASET_PATH))):
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        labels.append(label)
        for f in os.listdir(label_path):
            if f.endswith(".npy"):
                X.append(np.load(os.path.join(label_path, f)))
                y.append(idx)

    if len(X) == 0:
        return None, None, None

    return np.array(X), np.array(y), labels


def predict_gesture_knn(X_train, y_train, sample, k=7, distance_threshold=0.45):
    """k-NN classifier with majority voting"""
    if X_train is None or len(X_train) == 0:
        return None, 0.0, float('inf')

    dists = np.linalg.norm(X_train - sample, axis=1)
    sorted_idx = np.argsort(dists)
    k = min(k, len(dists))
    k_idx = sorted_idx[:k]
    k_labels = y_train[k_idx]
    k_dists = dists[k_idx]

    cnt = Counter(k_labels)
    pred_label = cnt.most_common(1)[0][0]
    votes_for_pred = cnt[pred_label]
    vote_ratio = votes_for_pred / k

    matched_distances = [d for i, d in zip(
        k_idx, k_dists) if y_train[i] == pred_label]
    avg_dist = float(np.mean(matched_distances)
                     ) if matched_distances else float(np.mean(k_dists))

    norm_factor = np.linalg.norm(sample) + 1e-8
    if (avg_dist / norm_factor) > distance_threshold or vote_ratio < 0.5:
        return None, vote_ratio, avg_dist

    return int(pred_label), vote_ratio, avg_dist


def get_speech_text(gesture_label: str, language: str) -> str:
    """Get speech text for a gesture in the specified language"""
    if gesture_label in GESTURE_SPEECH:
        return GESTURE_SPEECH[gesture_label].get(language, GESTURE_SPEECH[gesture_label].get("en", gesture_label))
    else:
        return translate_text(gesture_label, language)

# ----------------------------
# API Endpoints
# ----------------------------


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Silent Voice API is running", "frontend": "/static/index.html"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Silent Voice API", "tts": "gTTS enabled"}


@app.get("/api/gestures")
async def list_gestures():
    """Get list of all available gestures"""
    if not os.path.exists(DATASET_PATH):
        return {"gestures": []}

    gestures = []
    for label in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            sample_count = len(
                [f for f in os.listdir(label_path) if f.endswith(".npy")])
            gestures.append({
                "label": label,
                "sample_count": sample_count
            })

    return {"gestures": gestures}


@app.post("/api/recognize")
async def recognize_gesture(data: LandmarkData):
    """Recognize gesture from normalized landmarks"""
    X_train, y_train, labels = load_dataset()

    if X_train is None:
        raise HTTPException(
            status_code=404, detail="No gestures found. Please create some gestures first.")

    try:
        sample = np.array(data.landmarks, dtype=np.float32)
        if len(sample) != 63:
            raise HTTPException(
                status_code=400, detail=f"Expected 63 landmarks, got {len(sample)}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid landmark data: {str(e)}")

    pred_idx, vote_ratio, avg_dist = predict_gesture_knn(
        X_train, y_train, sample, k=7, distance_threshold=0.45)

    if pred_idx is None:
        return {
            "gesture": None,
            "speech_text": None,
            "confidence": 0.0,
            "message": "No valid gesture detected"
        }

    gesture_label = labels[pred_idx]
    speech_text = get_speech_text(gesture_label, data.language)

    return {
        "gesture": gesture_label,
        "speech_text": speech_text,
        "confidence": float(vote_ratio),
        "language": data.language
    }


@app.post("/api/create-gesture")
async def create_gesture_sample(data: GestureCreate):
    """Add a single sample to a gesture dataset"""
    label_path = os.path.join(DATASET_PATH, data.label)

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    try:
        sample = np.array(data.landmarks, dtype=np.float32)
        if len(sample) != 63:
            raise HTTPException(
                status_code=400, detail=f"Expected 63 landmarks, got {len(sample)}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid landmark data: {str(e)}")

    existing_files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    next_index = len(existing_files)

    file_path = os.path.join(label_path, f"{next_index}.npy")
    np.save(file_path, sample)

    return {
        "message": f"Sample saved for gesture '{data.label}'",
        "label": data.label,
        "sample_index": next_index,
        "total_samples": next_index + 1
    }


@app.post("/api/edit-gesture")
async def edit_gesture(data: GestureEdit):
    """Edit a gesture: rename it or add samples"""
    old_path = os.path.join(DATASET_PATH, data.old_label)

    if not os.path.exists(old_path):
        raise HTTPException(
            status_code=404, detail=f"Gesture '{data.old_label}' not found")

    if data.new_label and data.new_label != data.old_label:
        new_path = os.path.join(DATASET_PATH, data.new_label)

        if os.path.exists(new_path):
            raise HTTPException(
                status_code=400, detail=f"Gesture '{data.new_label}' already exists")

        os.rename(old_path, new_path)
        return {
            "message": f"Gesture renamed from '{data.old_label}' to '{data.new_label}'",
            "old_label": data.old_label,
            "new_label": data.new_label
        }

    if data.landmarks:
        try:
            sample = np.array(data.landmarks, dtype=np.float32)
            if len(sample) != 63:
                raise HTTPException(
                    status_code=400, detail=f"Expected 63 landmarks, got {len(sample)}")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid landmark data: {str(e)}")

        existing_files = [f for f in os.listdir(
            old_path) if f.endswith(".npy")]
        next_index = len(existing_files)

        file_path = os.path.join(old_path, f"{next_index}.npy")
        np.save(file_path, sample)

        return {
            "message": f"Sample added to gesture '{data.old_label}'",
            "label": data.old_label,
            "sample_index": next_index,
            "total_samples": next_index + 1
        }

    raise HTTPException(status_code=400, detail="No operation specified")


@app.post("/api/delete-gesture")
async def delete_gesture(data: GestureDelete):
    """Delete a gesture dataset"""
    label_path = os.path.join(DATASET_PATH, data.label)

    if not os.path.exists(label_path):
        raise HTTPException(
            status_code=404, detail=f"Gesture '{data.label}' not found")

    success = safe_rmtree(label_path)

    if not success:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete gesture '{data.label}'")

    return {
        "message": f"Gesture '{data.label}' deleted successfully",
        "label": data.label
    }


@app.post("/api/translate")
async def translate(data: TextTranslate):
    """Translate text to target language"""
    translated = translate_text(data.text, data.target_lang)
    return {
        "original": data.text,
        "translated": translated,
        "target_lang": data.target_lang
    }


@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    return {"languages": LANGUAGES}


@app.post("/api/speak")
async def generate_speech(data: SpeakRequest):
    """
    Generate speech audio using gTTS for all languages.
    Returns base64 encoded MP3 audio.
    """
    try:
        print(f"[TTS] Generating speech: '{data.text}' in {data.language}")

        # Create TTS
        tts = gTTS(text=data.text, lang=data.language, slow=False)

        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        print(f"[TTS] Success! Generated {len(audio_base64)} bytes")

        return {
            "success": True,
            "audio_base64": audio_base64,
            "text": data.text,
            "language": data.language
        }
    except Exception as e:
        print(f"[TTS ERROR] {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {str(e)}")

# ----------------------------
# For debugging: Test endpoint
# ----------------------------


@app.get("/api/test-tts")
async def test_tts():
    """Test TTS functionality"""
    tests = {
        "en": "Hello",
        "kn": "ನಮಸ್ಕಾರ",
        "hi": "नमस्ते",
        "te": "నమస్కారం"
    }

    results = {}
    for lang, text in tests.items():
        try:
            tts = gTTS(text=text, lang=lang)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            results[lang] = {"success": True,
                             "size": len(audio_buffer.getvalue())}
        except Exception as e:
            results[lang] = {"success": False, "error": str(e)}

    return {"test_results": results}
