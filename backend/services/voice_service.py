"""
EchoBond Voice Service
======================
Voice sample analysis (pitch detection) and text-to-speech synthesis
using Microsoft Edge Neural TTS voices, with pitch/rate adjustments
to approximate a family member's real voice.

Voice Cloning Approach:
  1. Family uploads/records a 5-60s voice sample of the person.
  2. We analyze the sample to extract fundamental frequency (F0/pitch).
  3. Based on gender (from pitch) we pick the best matching Edge neural voice.
  4. We adjust pitch offset (+Hz/-Hz) and speaking rate to approximate
     the person's natural voice characteristics.
  5. When the patient asks to hear from that person, we synthesize speech
     using their personalized voice profile.
"""

import asyncio
import io
import json
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Voice Profile Constants ─────────────────────────────
# Edge-TTS neural voice mapping by gender and language
VOICE_PROFILES = {
    "male": {
        "hi-IN": {"voice": "hi-IN-MadhurNeural",   "base_pitch": 125.0, "label": "Hindi Male (Madhur)"},
        "en-IN": {"voice": "en-IN-PrabhatNeural",   "base_pitch": 130.0, "label": "Indian English Male (Prabhat)"},
        "en-US": {"voice": "en-US-GuyNeural",       "base_pitch": 120.0, "label": "English Male (Guy)"},
        "en-GB": {"voice": "en-GB-RyanNeural",      "base_pitch": 118.0, "label": "British English Male (Ryan)"},
    },
    "female": {
        "hi-IN": {"voice": "hi-IN-SwaraNeural",     "base_pitch": 210.0, "label": "Hindi Female (Swara)"},
        "en-IN": {"voice": "en-IN-NeerjaNeural",    "base_pitch": 215.0, "label": "Indian English Female (Neerja)"},
        "en-US": {"voice": "en-US-JennyNeural",     "base_pitch": 200.0, "label": "English Female (Jenny)"},
        "en-GB": {"voice": "en-GB-SoniaNeural",     "base_pitch": 205.0, "label": "British English Female (Sonia)"},
    },
}

# Default / EchoBond system voice
SYSTEM_VOICE = "en-IN-NeerjaNeural"
SYSTEM_PITCH = "+0Hz"
SYSTEM_RATE = "-10%"

# Pitch boundaries for gender classification
MALE_PITCH_MAX = 165.0
FEMALE_PITCH_MIN = 155.0


def classify_gender(pitch_hz: float) -> str:
    """Classify voice gender from fundamental frequency."""
    if pitch_hz < MALE_PITCH_MAX:
        return "male"
    return "female"


def compute_pitch_offset(actual_pitch: float, base_pitch: float) -> str:
    """Compute edge-tts pitch offset string like '+12Hz' or '-8Hz'."""
    diff = actual_pitch - base_pitch
    # Clamp to reasonable range
    diff = max(-50, min(50, diff))
    sign = "+" if diff >= 0 else ""
    return f"{sign}{int(round(diff))}Hz"


def compute_rate_offset(actual_pitch: float) -> str:
    """Slightly slower for deeper voices, slightly faster for higher voices."""
    # Base rate offset is -5% (slightly slower for clarity with patients)
    base = -5
    # Adjust slightly based on pitch (elderly voices tend to be slower)
    if actual_pitch < 110:
        base = -12  # deeper/older = slower
    elif actual_pitch > 240:
        base = -3   # higher/younger = closer to normal
    sign = "+" if base >= 0 else ""
    return f"{sign}{base}%"


def analyze_audio_bytes(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze audio bytes (WAV format) to extract:
    - duration_s: length in seconds
    - pitch_hz: estimated fundamental frequency
    - gender: 'male' or 'female'
    - ok: whether the sample is usable
    - problem: error description if not ok
    """
    result = {
        "format": "wav",
        "duration_s": None,
        "pitch_hz": None,
        "gender": None,
        "ok": True,
        "problem": "",
    }

    try:
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

        duration = n_frames / sample_rate
        result["duration_s"] = round(duration, 2)

        if duration < 3:
            result["ok"] = False
            result["problem"] = (
                f"The clip is {round(duration)} seconds. "
                "Please record at least 5 seconds of clear speech."
            )
            return result

        if duration > 120:
            result["ok"] = False
            result["problem"] = (
                f"The clip is {round(duration)} seconds. "
                "Please trim it to 60 seconds or less."
            )
            return result

        # Convert to mono float samples
        if sample_width == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
        elif sample_width == 4:
            fmt = f"<{n_frames * n_channels}i"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
        elif sample_width == 1:
            samples = np.array(list(raw), dtype=np.float64) - 128.0
        else:
            result["problem"] = "Unusual audio format. Please use a standard WAV file."
            return result

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Normalize
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak

        # Pitch detection using autocorrelation on frames
        frame_size = int(sample_rate * 0.04)  # 40ms frames
        lo_lag = int(sample_rate / 400)        # 400 Hz max
        hi_lag = int(sample_rate / 60)         # 60 Hz min
        f0_values = []

        for start in range(0, len(samples) - frame_size, frame_size * 3):
            frame = samples[start:start + frame_size]
            energy = np.sum(frame ** 2)
            rms = math.sqrt(energy / frame_size)

            # Skip silent/quiet frames
            if rms < 0.02:
                continue

            # Autocorrelation
            best_corr = 0
            best_lag = 0
            auto_e = energy

            for lag in range(lo_lag, min(hi_lag, frame_size)):
                corr = np.sum(frame[:frame_size - lag] * frame[lag:frame_size])
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            if best_lag > 0 and auto_e > 0 and (best_corr / auto_e) > 0.3:
                f0_values.append(sample_rate / best_lag)

            if len(f0_values) >= 200:
                break

        if len(f0_values) >= 3:
            f0_values.sort()
            median_f0 = f0_values[len(f0_values) // 2]
            result["pitch_hz"] = round(median_f0, 1)
            result["gender"] = classify_gender(median_f0)
        else:
            result["problem"] = (
                "Couldn't detect a clear voice pitch. "
                "Please record a clip with clear speech (not music or noise)."
            )

    except Exception as e:
        result["ok"] = False
        result["problem"] = f"Couldn't read the audio file: {str(e)}"

    return result


def convert_to_wav(audio_bytes: bytes, original_name: str = "") -> bytes:
    """
    Try to ensure audio bytes are WAV format.
    If they already have a WAV header, return as-is.
    Otherwise, attempt basic conversion.
    """
    # Check for WAV header
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    # For other formats, we write raw and hope for the best
    # In production you'd use ffmpeg or pydub
    return audio_bytes


def build_voice_profile(
    person: Dict[str, Any],
    lang: str = "en-IN",
) -> Dict[str, Any]:
    """
    Build a complete voice synthesis profile for a person.

    Returns dict with:
      - voice: edge-tts voice name
      - pitch: pitch offset string
      - rate: rate offset string
      - gender: detected gender
      - label: human-readable voice description
      - cloned: whether this is a "cloned" (personalized) voice
    """
    pitch_hz = person.get("voice_pitch")
    has_voice = bool(person.get("has_voice"))
    consent = bool(person.get("consent_voice_clone"))

    if not consent or not has_voice or not pitch_hz:
        # No voice sample or consent -> use system voice
        return {
            "voice": SYSTEM_VOICE,
            "pitch": SYSTEM_PITCH,
            "rate": SYSTEM_RATE,
            "gender": None,
            "label": "EchoBond (default voice)",
            "cloned": False,
        }

    gender = classify_gender(pitch_hz)
    profiles = VOICE_PROFILES.get(gender, {})
    profile = profiles.get(lang) or profiles.get("en-IN") or profiles.get("hi-IN")

    if not profile:
        return {
            "voice": SYSTEM_VOICE,
            "pitch": SYSTEM_PITCH,
            "rate": SYSTEM_RATE,
            "gender": gender,
            "label": "EchoBond (default voice)",
            "cloned": False,
        }

    pitch_offset = compute_pitch_offset(pitch_hz, profile["base_pitch"])
    rate_offset = compute_rate_offset(pitch_hz)

    return {
        "voice": profile["voice"],
        "pitch": pitch_offset,
        "rate": rate_offset,
        "gender": gender,
        "label": f"{profile['label']} (tuned to {pitch_hz:.0f} Hz)",
        "cloned": True,
    }


async def synthesize_speech(
    text: str,
    voice: str = SYSTEM_VOICE,
    pitch: str = SYSTEM_PITCH,
    rate: str = SYSTEM_RATE,
) -> bytes:
    """
    Synthesize speech using Edge TTS.
    Returns MP3 audio bytes.
    """
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        pitch=pitch,
        rate=rate,
    )

    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    return b"".join(audio_chunks)


def synthesize_speech_sync(
    text: str,
    voice: str = SYSTEM_VOICE,
    pitch: str = SYSTEM_PITCH,
    rate: str = SYSTEM_RATE,
) -> bytes:
    """Synchronous wrapper for synthesize_speech."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Flask dev server)
        # Create a new thread to run the async code
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                synthesize_speech(text, voice, pitch, rate),
            )
            return future.result(timeout=30)
    else:
        return asyncio.run(synthesize_speech(text, voice, pitch, rate))


def speak_as_person(
    text: str,
    person: Dict[str, Any],
    lang: str = "en-IN",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Synthesize speech in a person's cloned voice.

    Returns:
        (audio_bytes, profile_info)
    """
    profile = build_voice_profile(person, lang)
    audio = synthesize_speech_sync(
        text=text,
        voice=profile["voice"],
        pitch=profile["pitch"],
        rate=profile["rate"],
    )
    return audio, profile


def speak_as_echobond(text: str) -> bytes:
    """Synthesize speech in EchoBond's default voice."""
    return synthesize_speech_sync(
        text=text,
        voice=SYSTEM_VOICE,
        pitch=SYSTEM_PITCH,
        rate=SYSTEM_RATE,
    )
