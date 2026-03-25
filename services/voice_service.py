import os
import io
import torch
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from TTS.api import TTS

# PyTorch 2.6 security fix: Allow XTTS config classes to be loaded
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
except ImportError:
    pass

# Languages officially supported by XTTSv2 voice cloning model
_XTTS_SUPPORTED = {
    'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru',
    'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'
}

# Edge-TTS voice names per language per gender
# Edge-TTS supports gender-specific neural voices for Kannada and other Indic languages
_EDGE_TTS_VOICES = {
    'kn': {'male': 'kn-IN-GaganNeural',   'female': 'kn-IN-SapnaNeural'},
    'ta': {'male': 'ta-IN-ValluvarNeural', 'female': 'ta-IN-PallaviNeural'},
    'te': {'male': 'te-IN-MohanNeural',    'female': 'te-IN-ShrutiNeural'},
    'ml': {'male': 'ml-IN-MidhunNeural',   'female': 'ml-IN-SobhanaNeural'},
    'bn': {'male': 'bn-IN-BashkarNeural',  'female': 'bn-IN-TanishaaNeural'},
    'gu': {'male': 'gu-IN-NiranjanNeural', 'female': 'gu-IN-DhwaniNeural'},
    'mr': {'male': 'mr-IN-ManoharNeural',  'female': 'mr-IN-AarohiNeural'},
}

# Default edge-tts fallback if language not in map
_EDGE_TTS_DEFAULT = {'male': 'en-IN-PrabhatNeural', 'female': 'en-IN-NeerjaNeural'}

# Per-language TTS parameters — syllable density and natural speaking rate
# syllables_per_word: avg syllables per translated word in that language
# natural_rate_sps:   syllables per second at a comfortable natural pace
# These calibrate the auto-speed formula so each language sounds natural
_LANG_TTS_PARAMS = {
    'en': {'syllables_per_word': 1.5, 'natural_rate_sps': 3.3},  # English: short, fast
    'hi': {'syllables_per_word': 2.1, 'natural_rate_sps': 2.8},  # Hindi: moderate density
    'kn': {'syllables_per_word': 2.5, 'natural_rate_sps': 2.4},  # Kannada: dense, slower
    'ta': {'syllables_per_word': 2.4, 'natural_rate_sps': 2.5},  # Tamil
    'te': {'syllables_per_word': 2.3, 'natural_rate_sps': 2.6},  # Telugu
    'mr': {'syllables_per_word': 2.0, 'natural_rate_sps': 2.9},  # Marathi
    'gu': {'syllables_per_word': 1.9, 'natural_rate_sps': 3.0},  # Gujarati
    'ar': {'syllables_per_word': 2.0, 'natural_rate_sps': 2.8},  # Arabic
    'zh-cn': {'syllables_per_word': 1.0, 'natural_rate_sps': 3.8},  # Mandarin: 1 char ≈ 1 syllable
    'ja': {'syllables_per_word': 1.8, 'natural_rate_sps': 3.2},  # Japanese
    'ko': {'syllables_per_word': 1.7, 'natural_rate_sps': 3.2},  # Korean
    # Default fallback for unsupported or unknown languages
    '_default': {'syllables_per_word': 1.8, 'natural_rate_sps': 3.0},
}

# Per-language BASE speed — the natural starting speed before per-segment fine-tuning.
# Each language has a culturally calibrated pace: English is brisk, Kannada is measured, etc.
_LANG_BASE_SPEED = {
    'en':    1.05,  # English: slightly brisk, matches natural fast English pace
    'hi':    1.00,  # Hindi: neutral — Devanagari TTS already well-paced
    'kn':    0.92,  # Kannada: dense agglutinative words — needs more room to breathe
    'ta':    0.94,  # Tamil: similar density to Kannada
    'te':    0.95,  # Telugu
    'ml':    0.93,  # Malayalam: very dense conjunct clusters
    'mr':    0.98,  # Marathi
    'gu':    1.00,  # Gujarati
    'ar':    0.97,  # Arabic
    'zh-cn': 1.10,  # Mandarin: monosyllabic, can go faster
    'ja':    1.02,  # Japanese
    'ko':    1.02,  # Korean
    '_default': 1.0,
}
# XTTS safe speed clamping range
_MIN_SPEED = 0.85
_MAX_SPEED = 1.45

# Pitch boundary between typical male and female F0 (Hz)
# Males: 85–180 Hz, Females: 165–255 Hz  → threshold at 165 Hz
_GENDER_PITCH_THRESHOLD_HZ = 165.0


def _detect_gender_from_audio(audio_path: str, start: float, end: float) -> str:
    """
    Estimate speaker gender using fundamental frequency (F0) analysis.
    Analyses a slice of the audio between start–end seconds.

    Uses librosa's yin algorithm for F0 estimation.
    Returns 'male' or 'female'.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, offset=start, duration=min(end - start, 10.0), mono=True)
        if len(y) < sr * 0.5:
            return 'male'  # too short to analyse — default
        # YIN algorithm: accurate for speech F0
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        # Filter out zeros (unvoiced frames)
        voiced_f0 = f0[f0 > 60]
        if len(voiced_f0) == 0:
            return 'male'
        median_f0 = float(np.median(voiced_f0))
        gender = 'female' if median_f0 >= _GENDER_PITCH_THRESHOLD_HZ else 'male'
        print(f"[GenderDetect] {start:.1f}s–{end:.1f}s | median F0={median_f0:.1f}Hz → {gender}")
        return gender
    except Exception as e:
        print(f"[GenderDetect] Failed: {e}. Defaulting to 'male'.")
        return 'male'


class VoiceCloningService:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.cloned_dir = self.work_dir / "cloned_audio"
        self.cloned_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[VoiceCloningService] Loading XTTSv2 Voice Cloning model... This may take a while")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def extract_speaker_sample(self, reference_audio: str, start: float, end: float) -> str:
        """Extract a small clean sample of the speaker for cloning, filtering out background noise."""
        audio = AudioSegment.from_file(reference_audio)
        sample = audio[start * 1000: end * 1000]
        sample = sample.normalize()
        sample = sample.high_pass_filter(100)
        sample = sample.low_pass_filter(8000)
        sample_path = str(self.work_dir / f"sample_{start}_{end}.wav")
        sample.export(sample_path, format="wav")
        return sample_path

    def _build_gender_sample_map(self, transcript: list, reference_audio: str) -> dict:
        """
        Build gender-keyed reference sample map: {'male': path, 'female': path}.
        Analyses each segment with a duration ≥ 3s, detects gender, and picks the
        best (longest) audio clip for each gender.
        Falls back to the single best sample if only one gender is detected.
        """
        gender_best = {'male': None, 'female': None}
        gender_best_dur = {'male': 0.0, 'female': 0.0}

        for seg in transcript:
            dur = seg['end'] - seg['start']
            if dur < 3.0 or dur > 15.0:
                continue
            gender = _detect_gender_from_audio(reference_audio, seg['start'], seg['end'])
            if dur > gender_best_dur[gender]:
                gender_best_dur[gender] = dur
                try:
                    sample_end = min(seg['start'] + 12.0, seg['end'])
                    gender_best[gender] = self.extract_speaker_sample(
                        reference_audio, seg['start'], sample_end
                    )
                except Exception as e:
                    print(f"[VoiceCloningService] Failed extracting {gender} sample: {e}")

        # Fallback: if one gender wasn't found, use the other for both
        fallback = gender_best.get('male') or gender_best.get('female')
        if fallback is None:
            # Ultimate fallback: use first 10s
            fallback = self.extract_speaker_sample(reference_audio, 0.0, 10.0)
        if gender_best['male'] is None:
            gender_best['male'] = fallback
        if gender_best['female'] is None:
            gender_best['female'] = fallback

        print(f"[VoiceCloningService] Gender sample map: "
              f"male={'✓' if gender_best['male'] else '✗'}, "
              f"female={'✓' if gender_best['female'] else '✗'}")
        return gender_best

    def _compute_segment_speed(self, text: str, duration_sec: float,
                               base_speed: float, language: str = 'en') -> float:
        """
        Compute per-segment auto-speed calibrated to the target language.
        Prioritizes natural-sounding speech over strict lip-sync timing.
        """
        if duration_sec <= 0.5:
            return base_speed

        # Character-based syllable estimation for better accuracy in Indic/Asian languages
        char_count = len(text.replace(" ", ""))
        word_count = len(text.split())
        
        if language == 'en':
            syllable_count = word_count * 1.5
            natural_rate_sps = 4.0
        elif language in ['hi', 'mr', 'gu', 'bn']:
            syllable_count = char_count * 0.5  # ~2 chars per syllable natively
            natural_rate_sps = 4.0
        elif language in ['kn', 'ta', 'te', 'ml']:
            syllable_count = char_count * 0.45 # Agglutinative, dense words
            natural_rate_sps = 4.5
        else:
            syllable_count = word_count * 2.0
            natural_rate_sps = 4.0

        estimated_natural_duration = syllable_count / natural_rate_sps

        if estimated_natural_duration <= 0:
            return base_speed

        # The speed required to fit into the original video duration
        required_speed = (estimated_natural_duration / duration_sec) * base_speed

        # 40% forcing to fit time, 60% natural speed (more natural, less chipmunk)
        blended_speed = 0.4 * required_speed + 0.6 * base_speed
        
        # Allow up to 1.45x for TTS frameworks which handle speedups okay natively
        clamped_speed = max(0.85, min(1.45, blended_speed))
        
        print(f"[VoiceCloningService] [{language}] Speed: {clamped_speed:.3f}x "
              f"(chars={char_count}, required={required_speed:.3f}, dur={duration_sec:.1f}s)")
        return clamped_speed

    def _generate_with_edge_tts(self, text: str, language: str, gender: str,
                                out_path: str, speed: float = 1.0) -> bool:
        """
        Generate speech using Microsoft Edge TTS (neural voices).
        Supports gender-specific voices and language-calibrated speaking rate.
        The `speed` parameter comes from _compute_segment_speed (language-aware).
        Edge-TTS accepts rate as a percentage string e.g. '+10%' or '-5%'.
        """
        try:
            import edge_tts
            voices = _EDGE_TTS_VOICES.get(language, _EDGE_TTS_DEFAULT)
            voice_name = voices.get(gender, voices.get('male'))

            # Convert speed multiplier to Edge-TTS rate percentage
            # speed=1.0 → '+0%', speed=1.2 → '+20%', speed=0.9 → '-10%'
            rate_pct = int(round((speed - 1.0) * 100))
            rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

            async def _run():
                communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_str)
                mp3_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_data += chunk["data"]
                return mp3_data

            mp3_data = asyncio.run(_run())
            if not mp3_data:
                raise ValueError("Edge-TTS returned empty audio")

            # Convert MP3 bytes → WAV via pydub
            audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            audio.export(out_path, format="wav")
            print(f"[VoiceCloningService] Edge-TTS ✓ voice='{voice_name}' gender='{gender}'")
            return True
        except Exception as e:
            print(f"[VoiceCloningService] Edge-TTS failed ({e}), falling back to gTTS")
            return self._generate_with_gtts_fallback(text, language, out_path)

    def _generate_with_gtts_fallback(self, text: str, language: str, out_path: str) -> bool:
        """Last-resort fallback using gTTS (no gender control)."""
        try:
            from gtts import gTTS
            tts_obj = gTTS(text=text, lang=language, slow=False)
            mp3_buffer = io.BytesIO()
            tts_obj.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            audio.export(out_path, format="wav")
            print(f"[VoiceCloningService] gTTS fallback ✓ language='{language}'")
            return True
        except Exception as e:
            print(f"[VoiceCloningService] gTTS also failed: {e}")
            return False

    def generate_speech(self, transcript: list, reference_audio: str, language: str = "en", speed: float = 1.0) -> list:
        """
        Generate gender-aware speech for each segment.

        For XTTS-supported languages (en, hi…):
          - Detect each segment's speaker gender via pitch analysis
          - Select a gender-matched XTTS reference sample so the cloned voice
            sounds like the correct gender.

        For XTTS-unsupported languages (kn, ta, te…):
          - Use Edge-TTS neural voices (gender-specific: e.g. kn-IN-GaganNeural ♂,
            kn-IN-SapnaNeural ♀) — no voice cloning, but correct gender voice.
        """
        use_xtts = language in _XTTS_SUPPORTED
        print(f"[VoiceCloningService] Generating {language} speech for {len(transcript)} segments "
              f"(xtts={'yes' if use_xtts else 'no/edge-tts'}, auto-speed enabled)...")

        # Build gender-keyed reference sample map (for XTTS languages)
        if use_xtts:
            gender_sample_map = self._build_gender_sample_map(transcript, reference_audio)
        else:
            gender_sample_map = {}

        # Select the language-specific base speed — each language has its own natural pace.
        # This replaces the single pipeline-level speed value with a per-language calibrated one.
        lang_base_speed = _LANG_BASE_SPEED.get(language, _LANG_BASE_SPEED['_default'])
        print(f"[VoiceCloningService] [{language}] Using language base speed: {lang_base_speed:.2f}x")

        cloned_segments = []
        for i, segment in enumerate(transcript):
            text = segment.get("text", "").strip()
            if not text:
                print(f"[VoiceCloningService] Skipping empty segment {i}")
                continue

            out_path = str(self.cloned_dir / f"segment_{i}.wav")
            segment_duration = segment["end"] - segment["start"]
            # Fine-tune per-segment speed using the language-specific base (not the generic pipeline speed)
            segment_speed = self._compute_segment_speed(
                text, segment_duration, base_speed=lang_base_speed, language=language
            )

            # Detect gender of this specific segment
            segment_gender = _detect_gender_from_audio(
                reference_audio, segment["start"], segment["end"]
            )
            print(f"[VoiceCloningService] Segment {i}: '{text[:35]}…' "
                  f"| gender={segment_gender} | speed={segment_speed:.3f}")

            # ── XTTS-supported language: gender-matched voice cloning ──────────
            if use_xtts:
                # Use the gender-matching reference sample
                reference_sample = gender_sample_map.get(segment_gender, gender_sample_map.get('male'))

                # Also try a dynamic sample from this exact segment if it's long enough
                if segment_duration >= 3.0:
                    try:
                        sample_end = min(segment["start"] + 12.0, segment["end"])
                        dynamic_sample = self.extract_speaker_sample(
                            reference_audio, segment["start"], sample_end
                        )
                        # Only use dynamic sample if it matches the detected gender
                        seg_gender_check = _detect_gender_from_audio(
                            reference_audio, segment["start"], min(segment["start"] + 5.0, segment["end"])
                        )
                        if seg_gender_check == segment_gender:
                            reference_sample = dynamic_sample
                    except Exception as e:
                        print(f"[VoiceCloningService] Dynamic sample failed: {e}")

                if not reference_sample or not os.path.exists(reference_sample):
                    print(f"[VoiceCloningService] ERROR: No valid reference sample for segment {i}")
                    continue

                try:
                    self.tts.tts_to_file(
                        text=text,
                        speaker_wav=reference_sample,
                        language=language,
                        file_path=out_path,
                        speed=segment_speed
                    )
                except TypeError:
                    self.tts.tts_to_file(
                        text=text, speaker_wav=reference_sample,
                        language=language, file_path=out_path
                    )
                except (AssertionError, Exception) as e:
                    print(f"[VoiceCloningService] XTTS error seg {i}: {e} → Edge-TTS fallback")
                    if not self._generate_with_edge_tts(text, language, segment_gender, out_path, speed=segment_speed):
                        continue

            # ── Non-XTTS language (Kannada etc.): gender-specific Edge-TTS ───
            else:
                if not self._generate_with_edge_tts(text, language, segment_gender, out_path, speed=segment_speed):
                    continue

            cloned_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "audio_path": out_path,
                "gender": segment_gender
            })

        return cloned_segments
