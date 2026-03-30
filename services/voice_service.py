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
_EDGE_TTS_VOICES = {
    'hi': {'male': 'hi-IN-MadhurNeural',   'female': 'hi-IN-SwaraNeural'},
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

# Per-language BASE speed
_LANG_BASE_SPEED = {
    'en':    1.05,
    'hi':    1.00,
    'kn':    0.92,
    'ta':    0.94,
    'te':    0.95,
    'ml':    0.93,
    'mr':    0.98,
    'gu':    1.00,
    'ar':    0.97,
    'zh-cn': 1.10,
    'ja':    1.02,
    'ko':    1.02,
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
    Accurately determine speaker gender using a robust multi-method F0 pipeline.

    Strategy:
      1. Use librosa.pyin (probabilistic YIN) — far more accurate than plain YIN
         for speech, especially short clips. It assigns a voiced probability to
         each frame so unvoiced/uncertain frames are automatically excluded.
      2. Statistical filtering: use the 25th–75th percentile (interquartile range)
         of voiced F0 frames to discard pitch outliers from background noise.
      3. Confidence check: if fewer than 15% of frames are confidently voiced,
         the sample is too short/noisy — run a second pass on a wider window
         before falling back to 'male'.
      4. Gender bands (Hz):
           Male:   65 – 180 Hz  → median in this range = male
           Female: 160 – 300 Hz → median in this range = female
           Overlap (160–180): leaned toward female because misclassifying a
           female as male is more noticeable to listeners.
    Returns 'male' or 'female'.
    """
    try:
        import librosa

        duration = min(end - start, 20.0)
        y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration, mono=True)

        if len(y) < sr * 0.3:
            print(f"[GenderDetect] {start:.1f}s–{end:.1f}s | Clip too short → defaulting to 'male'")
            return 'male'

        # ── Method 1: pyin (probabilistic YIN) ─────────────────────────────
        # Returns F0 per frame + voiced flag + voiced probability
        f0_frames, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz  — lowest male pitch
            fmax=librosa.note_to_hz('D5'),   # ~587 Hz — well above female max
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )

        # Keep only frames that pyin is confident are voiced (probability > 0.6)
        confident_voiced = (voiced_probs > 0.6) & voiced_flag & (f0_frames > 60)
        voiced_f0 = f0_frames[confident_voiced]

        # ── Confidence check ─────────────────────────────────────────────────
        voiced_ratio = len(voiced_f0) / max(len(f0_frames), 1)
        if voiced_ratio < 0.10 or len(voiced_f0) < 10:
            # Too little voiced speech detected — try a slightly looser threshold
            fallback_mask = (voiced_flag) & (f0_frames > 60) & (f0_frames < 600)
            voiced_f0 = f0_frames[fallback_mask]
            print(f"[GenderDetect] Low confidence ({voiced_ratio:.1%} voiced) — using fallback mask, "
                  f"n={len(voiced_f0)} frames")
            if len(voiced_f0) < 5:
                print(f"[GenderDetect] Insufficient voiced frames → defaulting to 'male'")
                return 'male'

        # ── IQR-based robust central tendency ────────────────────────────────
        # Use 25th–75th percentile to exclude pitch outliers from laughter,
        # background speech, music, etc.
        p25 = float(np.percentile(voiced_f0, 25))
        p75 = float(np.percentile(voiced_f0, 75))
        iqr_mask = (voiced_f0 >= p25) & (voiced_f0 <= p75)
        core_f0 = voiced_f0[iqr_mask] if iqr_mask.sum() > 3 else voiced_f0
        representative_f0 = float(np.median(core_f0))

        # ── Gender classification ─────────────────────────────────────────────
        # Female range: 160–300 Hz (lower bound 160 to catch contralto speakers)
        # Male   range:  65–180 Hz
        # Overlap 160–180 Hz → classify as female (less jarring to listeners)
        if representative_f0 >= 160.0:
            gender = 'female'
        else:
            gender = 'male'

        print(f"[GenderDetect] {start:.1f}s–{end:.1f}s | "
              f"F0 (IQR median)={representative_f0:.1f}Hz | "
              f"voiced={voiced_ratio:.1%} | → {gender}")
        return gender

    except Exception as e:
        print(f"[GenderDetect] Error: {e}. Defaulting to 'male'.")
        return 'male'


class VoiceCloningService:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.cloned_dir = self.work_dir / "cloned_audio"
        self.cloned_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[VoiceCloningService] Loading XTTSv2 Voice Cloning model... This may take a while")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def extract_speaker_sample(self, reference_audio: str, start: float, end: float, tag: str = "") -> str:
        """Extract a clean audio sample of the speaker for cloning."""
        audio = AudioSegment.from_file(reference_audio)
        sample = audio[start * 1000: end * 1000]
        sample = sample.normalize()
        sample = sample.high_pass_filter(50)    # Allows deep bass frequencies (was 100Hz)
        sample = sample.low_pass_filter(15000)  # Allows high clarity/air (was 8000Hz)
        filename = f"sample_{tag}_{start:.1f}_{end:.1f}.wav" if tag else f"sample_{start:.1f}_{end:.1f}.wav"
        sample_path = str(self.work_dir / filename)
        sample.export(sample_path, format="wav")
        return sample_path

    def _build_speaker_profiles(self, transcript: list, reference_audio: str) -> dict:
        """
        Build a persistent identity profile for each unique speaker_id in the transcript.

        For each speaker:
          - Find their longest continuous segment (best for pitch/gender analysis).
          - Analyse pitch over that segment to determine their gender once and for all.
          - Extract a clean reference audio sample from their longest segment for XTTS cloning.

        Returns:
            dict keyed by speaker_id:
            {
                'SPEAKER_01': {'gender': 'male',   'sample': '/path/to/sample.wav'},
                'SPEAKER_02': {'gender': 'female', 'sample': '/path/to/sample.wav'},
                ...
            }
        """
        # --- Group segments by speaker ---
        speaker_segments = {}
        for seg in transcript:
            sid = seg.get('speaker_id', 'SPEAKER_01')
            speaker_segments.setdefault(sid, []).append(seg)

        profiles = {}
        for sid, segs in speaker_segments.items():
            # Pick the longest segment for this speaker (best pitch signal)
            best_seg = max(segs, key=lambda s: s['end'] - s['start'])
            dur = best_seg['end'] - best_seg['start']

            # Determine gender once for this speaker using their best segment
            gender = _detect_gender_from_audio(reference_audio, best_seg['start'], best_seg['end'])

            # Extract a clean reference sample (cap at 12s for XTTS quality)
            sample = None
            if dur >= 2.0:
                try:
                    sample_end = min(best_seg['start'] + 12.0, best_seg['end'])
                    sample = self.extract_speaker_sample(
                        reference_audio, best_seg['start'], sample_end, tag=sid
                    )
                except Exception as e:
                    print(f"[VoiceCloningService] Could not extract sample for {sid}: {e}")

            # If no good segment found, use first 10s as fallback
            if sample is None:
                try:
                    sample = self.extract_speaker_sample(reference_audio, 0.0, 10.0, tag=f"{sid}_fallback")
                except Exception as e:
                    print(f"[VoiceCloningService] Fallback sample failed for {sid}: {e}")

            profiles[sid] = {'gender': gender, 'sample': sample}
            print(f"[VoiceCloningService] Speaker profile: {sid} → gender={gender}, "
                  f"sample={'✓' if sample else '✗'} (longest seg {dur:.1f}s)")

        return profiles

    def _compute_segment_speed(self, text: str, duration_sec: float,
                               base_speed: float, language: str = 'en') -> float:
        """Compute per-segment auto-speed calibrated to the target language."""
        if duration_sec <= 0.5:
            return base_speed

        char_count = len(text.replace(" ", ""))
        word_count = len(text.split())

        if language == 'en':
            syllable_count = word_count * 1.5
            natural_rate_sps = 4.0
        elif language in ['hi', 'mr', 'gu', 'bn']:
            syllable_count = char_count * 0.5
            natural_rate_sps = 4.0
        elif language in ['kn', 'ta', 'te', 'ml']:
            syllable_count = char_count * 0.45
            natural_rate_sps = 4.5
        else:
            syllable_count = word_count * 2.0
            natural_rate_sps = 4.0

        estimated_natural_duration = syllable_count / natural_rate_sps
        if estimated_natural_duration <= 0:
            return base_speed

        required_speed = (estimated_natural_duration / duration_sec) * base_speed
        
        # We heavily prioritize the natural base speed of the target language
        # to prevent unnatural chipmunk or slow-motion sounds.
        # We only apply a 15% pull towards the 'required' video duration speed.
        blended_speed = 0.15 * required_speed + 0.85 * base_speed
        
        clamped_speed = max(_MIN_SPEED, min(_MAX_SPEED, blended_speed))

        print(f"[VoiceCloningService] [{language}] Speed: {clamped_speed:.3f}x "
              f"(chars={char_count}, required={required_speed:.3f}, dur={duration_sec:.1f}s)")
        return clamped_speed

    def _generate_with_edge_tts(self, text: str, language: str, gender: str,
                                out_path: str, speed: float = 1.0) -> bool:
        """Generate speech using Microsoft Edge TTS with gender-specific neural voices."""
        try:
            import edge_tts
            voices = _EDGE_TTS_VOICES.get(language, _EDGE_TTS_DEFAULT)
            voice_name = voices.get(gender, voices.get('male'))

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
        Generate gender-consistent speech for each segment.

        Key fix: Gender and voice sample are determined ONCE per unique speaker_id
        (using their longest segment for the most reliable pitch analysis), then
        applied consistently to ALL that speaker's segments. This prevents a single
        speaker from randomly alternating between a male and female voice.

        For XTTS languages (en, hi...): clones the actual speaker's voice.
        For non-XTTS languages (kn, ta...): uses gender-matched Edge-TTS neural voices.
        """
        use_xtts = language in _XTTS_SUPPORTED
        print(f"[VoiceCloningService] Generating {language} speech for {len(transcript)} segments "
              f"(xtts={'yes' if use_xtts else 'no/edge-tts'}, persistent-speaker-profiles enabled)...")

        # ── Build ONE profile per speaker (gender + sample) — the core fix ─────
        speaker_profiles = self._build_speaker_profiles(transcript, reference_audio)
        print(f"[VoiceCloningService] Built profiles for {len(speaker_profiles)} unique speaker(s): "
              f"{list(speaker_profiles.keys())}")

        lang_base_speed = _LANG_BASE_SPEED.get(language, _LANG_BASE_SPEED['_default'])
        print(f"[VoiceCloningService] [{language}] Base speed: {lang_base_speed:.2f}x")

        cloned_segments = []
        for i, segment in enumerate(transcript):
            text = segment.get("text", "").strip()
            if not text:
                print(f"[VoiceCloningService] Skipping empty segment {i}")
                continue

            sid = segment.get("speaker_id", "SPEAKER_01")
            # Look up this speaker's persistent profile
            profile = speaker_profiles.get(sid, speaker_profiles.get("SPEAKER_01", {}))
            segment_gender = profile.get('gender', 'male')
            reference_sample = profile.get('sample')

            out_path = str(self.cloned_dir / f"segment_{i}.wav")
            segment_duration = segment["end"] - segment["start"]
            segment_speed = self._compute_segment_speed(
                text, segment_duration, base_speed=lang_base_speed, language=language
            )

            print(f"[VoiceCloningService] Seg {i} [{sid}]: '{text[:40]}…' "
                  f"| gender={segment_gender} | speed={segment_speed:.3f}")

            # We trick XTTS into enthusiastically speaking Kannada by transliterating 
            # the text into phonetic Devanagari (Hindi) which XTTS proudly supports!
            xtts_text = text
            xtts_language = language
            segment_use_xtts = use_xtts
            
            if language == 'kn':
                try:
                    from indic_transliteration import sanscript
                    # Transform Kannada Unicode -> Devanagari Unicode
                    xtts_text = sanscript.transliterate(text, sanscript.KANNADA, sanscript.DEVANAGARI)
                    xtts_language = 'hi'  # Trick the model into reading it phonetically
                    segment_use_xtts = True
                    print(f"[VoiceCloningService] Transliterated KN→HI for XTTS: '{xtts_text[:20]}…'")
                except ImportError:
                    print("[VoiceCloningService] indic-transliteration missing! Falling back to Edge-TTS.")
                    segment_use_xtts = False

            # ── XTTS path: voice cloning with dynamic emotion & phonetic mapping ──
            if segment_use_xtts:
                if not reference_sample or not os.path.exists(reference_sample):
                    print(f"[VoiceCloningService] No sample for {sid} — falling back to Edge-TTS")
                    if not self._generate_with_edge_tts(text, language, segment_gender, out_path, speed=segment_speed):
                        continue
                else:
                    # DYNAMIC EMOTION REFERENCING
                    # We pass the current segment's original audio to capture its exact emotion,
                    # combined with the primary reference sample to retain strict speaker identity.
                    current_emotion_sample = reference_sample
                    if segment_duration >= 2.0:
                        try:
                            # Extract the exact sentence from the original video
                            dynamic_sample = self.extract_speaker_sample(
                                reference_audio,
                                segment["start"],
                                segment["end"],
                                tag=f"seg_{i}_emotion"
                            )
                            # XTTS accepts a list: [Emotional Prosody Sample, Primary Identity Sample]
                            current_emotion_sample = [dynamic_sample, reference_sample]
                            print(f"[VoiceCloningService] Seg {i}: Infusing local emotion from original voice.")
                        except Exception as e:
                            print(f"[VoiceCloningService] Seg {i} emotion extract failed: {e}")

                    try:
                        self.tts.tts_to_file(
                            text=xtts_text,
                            speaker_wav=current_emotion_sample,
                            language=xtts_language,
                            file_path=out_path,
                            speed=segment_speed
                        )
                    except TypeError:
                        self.tts.tts_to_file(
                            text=xtts_text, speaker_wav=current_emotion_sample,
                            language=xtts_language, file_path=out_path
                        )
                    except Exception as e:
                        print(f"[VoiceCloningService] XTTS error seg {i}: {e} → Edge-TTS fallback")
                        if not self._generate_with_edge_tts(text, language, segment_gender, out_path, speed=segment_speed):
                            continue

            # ── Edge-TTS path: gender-specific neural voices (Kannada, etc.) ──
            else:
                if not self._generate_with_edge_tts(text, language, segment_gender, out_path, speed=segment_speed):
                    continue

            cloned_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "audio_path": out_path,
                "gender": segment_gender,
                "speaker_id": sid
            })

        return cloned_segments
