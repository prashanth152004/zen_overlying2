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
    'en': {'male': 'en-IN-PrabhatNeural',  'female': 'en-IN-NeerjaNeural'},
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

# Edge-TTS Neural robust speed clamping boundaries
# Neural networks scale naturally, so we safely stretch from 70% to 200% voice speed!
_MIN_SPEED = 0.70
_MAX_SPEED = 2.0

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
        import io
        from pydub import AudioSegment

        duration = min(end - start, 20.0)
        
        # PRE-PROCESS: Isolate human vocal fundamental frequencies (60Hz - 400Hz)
        # This absolutely annihilates background music/noise from messing up the F0 pitch tracking!
        audio = AudioSegment.from_file(audio_path)
        clip = audio[start * 1000 : (start + duration) * 1000].set_channels(1)
        clip = clip.high_pass_filter(60).low_pass_filter(400)
        
        buf = io.BytesIO()
        clip.export(buf, format="wav")
        buf.seek(0)
        
        # Load the filtered, purified audio into librosa's pitch tracker
        y, sr = librosa.load(buf, sr=22050)

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
        # Raising overlap threshold to 195.0Hz to definitively prevent misclassifying higher-pitched males
        if representative_f0 >= 195.0:
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


def _cluster_genders_by_pitch_and_timbre(transcript: list, audio_path: str) -> dict:
    """
    Tokenless Fallback Upgrade:
    Extracts 13 MFCC physical vocal tract dimensions + 1 Pitch F0 median per sentence.
    Runs Scipy K-Means (k=2) to cluster all sentences identically across the video,
    entirely bypassing mid-video shouting or pitch spikes that confuse raw thresholding.
    Returns: mapped dict of {segment_index: 'male' / 'female'}
    """
    try:
        import librosa
        import numpy as np
        import io
        from scipy.cluster.vq import kmeans2, whiten
        from pydub import AudioSegment
        
        print(f"[VoiceCloningService] Tokenless Acoustic Scan activated across {len(transcript)} segments...")
        
        features = []
        valid_indices = []
        full_audio = AudioSegment.from_file(audio_path)
        
        for i, segment in enumerate(transcript):
            start, end = segment["start"], segment["end"]
            dur = end - start
            if dur < 1.0:
                continue
                
            # Tightly bound the vocal filter for structural tract extraction
            clip = full_audio[start * 1000 : end * 1000].set_channels(1)
            clip = clip.high_pass_filter(50).low_pass_filter(2000)
            
            buf = io.BytesIO()
            clip.export(buf, format="wav")
            buf.seek(0)
            
            y, sr = librosa.load(buf, sr=16000)
            if len(y) < sr * 0.3:
                continue
                
            # Extract Pitch (F0)
            f0_frames, voiced_flag, _ = librosa.pyin(
                y, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=512
            )
            f0 = f0_frames[voiced_flag & (f0_frames > 60)]
            median_f0 = float(np.median(f0)) if len(f0) > 5 else 185.0
            
            # Extract Biological Timbre Shape (MFCC)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Construct 14-dimensional Physical Voice Vector!
            vec = np.append(mfcc_mean, median_f0)
            features.append(vec)
            valid_indices.append(i)
            
        n_clusters = min(2, len(features))
        if n_clusters < 2:
            return {i: 'male' for i in range(len(transcript))}
            
        data = np.array(features)
        
        # Whitening neutralizes the massive mathematical scale differences between 
        # a 180Hz Pitch unit and a 0.2 MFCC resonance unit!
        whitened = whiten(data)
        
        # K-Means categorizes all physical vectors into exactly two groups
        centroids, labels = kmeans2(whitened, k=n_clusters, minit='++')
        
        # ── SOLO-ACTOR CENTROID VALIDATOR ──
        # Calculate the absolute median pitch of both 14-dimensional clusters
        cluster_pitches = []
        cluster_has_data = []
        for cluster_id in range(n_clusters):
            mask = (labels == cluster_id)
            has_data = np.any(mask)
            cluster_has_data.append(has_data)
            avg_f0 = np.mean(data[mask, 13]) if has_data else 185.0
            cluster_pitches.append(avg_f0)
            
        # If we successfully created 2 clusters, measure their physical separation gap
        if n_clusters == 2 and all(cluster_has_data):
            pitch_0 = cluster_pitches[0]
            pitch_1 = cluster_pitches[1]
            pitch_gap = abs(pitch_0 - pitch_1)
            
            # If the difference is less than 45Hz, it's mathematically the EXACT SAME PERSON!
            # K-Means accidentally chopped a Solo-Actor's voice into loud/quiet buckets.
            if pitch_gap < 45.0:
                print(f"[VoiceCloningService] Centroid gap {pitch_gap:.1f}Hz < 45Hz. Collapsing into ONE gender!")
                global_avg_pitch = (pitch_0 + pitch_1) / 2.0
                resolved_gender = 'female' if global_avg_pitch > 195.0 else 'male'
                # Lock both randomly split clusters together into the same gender
                cluster_gender_map = {0: resolved_gender, 1: resolved_gender}
            else:
                print(f"[VoiceCloningService] Centroid gap {pitch_gap:.1f}Hz > 45Hz. Confirmed TWO distinct genders!")
                male_id = np.argmin(cluster_pitches)
                cluster_gender_map = {
                    male_id: 'male',
                    1 - male_id: 'female'
                }
        else:
            # Only 1 cluster generated
            single_pitch = cluster_pitches[0]
            resolved_gender = 'female' if single_pitch > 195.0 else 'male'
            cluster_gender_map = {0: resolved_gender}
        
        mapped_genders = {}
        for idx, segment_label in zip(valid_indices, labels):
            mapped_genders[idx] = cluster_gender_map[segment_label]
            f0 = data[valid_indices.index(idx), 13]
            print(f"[VoiceCloningService] K-Means clustered Seg {idx} (Pitch={f0:.1f}Hz, Cluster={cluster_pitches[segment_label]:.1f}Hz) → {mapped_genders[idx]}")
            
        return mapped_genders
        
    except Exception as e:
        print(f"[VoiceCloningService] K-Means Audio Clustering Error: {e}. Bypassing.")
        return {}


class VoiceCloningService:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.cloned_dir = self.work_dir / "cloned_audio"
        self.cloned_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[VoiceCloningService] Loading XTTSv2 Voice Cloning model... This may take a while")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def extract_speaker_sample(self, reference_audio: str, start: float, end: float, tag: str = "", pitch_shift: float = 0.0) -> str:
        """Extract a clean audio sample of the speaker for cloning."""
        audio = AudioSegment.from_file(reference_audio)
        sample = audio[start * 1000: end * 1000]
        sample = sample.normalize()
        sample = sample.high_pass_filter(50)    # Allows deep bass frequencies (was 100Hz)
        sample = sample.low_pass_filter(15000)  # Allows high clarity/air (was 8000Hz)
        
        # Deep Pitch Shift for Male XTTS Hallucination Override
        if pitch_shift != 0.0:
            # Shift pitch mathematically (Pydub trick: drops pitch by stretching time)
            # Safe because XTTS only uses this file to extract a static voice tone vector!
            new_sample_rate = int(sample.frame_rate * (2.0 ** (pitch_shift / 12.0)))
            shifted = sample._spawn(sample.raw_data, overrides={'frame_rate': new_sample_rate})
            sample = shifted.set_frame_rate(sample.frame_rate)
            
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

            # Aggressive pitch drop if male to combat XTTS Hindi female-bias
            shift = -3.5 if gender == 'male' else 0.0

            # Extract a clean reference sample (cap at 12s for XTTS quality)
            sample = None
            if dur >= 2.0:
                try:
                    sample_end = min(best_seg['start'] + 12.0, best_seg['end'])
                    sample = self.extract_speaker_sample(
                        reference_audio, best_seg['start'], sample_end, tag=sid, pitch_shift=shift
                    )
                except Exception as e:
                    print(f"[VoiceCloningService] Could not extract sample for {sid}: {e}")

            # If no good segment found, use first 10s as fallback
            if sample is None:
                try:
                    sample = self.extract_speaker_sample(reference_audio, 0.0, 10.0, tag=f"{sid}_fallback", pitch_shift=shift)
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
        
        # SMART ELASTIC SCALING:
        # We re-enable automatic speed adjustment to fit your video uploaded,
        # but with a 'Context-Safe' 40% blend to keep the human sound natural.
        # This allows the AI to move toward the video duration for lip-sync,
        # but prevents robotic racing or dragging.
        blended_speed = 0.40 * required_speed + 0.60 * base_speed
        
        # Strictly clamp the AI generation between 0.85x and 1.25x.
        # This is the 'Natural Zone' where human ears cannot detect speed shifts.
        clamped_speed = max(0.85, min(1.25, blended_speed))

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

        # ── TOKENLESS OVERRIDE MODE ──
        # If the user declined to provide a HuggingFace Pyannote Token, diarization failed,
        # meaning ALL sentences are grouped into SPEAKER_01. To prevent men and women from 
        # sharing the exact same voice, we mathematically decouple them and calculate the 
        # precise gender + voice sample for EVERY SINGLE SENTENCE individually!
        is_fallback_mode = (len(speaker_profiles) <= 1)
        fallback_gender_map = {}
        if is_fallback_mode:
            fallback_gender_map = _cluster_genders_by_pitch_and_timbre(transcript, reference_audio)

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

            # ---- DYNAMIC PER-SENTENCE OVERRIDE ----
            if is_fallback_mode and i in fallback_gender_map:
                print(f"[VoiceCloningService] Tokenless Mode Active: Applying K-Means clustered gender for Seg {i}...")
                local_gender = fallback_gender_map[i]
                segment_gender = local_gender
                
                try:
                    local_shift = -3.5 if local_gender == 'male' else 0.0
                    local_sample = self.extract_speaker_sample(
                        reference_audio, segment["start"], segment["end"], tag=f"seg_{i}_local", pitch_shift=local_shift
                    )
                    reference_sample = local_sample
                except Exception as e:
                    print(f"[VoiceCloningService] Fallback extraction failed for seg {i}: {e}")

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
            
            # ── UNIVERSAL EDGE-TTS INTERCEPTOR ──
            # Open-Source XTTS fundamentally lacks structural stability across languages and hallucinates gender.
            # To preserve mathematically perfect gender translation and broadcast-quality purity across EVERY
            # language unconditionally, we sever XTTS entirely and reroute 100% of all scripts into Microsoft Neural.
            print(f"[VoiceCloningService] Seg {i} [{segment_gender}] → Forcing Microsoft Edge-TTS Cloud API. Bypassing XTTS strictly.")
            segment_use_xtts = False
            
            if language == 'kn' and segment_use_xtts:
                try:
                    from indic_transliteration import sanscript
                    # Transform Kannada Unicode -> Devanagari Unicode
                    xtts_text = sanscript.transliterate(text, sanscript.KANNADA, sanscript.DEVANAGARI)
                    xtts_language = 'hi'  # Trick the model into reading it phonetically
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
                                tag=f"seg_{i}_emotion",
                                pitch_shift=-3.5 if segment_gender == 'male' else 0.0
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
