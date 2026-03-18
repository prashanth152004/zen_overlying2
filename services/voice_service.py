import os
import torch
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

# Average syllables per word in English/Hindi TTS output (tuned empirically)
_AVG_SYLLABLES_PER_WORD = 1.5
# XTTS safe speed range
_MIN_SPEED = 0.85
_MAX_SPEED = 1.35


class VoiceCloningService:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.cloned_dir = self.work_dir / "cloned_audio"
        self.cloned_dir.mkdir(exist_ok=True)
        # Assuming CUDA if available otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("[VoiceCloningService] Loading XTTSv2 Voice Cloning model... This may take a while")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def extract_speaker_sample(self, reference_audio: str, start: float, end: float) -> str:
        """Extract a small clean sample of the speaker for cloning, filtering out background noise."""
        audio = AudioSegment.from_file(reference_audio)
        sample = audio[start * 1000 : end * 1000]
        
        # Pre-process the reference sample so the AI doesn't clone background noise/muddiness
        # 1. Normalize the volume so the model hears the voice clearly
        sample = sample.normalize()
        # 2. High-pass filter to remove low rumble, wind noise, mic bumps
        sample = sample.high_pass_filter(100)
        # 3. Low-pass filter to remove high-frequency background hiss/static
        sample = sample.low_pass_filter(8000)
        
        sample_path = str(self.work_dir / f"sample_{start}_{end}.wav")
        sample.export(sample_path, format="wav")
        return sample_path

    def _pick_best_speaker_sample(self, transcript: list, reference_audio: str) -> str:
        """Pick the longest available clean speaker segment as the default reference (capped at 12s)."""
        best_start, best_end = 0.0, 0.0
        best_duration = 0.0
        for seg in transcript:
            dur = seg["end"] - seg["start"]
            if dur > best_duration and dur <= 12.0:
                best_duration = dur
                best_start = seg["start"]
                best_end = seg["end"]
        
        # If no good segment found, fall back to first 10 seconds
        if best_duration < 3.0:
            best_start, best_end = 0.0, min(10.0, 10.0)
        
        print(f"[VoiceCloningService] Best speaker reference sample: {best_start:.1f}s → {best_end:.1f}s ({best_duration:.1f}s)")
        return self.extract_speaker_sample(reference_audio, best_start, best_end)

    def _compute_segment_speed(self, text: str, duration_sec: float, base_speed: float) -> float:
        """
        Compute the ideal TTS speed so the synthesized speech fits naturally within
        the original segment duration.

        Strategy:
          1. Estimate how many seconds the text would take at base_speed (using XTTS
             average of ~3.2 syllables/sec at speed=1.0).
          2. Compute ratio = estimated_duration / available_duration.
          3. Clamp to [_MIN_SPEED, _MAX_SPEED] to stay within XTTS safe range.
          4. Blend 70% auto + 30% base so user baseline still has some influence.
        """
        if duration_sec <= 0.5:
            return base_speed  # too short to compute meaningfully

        word_count = len(text.split())
        syllable_count = word_count * _AVG_SYLLABLES_PER_WORD
        # At speed=1.0, XTTS speaks ~3.2 syllables/second
        estimated_natural_duration = syllable_count / (3.2 * base_speed)

        if estimated_natural_duration <= 0:
            return base_speed

        ratio = estimated_natural_duration / duration_sec
        # Blend: favour auto-computed but let the user baseline nudge it slightly
        auto_speed = base_speed * ratio
        blended_speed = 0.7 * auto_speed + 0.3 * base_speed
        clamped_speed = max(_MIN_SPEED, min(_MAX_SPEED, blended_speed))

        print(f"[VoiceCloningService] Per-segment speed: {clamped_speed:.3f}x "
              f"(words={word_count}, dur={duration_sec:.1f}s, estimated={estimated_natural_duration:.1f}s, "
              f"auto={auto_speed:.3f}, base={base_speed:.2f})")
        return clamped_speed

    def generate_speech(self, transcript: list, reference_audio: str, language: str = "en", speed: float = 1.0) -> list:
        """
        Generate speech cloned from the original speaker voice with automatic
        per-segment speed adjustment so each clip fits the original video timing.

        Args:
            speed: Baseline TTS speed multiplier (used as an anchor for per-segment
                   auto-speed blending). 1.0 = neutral.
        """
        print(f"[VoiceCloningService] Generating {language} speech for {len(transcript)} segments "
              f"(base_speed={speed}, auto-speed per segment enabled)...")
        
        # Pick the best (longest/cleanest) speaker sample from the transcript for reference
        default_sample = self._pick_best_speaker_sample(transcript, reference_audio)

        cloned_segments = []
        for i, segment in enumerate(transcript):
            text = segment.get("text", "").strip()
            if not text:
                print(f"[VoiceCloningService] Skipping empty text for segment {i}")
                continue

            out_path = str(self.cloned_dir / f"segment_{i}.wav")
            
            # --- Auto-speed: compute per-segment ideal speed ---
            segment_duration = segment["end"] - segment["start"]
            segment_speed = self._compute_segment_speed(text, segment_duration, base_speed=speed)

            # Extract a dynamic sample for this specific segment to better match speaker tone
            current_sample = default_sample
            if segment_duration >= 3.0:
                # XTTS prefers >3s samples for good cloning
                try:
                    sample_end = min(segment["start"] + 12.0, segment["end"])
                    current_sample = self.extract_speaker_sample(reference_audio, segment["start"], sample_end)
                except Exception as e:
                    print(f"[VoiceCloningService] Failed to extract dynamic sample for segment {i}, "
                          f"falling back to best sample. Error: {e}")
            
            print(f"[VoiceCloningService] Generating segment {i}: '{text[:40]}...' "
                  f"(speed={segment_speed:.3f}, dur={segment_duration:.1f}s)")
            
            if not os.path.exists(current_sample):
                print(f"[VoiceCloningService] ERROR: Speaker sample not found at {current_sample}")
                raise FileNotFoundError(f"Speaker sample missing: {current_sample}")

            # Use original voice to speak translated text with auto-computed speed
            try:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=current_sample,
                    language=language,
                    file_path=out_path,
                    speed=segment_speed
                )
            except TypeError:
                # Older TTS versions may not support speed parameter — fallback gracefully
                print(f"[VoiceCloningService] WARNING: This TTS version does not support 'speed'. Using default.")
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=current_sample,
                    language=language,
                    file_path=out_path
                )
            except Exception as e:
                print(f"[VoiceCloningService] TTS failed for segment {i}: {e}")
                raise e
            
            cloned_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "audio_path": out_path
            })
            
        return cloned_segments
