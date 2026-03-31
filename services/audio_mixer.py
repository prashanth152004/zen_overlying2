import os
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import scipy.signal
import subprocess

class AudioMixerEngine:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.mixed_dir = self.work_dir / "mixed_audio"
        self.mixed_dir.mkdir(exist_ok=True)

    def apply_eq_cut(self, audio_data, sr):
        """Apply a slight mid-frequency dip (1-3 kHz) to push background behind translated speech."""
        nyquist = sr / 2.0
        low = 1000.0 / nyquist
        high = 3000.0 / nyquist
        b, a = scipy.signal.butter(2, [low, high], btype='bandstop')
        filtered = scipy.signal.lfilter(b, a, audio_data)
        return filtered

    def normalize_loudness(self, audio_data, sr, target_lufs):
        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(audio_data)
        normalized = pyln.normalize.loudness(audio_data, current_loudness, target_lufs)
        return normalized

    def _build_atempo_filter(self, ratio: float) -> str:
        """
        Build a chained FFmpeg atempo filter string to support extreme stretch ratios.
        atempo only supports values between 0.5 and 100.0.
        For ratios > 2.0 or < 0.5, we chain multiple atempo filters.
        E.g. ratio=3.0 → "atempo=2.0,atempo=1.5"
             ratio=0.25 → "atempo=0.5,atempo=0.5"
        """
        filters = []
        remaining = ratio
        
        # Speed up: chain values up to 2.0 per step
        if remaining > 1.0:
            while remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2.0
            filters.append(f"atempo={remaining:.4f}")
        # Slow down: chain values down to 0.5 per step
        else:
            while remaining < 0.5:
                filters.append("atempo=0.5")
                remaining /= 0.5
            filters.append(f"atempo={remaining:.4f}")
        
        return ",".join(filters)

    def _apply_lip_sync_stretch(self, audio_path: str, current_ms: int, target_ms: int, segment_idx: int) -> AudioSegment:
        """
        Apply precise atempo time-stretching for lip sync.
        Strips artificial TTS silence before stretching so audio perfectly matches original mouth boundaries.
        """
        if target_ms <= 0 or current_ms <= 0:
            return AudioSegment.from_file(audio_path)

        audio = AudioSegment.from_file(audio_path)
        
        # --- Trim leading/trailing silence for accurate lip-sync ---
        # TTS models often add ~200-500ms of dead air which ruins the lip-sync ratio
        def trim_silence(sound: AudioSegment, threshold_db=-45.0, chunk_ms=10):
            if len(sound) == 0:
                return sound
            start_trim = 0
            for i in range(0, len(sound), chunk_ms):
                if sound[i:i+chunk_ms].dBFS > threshold_db:
                    start_trim = max(0, i - chunk_ms)
                    break
            end_trim = len(sound)
            for i in range(len(sound), 0, -chunk_ms):
                if sound[max(0, i-chunk_ms):i].dBFS > threshold_db:
                    end_trim = min(len(sound), i + chunk_ms)
                    break
            return sound[start_trim:end_trim] if end_trim > start_trim else sound

        audio_trimmed = trim_silence(audio)
        current_ms_trimmed = len(audio_trimmed)

        if current_ms_trimmed <= 0:
            return audio

        ratio = current_ms_trimmed / target_ms

        # To guarantee the lip-sync is absolutely structurally locked to the original video,
        # we aggressively permit mathematical time-stretching up to 2.0x ! 
        # This violently resolves any misalignments that slipped through the Edge-TTS layer.
        ratio_clamped = max(0.65, min(2.0, ratio))

        if abs(ratio_clamped - 1.0) < 0.02:
            return audio_trimmed

        # Save trimmed clip so ffmpeg uses tightly cropped speech
        trimmed_path = str(self.work_dir / f"trimmed_{segment_idx}.wav")
        audio_trimmed.export(trimmed_path, format="wav")

        atempo_filter = self._build_atempo_filter(ratio_clamped)
        stretched_path = str(self.work_dir / f"stretched_{segment_idx}.wav")

        print(f"[AudioMixerEngine] Lip-Sync segment {segment_idx}: ratio={ratio:.3f}x → filter '{atempo_filter}' ({current_ms_trimmed}ms→{target_ms}ms)")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", trimmed_path,
            "-filter:a", atempo_filter,
            stretched_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stretched = AudioSegment.from_file(stretched_path)
            print(f"[AudioMixerEngine] Lip-Sync success: {current_ms_trimmed}ms → {len(stretched)}ms (target: {target_ms}ms)")
            return stretched
        except Exception as e:
            print(f"[AudioMixerEngine] WARNING: atempo failed for segment {segment_idx}: {e}. Using raw audio.")
            return audio_trimmed

    def _enhance_voice_clarity(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply gentle voice clarity enhancement to maintain the natural, human tone.
        We avoid heavy EQ and aggressive compression to keep the neural TTS sounding like the original.
        """
        # 1. Normalize baseline to prevent clipping before processing
        audio = audio.normalize()

        # 2. Gentle High-pass/Low-pass to clean up extreme sub-rumble and synthetic high-hiss
        audio = audio.high_pass_filter(60)
        audio = audio.low_pass_filter(15000)

        # 3. Light dynamic compression just to even out TTS spikes, keeping it natural
        audio = compress_dynamic_range(
            audio,
            threshold=-15.0,
            ratio=2.0,
            attack=5.0,
            release=50.0
        )

        # 4. Final slight normalize to ensure it fits the mix perfectly (-1.0dBFS headroom)
        audio = audio.normalize(headroom=1.0)

        return audio

    def mix_audio(self, primary_segments: list, secondary_audio: str, video_duration: float, bg_lufs: float = -21.0, fg_gain: float = 0.0, language: str = "en") -> tuple:
        """
        Primary (Translated): Adjusted by fg_gain.
        Secondary (Kannada Original): Normalized to bg_lufs relative to Translated, EQ dip.
        """
        print(f"[AudioMixerEngine] Mixing Cloned Translated and Original Kannada audio (BG: {bg_lufs} LUFS, FG: {fg_gain} dB)...")
        
        # Load secondary (Kannada)
        data_sec, sr_sec = sf.read(secondary_audio)
        if len(data_sec.shape) > 1:
            data_sec = data_sec.mean(axis=1) # force mono
            
        # Apply EQ and ducking
        data_sec = self.apply_eq_cut(data_sec, sr_sec)
        
        if bg_lufs > -40.0:
            data_sec = self.normalize_loudness(data_sec, sr_sec, bg_lufs)
        else:
            data_sec = data_sec * 0.01 
            
        import io
        buf = io.BytesIO()
        sf.write(buf, data_sec, sr_sec, format='WAV')
        buf.seek(0)
        background = AudioSegment.from_file(buf, format="wav")
        
        # Create a blank canvas for primary (Foreground voices) with 44.1kHz proper frame rate
        foreground = AudioSegment.silent(duration=int(video_duration * 1000 + 2000), frame_rate=44100)
        
        print(f"[AudioMixerEngine] Overlaying {len(primary_segments)} translated segments...")
        for i, seg in enumerate(primary_segments):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg.get("end", seg["start"] + 3) * 1000)
            target_duration_ms = end_ms - start_ms
            
            if not os.path.exists(seg["audio_path"]):
                print(f"[AudioMixerEngine] WARNING: Audio file missing for segment {i}: {seg['audio_path']}")
                continue
            
            # --- IMPROVED LIP-SYNC: Apply stretch for any mismatch, with chained atempo ---
            raw_audio = AudioSegment.from_file(seg["audio_path"])
            current_duration_ms = len(raw_audio)
            
            en_audio = self._apply_lip_sync_stretch(
                audio_path=seg["audio_path"],
                current_ms=current_duration_ms,
                target_ms=target_duration_ms,
                segment_idx=i
            )
            
            # --- IMPROVED CLARITY: Multi-stage voice enhancement chain ---
            en_audio = self._enhance_voice_clarity(en_audio)
            
            # Apply user-requested foreground gain on top of enhanced audio
            if fg_gain != 0.0:
                en_audio = en_audio + fg_gain
                
            print(f"[AudioMixerEngine] Mixing segment {i} at {start_ms}ms (Final Duration: {len(en_audio)}ms)")
            foreground = foreground.overlay(en_audio, position=start_ms)
            
        # Mix them: Translated (foreground) + Original (background)
        final_mix = background.overlay(foreground)
        
        out_path_mixed = str(self.mixed_dir / f"final_mixed_{language}.wav")
        final_mix.export(out_path_mixed, format="wav")
        
        out_path_fg = str(self.mixed_dir / f"final_foreground_{language}.wav")
        foreground.export(out_path_fg, format="wav")
        
        return out_path_mixed, out_path_fg


