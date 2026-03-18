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
        Supports stretch ratios in the range [0.25x – 4.0x] using chained atempo.
        Applies stretch for ANY timing mismatch (no threshold).
        """
        if target_ms <= 0 or current_ms <= 0:
            return AudioSegment.from_file(audio_path)

        ratio = current_ms / target_ms

        # Clamp to supported range [0.25x – 4.0x]
        ratio_clamped = max(0.25, min(4.0, ratio))

        if abs(ratio_clamped - 1.0) < 0.01:
            # No meaningful stretch needed
            return AudioSegment.from_file(audio_path)

        atempo_filter = self._build_atempo_filter(ratio_clamped)
        stretched_path = str(self.work_dir / f"stretched_{segment_idx}.wav")

        print(f"[AudioMixerEngine] Lip-Sync segment {segment_idx}: ratio={ratio:.3f}x → filter '{atempo_filter}' ({current_ms}ms→{target_ms}ms)")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-filter:a", atempo_filter,
            stretched_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stretched = AudioSegment.from_file(stretched_path)
            print(f"[AudioMixerEngine] Lip-Sync success: {current_ms}ms → {len(stretched)}ms (target: {target_ms}ms)")
            return stretched
        except Exception as e:
            print(f"[AudioMixerEngine] WARNING: atempo failed for segment {segment_idx}: {e}. Using raw audio.")
            return AudioSegment.from_file(audio_path)

    def _apply_presence_boost(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a +3dB peaking EQ boost in the 2–4kHz speech presence band.
        This makes the dubbed voice cut through a mix more intelligibly.
        Also apply a gentle de-essing notch at 6–8kHz to soften harsh sibilants.
        """
        nyquist = sr / 2.0

        # --- Presence boost: 2–4kHz band-pass shelf (+3dB) ---
        # We add a fraction of the bandpass signal back to the original
        low_p, high_p = 2000.0 / nyquist, 4000.0 / nyquist
        low_p = max(0.001, min(low_p, 0.999))
        high_p = max(0.001, min(high_p, 0.999))
        if low_p < high_p:
            b_bp, a_bp = scipy.signal.butter(2, [low_p, high_p], btype='band')
            presence_band = scipy.signal.lfilter(b_bp, a_bp, audio_data)
            audio_data = audio_data + 0.4 * presence_band   # +~3dB boost

        # --- De-essing: gentle notch at 6–8kHz ---
        low_d, high_d = 6000.0 / nyquist, 8000.0 / nyquist
        low_d = max(0.001, min(low_d, 0.999))
        high_d = max(0.001, min(high_d, 0.999))
        if low_d < high_d:
            b_n, a_n = scipy.signal.butter(2, [low_d, high_d], btype='bandstop')
            audio_data = scipy.signal.lfilter(b_n, a_n, audio_data)

        print("[AudioMixerEngine] Presence boost (2–4kHz +3dB) and de-essing (6–8kHz notch) applied.")
        return audio_data

    def _enhance_voice_clarity(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply a multi-stage voice clarity enhancement chain to synthesized speech:
        1. Normalize baseline volume
        2. High-pass filter at 80Hz   — removes low rumble and TTS mud
        3. Presence boost 2–4kHz      — makes speech intelligible and bright
        4. De-essing 6–8kHz notch     — softens harsh sibilant artifacts from TTS
        5. Low-pass filter at 14kHz   — removes unnatural synthetic hiss (raised from 12kHz)
        6. Tighter dynamic compression— fast 2ms attack, evens out TTS amplitude spikes
        7. Makeup normalize           — brings compressed signal back, with small headroom
        """
        # 1. Normalize baseline
        audio = audio.normalize()

        # 2. High-pass: remove low-frequency rumble/mud below 80Hz
        audio = audio.high_pass_filter(80)

        # 3 & 4. Presence boost + De-essing (requires numpy/scipy processing)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # Normalize to float [-1, 1] for scipy
        max_val = float(2 ** (audio.sample_width * 8 - 1))
        samples = samples / max_val
        if audio.channels == 2:
            samples = samples.reshape(-1, 2)
            samples[:, 0] = self._apply_presence_boost(samples[:, 0], audio.frame_rate)
            samples[:, 1] = self._apply_presence_boost(samples[:, 1], audio.frame_rate)
            samples = samples.flatten()
        else:
            samples = self._apply_presence_boost(samples, audio.frame_rate)
        # Clip and convert back to integer samples
        samples = np.clip(samples, -1.0, 1.0)
        samples_int = (samples * max_val).astype(np.int16)
        audio = audio._spawn(samples_int.tobytes())

        # 5. Low-pass: remove harsh synthetic hiss above 14kHz (raised from 12kHz)
        audio = audio.low_pass_filter(14000)

        # 6. Compression: tighter for TTS (threshold=-20dB, ratio=3.5, fast attack=2ms)
        audio = compress_dynamic_range(
            audio,
            threshold=-20.0,
            ratio=3.5,
            attack=2.0,
            release=80.0
        )

        # 7. Makeup gain: normalize with -0.5dBFS headroom to prevent clipping in mix
        audio = audio.normalize(headroom=0.5)

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
            
        sec_path_temp = str(self.work_dir / "temp_sec.wav")
        sf.write(sec_path_temp, data_sec, sr_sec)
        
        background = AudioSegment.from_file(sec_path_temp)
        
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


