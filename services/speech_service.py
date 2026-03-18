import os
import json
from faster_whisper import WhisperModel

class SpeechService:
    def __init__(self, model_size="small", hf_token=None):
        """
        Initialize Whisper model + optional Pyannote diarization.
        
        Args:
            hf_token: Hugging Face access token. Required to use Pyannote speaker diarization.
                      Get one at: https://hf.co/settings/tokens
                      Also accept the model at: https://hf.co/pyannote/speaker-diarization-3.1
        """
        self.model_size = model_size
        self.hf_token = hf_token

        print(f"[SpeechService] Loading Faster-Whisper '{model_size}' model...")
        self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # Initialize Pyannote diarization pipeline if HF token is provided
        self.diarization_pipeline = None
        if hf_token:
            try:
                from pyannote.audio import Pipeline
                print("[SpeechService] Loading Pyannote speaker-diarization-3.1...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                print("[SpeechService] Pyannote diarization pipeline loaded successfully.")
            except Exception as e:
                print(f"[SpeechService] WARNING: Could not load Pyannote pipeline: {e}")
                print("[SpeechService] Falling back to single-speaker mode (SPEAKER_01).")
                self.diarization_pipeline = None
        else:
            print("[SpeechService] No HF token provided. Skipping Pyannote — using single-speaker mode.")

    def _get_speaker_at_time(self, diarization, midpoint: float) -> str:
        """
        Given a pyannote diarization result, find which speaker is active at `midpoint` seconds.
        Returns the speaker label string (e.g. 'SPEAKER_00'), or 'SPEAKER_01' as fallback.
        """
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= midpoint <= turn.end:
                return speaker
        return "SPEAKER_01"

    def transcribe_and_diarize(self, audio_path: str, language="kn") -> list:
        """
        Transcribe audio and segment by speaker using Pyannote (if token provided).
        Returns a list of dicts: { start, end, speaker_id, text }
        """
        print(f"[SpeechService] Transcribing {audio_path} in '{language}'...")
        
        segments_gen, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            task="translate",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )
        
        # Collect all Whisper segments first (generator is lazy)
        whisper_segments = list(segments_gen)
        print(f"[SpeechService] Whisper found {len(whisper_segments)} segments.")

        # Run Pyannote diarization if available
        diarization_result = None
        if self.diarization_pipeline is not None:
            try:
                print("[SpeechService] Running Pyannote speaker diarization...")
                diarization_result = self.diarization_pipeline(audio_path)
                print("[SpeechService] Diarization complete.")
            except Exception as e:
                print(f"[SpeechService] WARNING: Diarization failed: {e}. Using SPEAKER_01 fallback.")
                diarization_result = None

        transcript = []
        for segment in whisper_segments:
            # Determine speaker using midpoint of segment
            if diarization_result is not None:
                midpoint = (segment.start + segment.end) / 2.0
                speaker_label = self._get_speaker_at_time(diarization_result, midpoint)
            else:
                speaker_label = "SPEAKER_01"

            print(f"[SpeechService] [{speaker_label}] {segment.start:.2f}s-{segment.end:.2f}s | {segment.text[:40]}...")

            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "speaker_id": speaker_label,
                "text": segment.text.strip()
            })

        if not transcript:
            print("[SpeechService] WARNING: No speech segments detected!")

        return transcript
