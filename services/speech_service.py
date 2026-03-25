import os
from faster_whisper import WhisperModel

# Languages Whisper can directly transcribe (task="transcribe" keeps original text)
# All others will use task="translate" to get English output
_TRANSCRIBE_NATIVELY = {"en", "hi", "kn"}

# Human-readable names
LANG_NAMES = {
    "kn": "Kannada",
    "en": "English",
    "hi": "Hindi",
}


class SpeechService:
    def __init__(self, model_size="small", hf_token=None):
        """
        Initialize Whisper model + optional Pyannote diarization.

        Args:
            hf_token: Hugging Face access token. Required to use Pyannote speaker diarization.
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

    def detect_language(self, audio_path: str) -> str:
        """
        Detect the spoken language in the audio using Whisper.
        Analyses up to the first 30 seconds for speed.
        Returns an ISO language code e.g. 'kn', 'en', 'hi'.
        Defaults to 'kn' if detection fails.
        """
        print(f"[SpeechService] Auto-detecting language in: {audio_path}")
        try:
            _, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=1,
                task="transcribe",
                vad_filter=True,
            )
            detected = info.language
            confidence = round(info.language_probability * 100, 1)
            lang_name = LANG_NAMES.get(detected, detected.upper())
            print(f"[SpeechService] Detected language: '{detected}' ({lang_name}) — confidence {confidence}%")
            return detected
        except Exception as e:
            print(f"[SpeechService] Language detection failed: {e}. Defaulting to 'kn'.")
            return "kn"

    def _get_speaker_at_time(self, diarization, midpoint: float) -> str:
        """Find which speaker is active at midpoint seconds."""
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= midpoint <= turn.end:
                return speaker
        return "SPEAKER_01"

    def transcribe_and_diarize(self, audio_path: str, language: str = None) -> tuple:
        """
        Transcribe audio and segment by speaker using Pyannote (if token provided).

        For English input: task="transcribe" → keeps original English text.
        For Kannada/Hindi: task="translate" → Whisper outputs English (pipeline then
        translates from English to the desired target language).

        Args:
            language: ISO code ('en', 'hi', 'kn') or None for auto-detect.

        Returns:
            (transcript, detected_language_code)
            transcript = list of { start, end, speaker_id, text }
        """
        if language is None:
            language = self.detect_language(audio_path)

        lang_name = LANG_NAMES.get(language, language.upper())

        # English input: transcribe natively to keep English text.
        # Kannada / Hindi: translate to English via Whisper so the
        # translation_service can then map English → target (hi/kn).
        if language == "en":
            task = "transcribe"
            whisper_lang = "en"
        elif language == "hi":
            task = "translate"   # Whisper: Hindi audio → English text
            whisper_lang = "hi"
        else:
            # Kannada (and any other unsupported language)
            task = "translate"   # Whisper: Kannada audio → English text
            whisper_lang = language

        print(f"[SpeechService] Transcribing ({lang_name}) with task='{task}'...")

        segments_gen, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language=whisper_lang,
            task=task,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )

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

        transcript = []
        for segment in whisper_segments:
            if diarization_result is not None:
                midpoint = (segment.start + segment.end) / 2.0
                speaker_label = self._get_speaker_at_time(diarization_result, midpoint)
            else:
                speaker_label = "SPEAKER_01"

            print(f"[SpeechService] [{speaker_label}] {segment.start:.2f}s-{segment.end:.2f}s | {segment.text[:50]}...")

            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "speaker_id": speaker_label,
                "text": segment.text.strip()
            })

        if not transcript:
            print("[SpeechService] WARNING: No speech segments detected!")

        return transcript, language
