import os
from pathlib import Path
from services.video_service import VideoService
from services.speech_service import SpeechService, LANG_NAMES
from services.translation_service import TranslationEngine
from services.voice_service import VoiceCloningService
from services.audio_mixer import AudioMixerEngine
from services.subtitle_service import SubtitleEngine
from services.qc_service import QualityControlEngine

# All three supported languages
ALL_LANGS = ["en", "hi", "kn"]

# TTS language codes (what XTTS accepts)
TTS_LANG_CODE = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
}


class TranslationPipeline:
    def __init__(
        self,
        work_dir="./workspace",
        bg_lufs=-25.0,
        fg_gain=0.0,
        translation_model="deep_translator",
        sarvam_api_key=None,
        tts_speed=1.0,
        hf_token=None,
        source_lang=None,            # None = auto-detect
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.bg_lufs = bg_lufs
        self.fg_gain = fg_gain
        self.translation_model = translation_model
        self.sarvam_api_key = sarvam_api_key
        self.tts_speed = tts_speed
        self.hf_token = hf_token
        self.source_lang = source_lang   # None means auto-detect later

        self.video_service = VideoService(self.work_dir)
        self.speech_service = SpeechService(hf_token=self.hf_token)
        self.translation_service = TranslationEngine(model=self.translation_model, api_key=self.sarvam_api_key)
        self.voice_service = VoiceCloningService(self.work_dir)
        self.audio_mixer = AudioMixerEngine(self.work_dir)
        self.subtitle_engine = SubtitleEngine(self.work_dir)
        self.qc_engine = QualityControlEngine()

    def run(self, input_video_path: str):
        print(f"--- Starting Multi-Language Pipeline for: {input_video_path} ---")

        # ── Stage 1: Ingest ───────────────────────────────────────────────
        print("Stage 1: Video Ingest & Audio Extraction")
        audio_path, video_metadata = self.video_service.ingest_video(input_video_path)

        # ── Stage 2: Transcription + Language Detection ───────────────────
        print("Stage 2: Transcription & Language Detection")
        english_transcript, detected_lang = self.speech_service.transcribe_and_diarize(
            audio_path, language=self.source_lang
        )
        lang_name = LANG_NAMES.get(detected_lang, detected_lang.upper())
        print(f"[Pipeline] Source language: {detected_lang} ({lang_name})")

        # ── Determine target languages (everything except the source) ──────
        target_langs = [l for l in ALL_LANGS if l != detected_lang]
        print(f"[Pipeline] Target languages: {target_langs}")

        # ── Build results structure ────────────────────────────────────────
        original_track_name = f"Original {lang_name}"
        results = {
            "detected_lang": detected_lang,
            "lang_name": lang_name,
            "qc_report": {},
            "videos": {
                original_track_name: input_video_path,
            },
            "subtitles": {},
            "audio_tracks": {
                original_track_name: audio_path
            }
        }

        # Generate subtitles for the original language too
        # (transcript is in English if source was kn/hi, otherwise in the source lang)
        original_subtitle_path = self.subtitle_engine.generate_subtitles(
            english_transcript, language=detected_lang
        )
        results["subtitles"][detected_lang] = original_subtitle_path

        # ── Stage 3–5 Loop: per target language ───────────────────────────
        first_en_fg_path = None   # keep for QC

        for target_lang in target_langs:
            target_name = LANG_NAMES.get(target_lang, target_lang.upper())
            print(f"\n--- Processing Target: {target_lang} ({target_name}) ---")

            # Translate English transcript → target
            # (For English source the english_transcript IS the transcript;
            #  for kn/hi Whisper already output English via task=translate)
            transcript = self.translation_service.translate_transcript(
                english_transcript, source="en", target=target_lang
            )

            # Determine XTTS language code
            tts_lang = TTS_LANG_CODE.get(target_lang, "en")

            # Generate voice-cloned speech
            cloned_audio_segments = self.voice_service.generate_speech(
                transcript,
                reference_audio=audio_path,
                language=tts_lang,
                speed=self.tts_speed
            )

            # Mix audio
            mixed_audio_path, fg_audio_path = self.audio_mixer.mix_audio(
                primary_segments=cloned_audio_segments,
                secondary_audio=audio_path,
                video_duration=video_metadata["duration"],
                bg_lufs=self.bg_lufs,
                fg_gain=self.fg_gain,
                language=target_lang
            )

            # Generate subtitles
            subtitle_path = self.subtitle_engine.generate_subtitles(transcript, language=target_lang)
            results["subtitles"][target_lang] = subtitle_path

            # Mux: Dub (foreground only)
            final_video_dub = self.video_service.render_final_video(
                input_video_path, fg_audio_path, language=f"{target_lang}_fg"
            )
            # Mux: Overlaying (mixed)
            final_video_mixed = self.video_service.render_final_video(
                input_video_path, mixed_audio_path, language=f"{target_lang}_mixed"
            )

            dub_key = f"{target_name} Dub"
            overlay_key = f"{target_name} (Overlaying)"

            results["videos"][dub_key] = final_video_dub
            results["videos"][overlay_key] = final_video_mixed
            results["audio_tracks"][dub_key] = fg_audio_path
            results["audio_tracks"][overlay_key] = mixed_audio_path

            if target_lang == "en" or first_en_fg_path is None:
                first_en_fg_path = final_video_dub

        # ── Master multi-track video ──────────────────────────────────────
        print("\n--- Rendering Master Multi-Track Video ---")
        multitrack_video_path = self.video_service.render_multitrack_video(
            input_video_path=input_video_path,
            audio_tracks=results["audio_tracks"],
            subtitle_tracks=results["subtitles"]
        )
        results["master_video"] = multitrack_video_path

        # ── QC ────────────────────────────────────────────────────────────
        print("Stage QC: Quality Control")
        qc_video = first_en_fg_path or list(results["videos"].values())[1]
        results["qc_report"] = self.qc_engine.run_checks(
            video_path=qc_video,
            transcript=english_transcript,
            audio_path=audio_path
        )

        print("\n--- Pipeline Complete ---")
        return results
