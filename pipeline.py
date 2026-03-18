import os
from pathlib import Path
from services.video_service import VideoService
from services.speech_service import SpeechService
from services.translation_service import TranslationEngine
from services.voice_service import VoiceCloningService
from services.audio_mixer import AudioMixerEngine
from services.subtitle_service import SubtitleEngine
from services.qc_service import QualityControlEngine

class TranslationPipeline:
    def __init__(self, work_dir="./workspace", bg_lufs=-25.0, fg_gain=0.0, translation_model="deep_translator", sarvam_api_key=None, tts_speed=1.1, hf_token=None):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.bg_lufs = bg_lufs
        self.fg_gain = fg_gain
        self.translation_model = translation_model
        self.sarvam_api_key = sarvam_api_key
        self.tts_speed = tts_speed
        self.hf_token = hf_token
        
        self.video_service = VideoService(self.work_dir)
        self.speech_service = SpeechService(hf_token=self.hf_token)
        self.translation_service = TranslationEngine(model=self.translation_model, api_key=self.sarvam_api_key)
        self.voice_service = VoiceCloningService(self.work_dir)
        self.audio_mixer = AudioMixerEngine(self.work_dir)
        self.subtitle_engine = SubtitleEngine(self.work_dir)
        self.qc_engine = QualityControlEngine()

    def run(self, input_video_path: str):
        print(f"--- Starting Pipeline for {input_video_path} (Multi-track Output) ---")
        
        # Stage 1: Ingest
        print("Stage 1: Video Ingest & Audio Extraction")
        audio_path, video_metadata = self.video_service.ingest_video(input_video_path)
        
        # Stage 2: Speech Recognition (Kannada)
        print("Stage 2: Kannada Speech Recognition & Diarization")
        # Whisper translates directly to English
        english_transcript = self.speech_service.transcribe_and_diarize(audio_path, language="kn")
        
        # We need to process both English and Hindi
        results = {
            "qc_report": {},
            "videos": {
                "Original Kannada": input_video_path,
            },
            "subtitles": {},
            "audio_tracks": {
                "Original Kannada": audio_path
            }
        }

        # Stage 3-5 Loop: Translation, Voice Cloning, Mixing, Subtitles, Muxing
        for target_lang in ["en", "hi"]:
            print(f"--- Processing Target: {target_lang} ---")
            
            # Translate English transcript to target (if target is 'en' it's a pass-through)
            transcript = self.translation_service.translate_transcript(
                english_transcript, source="en", target=target_lang
            )
            
            # Generate speech
            cloned_audio_segments = self.voice_service.generate_speech(
                transcript, reference_audio=audio_path, language=target_lang, speed=self.tts_speed
            )
            
            # Mix audio
            mixed_audio_path, fg_audio_path = self.audio_mixer.mix_audio(
                primary_segments=cloned_audio_segments,
                secondary_audio=audio_path,
                video_duration=video_metadata['duration'],
                bg_lufs=self.bg_lufs,
                fg_gain=self.fg_gain,
                language=target_lang
            )
            
            # Generate WebVTT subtitles
            subtitle_path = self.subtitle_engine.generate_subtitles(transcript, language=target_lang)
            results["subtitles"][target_lang] = subtitle_path
            
            # Mux to create video with dubbed audio (mixed/overlaying)
            final_video_path_mixed = self.video_service.render_final_video(
                input_video_path, 
                mixed_audio_path,
                language=f"{target_lang}_mixed"
            )
            
            # Mux to create video with dubbed audio (foreground only)
            final_video_path_fg = self.video_service.render_final_video(
                input_video_path, 
                fg_audio_path,
                language=f"{target_lang}_fg"
            )
            
            # Store the resulting playback video tracks and audio tracks for multitrack render
            if target_lang == "en":
                results["videos"]["English Dub"] = final_video_path_fg
                results["videos"]["English (overlaying)"] = final_video_path_mixed
                results["audio_tracks"]["English Dub"] = fg_audio_path
                results["audio_tracks"]["English (overlaying)"] = mixed_audio_path
            elif target_lang == "hi":
                results["videos"]["Hindi Dub"] = final_video_path_fg
                results["videos"]["Hindi (overlaying)"] = final_video_path_mixed
                results["audio_tracks"]["Hindi Dub"] = fg_audio_path
                results["audio_tracks"]["Hindi (overlaying)"] = mixed_audio_path
            else:
                results["videos"][f"{target_lang} Dub"] = final_video_path_fg
                results["videos"][f"{target_lang} (overlaying)"] = final_video_path_mixed
                results["audio_tracks"][f"{target_lang} Dub"] = fg_audio_path
                results["audio_tracks"][f"{target_lang} (overlaying)"] = mixed_audio_path
        
        # Stage X: Generate Master Multi-Track Video
        print("--- Rendering Master Multi-Track Video ---")
        multitrack_video_path = self.video_service.render_multitrack_video(
            input_video_path=input_video_path,
            audio_tracks=results["audio_tracks"],
            subtitle_tracks=results["subtitles"]
        )
        results["master_video"] = multitrack_video_path
        
        # Stage 9: Quality Control (run on English just to get the report structure)
        print("Stage 9: Quality Control")
        qc_report = self.qc_engine.run_checks(
            video_path=results["videos"]["English Dub"],
            transcript=english_transcript,
            audio_path=audio_path # Simplified QC check
        )
        results["qc_report"] = qc_report
        
        print("--- Pipeline Complete ---")
        return results
