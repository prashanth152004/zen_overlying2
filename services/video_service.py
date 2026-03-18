import os
import subprocess
import cv2
from pathlib import Path

class VideoService:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def get_video_metadata(self, video_path: str) -> dict:
        """Extract frame rate, resolution, duration via OpenCV."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration
        }

    def ingest_video(self, input_video_path: str):
        """Extract audio using FFmpeg and gather metadata."""
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video not found: {input_video_path}")
            
        metadata = self.get_video_metadata(input_video_path)
        print(f"[VideoService] Ingested video: {metadata['width']}x{metadata['height']} @ {metadata['fps']}fps, {metadata['duration']:.2f}s")
        
        audio_out_path = self.work_dir / "extracted_audio.wav"
        
        # FFmpeg extract audio: PCM 16-bit, 44100Hz mono
        cmd = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            str(audio_out_path)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[VideoService] Audio extracted to {audio_out_path}")
        
        return str(audio_out_path), metadata

    def render_final_video(self, input_video_path: str, mixed_audio_path: str, language: str) -> str:
        """Mux the mixed audio back into the original video (without re-encoding video)."""
        output_path = str(self.work_dir / f"final_output_{language}.mp4")
        
        cmd = [
            "ffmpeg", "-y", 
            "-i", input_video_path,
            "-i", mixed_audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
            output_path
        ]
        
        print(f"[VideoService] Fast-muxing audio ({language}) into video...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path

    def render_multitrack_video(self, input_video_path: str, audio_tracks: dict, subtitle_tracks: dict) -> str:
        """Mux the original video with all generated audio tracks and subtitles into a single MKV."""
        output_path = str(self.work_dir / "master_multitrack_output.mkv")
        
        cmd = ["ffmpeg", "-y", "-i", input_video_path]
        
        # 1. Add audio inputs
        audio_names = list(audio_tracks.keys())
        for name in audio_names:
            cmd.extend(["-i", audio_tracks[name]])
            
        # 2. Add subtitle inputs
        sub_langs = list(subtitle_tracks.keys())
        for lang in sub_langs:
            cmd.extend(["-i", subtitle_tracks[lang]])
            
        # 3. Map video
        cmd.extend(["-map", "0:v:0"]) # specifically map the first video stream of the first input
        
        # 4. Map audio
        for i, name in enumerate(audio_names):
            cmd.extend(["-map", f"{i+1}:a"])
            
        # 5. Map subtitles
        offset = len(audio_names) + 1
        for i, lang in enumerate(sub_langs):
            cmd.extend(["-map", f"{offset+i}:s?"]) # '?' makes it optional just in case
            
        # 6. Set titles and languages for audio
        for i, name in enumerate(audio_names):
            lang_code = "eng" if "English" in name else "hin" if "Hindi" in name else "kan"
            cmd.extend([f"-metadata:s:a:{i}", f"language={lang_code}"])
            cmd.extend([f"-metadata:s:a:{i}", f"title={name}"])

        # 7. Set titles and languages for subtitles
        for i, lang in enumerate(sub_langs):
            lang_code = "eng" if lang == "en" else "hin" if lang == "hi" else "kan"
            cmd.extend([f"-metadata:s:s:{i}", f"language={lang_code}"])
            cmd.extend([f"-metadata:s:s:{i}", f"title={lang.upper()} Subtitles"])

        # 8. Set stream dispositions (Optional: set Original Kannada as default)
        cmd.extend(["-disposition:a:0", "default"])

        # 9. Codecs
        # Use aac for audio, copy for video, srt for subtitles (MKV supports SRT natively)
        cmd.extend(["-c:v", "copy", "-c:a", "aac", "-ar", "44100", "-b:a", "192k", "-c:s", "srt"])
        
        cmd.append(output_path)
        
        print(f"[VideoService] Rendering master multitrack MKV video...")
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[VideoService] Successfully rendered master video at {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[VideoService] FFmpeg failed during multitrack muxing: {e}")
            raise
            
        return output_path
