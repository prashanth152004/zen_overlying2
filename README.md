# National Portal for Media Translation

A production-grade, Streamlit-based AI video translation application. This pipeline automatically transcribes, translates, and dubs Kannada civic communication videos into high-quality **English** and **Hindi** videos.

### Features
* **Netflix-Style Player:** Watch videos with a consolidated, Netflix-style UI. Includes real-time audio track switching (Original, English Dub, Hindi Dub), native WebVTT Closed Captions (CC), and 10-second skip controls right in the browser.
* **Offline First Architecture:** Uses Faster-Whisper and Coqui XTTSv2 running completely locally to ensure sensitive civic data never leaves your infrastructure. 
* **Speaker Diarization & Voice Cloning:** Identifies different speakers and copies the specific tone and cadence of the original speaker for each segment of dialogue.
* **Perfect Lip-Sync (FFmpeg `atempo`):** The engine perfectly stretches and condenses translated audio down to the millisecond to match the original speaker's mouth movements.
* **Broadcast-Quality Audio Clarity:** Reference audio is aggressively de-noised (High/Low pass) before cloning. Synthetic voices are enhanced using dynamic range compression, high-pass filtering, and makeup gain to punch clearly through the mix.
* **Audio Ducking:** Dynamically balances the AI synthesized voice with the original background audio (defaulting to -25 LUFS) so original sound effects or music remain audible underneath.
* **Master Multi-Track `.MKV` Export:** Automatically multiplexes all generated audio and subtitle tracks back into a single, comprehensive Master MKV file that players like VLC or native system players can switch between natively.
* **Formal Government Aesthetics:** Features a clean, accessible UI built around Indian civic standards (Tricolor accents, Roboto typography).

## System Architecture

The overarching pipeline process is completely automated:
1. **Video Ingest**: Extracts the main audio track.
2. **Speech Recognition (Kannada)**: Uses `faster-whisper` (`task="translate"`) to natively generate a highly accurate English transcript and runs Pyannote Diarization.
3. **Deep Translation**: Uses `deep-translator` to translate the English baseline transcript into Hindi. 
4. **Voice Synthesis**: Dynamically extracts speaker samples, cleans them of background noise, and feeds them into `Coqui XTTSv2` to synthetically recreate the speaker in English and Hindi.
5. **Audio Mixing & Processing**: Applies EQ, compression, and LUFS ducking using `pydub` & `pyloudnorm`. Includes isolated "Dub" tracks and background-mixed "Overlaying" tracks.
6. **Subtitles**: Converts transcripts to native `.vtt` WebVTT formatting.
7. **Fast Multiplexing**: Uses `ffmpeg-python` to remux the individual streams together instantaneously into `.mp4` display files and the final master `.mkv` container.

## Setup & Installation

**Prerequisites:**
* Python 3.11+
* FFmpeg (must be installed on the system and available in PATH)

### Option A: Local Installation
```bash
# Clone the repository
git clone https://github.com/prashanth152004/zen_overlying_new.git
cd zen_overlying_new

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option B: Docker Deployment (Recommended)
You can easily deploy the complete environment using Docker Compose. This ensures all system dependencies (like FFmpeg) and Python libraries are isolated and perfectly configured.

```bash
# Ensure Docker Desktop is running, then build and start the container
docker-compose up --build
```
The application will be accessible at `http://localhost:8501`.

*(Note: Ensure your system has adequate graphical acceleration / GPU drivers installed so PyTorch/XTTSv2 can run its tensor operations rapidly).*
