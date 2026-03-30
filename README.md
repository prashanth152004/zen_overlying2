# National Media Translation Portal

An advanced, Dockerized AI pipeline for regional video translation and civic voice cloning. This application seamlessly ingests video content in English, Hindi, or Kannada, transcribes the speech, translates it contextually, and perfectly clones the original speakers' voices and emotional inflections into the target languages.

## 🚀 Key Features

* **Multi-Language Support**: Automatically detects input language and translates/dubs the video into the remaining supported languages (English, Hindi, Kannada).
* **Conversational Voice Cloning (XTTSv2)**: Generates highly realistic, zero-shot voice clones. Includes a unique **Dynamic Emotion Referencing** system that captures the exact excitement, pauses, and pacing of the original sentence, completely removing the "robotic reading" effect.
* **Phonetic Language Unlock**: Natively tricks XTTS into speaking unsupported languages like Kannada by mapping the phonetic Devanagari script over the Kannada Unicode characters.
* **Speaker Diarization (Pyannote)**: Separates overlapping voices (e.g., Male vs. Female) and assigns them persistent, unique voice profiles throughout the entire video.
* **Contextual Batch Translation**: Bypasses naive word-for-word translation limiters by sending grouped sentence blocks (via Google Deep-Translator/Sarvam APIs), achieving natural sentence flow.
* **Audio Engineering**: Includes precise background audio "ducking" (LUFS adjustments), multi-track mixing, and a custom Netflix-style video interface to instantly test out multiple audio tracks.

## 🛠️ Technical Architecture

| Stage | Technology / Component | Description |
|---|---|---|
| **1. Ingest & Detect** | `Faster-Whisper` | VAD filtering and blazing-fast ASR transcription. Auto-detects spoken language (en, hi, kn). |
| **2. Speaker Separation** | `Pyannote.audio` (HF Token) | Isolates individual speakers and groups their vocal segments to lock in unique genders and identity clones. |
| **3. Translation** | `Deep-Translator` / `Sarvam AI` | Translates the foundational English transcript into target scripts, utilizing intelligent double-newline batching to keep conversation context intact. |
| **4. Emotional Cloning** | `Coqui XTTSv2` / `Edge-TTS` | Extracts the exact matching sentence audio from the original video, blending it with the speaker's main profile to mimic exact intonation and cadence. |
| **5. Subtitles & Muxing** | `FFmpeg` / `Pysubs2` | Bakes high-contrast ASS subtitles and mathematically layers the newly generated AI speech exactly over the silenced original actors, retaining background ambiance. |

## 📦 Local Setup & Docker Installation

The entire application runs inside an isolated Docker container, avoiding massive dependency conflicts between PyTorch, LLVM, Numba, and system packages.

### Prerequisites
* Docker Desktop installed with at least **8GB - 12GB+ RAM** allocated (Settings -> Resources).
* Hugging Face Access Token (required for Pyannote Speaker Diarization).

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/prashanth152004/zen_overlying2.git
   cd zen_overlying2
   ```

2. Build and launch the container cluster:
   ```bash
   docker compose up --build
   ```
   *(Note: The first time downloading the models will require ~4GB of storage space in the `.cache/huggingface` volume).*

3. Open your browser and navigate to the Streamlit Control Center:
   ```text
   http://localhost:8501
   ```

4. **Paste your Hugging Face Token** in the sidebar to activate the multi-speaker AI, drop your video in, and click "Start Production Pipeline"!

## ⚠️ Important Configurations for 8GB Systems

To prevent Docker Out-Of-Memory (OOM) crashes, the `pipeline.py` script has been aggressively fine-tuned to sequentially load and garbage-collect heavy PyTorch models. It ensures the 2.5GB Whisper model is deleted from system memory *before* the 3GB XTTS engine is loaded.

## 📄 License
This architecture utilizes open-source AI frameworks (Whisper, XTTSv2, Pyannote). Ensure compliance with respective model licenses and Coqui's TTS Terms of Service.
