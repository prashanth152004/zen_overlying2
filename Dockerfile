# ── Stage: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies: FFmpeg (video processing) + build tools for native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    curl \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Install dependencies in stages to avoid pip resolver depth issues with TTS + numba.
COPY requirements.txt .
# Stage A: Lock heavy AI frameworks first so pip doesn't try to resolve them later
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numba==0.61.0 llvmlite==0.44.0 && \
    pip install --no-cache-dir TTS==0.22.0 && \
    pip install --no-cache-dir faster-whisper==1.0.3
# Stage B: Install remaining (lighter) packages normally with full dependency resolution
RUN pip install --no-cache-dir \
        "pyannote.audio>=3.3.1" \
        "deep-translator==1.11.4" \
        "gtts>=2.5.0" \
        "librosa>=0.10.0" \
        "edge-tts>=6.1.0" \
        "pydub==0.25.1" \
        "pyloudnorm==0.1.1" \
        "soundfile==0.12.1" \
        "ffmpeg-python==0.2.0" \
        "opencv-python-headless==4.9.0.80" \
        "streamlit==1.32.2" \
        "python-dotenv==1.0.1" \
        numpy scipy requests tqdm \
        "google-generativeai==0.8.3"

# Do NOT copy source code here — it is volume-mounted at runtime for live reload.
# This means any file you edit locally is instantly reflected inside the container.

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit with file watcher enabled for live reload on code changes
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=poll", \
     "--server.runOnSave=true"]
