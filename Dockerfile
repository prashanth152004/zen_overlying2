# ── Stage: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies: FFmpeg + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    curl \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dramatically faster pip resolution
RUN pip install --upgrade pip uv

WORKDIR /app
COPY requirements.txt .

# ── Stage A: Pin all conflict-prone packages first ────────────────────────────
# Install PyTorch (CPU) first so TTS/pyannote don't fight over it
RUN uv pip install --system --no-cache \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pin numba + llvmlite before anything else pulls them in transitively
RUN uv pip install --system --no-cache \
    "numba==0.61.0" "llvmlite==0.44.0"

# Install librosa & pyannote with --no-deps so they can't re-resolve numba
RUN uv pip install --system --no-cache --no-deps \
    "librosa==0.10.2" \
    "pyannote.audio==3.3.2"

# Install the transitive deps that librosa/pyannote actually need (excluding numba already installed)
RUN uv pip install --system --no-cache \
    audioread decorator pooch soxr lazy-loader \
    asteroid-filterbanks einops huggingface_hub omegaconf \
    pytorch-metric-learning speechbrain \
    pyannote.core pyannote.database pyannote.metrics pyannote.pipeline

# ── Stage B: Remaining packages (lighter, no numba conflicts) ─────────────────
RUN uv pip install --system --no-cache \
    "transformers<4.42.0" \
    "TTS==0.22.0" \
    "faster-whisper==1.0.3" \
    "deep-translator==1.11.4" \
    "gtts>=2.5.0" \
    "edge-tts>=6.1.0" \
    "pydub==0.25.1" \
    "pyloudnorm==0.1.1" \
    "soundfile==0.12.1" \
    "ffmpeg-python==0.2.0" \
    "opencv-python-headless==4.9.0.80" \
    "streamlit==1.32.2" \
    "python-dotenv==1.0.1" \
    numpy scipy requests tqdm \
    "google-generativeai==0.8.3" \
    "indic-transliteration==2.3.61"

# Source code is volume-mounted (live reload) — do not COPY here.
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=poll", \
     "--server.runOnSave=true"]
