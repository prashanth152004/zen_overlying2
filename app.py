# app.py

import streamlit as st
import streamlit.components.v1 as components
import os
import time
from pipeline import TranslationPipeline
from services.player_service import get_netflix_player_html

# --- UI Configuration ---
st.set_page_config(
    page_title="National Media Translation Portal",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Indian Government Theme ---
st.markdown("""
    <style>
    /* Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }

    /* Top Tricolor Banner */
    .block-container {
        border-top: 8px solid;
        border-image: linear-gradient(to right, #FF9933 33.33%, #FFFFFF 33.33%, #FFFFFF 66.66%, #138808 66.66%) 1;
        padding-top: 2rem;
    }

    /* Light Theme Background */
    .main {
        background-color: #F8F9FA;
        color: #1A1A1A;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Official Headings */
    h1, h2, h3 {
        color: #000080; /* Navy Blue */
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h1 {
        border-bottom: 2px solid #FF9933;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Card Styling */
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 4px;
        border-left: 5px solid #000080;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Primary Buttons (Saffron/Green accents depending on action) */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #138808; /* Indian Green */
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background-color: #0F6B06;
        box-shadow: 0 4px 8px rgba(19,136,8,0.3);
        border: none;
        color: white;
    }

    /* Progress bar */
    .stProgress .st-bo {
        background-color: #000080;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 4px;
        border-left: 4px solid #FF9933;
    }
    
    /* Top Header Official Details */
    .gov-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #F0F0F0;
        padding: 5px 20px;
        font-size: 12px;
        color: #333;
        border-bottom: 1px solid #CCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Technical Architecture & Settings ---
with st.sidebar:
    st.markdown("### 🏛️ Directorate of Media Technology")
    
    with st.expander("🎚️ Advanced Audio Controls", expanded=True):
        st.write("Determine sound balance and TTS speed.")
        bg_lufs = st.slider(
            "Original Background Loudness (LUFS)",
            min_value=-40.0, max_value=-10.0, value=-25.0, step=1.0,
            help="-25 LUFS is standard for 'ducking' audio heavily under speech. -40 essentially mutes the background."
        )
        fg_gain = st.slider(
            "Translated Speech Boost (dB)",
            min_value=-10.0, max_value=10.0, value=0.0, step=1.0,
            help="Increase or decrease the loudness of the AI generated translated voice."
        )
        st.info("🗣️ **Voice Speed: Auto-Adjusted**\n\nThe pipeline automatically computes the ideal speed for each segment based on word count and available duration — no manual tuning needed.")

    with st.expander("🌍 AI Translation Model", expanded=True):
        st.write("Select translation engine:")
        translation_model_choice = st.selectbox(
            "Translation Model",
            options=["Deep Translation (Free/Open)", "Sarvam AI (High Accuracy/Paid)"],
            index=0,
            label_visibility="collapsed"
        )
        sarvam_api_key = None
        if translation_model_choice == "Sarvam AI (High Accuracy/Paid)":
            sarvam_api_key = st.text_input(
                "Sarvam AI Subscription Key",
                type="password",
                help="Required to use the Sarvam AI translation API."
            )
        internal_model_key = "sarvam_ai" if translation_model_choice == "Sarvam AI (High Accuracy/Paid)" else "deep_translator"

    with st.expander("🤗 Speaker Diarization (Pyannote)", expanded=False):
        st.write("Enable real multi-speaker detection via Hugging Face Pyannote.")
        hf_token = st.text_input(
            "Hugging Face Access Token",
            type="password",
            placeholder="hf_...",
            help="Get your token at: https://hf.co/settings/tokens  Also accept the model at: https://hf.co/pyannote/speaker-diarization-3.1"
        )
        if hf_token:
            st.success("✅ HF Token set — Pyannote diarization enabled!")
        else:
            st.info("ℹ️ No token — using single-speaker mode (SPEAKER_01).")

    st.divider()
    
    st.title("System Architecture")
    
    with st.expander("🎥 Stage 1: Media Ingest", expanded=False):
        st.write("Secure processing via FFmpeg for high-fidelity PCM audio extraction and video encoding protocols.")
        
    with st.expander("🗣️ Stage 2: Civic ASR Engine", expanded=False):
        st.write("Regional-optimized neural modeling (Faster-Whisper) for accurate Kannada speech recognition and timestamp generation.")
        
    with st.expander("🌐 Stage 3: Regional Translation", expanded=False):
        st.write("Integrated translation matrix mapping regional languages (Kannada) to administrative English.")
        
    with st.expander("🧠 Stage 4: Acoustic Signature Replication", expanded=False):
        st.write("Utilizing XTTSv2 frameworks to clone and synthesize vocal identity for seamless dubbing.")
        
    with st.expander("🎚️ Stage 5: Audio Mastering", expanded=False):
        st.write("Dynamic range compression and 1-3kHz EQ attenuation to embed original ambiance behind synthesized speech.")
        
    with st.expander("📝 Stage 6: Universal Accessibility Subtitles", expanded=False):
        st.write("Advanced Substation Alpha (ASS) format utilized for mobile-optimized, high-contrast captioning.")

    st.divider()
    st.info("Authorized Personnel Only. System governed by the National Data Processing Guidelines.")

# --- Main Page Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("National Portal for Civic Media Translation")
    st.subheader("Official Platform for Regional Video Processing")
    
    st.markdown("""
    ### Processing Guidelines:
    1. **Upload**: Submit the official Kannada civic feedback or address video.
    2. **Process**: The National AI Pipeline will transcribe, translate, and synthesize the English and Hindi equivalents simultaneously.
    3. **Review**: Access the finalized official broadcast file with compliant multi-track subtitles.
    """)

    with st.expander("🤖 AI Models & Tools Used", expanded=False):
        st.markdown("""
        **1. Faster-Whisper**
        * **Purpose:** Speech Recognition & Translation
        * **Description:** An optimized engine used to transcribe the original Kannada audio and natively translate it into high-accuracy English transcripts.
        * **Steps:** Extract Audio `->` Perform Voice Activity Detection (VAD) `->` Generate English Text & Timestamps.

        **2. Pyannote.audio**
        * **Purpose:** Speaker Diarization
        * **Description:** Neural network AI that detects multiple speakers in an audio track, allowing the pipeline to map translated dialogue back to the correct original speaker.
        * **Steps:** Acoustic Feature Extraction `->` Speaker Segmentation `->` Clustering Speaker Turns.

        **3. Deep-Translator (Free)**
        * **Purpose:** Transcript Translation
        * **Description:** Free, open-source AI translation service used to convert the baseline English transcript into other regional languages.
        * **Steps:** Read English Transcript `->` Translation Execution `->` Output Target Transcript.

        **4. Sarvam AI (Paid/High Accuracy)**
        * **Purpose:** Contextual Translation
        * **Description:** State-of-the-art regional LLM used for high-accuracy, context-aware translation of administrative English into Indian regional languages like Hindi.
        * **Steps:** Format Payload `->` API Request (Indic LLM) `->` Parse Translated JSON.

        **5. Coqui XTTSv2 (Local)**
        * **Purpose:** Voice Cloning & Text-to-Speech (TTS)
        * **Description:** A zero-shot voice generation model that replicates the original speaker's tone, pitch, and cadence into a new language using a dynamically selected, clean reference audio sample.
        * **Steps:** Find Best Speaker Sample `->` Extract Acoustic Signature `->` Synthesize Translated Speech (Paced via TTS Speed).
        
        **6. FFmpeg & Pydub**
        * **Purpose:** Audio Processing, Clarity & Multiplexing
        * **Description:** Advanced audio engineering tools. Used for broadcast-quality voice clarity (EQ, Compression, Low/High Pass) and precise lip-syncing using chained `atempo` time-stretching.
        * **Steps:** Enhance Voice Clarity `->` Apply Chained Atempo Stretch `->` Background Ducking `->` Multiplex Master Video.
        """)

    uploaded_file = st.file_uploader("Upload Official Video File (.mp4, .mov)", type=["mp4", "mov", "mkv"])

with col2:
    if uploaded_file is not None:
        st.markdown("### Preview")
        st.video(uploaded_file)
        
        if st.button("🚀 Start Production Pipeline"):
            temp_dir = "workspace"
            os.makedirs(temp_dir, exist_ok=True)
            in_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(in_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- Pipeline execution ---
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                pipeline = TranslationPipeline(
                    work_dir=temp_dir, 
                    bg_lufs=bg_lufs, 
                    fg_gain=fg_gain,
                    translation_model=internal_model_key,
                    sarvam_api_key=sarvam_api_key,
                    tts_speed=1.0,  # Auto-adjusted per segment in voice_service
                    hf_token=hf_token if hf_token else None
                )
                
                status_container.info("🔄 Processing National Assets (Multi-Language Generation)...")
                progress_bar.progress(30)
                
                results = pipeline.run(in_path)
                
                progress_bar.progress(100)
                status_container.success("✅ Multi-Language Media Generation Complete!")
                
                # Store results in session_state so UI interactions don't reload the pipeline
                st.session_state["pipeline_results"] = results
                st.session_state["uploaded_filename"] = uploaded_file.name
                    
            except Exception as e:
                status_container.error(f"❌ Pipeline Failed: {str(e)}")
                st.exception(e)

# --- Netflix-Style Multi-Track Player ---
if "pipeline_results" in st.session_state:
    st.divider()
    st.markdown("### 🎬 Official Output Media Player")
    
    results = st.session_state["pipeline_results"]
    
    # Load subtitle dictionaries based on our keys
    subtitle_dict = {}
    if "en" in results["subtitles"]:
        subtitle_dict["English"] = results["subtitles"]["en"]
    if "hi" in results["subtitles"]:
        subtitle_dict["Hindi"] = results["subtitles"]["hi"]
    
    # Generate custom Netflix-style HTML
    player_html = get_netflix_player_html(
        videos_dict=results["videos"],
        subtitles_dict=subtitle_dict,
        default_audio="English Dub"
    )
    
    # Render full width player
    components.html(player_html, height=500, scrolling=False)
    
    # Render Download Buttons Below Player
    st.markdown("#### Media Downloads")
    
    # Master Video Download
    if "master_video" in results and os.path.exists(results["master_video"]):
        with open(results["master_video"], "rb") as file:
            st.download_button(
                label="🌟 Download Master Multi-Track Video (.mkv)",
                data=file,
                file_name=f"Master_{st.session_state['uploaded_filename'].split('.')[0]}.mkv",
                mime="video/x-matroska",
                use_container_width=True,
                type="primary"
            )
            
        st.markdown("*The Master SDK Video contains all isolated/overlaid translated audio tracks and subtitles multiplexed natively into a single file. Perfect for offline broadcast and multi-track players like VLC.*")
        st.divider()
        st.markdown("##### Individual Track Downloads")

    dl_cols = st.columns(len(results["videos"]))
    
    for i, (audio_track, video_path) in enumerate(results["videos"].items()):
        with dl_cols[i]:
            with open(video_path, "rb") as file:
                st.download_button(
                    label=f"⬇️ Download {audio_track}",
                    data=file,
                    file_name=f"{audio_track.replace(' ', '_')}_{st.session_state['uploaded_filename']}",
                    mime="video/mp4",
                    use_container_width=True
                )
            
    with st.expander("📊 Official Data Processing Report"):
        st.json(results["qc_report"])
