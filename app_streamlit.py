import streamlit as st
import os
import time
import tempfile
import uuid
import hashlib
import io
import numpy as np
from pydub import AudioSegment
from model.deepfake_detector import DeepfakeDetector
from audiorecorder import audiorecorder

st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton > button[kind="primary"] {
        background-color: #4f8bf9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #3b6fd4;
    }
    .stButton > button[kind="primary"]:disabled {
        background-color: #3a3d47;
        color: #666;
    }
    .stButton > button[kind="secondary"] {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #3a3d47;
        border-radius: 8px;
    }
    .stFileUploader > div {
        background-color: #1a1d24;
        border: 2px dashed #3a3d47;
        border-radius: 12px;
        padding: 20px;
    }
    .card {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 12px;
        color: #4f8bf9;
        font-size: 18px;
    }
    .card p, .card li {
        color: #b0b0b0;
        font-size: 14px;
    }
    .prediction-badge {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        width: 100%;
    }
    .prediction-real {
        background-color: #1a4d2e;
        color: #4ade80;
        border: 2px solid #2d6a4f;
    }
    .prediction-fake {
        background-color: #4d1a1a;
        color: #f87171;
        border: 2px solid #6a2d2d;
    }
    .prediction-unknown {
        background-color: #4d4d1a;
        color: #fbbf24;
        border: 2px solid #6a6a2d;
    }
    .metric-card {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        height: 100%;
    }
    .metric-card .label {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #fafafa;
        font-size: 28px;
        font-weight: bold;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #b0b0b0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #4f8bf9;
        border-bottom: 2px solid #4f8bf9;
    }
    .download-btn {
        background-color: #1a4d2e;
        color: #4ade80;
        border: 2px solid #2d6a4f;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .audio-section {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .audio-section h3 {
        margin-top: 0;
        color: #4f8bf9;
    }
    .recording-section {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .recording-section h3 {
        margin-top: 0;
        color: #f59e0b;
    }
    div[data-testid="stAudio"] {
        background-color: #1a1d24;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning {
        background-color: #523a28;
        border: 1px solid #6a4d35;
        border-radius: 8px;
    }
    .stSuccess {
        background-color: #1a4d2e;
        border: 1px solid #2d6a4f;
        border-radius: 8px;
    }
    .stError {
        background-color: #4d1a1a;
        border: 1px solid #6a2d2d;
        border-radius: 8px;
    }
    .stInfo {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 8px;
    }
    hr {
        border-color: #2a2d36;
    }
    .processing-status {
        background-color: #1a1d24;
        border: 1px solid #4f8bf9;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    .processing-status .spinner-text {
        color: #4f8bf9;
        font-size: 16px;
        font-weight: 600;
    }
    .debug-info {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 12px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'audio_hash' not in st.session_state:
    st.session_state.audio_hash = None

@st.cache_resource
def load_model():
    detector = DeepfakeDetector(mode="inference")
    if os.path.exists("model_files/model.pth"):
        detector.load_model("model_files/model.pth")
        return detector
    return None

detector = load_model()

st.title("🎙️ Audio Deepfake Detection System")
st.markdown("Upload or record audio to analyze for deepfake artifacts.")

tab_upload, tab_record = st.tabs(["📁 Upload File", "🎤 Record Audio"])

audio_data = None
audio_name = None
audio_hash = None

with tab_upload:
    st.markdown('<div class="audio-section">', unsafe_allow_html=True)
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['flac', 'wav', 'mp3', 'm4a', 'ogg'], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
        audio_bytes = io.BytesIO(uploaded_file.getvalue())
        audio_segment = AudioSegment.from_file(audio_bytes, format=file_ext)
        audio_data = audio_segment.export(format='wav').read()
        audio_name = uploaded_file.name
        audio_hash = hashlib.md5(audio_data).hexdigest()
        
        st.markdown(f"**File**: {audio_name} ({len(audio_data)} bytes)")
        st.audio(audio_data)

with tab_record:
    st.markdown('<div class="recording-section">', unsafe_allow_html=True)
    st.markdown("### Record Audio")
    st.markdown("Click the button below to start recording. Speak clearly into your microphone.")
    recorded_audio = audiorecorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        pause_prompt="⏸️ Pause",
        show_visualizer=True,
        key="recorder"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if recorded_audio is not None and len(recorded_audio) > 0:
        audio_data = recorded_audio.export(format="wav").read()
        audio_name = "recording.wav"
        audio_hash = hashlib.md5(audio_data).hexdigest()
        
        st.markdown(f"**Recording**: {len(audio_data)} bytes")
        st.audio(audio_data)

if audio_data is not None:
    st.markdown("---")
    
    col_btn, col_status = st.columns([3, 1])
    with col_btn:
        analyze_button = st.button(
            "🔍 Analyze Audio", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.is_processing
        )
    with col_status:
        if st.session_state.is_processing:
            st.markdown("⏳ **Processing...**")

if audio_data is not None and analyze_button and audio_hash != st.session_state.audio_hash:
    st.session_state.is_processing = True
    st.session_state.audio_hash = audio_hash
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        st.markdown("---")
        st.markdown("### 🔬 Analysis Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        result = detector.analyze_file(tmp_path)
        
        if "error" in result:
            status_text.error(f"❌ {result.get('error', 'Error processing audio')}")
            st.session_state.is_processing = False
        else:
            y = result["audio"]
            sr = result["sr"]
            mfcc = result["mfcc"]
            info = result["audio_info"]
            
            chunks = detector.get_chunks(y, sr)
            num_chunks = len(chunks)
            
            predictions = []
            for i, chunk in enumerate(chunks):
                status_text.text(f"📊 Processing chunk {i+1}/{num_chunks}...")
                progress_bar.progress((i + 1) / num_chunks)
                
                features = detector.preprocess_audio(chunk)
                pred = detector.predict(features)
                predictions.append(pred)
                
                st.write(f"   Chunk {i+1}: prediction = {pred:.4f}")
            
            avg_prediction = np.mean(predictions)
            prediction = "REAL" if avg_prediction < 0.5 else "FAKE"
            confidence = float(abs(0.5 - avg_prediction) * 2 * 100)
            probability = float(avg_prediction)
            
            status_text.text("✅ Analysis Complete!")
            progress_bar.progress(100)
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            pred_class = "prediction-real" if prediction == "REAL" else ("prediction-fake" if prediction == "FAKE" else "prediction-unknown")
            st.markdown(f'<div class="prediction-badge {pred_class}">{prediction}</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col2:
                st.metric("Probability", f"{probability:.3f}")
            with col3:
                st.metric("Chunks Processed", f"{num_chunks}")
            
            if result.get("warning"):
                st.warning(f"⚠️ {result['warning']}")
            
            st.markdown("### 📈 Visualizations")
            col_wave, col_mfcc = st.columns(2)
            
            with col_wave:
                fig = detector.generate_waveform_plot(y, sr)
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            
            with col_mfcc:
                fig = detector.generate_mfcc_plot(mfcc)
                st.pyplot(fig)
                plt.close(fig)
            
            st.markdown("### 📋 Details")
            tab1, tab2, tab3 = st.tabs(["📁 File Info", "📊 Statistics", "🤖 Model Info"])
            
            with tab1:
                st.markdown(f"""
                | Field | Value |
                |-------|-------|
                | **Filename** | {info['filename']} |
                | **Duration** | {info['duration']:.2f}s |
                | **Sample Rate** | {info['sample_rate']} Hz |
                | **Channels** | {info['channels']} |
                | **Format** | {info['format']} |
                """)
            
            with tab2:
                stats = detector.get_feature_stats(y, sr, mfcc)
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Mean MFCC", f"{float(stats['mean_mfcc']):.4f}")
                col_s2.metric("Frames", f"{int(stats['frame_count'])}")
                col_s3.metric("Energy", f"{float(stats['energy']):.4f}")
                col_s4.metric("Spectral Centroid", f"{float(stats['spectral_centroid']):.0f} Hz")
            
            with tab3:
                st.markdown("""
                | Field | Value |
                |-------|-------|
                | **Architecture** | Lightweight CNN (PyTorch) |
                | **Training Dataset** | ASVspoof-derived (~2.5GB) |
                | **Model Version** | 1.0 |
                | **Input Features** | MFCC (40) + Delta + Delta-Delta |
                | **Total Parameters** | ~423K |
                """)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pdf_dir = "temp"
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(pdf_dir, f"report_{timestamp}.pdf")
            
            result["prediction"] = prediction
            result["confidence"] = confidence
            result["probability"] = probability
            detector.generate_pdf(tmp_path, result, pdf_path)
            
            st.markdown("---")
            st.markdown("### 📄 Download Report")
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name=f"deepfake_report_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            
            st.success("✅ Analysis complete! All processing finished successfully.")
            
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    finally:
        st.session_state.is_processing = False

elif audio_data is None:
    st.info("👆 Upload an audio file or record audio, then click Analyze")

st.markdown("---")
st.markdown("*Audio Deepfake Detection System - Powered by PyTorch*")
