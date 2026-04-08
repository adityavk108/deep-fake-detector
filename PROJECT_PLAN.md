# Audio Deepfake Detection System - Project Plan

## 1. Project Objectives

1. Develop an audio deepfake detection model by extracting MFCCs and temporal derivatives from speech samples, training a lightweight CNN to classify audio as human or AI-generated.
2. Design and implement a web-based interface using Streamlit that processes uploaded audio files, performs feature extraction and model inference, and returns a probability-based prediction score.
3. Generate an automated analytical report (PDF) for cybersecurity auditing, including extracted audio features, model confidence score, and technical metadata.

---

## 2. Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **ML Framework** | PyTorch | Switched from TensorFlow (GPU detection issues in WSL2) |
| **Inference** | PyTorch | Native PyTorch inference |
| **Audio Processing** | Librosa | Handles all audio formats |
| **Numerical** | NumPy | Array operations |
| **Visualization** | Matplotlib | Waveform + MFCC plots |
| **PDF Generation** | ReportLab | Automated report generation |
| **UI Framework** | Streamlit | Switched from Gradio (performance issues) |
| **Recording** | streamlit-audiorecorder | Browser-based audio recording |
| **Audio Format Conversion** | pydub | Convert MP3/FLAC to WAV for processing |

---

## 3. GPU Strategy

| Component | Decision |
|-----------|----------|
| **Environment** | WSL2 |
| **GPU** | RTX 3050 (4GB VRAM) |
| **RAM** | 32GB |
| **PyTorch GPU** | Auto-detected via `torch.cuda.is_available()` |
| **Batch Size** | 16 (fits in 4GB VRAM) |

**Note:** TensorFlow had GPU detection issues in WSL2. Switched to PyTorch which auto-detects GPU correctly.

---

## 4. Dataset Structure

```
dataset/
├── train/
│   ├── real/    # .flac files, label = 0
│   └── fake/    # .flac files, label = 1
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Dataset Size:** ~3GB
**Format:** FLAC files for training
**Inference:** Accepts all formats (FLAC, WAV, MP3, M4A, OGG) - converted to WAV internally

---

## 5. Project Structure

```
Mini-Project/
├── model/
│   ├── deepfake_detector.py    # Core ML class (train + inference + PDF)
│   └── __init__.py
├── app_streamlit.py            # Streamlit UI
├── train.py                    # Training script
├── temp/                       # Temporary PDFs (auto-cleaned)
├── dataset/                    # Training data
├── model_files/                # Saved models (model.pth)
├── plans/                      # Implementation plans
├── PRD.md                      # Product Requirements Document
├── PROJECT_PLAN.md             # This file
├── README.md                   # Setup + usage instructions
└── requirements.txt
```

---

## 6. Architecture (Streamlit-Based)

### 6.1 Core Principle

Streamlit serves as the UI. No separate API layer, no frontend framework.

| Aspect | How It Works |
|--------|-------------|
| **No Database** | Zero storage of predictions, users, or history |
| **PDF Storage** | Saved to `temp/` folder with unique filename |
| **Plots (Waveform, MFCC)** | Displayed directly in Streamlit interface via Matplotlib |
| **File Handling** | Uploaded audio saved to temp → processed → deleted after response |
| **Session State** | Minimal - tracks processing state and audio hash for cache busting |
| **Cache Busting** | MD5 hash of audio data prevents stale results |

### 6.2 Streamlit Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  🎙️ Audio Deepfake Detection System                    │
│  Upload or record audio to analyze for deepfake        │
├─────────────────────────────────────────────────────────┤
│  [📁 Upload File]  [🎤 Record Audio]                   │
│  ─────────────────────────────────────────────────────  │
│  File: example.mp3 (4,750,892 bytes)                   │
│  [▶ Audio Player]                                      │
│  ─────────────────────────────────────────────────────  │
│  [🔍 Analyze Audio]                          ⏳ Process │
├─────────────────────────────────────────────────────────┤
│  🔬 Analysis Progress                                  │
│  [████████████░░░░░░░░░░░░] 75%                        │
│  📊 Processing chunk 3/4...                            │
│     Chunk 1: prediction = 0.1234                        │
│     Chunk 2: prediction = 0.2345                       │
│     Chunk 3: prediction = 0.3456                       │
├─────────────────────────────────────────────────────────┤
│  📊 Results                                            │
│  ┌─────────────────────────────────────────────────┐  │
│  │                      REAL                        │  │
│  └─────────────────────────────────────────────────┘  │
│  Confidence    │ Probability    │ Chunks Processed    │
│     87.3%       │    0.437        │        4            │
├─────────────────────────────────────────────────────────┤
│  📈 Visualizations                                     │
│  [Waveform Plot]    │    [MFCC Plot]                  │
├─────────────────────────────────────────────────────────┤
│  📋 Details                                            │
│  [📁 File Info] [📊 Statistics] [🤖 Model Info]        │
│  ─────────────────────────────────────────────────────  │
│  [📥 Download PDF Report]                             │
│  ✅ Analysis complete!                                │
└─────────────────────────────────────────────────────────┘
```

### 6.3 User Flow

1. User opens `http://localhost:8501` → sees Streamlit interface
2. Uploads audio file OR records audio
3. Clicks "Analyze Audio"
4. Progress bar shows chunk processing
5. Results displayed:
   - Prediction (Real/Fake) with confidence
   - Waveform plot
   - MFCC plot
   - File metadata
   - Feature statistics
   - PDF download button
6. User clicks "Download PDF Report" → PDF downloaded

---

## 7. ML Pipeline

### 7.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `sample_rate` | 16000 Hz |
| `n_mfcc` | 40 |
| `n_fft` | 512 |
| `hop_length` | 160 |
| `max_len` | 400 frames |
| `batch_size` | 16 |
| `epochs` | 20 |
| `fixed_audio_duration` | 4 seconds |
| `learning_rate` | 0.0003 (Adam) |

### 7.2 Audio Preprocessing (Training)

1. **Load**: Librosa → mono → 16kHz resample → normalize amplitude
2. **Fixed Duration**:
   - If > 4 seconds → truncate (take first 4s)
   - If < 4 seconds → pad with zeros
3. **Feature Extraction**:
   - MFCC (40 coefficients)
   - Delta (first derivative)
   - Delta-Delta (second derivative)
   - Stack → shape: (40, T, 3)
4. **Pad/Truncate Features**:
   - Fix to (40, 400, 3)
   - Pad or truncate along time axis

**Note:** Training uses fixed 4-second clips for consistent batch sizes. Inference uses chunk + average (see 7.3).

### 7.3 Chunk + Average Inference Strategy

For audio longer than 4 seconds, instead of truncating, we split into overlapping chunks and average predictions.

**How it works:**
1. Load full audio (16kHz, mono, normalized)
2. Split into 4-second chunks with 50% overlap (2-second stride)
   - Example: 10-second audio → chunks at [0-4s], [2-6s], [4-8s], [6-10s]
3. For each chunk:
   - Extract MFCC + delta + delta-delta
   - Pad/truncate to (40, 400, 3)
   - Run PyTorch inference → get probability
4. Average all chunk probabilities → final prediction
5. If audio < 4 seconds: pad and predict normally

**Example:**
```
10-second audio → 4 chunks → predictions: [0.12, 0.08, 0.05, 0.10]
→ average: 0.0875 → prediction: "Real" (91.25% confidence)
```

### 7.4 Model Architecture (PyTorch Lightweight CNN)

```
Input: (3, 40, 400)  # Channels first for PyTorch
│
├── Conv1d(1, 32, kernel_size=3) + ReLU + BatchNorm + MaxPool
├── Conv1d(32, 64, kernel_size=3) + ReLU + BatchNorm + MaxPool
├── Conv1d(64, 128, kernel_size=3) + ReLU + BatchNorm + MaxPool
├── Conv1d(128, 256, kernel_size=3) + ReLU + BatchNorm
│
├── GlobalAveragePooling1D
│
├── Linear(256, 128) + ReLU
├── Dropout(0.4)
│
└── Linear(128, 1) + Sigmoid → Output (0-1 probability)
```

**Total Parameters:** ~423K

**Compilation:**
- Optimizer: Adam (lr=0.0003)
- Loss: binary_crossentropy
- Metrics: accuracy, precision, recall

### 7.5 Training Pipeline

1. Load `dataset/train` → precompute ALL MFCCs into memory (X_train, y_train)
2. Load `dataset/val` → precompute ALL MFCCs into memory (X_val, y_val)
3. Build PyTorch model
4. Train with:
   - `epochs=20`
   - `batch_size=16`
   - `EarlyStopping(monitor='val_loss', patience=5)`
5. Save `model.pth`
6. Evaluate on `dataset/test`
7. Compute metrics: accuracy, precision, recall, F1 score

**Training Results:**
- Accuracy: 89.68%
- Precision: 99.26%
- Recall: 80.04%

### 7.6 Inference Pipeline (via Streamlit)

1. User uploads audio file or records audio
2. Convert to WAV using pydub (important for MP3/FLAC)
3. Save to temp file
4. `load_audio()` → raw audio (16kHz, mono, normalized)
5. If audio > 4s: chunk + average strategy (see 7.3)
6. If audio <= 4s: single prediction
7. For each chunk (or single):
   - `extract_features()` → MFCC + delta + delta-delta
   - `pad_or_truncate()` → (40, 400)
   - `predict()` → PyTorch inference → probability
8. Average probabilities (if multiple chunks)
9. Generate waveform plot → display in Streamlit
10. Generate MFCC plot → display in Streamlit
11. Generate PDF → save to `temp/` folder → return as downloadable file
12. Return all results to Streamlit interface

---

## 8. Streamlit App (`app_streamlit.py`)

### 8.1 Components

| Streamlit Component | Purpose |
|-------------------|---------|
| `st.file_uploader` | Audio file upload (supports all formats) |
| `st.audio` | Audio playback |
| `audiorecorder` | Browser-based audio recording |
| `st.progress` | Progress bar for chunk processing |
| `st.empty` | Dynamic status updates |
| `st.pyplot` | Display waveform and MFCC plots |
| `st.metric` | Display metrics |
| `st.tabs` | Organize details sections |
| `st.download_button` | PDF download |

### 8.2 App Features

- **Dark theme** with custom CSS
- **Tab-based UI**: Upload File / Record Audio
- **Progress display**: Shows "Processing chunk X of Y" for each chunk
- **Cache busting**: MD5 hash of audio prevents stale results
- **Session state**: Minimal tracking for is_processing and audio_hash
- **Audio format conversion**: Uses pydub to convert MP3/FLAC to WAV

### 8.3 Key Implementation Details

```python
# Audio format conversion (critical for MP3/FLAC uploads)
file_ext = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
audio_bytes = io.BytesIO(uploaded_file.getvalue())
audio_segment = AudioSegment.from_file(audio_bytes, format=file_ext)
audio_data = audio_segment.export(format='wav').read()

# Cache busting using hash
audio_hash = hashlib.md5(audio_data).hexdigest()
if audio_hash != st.session_state.audio_hash:
    # Process new audio
    st.session_state.audio_hash = audio_hash
```

---

## 9. PDF Report Contents

**Layout:** Clean white professional design (not dark theme)

### 9.1 Report Structure

| Section | Content |
|---------|---------|
| **Header** | Report title, unique Report ID (timestamp-based) |
| **File Information** | Original filename, file size, duration, sample rate, channels, format |
| **Prediction Summary** | Result (Real/Fake), confidence percentage, probability score |
| **Audio Analysis - Waveform** | Matplotlib-generated waveform visualization |
| **Audio Analysis - MFCC** | Matplotlib-generated MFCC feature visualization |
| **Feature Statistics** | Mean MFCC values, frame count, audio energy, spectral centroid |
| **Model Information** | Architecture, training dataset, model version |
| **Technical Explanation** | MFCCs explanation, deepfake detection methodology |
| **Disclaimer** | AI-generated analysis disclaimer |

---

## 10. Class Design: `DeepfakeDetector`

### 10.1 Initialization

```python
__init__(self, mode="inference")
```

- Detects GPU via PyTorch
- If `mode="inference"`: loads model, sets eval mode
- If `mode="train"`: initializes fresh model

### 10.2 Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `load_audio(file_path)` | file path | (y, sr) | Load, resample, normalize |
| `preprocess(file_path)` | file path | dict | Full preprocessing pipeline |
| `preprocess_audio(y)` | audio array | tensor | MFCC + delta + delta-delta |
| `build_model()` | - | PyTorch model | Build CNN architecture |
| `load_dataset(split_path)` | folder path | X, y tensors | Precompute MFCCs |
| `train(dataset_root)` | dataset root path | - | Train model, save metrics |
| `load_model(model_path)` | path | - | Load trained .pth file |
| `predict(features)` | features tensor | float (0-1) | PyTorch inference |
| `get_chunks(y, sr)` | audio, sr | list of chunks | Split audio into 4s chunks |
| `analyze_file(file_path)` | file path | dict | Complete analysis |
| `generate_waveform_plot(y, sr)` | audio, sr | Matplotlib figure | Waveform plot |
| `generate_mfcc_plot(mfcc)` | mfcc array | Matplotlib figure | MFCC plot |
| `get_feature_stats(y, sr, mfcc)` | audio, sr, mfcc | dict | Feature statistics |
| `generate_pdf(file_path, result, save_path)` | file, result, path | - | Generate PDF report |

---

## 11. Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| TensorFlow GPU not detected in WSL2 | Switched to PyTorch which auto-detects GPU |
| Gradio performance issues | Switched to Streamlit |
| MP3/FLAC files not processing correctly | Added pydub conversion to WAV |
| Caching causing fake results (26s→6.3s) | Added MD5 hash of audio to trigger re-runs |
| Empty audio (44 bytes) from recorder | Check `len(audio_data) < 100` before processing |
| Recordings showing as FAKE | Model trained on clean studio audio; recordings have different characteristics |

---

## 12. Output Files

| File | Location | Purpose |
|------|----------|---------|
| `model.pth` | `model_files/` | Trained PyTorch model (1.7MB) |
| PDF reports | `temp/` | Analysis reports |

---

## 13. Dependencies

```
torch>=2.0
librosa>=0.10
numpy>=1.24
matplotlib>=3.7
reportlab>=4.0
streamlit>=1.28
streamlit-audiorecorder>=0.0.6
pydub>=0.25
scikit-learn>=1.3
```

---

## 14. Setup Instructions

### Training

1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (WSL2) or `venv\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
4. Train model: `python train.py`
5. Model saved as `model_files/model.pth`

### Running the App

1. Activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app_streamlit.py`
4. Open: `http://localhost:8501`

---

## 15. Validation Checklist

- [x] WSL2 working
- [x] Dataset downloaded and structured correctly
- [x] Tech stack finalized (PyTorch + Streamlit)
- [x] UI layout approved
- [x] PDF report format finalized
- [x] Model architecture finalized
- [x] Training pipeline works (89.68% accuracy)
- [x] Chunk + average inference strategy implemented
- [x] Audio format conversion working
- [x] Cache busting implemented

---

*Document Version: 5.0*
*Last Updated: 2026-04-08*
*Changes from v4.0: Switched from TensorFlow to PyTorch, Gradio to Streamlit, added audio recording, fixed MP3 conversion, added cache busting via hash*
