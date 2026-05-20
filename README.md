# Audio Deepfake Detection System

A comprehensive, web-based cybersecurity utility designed to analyze audio files and ascertain whether they are human-generated or synthesized by AI (deepfakes). The system processes audio through an advanced machine learning pipeline, providing actionable insights, visualizations, and downloadable audit reports.

<img width="1600" height="762" alt="image" src="https://github.com/user-attachments/assets/7a2ac199-9c28-4492-b25f-4540b6b3b78b" />

## Features

- **Resilient audio preprocessing pipeline**: Standardizes a wide variety of audio with varying sampling rates and channels using smart chunking, and converts them into MFCC for inference.

- **Deepfake Inference Model**: Powered by a custom PyTorch CNN trained on datasets adapted from ASVspoof.

- **Smart Chunking Strategy**: Seamlessly handles long-form audio by splitting inputs >4 seconds into overlapping windows and averaging the predictive probabilities.

- **Pre-Inference Validation**: Intelligently skips model execution for empty/silent files, rejects corrupted files, and flags non-speech audio (e.g., ambient noise or music) to save GPU resources.

- **Automated PDF reports**: Generates downloadable PDF audit reports detailing feature statistics, confidence scores, and system metadata for cybersecurity compliance.
  
- **Waveform Visualizations**: Automatically plots full waveform graphs and Mel-Frequency Cepstral Coefficient (MFCC) feature heatmaps.


- **GPU Acceleration**: PyTorch with CUDA for fast inference
  
## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch |
| UI Framework | Streamlit |
| Visualization | Matplotlib |
| PDF Generation | ReportLab |
| Audio Processing | Librosa |
| Environment | venv |

## System Architecture
The application is built on a streamlined, database-free architecture utilizing Streamlit for the frontend and PyTorch for the machine learning backend.

<img width="1757" height="471" alt="pipe" src="https://github.com/user-attachments/assets/2e58269c-7326-470d-a0d4-3e8b8f978d63" />

### Web Architecture (Streamlit)
The application utilizes a decoupled client-server architecture, specifically separating the frontend interface from the heavy processing overhead of the detection engine to ensure the UI remains responsive. Designed as a vertical, single-page web application using Streamlit, the interface prioritizes a clean, clutter-free layout for immediate user feedback.

<img width="1161" height="248" alt="web" src="https://github.com/user-attachments/assets/677a1e7e-3350-4cf9-9d3b-dcdcbd7ca392" />

Key frontend components include:

- **Ingestion:** A drag-and-drop upload zone supporting standard audio formats including WAV, MP3, FLAC, M4A, and OGG.


- **Results Dashboard:** Displays instant, color-coded binary predictions ("REAL" in green or "FAKE" in red), alongside a granular confidence percentage and probability score (0 to 1).


- **Visual Forensics:** Dynamically renders a time-domain waveform plot (amplitude variations) and a 2D MFCC heatmap (frequency features across time) to visually explain the underlying acoustic structures driving the model's decision.


- **Metadata & Export:** Automatically extracts and displays intrinsic payload characteristics (file name, duration, sample rate, size) and provides a one-click PDF audit report download.

### Model Architecture (PyTorch CNN)
The core deep learning model treats audio feature extraction as an image classification problem. It processes 3-channel feature maps (MFCC, Delta, and Delta-Delta coefficients) using a 2D Convolutional Neural Network.

<img width="2005" height="291" alt="model" src="https://github.com/user-attachments/assets/128c6328-aa73-443f-a9a8-f690d408b678" />

Network dimensions:
- Input Shape: (3, 40, 400) corresponding to (Channels, n_mfcc, max_len).
- Conv2D(3, 32) → Batch Normalization → ReLU → Max Pooling
- Conv2D(32, 64) → Batch Normalization → ReLU → Max Pooling
- Conv2D(64, 128) → Batch Normalization → ReLU → Max Pooling
- Conv2D(128, 256) → Batch Normalization → ReLU
- Global Average Pooling 2D
- Dense(256, 128) → ReLU → Dropout(0.4)
- Dense(128, 1) → Sigmoid

## Flow
The pipeline follows a strict sequence from file ingestion to forensic output, heavily gated by an upfront validation layer.

1. **Upload & Pre-Check Validation:** The user uploads an audio file via the Streamlit interface. Before any tensor operations occur, the file enters an inline validation engine that checks for corruption, sub-minimal duration (under 1 second), and absolute silence. It also utilizes spectral centroids to flag non-speech audio (like ambient noise or music), preventing skewed predictions.


2. **Preprocessing & Feature Extraction:** Validated files are standardized to a clean 16,000 Hz mono signal using Librosa. The engine extracts 40 Mel-Frequency Cepstral Coefficients (MFCCs), plus their temporal derivatives (delta and delta-delta), forming a 3-channel feature map.


3. **Chunk Processing:** To handle files exceeding the model's fixed 4-second input window, the pipeline splits long-form audio into smaller, overlapping temporal frames.


4. **Inference:** The feature maps are passed to the PyTorch CNN. A hardware handler dynamically delegates tensor calculations to a CUDA-enabled GPU if available, falling back to the CPU if necessary.


5. **Aggregation & Output:** For chunked files, the system averages the resulting probability scores to form a final classification. The ultimate prediction, confidence metrics, and generated visualizations are pushed to the Streamlit UI, while the ReportLab module compiles all technical metadata into a structured, downloadable PDF report.

## Quick Start

### 1. Create Virtual Environment

```bash
cd /root/projects/Mini-Project
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch
pip install librosa matplotlib reportlab scikit-learn streamlit
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load dataset from `dataset/{train,val,test}/{real,fake}/`
- Train the CNN model (20 epochs, batch_size=8)
- Save `model_files/model.pth`
- Save metrics to `trained.txt`

### 4. Run the App

**Option A: Streamlit (Recommended - Faster, Better UI)**

```bash
streamlit run app_streamlit.py
```

Open http://localhost:8501 in your browser.

**Option B: Gradio (Legacy)**

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
audio-deepfake-detector/
├── model/
│   └── deepfake_detector.py    # DeepfakeDetector class (PyTorch)
├── app.py                       # Gradio UI (legacy)
├── app_streamlit.py             # Streamlit UI (recommended)
├── train.py                     # Training script
├── temp/                        # Temporary PDFs (auto-cleaned)
├── model_files/                 # Saved models
│   └── model.pth
├── dataset/                     # Training data
│   ├── train/real/, fake/
│   ├── val/real/, fake/
│   └── test/real/, fake/
├── requirements.txt
├── README.md
├── PRD.md
└── plans/
    └── audio-deepfake-detection.md
```

## Dataset Structure

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

## Pre-Checks (Before Model Inference)

The system performs these checks before running the model to save GPU resources:

1. **Empty/Silent** → Returns "Unknown" if energy < 0.01
2. **Duration** → Returns error if audio < 1 second
3. **Corrupted** → Returns error if file can't be loaded
4. **Non-Speech** → Warns if audio may not contain speech (music/ambient)

## Model Architecture (PyTorch)

- **Input**: (3, 40, 400) - MFCC + Delta + Delta-Delta
- **CNN**: 4 Conv2D layers (32→64→128→256) with BatchNorm and MaxPool
- **Pooling**: GlobalAveragePooling2D
- **Dense**: 128 units with Dropout(0.4)
- **Output**: Sigmoid (probability 0-1)
- **Total Parameters**: ~423K

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| sample_rate | 16000 Hz |
| n_mfcc | 40 |
| n_fft | 512 |
| hop_length | 160 |
| max_len | 400 frames |
| batch_size | 8 |
| epochs | 20 |
| learning_rate | 0.0003 |

## PDF Report Contents

The generated PDF includes:
- Report ID and timestamp
- File information (filename, size, duration, sample rate, channels, format)
- Prediction summary (Real/Fake with color indicator, confidence %, probability score)
- Waveform visualization
- MFCC heatmap
- Model information (architecture, dataset, version)
- Technical explanation (MFCCs, how detection works)
- Disclaimer

## Error Handling

| Case | Behavior |
|------|----------|
| Empty/Silent audio | Returns "Unknown" (skips model) |
| Audio <1s | Returns error "Too short" |
| Corrupted file | Returns error message |
| Non-speech audio | Runs model + adds warning |
| Model not found | Shows "Model not loaded" |

## GPU Support

PyTorch automatically detects NVIDIA GPUs in WSL2. The app will use GPU for inference if available.

Training with GPU:
- Uses CUDA for fast training
- Model runs on RTX 3050 (4GB VRAM) with batch_size=8

## Training Results

After training, metrics are saved to `trained.txt`:

- Accuracy: ~89.68%
- Precision: ~99.26%
- Recall: ~80.04%
- F1 Score: ~88.62%

