# Audio Deepfake Detection System

A web-based audio deepfake detection system that analyzes audio files to determine if they are human-generated or AI-generated.

## Features

- **Audio Upload**: Drag-and-drop audio files (FLAC, WAV, MP3, M4A, OGG)
- **Deepfake Detection**: PyTorch CNN model detects AI-generated audio
- **Visualizations**: Waveform and MFCC heatmap plots
- **PDF Reports**: Downloadable audit reports for cybersecurity compliance
- **Smart Pre-checks**: Validates audio before model inference (empty/silent, duration, corrupted, non-speech detection)
- **Chunk + Average**: Handles audio longer than 4 seconds by splitting into overlapping chunks
- **GPU Acceleration**: PyTorch with CUDA for fast inference

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch |
| Audio Processing | Librosa |
| Visualization | Matplotlib |
| PDF Generation | ReportLab |
| UI Framework | Streamlit |
| Environment | venv |

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

