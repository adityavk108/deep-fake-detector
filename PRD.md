# PRD: Audio Deepfake Detection System

## Problem Statement

As a cybersecurity analyst, I need to verify whether an audio sample is human-generated or AI-generated (deepfake) to assist in auditing audio content for authenticity. Currently, there is no easy way to analyze uploaded audio files for deepfake detection without specialized ML expertise.

## Solution

A web-based audio deepfake detection system that:
1. Accepts audio files (FLAC, WAV, MP3, M4A, OGG) via drag-and-drop
2. Extracts MFCC (Mel-Frequency Cepstral Coefficients) features + temporal derivatives
3. Runs inference through a lightweight CNN trained on ASVspoof-derived dataset
4. Returns a probability score (Real vs Fake) with confidence percentage
5. Displays waveform and MFCC heatmap visualizations
6. Generates a downloadable PDF audit report

## User Stories

1. As a cybersecurity analyst, I want to upload an audio file via drag-and-drop, so that I can quickly check if it contains deepfake artifacts
2. As a cybersecurity analyst, I want to see a clear "Real" or "Fake" prediction with confidence percentage, so that I can make informed decisions
3. As a cybersecurity analyst, I want to view the audio waveform, so that I can visually inspect the audio structure
4. As a cybersecurity analyst, I want to view the MFCC heatmap, so that I can understand the spectral features analyzed
5. As a cybersecurity analyst, I want to download a PDF audit report, so that I can document my findings for compliance
6. As a cybersecurity analyst, I want the system to warn me when audio is non-speech, so that I don't misinterpret results
7. As a system administrator, I want the app to automatically clean up old PDF files, so that storage doesn't fill up

## Implementation Decisions

### Architecture
- Single-page Gradio application (no separate frontend/backend)
- In-memory processing (no database, no user data stored)
- Temporary PDF storage with 1-hour TTL auto-cleanup

### ML Pipeline
- Feature extraction: MFCC (40 coefficients) + Delta + Delta-Delta
- Model: Lightweight CNN (4 conv layers + dense)
- Inference: TensorFlow Lite for fast, lightweight execution
- Chunk + average strategy for audio >4 seconds

### Pre-Check Pipeline (Before Model Inference)
```
Input Audio
    ↓
┌─────────────────────┐
│ Empty/Silent Check  │──yes──→ Return "Unknown" (skip model)
│ (energy < 0.01)     │
└─────────────────────┘
    ↓ no
┌─────────────────────┐
│ Duration Check      │──yes──→ Return "Too short" error
│ (duration < 1s)    │
└─────────────────────┘
    ↓ no
┌─────────────────────┐
│ Corrupted File      │──yes──→ Return error message
│ (librosa.load)     │
└─────────────────────┘
    ↓ no
┌─────────────────────┐
│ Non-Speech Check   │──yes──→ Run model + add WARNING flag
│ (energy + centroid)│
└─────────────────────┘
    ↓ no
   Run Model → Return prediction
```

### Non-Speech Detection
- Method: Energy threshold + Spectral centroid check
- Thresholds: energy > 0.05 AND 1000 < spectral_centroid < 8000
- If non-speech detected: run model anyway, add warning to results

### Supported Formats
- Input: FLAC, WAV, MP3, M4A, OGG (converted internally by librosa)
- Output: PDF report

### UI Layout
- Single column layout with:
  - Audio upload component
  - Analyze button
  - Prediction text + confidence number
  - Waveform plot
  - MFCC heatmap
  - File metadata (duration, sample rate, channels, format)
  - Feature statistics (mean MFCC, energy)
  - PDF download button

### PDF Report Contents
- Header: Report ID (timestamp-based), generation timestamp
- File Information: filename, size, duration, sample rate, channels, format
- Prediction Summary: Real/Fake with color indicator, confidence %, probability score
- Waveform visualization (embedded PNG)
- MFCC heatmap (embedded PNG)
- Feature Statistics: mean MFCC, frame count, energy, spectral centroid
- Model Information: architecture, dataset size, version
- Technical Explanation: What are MFCCs, how detection works
- Disclaimer: "This analysis is AI-generated and should be used as an auxiliary tool for cybersecurity auditing."

### Dataset Structure
```
dataset/
├── train/
│   ├── real/    # .flac files
│   └── fake/    # .flac files
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| sample_rate | 16000 Hz |
| n_mfcc | 40 |
| n_fft | 512 |
| hop_length | 160 |
| max_len | 400 frames |
| batch_size | 16 |
| epochs | 20 |
| fixed_audio_duration | 4 seconds |
| learning_rate | 0.0003 |

## Testing Decisions

No tests required for this project.

## Out of Scope

- Real-time audio recording analysis (batch upload only)
- Model training via UI (command-line training only)
- User authentication/history storage
- Cloud deployment
- CNN+LSTM hybrid architecture (future enhancement)
- Data augmentation (future enhancement)
- Multi-language support

## Further Notes

- GPU: RTX 3050 (4GB VRAM), batch size 16
- Training data: ~2.5GB ASVspoof-derived dataset (FLAC format)
- Fixed audio duration: 4 seconds for training
- Sample rate: 16kHz, Mono
- MFCC: 40 coefficients, n_fft=512, hop_length=160
- Max feature length: 400 frames
- Learning rate: 0.0003 (Adam optimizer)
- Epochs: 20 with early stopping (patience=5)
