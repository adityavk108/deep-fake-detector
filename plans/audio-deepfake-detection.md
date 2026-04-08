# Plan: Audio Deepfake Detection System

> Source PRD: /root/projects/Mini-Project/PRD.md

## Architectural decisions

Durable decisions that apply across all phases:

- **UI Framework**: Gradio (single-page, no backend)
- **ML Framework**: TensorFlow + Keras
- **Inference**: TensorFlow Lite (for fast inference)
- **Audio Processing**: Librosa
- **PDF Generation**: ReportLab
- **Pre-checks**: Energy threshold + spectral centroid (before model inference)
- **Chunk Strategy**: 4-second chunks with 50% overlap for audio >4s
- **Model Architecture**: Lightweight CNN (4 conv layers + dense)

---

## Phase 1: Project Setup + Basic Gradio UI

**User stories**: Project infrastructure (no user story)

### What to build

Create the project structure, install dependencies, and get a running Gradio app that displays the interface (even if non-functional yet).

### Acceptance criteria

- [ ] `requirements.txt` created with all dependencies (tensorflow, librosa, numpy, matplotlib, reportlab, gradio, scikit-learn)
- [ ] Directory structure created:
  - `model/` - for DeepfakeDetector class
  - `temp/` - for PDF storage
  - `model_files/` - for saved models
  - `dataset/` - placeholder for user-provided dataset
- [ ] `app.py` created with basic Gradio interface that runs without errors
- [ ] App accessible at localhost:7860

---

## Phase 2: Audio Loading + Pre-checks

**User stories**: 1 (partial), pre-check pipeline

### What to build

DeepfakeDetector class with audio loading and pre-check pipeline that runs BEFORE model inference to save resources.

### Acceptance criteria

- [ ] `load_audio()` method: loads audio, resamples to 16kHz, normalizes, converts to mono
- [ ] Empty/Silent check: energy < 0.01 → returns "Unknown" (skips model)
- [ ] Duration check: <1s → returns "Too short" error
- [ ] Corrupted file check: librosa.load fails → returns error message
- [ ] Non-speech check: energy + spectral centroid → returns warning flag (but runs model anyway)

---

## Phase 3: Feature Extraction + Model Architecture

**User stories**: 2 (partial - needs features first)

### What to build

MFCC feature extraction pipeline and CNN model architecture definition.

### Acceptance criteria

- [ ] `extract_features()`: MFCC (40) + Delta + Delta-Delta → produces (40, T, 3) shape
- [ ] `pad_or_truncate()`: fixes features to (40, 400, 3)
- [ ] `build_model()`: CNN with 4 conv layers + dense (as specified in PRD)
- [ ] Chunk + average strategy: for audio >4s, split into 4s chunks with 50% overlap, average predictions
- [ ] `preprocess()`: full pipeline from file to features ready for model

---

## Phase 4: Model Training

**User stories**: Core functionality (needs dataset)

### What to build

Load dataset, train model, save outputs.

### Acceptance criteria

- [ ] `load_dataset()`: loads audio from dataset/{train,val,test}/{real,fake}/, precomputes MFCCs
- [ ] `train()`: trains with epochs=20, batch_size=16, early stopping (patience=5)
- [ ] `model.h5` saved to model_files/
- [ ] `convert_to_tflite()`: generates model.tflite
- [ ] `trained.txt` created with metrics (accuracy, precision, recall, F1 on test set)

---

## Phase 5: Full Inference + Visualization

**User stories**: 2, 3, 4

### What to build

Full inference pipeline integrated with Gradio - upload audio, get prediction + visualizations.

### Acceptance criteria

- [ ] `predict()`: TFLite inference returns probability (0-1)
- [ ] `analyze_file()`: full pipeline (pre-checks → features → predict → chunk average if needed)
- [ ] `generate_waveform_plot()`: Matplotlib waveform → displayed in Gradio
- [ ] `generate_mfcc_plot()`: Matplotlib MFCC heatmap → displayed in Gradio
- [ ] Upload audio → prediction displays (Real/Fake + confidence percentage)
- [ ] Waveform plot displays in UI
- [ ] MFCC heatmap displays in UI
- [ ] File metadata displays (duration, sample rate, channels, format)

---

## Phase 6: PDF Generation

**User stories**: 5

### What to build

PDF report generation with all sections required for cybersecurity auditing.

### Acceptance criteria

- [ ] PDF header: Report ID (timestamp-based), generation timestamp
- [ ] File Information section: filename, size, duration, sample rate, channels, format
- [ ] Prediction Summary: Real/Fake with color indicator (green=Real, red=Fake), confidence %, probability score
- [ ] Waveform visualization embedded (PNG from Matplotlib)
- [ ] MFCC heatmap embedded (PNG from Matplotlib)
- [ ] Feature Statistics: mean MFCC, frame count, energy, spectral centroid
- [ ] Model Information: architecture type, dataset size, model version
- [ ] Technical Explanation: What are MFCCs, how deepfake detection works
- [ ] Disclaimer: "This analysis is AI-generated and should be used as an auxiliary tool for cybersecurity auditing."
- [ ] PDF downloadable from Gradio interface
- [ ] PDF saved to temp/ folder

---

## Phase 7: PDF Cleanup + Polish

**User stories**: 7

### What to build

Automatic cleanup of old PDFs and final polish.

### Acceptance criteria

- [ ] On app startup: delete all files in temp/
- [ ] Periodic cleanup: background thread deletes PDFs older than 1 hour
- [ ] Filename includes timestamp for easy age calculation
- [ ] Graceful handling when temp/ doesn't exist

---

## Dependencies Between Phases

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
   │            │            │            │            │            │
   ▼            ▼            ▼            ▼            ▼            ▼
 Setup      Audio IO     Features    Training    Inference    PDF        Cleanup
```

- Phase 1: Must complete first (foundation)
- Phase 2: Depends on Phase 1 (needs app structure)
- Phase 3: Independent (can build model before data)
- Phase 4: Depends on Phase 3 (needs model architecture)
- Phase 5: Depends on Phase 4 (needs trained model)
- Phase 6: Depends on Phase 5 (needs analysis pipeline)
- Phase 7: Independent (can add anytime, polish)
