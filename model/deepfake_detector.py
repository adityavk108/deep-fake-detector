import os
import numpy as np
import librosa
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepfakeDetector(nn.Module):
    def __init__(self, model_path=None, mode="inference"):
        super().__init__()
        self.model_path = model_path
        self.mode = mode
        self.sample_rate = 16000
        self.n_mfcc = 40
        self.n_fft = 512
        self.hop_length = 160
        self.max_len = 400
        self.fixed_duration = 4.0
        
        self.energy_threshold_silent = 0.01
        self.energy_threshold_speech = 0.05
        self.spectral_centroid_min = 1000
        self.spectral_centroid_max = 8000
        self.min_duration = 1.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"DeepfakeDetector initialized in {mode} mode on {self.device}")

    def load_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            if y.size == 0:
                raise ValueError("Audio file is empty or corrupted - no audio data found")
            
            y = librosa.util.normalize(y)
            logger.info(f"Loaded audio: {os.path.basename(file_path)}, duration: {len(y)/sr:.2f}s")
            return y, sr
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise ValueError(f"Could not load audio file: {str(e)}")

    def check_empty(self, y, sr):
        energy = np.mean(librosa.feature.rms(y=y))
        if energy < self.energy_threshold_silent:
            return True, "Unknown - audio appears to be silent or empty"
        return False, None

    def check_duration(self, y, sr):
        duration = len(y) / sr
        if duration < self.min_duration:
            return True, f"Rejected - audio too short ({duration:.2f}s < {self.min_duration}s)"
        return False, None

    def check_corrupted(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return False, None
        except Exception as e:
            return True, f"Error - could not load audio: {str(e)}"

    def check_non_speech(self, y, sr):
        mean_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        is_speech = mean_energy > self.energy_threshold_speech and \
                    self.spectral_centroid_min < spectral_centroid < self.spectral_centroid_max
        
        if not is_speech:
            warning = "Warning: audio may not contain speech (music or ambient sound detected)"
            return True, warning
        return False, None

    def get_audio_info(self, file_path, y, sr):
        duration = len(y) / sr
        file_size = os.path.getsize(file_path)
        
        return {
            "filename": os.path.basename(file_path),
            "file_size": file_size,
            "duration": duration,
            "sample_rate": sr,
            "channels": 1,
            "format": os.path.splitext(file_path)[1].upper().replace('.', '')
        }

    def get_feature_stats(self, y, sr, mfcc):
        mean_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        return {
            "mean_mfcc": np.mean(mfcc),
            "frame_count": mfcc.shape[1],
            "energy": mean_energy,
            "spectral_centroid": spectral_centroid
        }

    def preprocess(self, file_path):
        is_corrupted, error_msg = self.check_corrupted(file_path)
        if is_corrupted:
            return {"error": error_msg}
        
        y, sr = self.load_audio(file_path)
        
        is_empty, empty_msg = self.check_empty(y, sr)
        if is_empty:
            return {"error": empty_msg, "status": "unknown"}
        
        is_short, short_msg = self.check_duration(y, sr)
        if is_short:
            return {"error": short_msg}
        
        is_non_speech, speech_warning = self.check_non_speech(y, sr)
        
        audio_info = self.get_audio_info(file_path, y, sr)
        
        result = {
            "audio": y,
            "sr": sr,
            "audio_info": audio_info,
            "is_non_speech": is_non_speech,
            "warning": speech_warning if is_non_speech else None
        }
        
        return result

    def analyze_file(self, file_path):
        preprocess_result = self.preprocess(file_path)
        
        if "error" in preprocess_result:
            return preprocess_result
        
        audio_info = self.get_audio_info(file_path, preprocess_result["audio"], preprocess_result["sr"])
        mfcc = librosa.feature.mfcc(y=preprocess_result["audio"], sr=preprocess_result["sr"], n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        
        return {
            "status": "ready",
            "audio": preprocess_result["audio"],
            "sr": preprocess_result["sr"],
            "audio_info": audio_info,
            "mfcc": mfcc,
            "is_non_speech": preprocess_result["is_non_speech"],
            "warning": preprocess_result.get("warning")
        }

    def extract_features(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        
        features = np.stack([mfcc, delta, delta_delta], axis=-1)
        logger.info(f"Extracted features shape: {features.shape}")
        
        return features

    def pad_or_truncate(self, features):
        num_frames = features.shape[1]
        
        if num_frames < self.max_len:
            pad_width = ((0, 0), (0, self.max_len - num_frames), (0, 0))
            features = np.pad(features, pad_width, mode='constant')
        elif num_frames > self.max_len:
            features = features[:, :self.max_len, :]
        
        return features

    def build_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
        logger.info("PyTorch model built successfully on " + str(self.device))
        return self

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

    def get_chunks(self, y, sr):
        chunk_duration = self.fixed_duration
        chunk_samples = int(chunk_duration * sr)
        hop_samples = chunk_samples // 2
        
        chunks = []
        for start in range(0, len(y), hop_samples):
            end = start + chunk_samples
            chunk = y[start:end]
            if len(chunk) < chunk_samples:
                if start == 0 and len(chunk) > 0:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    chunks.append(chunk)
                    break
                elif len(chunk) > 0:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    chunks.append(chunk)
                break
            else:
                chunks.append(chunk)
        
        if len(chunks) == 0 and len(y) > 0:
            chunk = np.pad(y, (0, chunk_samples - len(y)))
            chunks.append(chunk)
        
        return chunks

    def preprocess_audio(self, y):
        features = self.extract_features(y)
        features = self.pad_or_truncate(features)
        features = np.transpose(features, (2, 0, 1))
        features = np.expand_dims(features, axis=0)
        return torch.FloatTensor(features).to(self.device)

    def load_dataset(self, split_path):
        import glob
        
        real_files = glob.glob(os.path.join(split_path, 'real', '*.flac'))
        fake_files = glob.glob(os.path.join(split_path, 'fake', '*.flac'))
        
        logger.info(f"Loading {len(real_files)} real and {len(fake_files)} fake files from {split_path}")
        
        X, y = [], []
        
        all_files = real_files + fake_files
        labels = [0] * len(real_files) + [1] * len(fake_files)
        
        for i, (file_path, label) in enumerate(zip(all_files, labels)):
            try:
                y_audio, sr = self.load_audio(file_path)
                
                duration = len(y_audio) / sr
                if duration > self.fixed_duration:
                    y_audio = y_audio[:int(self.fixed_duration * sr)]
                elif duration < self.fixed_duration:
                    y_audio = np.pad(y_audio, (0, int(self.fixed_duration * sr) - len(y_audio)))
                
                features = self.preprocess_audio(y_audio)
                X.append(features)
                y.append(label)
                
                if (i + 1) % 500 == 0:
                    logger.info(f"Processed {i+1}/{len(all_files)} files")
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
        
        X = torch.cat(X, dim=0)  # Keep on CPU
        y = torch.tensor(y, dtype=torch.float32)
        
        logger.info(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"Class distribution: Real={torch.sum(y==0).item()}, Fake={torch.sum(y==1).item()}")
        
        return X, y

    def train_model(self, dataset_root, epochs=20, batch_size=16, lr=0.0003):
        self.build_model()
        self.to(self.device)
        
        logger.info("Loading datasets...")
        
        X_train, y_train = self.load_dataset(os.path.join(dataset_root, 'train'))
        X_val, y_val = self.load_dataset(os.path.join(dataset_root, 'val'))
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train_shuffled[i:i+batch_size].to(self.device)
                batch_y = y_train_shuffled[i:i+batch_size].to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((outputs > 0.5).float() == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            train_acc = train_correct / train_total
            
            self.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_x = X_val[i:i+batch_size].to(self.device)
                    batch_y = y_val[i:i+batch_size].to(self.device).unsqueeze(1)
                    
                    outputs = self(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_correct += ((outputs > 0.5).float() == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / (len(X_val) // batch_size + 1)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model("model_files/model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Loading best model for evaluation...")
        self.load_model("model_files/model.pth")
        
        logger.info("Loading test set...")
        X_test, y_test = self.load_dataset(os.path.join(dataset_root, 'test'))
        
        self.eval()
        test_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_x = X_test[i:i+batch_size].to(self.device)
                outputs = self(batch_x)
                preds = (outputs > 0.5).int().flatten().cpu().numpy()
                test_preds.extend(preds)
        
        y_test_np = y_test.int().numpy()
        
        acc = accuracy_score(y_test_np, test_preds)
        prec = precision_score(y_test_np, test_preds)
        rec = recall_score(y_test_np, test_preds)
        f1 = f1_score(y_test_np, test_preds)
        
        with open('trained.txt', 'w') as f:
            f.write(f"Model Training Complete\n")
            f.write(f"=======================\n\n")
            f.write(f"Test Set Metrics:\n")
            f.write(f"- Accuracy:  {acc:.4f}\n")
            f.write(f"- Precision: {prec:.4f}\n")
            f.write(f"- Recall:    {rec:.4f}\n")
            f.write(f"- F1 Score:  {f1:.4f}\n")
            f.write(f"\nDataset Info:\n")
            f.write(f"- Train samples: {len(X_train)}\n")
            f.write(f"- Val samples:   {len(X_val)}\n")
            f.write(f"- Test samples:  {len(X_test)}\n")
            f.write(f"\nTraining Params:\n")
            f.write(f"- Epochs: {epochs}\n")
            f.write(f"- Batch Size: {batch_size}\n")
            f.write(f"- Learning Rate: {lr}\n")
        
        logger.info("Metrics saved to trained.txt")
        
        return acc, prec, rec, f1

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_mfcc': self.n_mfcc,
            'max_len': self.max_len
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.build_model()
        self.load_state_dict(checkpoint['model_state_dict'])
        self.n_mfcc = checkpoint['n_mfcc']
        self.max_len = checkpoint['max_len']
        self.to(self.device)
        self.eval()
        logger.info(f"Model loaded from {path}")

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            features = features.to(self.device)
            pred = self(features).item()
        return pred

    def run(self, file_path):
        preprocess_result = self.preprocess(file_path)
        
        if "error" in preprocess_result:
            return preprocess_result
        
        y = preprocess_result["audio"]
        sr = preprocess_result["sr"]
        
        chunks = self.get_chunks(y, sr)
        
        predictions = []
        for chunk in chunks:
            features = self.preprocess_audio(chunk)
            pred = self.predict(features)
            predictions.append(pred)
        
        avg_prediction = np.mean(predictions)
        
        mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        
        return {
            "prediction": "REAL" if avg_prediction < 0.5 else "FAKE",
            "probability": float(avg_prediction),
            "confidence": float(abs(0.5 - avg_prediction) * 2 * 100),
            "mfcc": mfcc_full,
            "is_non_speech": preprocess_result["is_non_speech"],
            "warning": preprocess_result.get("warning")
        }

    def generate_waveform_plot(self, y, sr):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 3))
        times = np.arange(len(y)) / sr
        ax.plot(times, y, linewidth=0.5, color='steelblue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig

    def generate_mfcc_plot(self, mfcc):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Frame')
        ax.set_ylabel('MFCC Coefficient')
        ax.set_title('MFCC Feature Heatmap')
        plt.colorbar(img, ax=ax, label='Amplitude')
        plt.tight_layout()
        
        return fig

    def generate_pdf(self, file_path, result, save_path):
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import matplotlib.pyplot as plt
        import io
        import time
        
        audio_info = result.get("audio_info", {})
        mfcc = result.get("mfcc", np.zeros((40, 100)))
        
        waveform_buffer = io.BytesIO()
        fig = self.generate_waveform_plot(result["audio"], result["sr"])
        fig.savefig(waveform_buffer, format='png', dpi=100)
        plt.close(fig)
        waveform_buffer.seek(0)
        
        mfcc_buffer = io.BytesIO()
        fig = self.generate_mfcc_plot(mfcc)
        fig.savefig(mfcc_buffer, format='png', dpi=100)
        plt.close(fig)
        mfcc_buffer.seek(0)
        
        doc = SimpleDocTemplate(save_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.Color(0.118, 0.227, 0.373))
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, textColor=colors.Color(0.118, 0.227, 0.373))
        
        story = []
        
        report_id = f"RPT-{time.strftime('%Y%m%d-%H%M%S')}"
        story.append(Paragraph(f"Audio Deepfake Detection Report", title_style))
        story.append(Paragraph(f"Report ID: {report_id}", styles['Normal']))
        story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("File Information", heading_style))
        file_data = [
            ['Field', 'Value'],
            ['Filename', audio_info.get('filename', 'N/A')],
            ['File Size', f"{audio_info.get('file_size', 0) / 1024:.2f} KB"],
            ['Duration', f"{audio_info.get('duration', 0):.2f} s"],
            ['Sample Rate', f"{audio_info.get('sample_rate', 0)} Hz"],
            ['Channels', str(audio_info.get('channels', 1))],
            ['Format', audio_info.get('format', 'N/A')]
        ]
        file_table = Table(file_data, colWidths=[2*inch, 3*inch])
        file_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.118, 0.227, 0.373)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.9, 0.9, 0.9))
        ]))
        story.append(file_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Prediction Summary", heading_style))
        prediction = result.get("prediction", "Unknown")
        probability = result.get("probability", 0)
        confidence = result.get("confidence", 0)
        
        pred_color = colors.Color(0.153, 0.682, 0.376) if prediction == "REAL" else colors.Color(0.906, 0.298, 0.235)
        
        pred_data = [
            ['Result', prediction],
            ['Confidence', f"{confidence:.1f}%"],
            ['Probability Score', f"{probability:.3f}"]
        ]
        pred_table = Table(pred_data, colWidths=[2*inch, 3*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pred_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.9, 0.9, 0.9))
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Audio Analysis - Waveform", heading_style))
        story.append(Image(waveform_buffer, width=5*inch, height=1.5*inch))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Audio Analysis - MFCC Heatmap", heading_style))
        story.append(Image(mfcc_buffer, width=5*inch, height=2*inch))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Model Information", heading_style))
        model_data = [
            ['Field', 'Value'],
            ['Architecture', 'Lightweight CNN (PyTorch)'],
            ['Training Dataset', 'ASVspoof-derived (~2.5GB)'],
            ['Model Version', '1.0'],
            ['Input Features', 'MFCC (40) + Delta + Delta-Delta']
        ]
        model_table = Table(model_data, colWidths=[2*inch, 3*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.118, 0.227, 0.373)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.9, 0.9, 0.9))
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Technical Explanation", heading_style))
        tech_text = """
        <b>MFCC (Mel-Frequency Cepstral Coefficients)</b> are features that represent the spectral envelope of audio. 
        They are commonly used in speech recognition and audio classification tasks because they approximate how humans perceive sound.
        
        <b>How Deepfake Detection Works:</b> The system extracts MFCC features from the audio and analyzes temporal patterns. 
        AI-generated audio often shows subtle artifacts in the spectral features that differ from natural speech patterns.
        
        <b>Model Architecture:</b> Lightweight CNN with 4 convolutional layers, BatchNormalization, and GlobalAveragePooling.
        The model was trained on the ASVspoof dataset to distinguish between real and AI-generated (fake) audio samples.
        """
        story.append(Paragraph(tech_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Disclaimer", heading_style))
        disclaimer = "This analysis is AI-generated and should be used as an auxiliary tool for cybersecurity auditing. Results are probabilistic and should not be considered definitive proof of authenticity or manipulation."
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        footer = Paragraph(f"Generated by Audio Deepfake Detection System | {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(footer)
        
        doc.build(story)
        
        logger.info(f"PDF generated: {save_path}")
        return save_path
