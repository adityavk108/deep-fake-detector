import gradio as gr
from model.deepfake_detector import DeepfakeDetector
import os
import time
import threading

detector = DeepfakeDetector(mode="inference")

MODEL_PATH = "model_files/model.pth"

if os.path.exists(MODEL_PATH):
    detector.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

def cleanup_old_pdfs(max_age_seconds=3600):
    if not os.path.exists("temp"):
        return
    current_time = time.time()
    try:
        for file in os.listdir("temp"):
            file_path = os.path.join("temp", file)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
    except Exception:
        pass

def periodic_cleanup():
    while True:
        time.sleep(3600)
        try:
            cleanup_old_pdfs()
        except Exception:
            pass

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

def analyze(audio_file):
    cleanup_old_pdfs()
    
    if audio_file is None:
        return (
            "Please upload an audio file", None, None, None,
            "### File Information\nUpload an audio file to see details",
            "### Feature Statistics\nUpload an audio file to see statistics",
            None,
            gr.Row(visible=False),
            gr.Row(visible=False),
            gr.Row(visible=False),
            gr.Markdown(visible=True),
            gr.Markdown(visible=False)
        )
    
    result = detector.analyze_file(audio_file)
    
    if "error" in result:
        error_msg = result["error"]
        status_msg = result.get("status", "error")
        if status_msg == "unknown":
            status_msg = "Unknown"
        
        return (
            status_msg, None, None, None,
            f"### File Information\n{error_msg}",
            "### Feature Statistics\nN/A",
            None,
            gr.Row(visible=False),
            gr.Row(visible=False),
            gr.Row(visible=False),
            gr.Markdown(visible=False),
            gr.Markdown(visible=True)
        )
    
    y = result["audio"]
    sr = result["sr"]
    mfcc = result["mfcc"]
    info = result["audio_info"]
    
    file_info_md = f"""### File Information
- **Filename**: {info['filename']}
- **Duration**: {info['duration']:.2f}s
- **Sample Rate**: {info['sample_rate']} Hz
- **Channels**: {info['channels']}
- **Format**: {info['format']}"""
    
    warning_md = ""
    if result.get("warning"):
        warning_md = f"\n\n> ⚠️ {result['warning']}"
    
    feature_stats = detector.get_feature_stats(y, sr, mfcc)
    stats_md = f"""### Feature Statistics
- **Energy**: {float(feature_stats['energy']):.4f}
- **Spectral Centroid**: {float(feature_stats['spectral_centroid']):.2f} Hz"""
    
    prediction = "Model not loaded"
    confidence = 0
    prob = 0
    
    if os.path.exists(MODEL_PATH):
        run_result = detector.run(audio_file)
        prediction = run_result.get("prediction", "Unknown")
        confidence = run_result.get("confidence", 0)
        prob = run_result.get("probability", 0)
        
        if run_result.get("warning"):
            warning_md += f"\n\n> ⚠️ {run_result['warning']}"
    
    waveform_fig = detector.generate_waveform_plot(y, sr)
    mfcc_fig = detector.generate_mfcc_plot(mfcc)
    
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join("temp", f"report_{timestamp}.pdf")
    
    result["prediction"] = prediction
    result["confidence"] = confidence
    result["probability"] = prob
    detector.generate_pdf(audio_file, result, pdf_path)
    
    return (
        prediction, confidence, waveform_fig, mfcc_fig,
        file_info_md + warning_md,
        stats_md,
        pdf_path,
        gr.Row(visible=True),
        gr.Row(visible=True),
        gr.Row(visible=True),
        gr.Markdown(visible=False),
        gr.Markdown(visible=False)
    )

with gr.Blocks(title="Audio Deepfake Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Audio Deepfake Detection System")
    gr.Markdown("Upload an audio file to analyze for deepfake artifacts.")
    
    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
        with gr.Column(scale=1):
            analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
    
    placeholder = gr.Markdown("### 👆 Upload an audio file and click **Analyze** to see results")
    
    with gr.Group(visible=False) as results_group:
        gr.Markdown("### 📊 Analysis Results")
        
        with gr.Row():
            with gr.Column(scale=1):
                prediction_output = gr.Label(label="Prediction")
                confidence_output = gr.Number(label="Confidence (%)", precision=1)
            
            with gr.Column(scale=2):
                waveform_plot = gr.Plot(label="Waveform")
        
        with gr.Row():
            mfcc_plot = gr.Plot(label="MFCC Heatmap")
    
    with gr.Group(visible=False) as details_group:
        with gr.Row():
            with gr.Column():
                metadata_output = gr.Markdown()
            
            with gr.Column():
                stats_output = gr.Markdown()
    
    with gr.Group(visible=False) as pdf_group:
        gr.Markdown("### 📄 Download Report")
        pdf_output = gr.File(label="Analysis Report")
    
    analyze_btn.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[
            prediction_output, confidence_output, waveform_plot, mfcc_plot,
            metadata_output, stats_output, pdf_output,
            results_group, details_group, pdf_group,
            placeholder, placeholder
        ]
    )

if __name__ == "__main__":
    cleanup_old_pdfs()
    demo.launch(server_name="0.0.0.0", server_port=7860)
