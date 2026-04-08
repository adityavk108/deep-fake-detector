import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.deepfake_detector import DeepfakeDetector

if __name__ == "__main__":
    import torch
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, using CPU")
    
    detector = DeepfakeDetector(mode="train")
    
    print("Starting training...")
    acc, prec, rec, f1 = detector.train_model(
        dataset_root="dataset",
        epochs=20,
        batch_size=8,
        lr=0.0003
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: model_files/model.pth")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
