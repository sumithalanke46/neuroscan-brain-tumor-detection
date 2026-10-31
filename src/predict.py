"""
NeuroScan-AI: Reusable Prediction Engine
Loads the trained deep neural model and provides robust single/batch
inference for clinical brain MRI scans with confidence scores and error handling.
"""

import os
import time
import io
import torch
from PIL import Image

from src.dataset import CLASS_NAMES, CLASS_DISPLAY_NAMES, get_transforms
from src.model import build_model


SEVERITY_MAPPINGS = {
    'glioma': {
        'status': 'Tumor Detected',
        'severity': 'High Severity - Infiltrative glial tissue malignancy requiring urgent neuro-oncology workup.',
        'color': '#d62728'
    },
    'meningioma': {
        'status': 'Tumor Detected',
        'severity': 'Moderate-to-High Severity - Primary meningeal tumor requiring neurosurgical review.',
        'color': '#ff7f0e'
    },
    'pituitary': {
        'status': 'Tumor Detected',
        'severity': 'Moderate Severity - Sellar/pituitary mass requiring endocrine and neurosurgical assessment.',
        'color': '#9467bd'
    },
    'no_tumor': {
        'status': 'Normal Scan',
        'severity': 'Normal - No prominent intracranial neoplastic mass lesion identified in current view.',
        'color': '#2ca02c'
    }
}


class NeuroScanPredictor:
    """Predictor class encapsulating model loading, validation, and inference."""
    def __init__(self, model_path: str = "models/neuroscan_best_model.pt", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get("model_type", "transfer")
        self.model = build_model(model_type=model_type, num_classes=4, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        _, self.transform = get_transforms()
        print(f"NeuroScanPredictor successfully initialized on {self.device} with model: {model_path}")

    def predict(self, input_image) -> dict:
        """
        Executes prediction on an input image.
        Args:
            input_image: File path (str), PIL.Image, or raw bytes.
        Returns:
            Dictionary with prediction results, probabilities, and clinical annotations.
        """
        # Validate and open image
        try:
            if isinstance(input_image, str):
                if not os.path.exists(input_image):
                    return {"success": False, "error": f"Image file not found: {input_image}"}
                image = Image.open(input_image).convert('RGB')
            elif isinstance(input_image, bytes):
                image = Image.open(io.BytesIO(input_image)).convert('RGB')
            elif isinstance(input_image, Image.Image):
                image = input_image.convert('RGB')
            else:
                return {"success": False, "error": f"Unsupported input type: {type(input_image)}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to decode or parse image: {str(e)}"}

        # Preprocessing & Inference
        start_t = time.time()
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        latency_ms = (time.time() - start_t) * 1000.0
        conf, pred_idx = torch.max(probs, dim=0)

        pred_class = CLASS_NAMES[pred_idx.item()]
        display_name = CLASS_DISPLAY_NAMES[pred_class]
        confidence_pct = round(conf.item() * 100.0, 2)
        sev_info = SEVERITY_MAPPINGS[pred_class]

        class_probabilities = {}
        for idx, cname in enumerate(CLASS_NAMES):
            class_probabilities[CLASS_DISPLAY_NAMES[cname]] = round(probs[idx].item() * 100.0, 2)

        return {
            "success": True,
            "predicted_class": pred_class,
            "display_name": display_name,
            "confidence_score": confidence_pct,
            "is_tumor_detected": (pred_class != "no_tumor"),
            "diagnostic_status": sev_info['status'],
            "clinical_notes": sev_info['severity'],
            "badge_color": sev_info['color'],
            "class_probabilities": class_probabilities,
            "inference_time_ms": round(latency_ms, 2)
        }


if __name__ == "__main__":
    # Test on a sample test image
    test_dir = os.path.join("data", "test", "glioma")
    if os.path.exists(test_dir):
        samples = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")]
        if samples and os.path.exists("models/neuroscan_best_model.pt"):
            predictor = NeuroScanPredictor()
            res = predictor.predict(samples[0])
            print("Sample Inference Result:")
            print(res)
        else:
            print("Model weights not yet trained. Run training first.")
