import onnxruntime as ort
import numpy as np
import os

class ModelRegistry:
    def __init__(self, model_path: str = "best_model.onnx"):
        self.model_path = model_path
        self.session = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file {self.model_path} not found. Inference will fail.")
            return None

        # Load the ONNX model into the runtime session
        return ort.InferenceSession(self.model_path)

    def predict(self, features: np.ndarray):
        if self.session is None:
            raise FileNotFoundError("Model not loaded. Please run benchmark.py first.")

        # Ensure features is a numpy array and has the correct shape [1, input_dim]
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get input name from session
        input_name = self.session.get_inputs()[0].name

        # Run inference
        preds = self.session.run(None, {input_name: features.astype(np.float32)})
        return preds[0]

# Singleton instance
registry = ModelRegistry()
