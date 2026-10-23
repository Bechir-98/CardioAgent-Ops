from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import librosa
from preprocessing import resample_audio, get_windows
from features import extract_features, FeatureConfig
from model_registry import registry

app = FastAPI(title="Audio Engine Service")

@app.post("/predict")
async def predict(file: UploadFile = File(...), window_size: int = 2):
    try:
        # 1. Read and Load Audio
        audio_bytes = await file.read()
        # Note: In production, use a temporary file or io.BytesIO for librosa
        import io
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # 2. Preprocessing
        resampled = resample_audio(audio, sr)

        # 3. Feature Extraction (on the first window for simplicity)
        windows = get_windows(resampled, window_size)
        first_window = next(windows)

        features = extract_features(first_window, sr=2000)

        # 4. Inference via Model Registry
        prediction = registry.predict(features)

        return {
            "prediction": int(prediction[0]),
            "status": "success",
            "window_size": f"{window_size}s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
