"""CardioAgent audio-engine API: canonical v2 preprocessing + ONNX inference."""
from fastapi import FastAPI, File, HTTPException, UploadFile

from features import mel_spectrogram
from model_registry import registry
from preprocess import load_wav_bytes

app = FastAPI(title="Audio Engine Service", version="2.0")


@app.get("/")
async def root():
    return {"service": "audio-engine", "pipeline": "v2/2000Hz/2s/log-mel64x128"}


@app.get("/health")
async def health():
    return {
        "status": "ok" if registry.loaded else "degraded",
        "model_loaded": registry.loaded,
        "model_path": str(registry.model_path),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio = load_wav_bytes(await file.read())
        mel = mel_spectrogram(audio)
        result = registry.predict(mel)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}") from exc

    return {
        "status": "success",
        "prediction": result["prediction"],
        "label": result["label"],
        "probability": result["probability"],
        "model": registry.model_path.name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
