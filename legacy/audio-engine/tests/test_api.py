import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint_missing_model():
    # Test behavior when best_model.onnx doesn't exist
    # Should return 500 because registry raises FileNotFoundError
    response = client.post("/predict", files={"file": ("test.wav", b"fake audio")})
    assert response.status_code == 500
    assert "Model not loaded" in response.json()["detail"]
