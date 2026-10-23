import os
import mlflow
import numpy as np
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from models.decision_tree import AudioDecisionTree
from models.svm import AudioSVM
from models.custom_cnn import AudioCustomCNN
from models.transfer_learning import AudioTransferLearning
from models.transformer_cnn import AudioTransformerCNN

# Pathological class is assumed to be 1
PATHOLOGICAL_CLASS = 1

def export_to_onnx(model, X_sample, path):
    """Exports model to ONNX format."""
    if hasattr(model, 'model') and hasattr(model.model, 'fit'): # Sklearn
        initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
        onx = convert_sklearn(model.model, initial_types=initial_type)
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
    elif isinstance(model.model, torch.nn.Module): # PyTorch
        dummy_input = torch.FloatTensor(X_sample[:1])
        torch.onnx.export(model.model, dummy_input, path, opset_version=11)

def run_benchmarks(X, y):
    """
    Iterates through all models, logs to MLflow, and exports the best one.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("audio_engine_benchmarking")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "DecisionTree": AudioDecisionTree(),
        "SVM": AudioSVM(),
        "CustomCNN": AudioCustomCNN(X.shape[1]),
        "TransferLearning": AudioTransferLearning(X.shape[1]),
        "TransformerCNN": AudioTransformerCNN(X.shape[1])
    }

    best_recall = -1
    best_model_name = None
    best_model_obj = None

    for name, model_obj in models.items():
        with mlflow.start_run(run_name=name):
            model_obj.train(X_train, y_train)
            preds = model_obj.predict(X_test)
            recall = recall_score(y_test, preds, pos_label=PATHOLOGICAL_CLASS)

            mlflow.log_metric("pathological_recall", recall)
            print(f"Model {name} - Pathological Recall: {recall:.4f}")

            if recall > best_recall:
                best_recall = recall
                best_model_name = name
                best_model_obj = model_obj

    print(f"Best Model: {best_model_name} with Recall: {best_recall:.4f}")

    # Export best model to ONNX
    export_path = "best_model.onnx"
    export_to_onnx(best_model_obj, X_test, export_path)
    print(f"Exported {best_model_name} to {export_path}")

if __name__ == "__main__":
    # Mock data for standalone execution
    X_mock = np.random.rand(100, 150)
    y_mock = np.random.randint(0, 2, 100)
    run_benchmarks(X_mock, y_mock)
