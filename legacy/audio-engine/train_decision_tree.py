import os
import numpy as np
import librosa
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

# Import from the models directory
from models.decision_tree import AudioDecisionTree
from preprocessing import resample_audio
from features import extract_features

# Pathological class is assumed to be 1
PATHOLOGICAL_CLASS = 1
NORMAL_CLASS = 0

def save_model_to_mlflow(pathological_recall, normal_recall, accuracy, model_name="DecisionTree"):
    mlflow.set_tracking_uri("http://localhost:5000")

    # Create or select an experiment
    mlflow.set_experiment("CardioAgent_Experiments")

    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        # Decision Tree specific parameter
        mlflow.log_param("max_depth", 10)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("pathological_recall", pathological_recall)
        mlflow.log_metric("normal_recall", normal_recall)

    print("Experiment logged successfully!")

def train_only_decision_tree(X, y):
    """
    Dedicated script to train only the DecisionTree model and evaluate its performance.
    """
    print("--- Starting DecisionTree Dedicated Training ---")

    # 1. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples")

    # 2. Initialize the DecisionTree wrapper
    print("Initializing Decision Tree model...")
    model_wrapper = AudioDecisionTree(max_depth=10)

    # 3. Execute training
    model_wrapper.train(X_train, y_train)

    # 4. Evaluation
    print("\n--- Evaluating Model Performance ---")

    preds = model_wrapper.predict(X_test)

    pathological_recall = recall_score(y_test, preds, pos_label=PATHOLOGICAL_CLASS)
    normal_recall = recall_score(y_test, preds, pos_label=NORMAL_CLASS)

    accuracy = np.mean(preds == y_test) # numpy array comparison

    save_model_to_mlflow(pathological_recall, normal_recall, accuracy, model_name="DecisionTree")

    print(f"Pathological Recall: {pathological_recall:.4f}")
    print(f"Normal Recall: {normal_recall:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return model_wrapper

def load_real_dataset(base_path):
    """
    Loads .wav files from a directory structure and extracts features.
    Expected structure: base_path/train/normal and base_path/train/abnormal
    """
    X = []
    y = []
    categories = {'normal': 0, 'abnormal': 1}

    print(f"Loading dataset from: {base_path}")

    for category, label in categories.items():
        folder_path = os.path.join(base_path, 'train', category)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Processing {len(files)} files from {category}...")

        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                # 1. Load audio
                audio, sr = librosa.load(file_path, sr=None)

                # 2. Resample to 2000Hz as per pipeline specs
                audio_resampled = resample_audio(audio, sr, target_sr=2000)

                # 3. Extract features
                # Using all features (default mode) for the Decision Tree
                features = extract_features(audio_resampled, sr=2000)

                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # --- REAL DATA LOADING ---
    dataset_path = 'C:/Users/adem/Desktop/3Ginfo/nouvelle architecture/CardioAgent Ops/ourdataset/ourdataset'

    print("Loading real dataset...")
    try:
        X, y = load_real_dataset(dataset_path)

        if len(X) == 0:
            print("Error: No data was loaded. Please check the dataset path.")
        else:
            trained_model = train_only_decision_tree(X, y)
            print("\nTraining and evaluation successful.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
