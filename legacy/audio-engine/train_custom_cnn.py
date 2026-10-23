import os
import numpy as np
import torch
import librosa
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

# Import from the models directory
from models.custom_cnn import AudioCustomCNN
from preprocessing import resample_audio
from features import extract_features

# Pathological class is assumed to be 1
PATHOLOGICAL_CLASS = 1

def train_only_custom_cnn(X, y):
    """
    Dedicated script to train only the CustomCNN model and evaluate its performance.
    """
    print("--- Starting CustomCNN Dedicated Training ---")

    # 1. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples")

    # 2. Initialize the CustomCNN wrapper
    input_dim = X.shape[1]
    print(f"Initializing model with input_dim: {input_dim}")
    model_wrapper = AudioCustomCNN(input_dim=input_dim)

    # 3. Execute training
    # This calls the internal loop in custom_cnn.py
    model_wrapper.train(X_train, y_train)

    # 4. Evaluation
    print("\n--- Evaluating Model Performance ---")
 
    preds = model_wrapper.predict(X_test)

    recall = recall_score(y_test, preds, pos_label=PATHOLOGICAL_CLASS)
    print(f"Pathological Recall: {recall:.4f}")

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

                # 3. Extract features (the "diagramme")
                # Using mode='mfcc' to only use MFCC features for CNN training
                features = extract_features(audio_resampled, sr=2000, mode='mfcc')

                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # --- REAL DATA LOADING ---
    # Using absolute path to avoid issues with current working directory
    dataset_path = 'C:/Users/adem/Desktop/3Ginfo/nouvelle architecture/CardioAgent Ops/ourdataset/ourdataset'

    print("Loading real dataset...")
    try:
        X, y = load_real_dataset(dataset_path)

        if len(X) == 0:
            print("Error: No data was loaded. Please check the dataset path.")
        else:
            trained_model = train_only_custom_cnn(X, y)
            print("\nTraining and evaluation successful.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
