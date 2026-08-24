import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Root paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "merged_fatigue_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Could not find merged dataset at:\n   {INPUT_CSV}")
        return

    print("📄 Loading merged dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(df)} rows.")

    # 1. Separate Labels (y) and Features (X)
    # First 3 columns are: sample_id, label, timestamp
    y_raw = df["label"].values
    X_raw = df.iloc[:, 3:].values.astype(np.float32)  # 180 feature columns

    # 2. Encode text labels into numbers: alert=0, drowsy=1, yawning=2, distracted=3
    label_encoder = LabelEncoder()
    # Explicit class ordering for consistency
    classes = ["alert", "drowsy", "yawning", "distracted"]
    label_encoder.fit(classes)
    y_encoded = label_encoder.transform(y_raw)

    print("\n🏷️ Class Encodings:")
    for idx, cls_name in enumerate(label_encoder.classes_):
        print(f"   {cls_name:12s} -> {idx}")

    # 3. Train / Test Split (80% Train, 20% Test) with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded  # Ensures equal proportion of each class in train & test
    )

    # 4. Standardize Features (Mean = 0, Variance = 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Save preprocessed dataset as .npz (fast PyTorch loading)
    npz_path = os.path.join(OUTPUT_DIR, "preprocessed_fatigue_data.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )
    print(f"\n💾 Saved preprocessed data to:\n   {npz_path}")

    # 6. Save Scaler and Label Encoder for live real-time inference
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"💾 Saved scaler to:\n   {scaler_path}")
    print(f"💾 Saved label encoder to:\n   {encoder_path}")

    print("\n==============================")
    print("📊 PREPROCESSING SUMMARY")
    print("==============================")
    print(f"Train samples: {len(X_train)}  (80%)")
    print(f"Test samples : {len(X_test)}   (20%)")
    print(f"Feature shape: {X_train.shape[1]} inputs per sample")
    print("✅ Ready for PyTorch Model Training!")


if __name__ == "__main__":
    main()