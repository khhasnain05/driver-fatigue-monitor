import os
import glob
import pandas as pd

# Always point to project root / data folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "landmark_features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_FEATURE_COLS = 30 * 6  # 180
EXPECTED_TOTAL_COLS = 3 + EXPECTED_FEATURE_COLS  # sample_id, label, timestamp + 180


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "fatigue_dataset_*.csv")))
    if not files:
        print("❌ No CSV files found in data/landmark_features/")
        return

    print("📁 Found files:")
    for f in files:
        print("  -", os.path.basename(f))

    frames = []
    for f in files:
        df = pd.read_csv(f)
        print(f"\n📄 {os.path.basename(f)}")
        print(f"   rows: {len(df)} | cols: {len(df.columns)}")
        if "label" in df.columns:
            print("   labels:", df["label"].value_counts().to_dict())
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Re-number sample_id cleanly
    if "sample_id" in merged.columns:
        merged["sample_id"] = range(len(merged))

    print("\n==============================")
    print("📊 MERGED DATASET SUMMARY")
    print("==============================")
    print(f"Total samples: {len(merged)}")
    print(f"Total columns: {len(merged.columns)}")

    if "label" not in merged.columns:
        print("❌ Missing 'label' column. Something is wrong with CSV format.")
        return

    print("\nSamples per class:")
    counts = merged["label"].value_counts()
    for label, count in counts.items():
        print(f"  {label:12s}: {count}")

    # Column checks
    if len(merged.columns) != EXPECTED_TOTAL_COLS:
        print(f"\n⚠️ Column count is {len(merged.columns)}, expected {EXPECTED_TOTAL_COLS}")
        print("   (Training can still work, but tell me if this warning appears.)")
    else:
        print(f"\n✅ Column count correct ({EXPECTED_TOTAL_COLS})")

    # Null check
    nulls = merged.isna().sum().sum()
    print(f"✅ Missing values: {nulls}")

    # Save merged file
    out_path = os.path.join(OUTPUT_DIR, "merged_fatigue_dataset.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n💾 Saved merged file to:\n   {out_path}")

    # Minimum readiness check
    print("\n🎯 Readiness check (target ~50 each):")
    ready = True
    for cls in ["alert", "drowsy", "yawning", "distracted"]:
        c = int(counts.get(cls, 0))
        status = "✅" if c >= 40 else "⚠️ LOW"
        if c < 40:
            ready = False
        print(f"  {cls:12s}: {c:3d}  {status}")

    if ready:
        print("\n🚀 Dataset looks good for Phase 6 training!")
    else:
        print("\n⚠️ Some classes are low. You can still train, but accuracy may be weaker.")


if __name__ == "__main__":
    main()