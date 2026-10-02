import glob
import os
import pandas as pd
from candle_patterns.detection import detect_patterns

os.makedirs("data/labeled", exist_ok=True)

all_labels = []

for path in glob.glob("data/historical/*.csv"):
    symbol = os.path.basename(path).replace(".csv", "")

    print(f"Processing {symbol}...")

    df = pd.read_csv(path)
    patterns = detect_patterns(df)

    for p in patterns:
        all_labels.append({
            "symbol": symbol,
            "timestamp": p["timestamp"],
            "pattern": p["pattern"],
        })

    print(f"  {len(patterns)} detections")

labels = pd.DataFrame(all_labels)

labels.to_csv(
    "data/labeled/pattern_labels.csv",
    index=False
)

print("\nDone.")
print(f"Total detections: {len(labels)}")
print(f"Assets: {labels['symbol'].nunique()}")
print(f"Patterns: {labels['pattern'].nunique()}")
print("\nPattern counts:")
print(labels["pattern"].value_counts())