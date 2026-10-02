import glob
import os

import pandas as pd

from candle_patterns.detection import detect_patterns


PATTERNS = [
    "doji",
    "hammer",
    "bullish_engulfing",
    "bearish_engulfing",
    "three_white_soldiers",
    "three_black_crows",
    "spinning_top",
    "shooting_star",
    "hanging_man",
    "piercing_line",
    "morning_star",
    "evening_star",
    "dark_cloud_cover",
    "bullish_harami",
    "bearish_harami",
    "on_neck_line",
    "in_neck_line",
]


def create_features(df):
    df = df.copy()

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]

    df["upper_wick"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    )

    df["is_green"] = (
        df["close"] >= df["open"]
    ).astype(int)

    return df


def process_file(path):
    symbol = os.path.basename(path).replace(".csv", "")

    print(f"Processing {symbol}...")

    df = pd.read_csv(path)

    # Generate rule-based labels once
    detections = detect_patterns(df)

    # timestamp -> set of patterns
    label_map = {}

    for detection in detections:
        timestamp = detection["timestamp"]
        pattern = detection["pattern"]

        label_map.setdefault(timestamp, set()).add(pattern)

    # Create rule-relevant features
    df = create_features(df)

    feature_names = [
        "body",
        "range",
        "upper_wick",
        "lower_wick",
        "is_green",
    ]

    rows = []

    # 3-candle sliding window
    for i in range(2, len(df)):

        window = df.iloc[i - 2:i + 1]

        row = {
            "symbol": symbol,
            "timestamp": df.iloc[i]["timestamp"],
        }

        # 3 candles × 5 features = 15 input features
        for position, (_, candle) in enumerate(window.iterrows()):

            prefix = f"t{position - 2}"

            for feature in feature_names:
                row[f"{prefix}_{feature}"] = candle[feature]

        # 17 multi-label targets
        timestamp = df.iloc[i]["timestamp"]
        detected = label_map.get(timestamp, set())

        for pattern in PATTERNS:
            row[f"label_{pattern}"] = int(pattern in detected)

        rows.append(row)

    result = pd.DataFrame(rows)

    print(
        f"  candles={len(df)}, "
        f"windows={len(result)}, "
        f"positive_windows={sum(result[f'label_{p}'].sum() for p in PATTERNS)}"
    )

    return result


def main():

    files = glob.glob("data/historical/*.csv")

    all_data = []

    for path in files:
        all_data.append(process_file(path))

    dataset = pd.concat(all_data, ignore_index=True)

    os.makedirs("data/training", exist_ok=True)

    output = "data/training/candlestick_training_dataset.csv"

    dataset.to_csv(output, index=False)

    print("\nDataset created.")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Saved to: {output}")

    print("\nLabel counts:")

    for pattern in PATTERNS:
        print(
            f"{pattern}: "
            f"{dataset[f'label_{pattern}'].sum()}"
        )


if __name__ == "__main__":
    main()