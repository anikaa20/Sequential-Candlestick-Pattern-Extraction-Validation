
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# ============================================================
# 17 CANDLESTICK PATTERNS
# ============================================================

PATTERN_NAMES = [
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


# ============================================================
# MODEL CLASS
# ============================================================

class CandlePatternMLModels:

    def __init__(self):

        self.scaler = StandardScaler()
        self.feature_columns = None

        # ----------------------------
        # Random Forest
        # ----------------------------

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        # ----------------------------
        # XGBoost
        # ----------------------------

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        # ----------------------------
        # SVM
        # ----------------------------

        svm = SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

        self.models = {
            "Random Forest": MultiOutputClassifier(rf),
            "XGBoost": MultiOutputClassifier(xgb),
            "SVM": MultiOutputClassifier(svm),
        }


    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(self, df):

        self.feature_columns = [
            col
            for col in df.columns
            if col not in PATTERN_NAMES
            and col != "timestamp"
        ]

        X = df[self.feature_columns].copy()
        y = df[PATTERN_NAMES].copy()

        return X, y


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    def calculate_metrics(self, y_true, y_pred, y_prob):

        y_true_flat = y_true.values.ravel()
        y_pred_flat = y_pred.ravel()
        y_prob_flat = y_prob.ravel()

        metrics = {}

        metrics["Accuracy"] = accuracy_score(
            y_true_flat,
            y_pred_flat
        )

        metrics["Precision"] = precision_score(
            y_true_flat,
            y_pred_flat,
            zero_division=0
        )

        metrics["Recall"] = recall_score(
            y_true_flat,
            y_pred_flat,
            zero_division=0
        )

        metrics["F1"] = f1_score(
            y_true_flat,
            y_pred_flat,
            zero_division=0
        )

        # ROC-AUC can fail if a fold/test set contains
        # only one class, so handle safely.
        try:
            metrics["ROC-AUC"] = roc_auc_score(
                y_true_flat,
                y_prob_flat
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan

        # PR-AUC / Average Precision
        try:
            metrics["PR-AUC"] = average_precision_score(
                y_true_flat,
                y_prob_flat
            )
        except ValueError:
            metrics["PR-AUC"] = np.nan

        return metrics


    # ========================================================
    # TRAIN + EVALUATE
    # ========================================================

    def train(self, df, test_size=0.2):

        X, y = self.prepare_data(df)

        # ----------------------------------------------------
        # Chronological split
        # ----------------------------------------------------

        split_idx = int(len(df) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]

        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print("\nDataset information")
        print("-------------------")
        print(f"Total samples : {len(df)}")
        print(f"Training      : {len(X_train)}")
        print(f"Testing       : {len(X_test)}")
        print(f"Features      : {len(self.feature_columns)}")
        print(f"Patterns      : {len(PATTERN_NAMES)}")


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)


        summary_train = {}
        summary_test = {}
        pattern_f1 = {}


        # ----------------------------------------------------
        # Train each model
        # ----------------------------------------------------

        for name, model in self.models.items():

            print("\n" + "=" * 65)
            print(f"TRAINING {name.upper()}")
            print("=" * 65)

            # SVM requires scaled features.
            # Tree models do not.
            if name == "SVM":

                train_X = X_train_scaled
                test_X = X_test_scaled

            else:

                train_X = X_train
                test_X = X_test


            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(train_X, y_train)


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            train_pred = model.predict(train_X)
            test_pred = model.predict(test_X)


            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            train_prob_list = model.predict_proba(train_X)
            test_prob_list = model.predict_proba(test_X)

            train_prob = np.column_stack(
                [p[:, 1] for p in train_prob_list]
            )

            test_prob = np.column_stack(
                [p[:, 1] for p in test_prob_list]
            )


            # ------------------------------------------------
            # Overall metrics
            # ------------------------------------------------

            summary_train[name] = self.calculate_metrics(
                y_train,
                train_pred,
                train_prob
            )

            summary_test[name] = self.calculate_metrics(
                y_test,
                test_pred,
                test_prob
            )


            # ------------------------------------------------
            # Per-pattern F1
            # ------------------------------------------------

            pattern_f1[name] = {}

            for i, pattern in enumerate(PATTERN_NAMES):

                pattern_f1[name][pattern] = f1_score(
                    y_test.iloc[:, i],
                    test_pred[:, i],
                    zero_division=0
                )


        # ====================================================
        # SUMMARY TABLES
        # ====================================================

        train_table = pd.DataFrame(summary_train)
        test_table = pd.DataFrame(summary_test)

        print("\n\n")
        print("=" * 75)
        print("TRAINING DATA METRICS")
        print("=" * 75)

        print(train_table.round(4).to_string())

        print("\n\n")
        print("=" * 75)
        print("TEST DATA METRICS")
        print("=" * 75)

        print(test_table.round(4).to_string())


        # Save evaluation report
        train_table.insert(0, "Dataset", "Train")
        test_table.insert(0, "Dataset", "Test")

        evaluation_report = pd.concat(
        [train_table, test_table],
        ignore_index=True
        )

        evaluation_report.to_csv("evaluation_report.csv", index=False)

        print("\nEvaluation report saved to: evaluation_report.csv")


        # ====================================================
        # PER-PATTERN TEST F1
        # ====================================================

        pattern_table = pd.DataFrame(pattern_f1).T
        pattern_table = pattern_table.T

        print("\n\n")
        print("=" * 75)
        print("TEST F1 SCORE BY CANDLESTICK PATTERN")
        print("=" * 75)

        print(
            pattern_table.round(4).to_string()
        )


        return {
            "train_metrics": train_table,
            "test_metrics": test_table,
            "pattern_f1": pattern_table,
        }


    # ========================================================
    # PREDICT PATTERNS
    # ========================================================

    def predict(self, df):

        X = df[self.feature_columns].copy()

        X_scaled = self.scaler.transform(X)

        predictions = {}

        for name, model in self.models.items():

            if name == "SVM":

                pred = model.predict(X_scaled)

            else:

                pred = model.predict(X)

            predictions[name] = pd.DataFrame(
                pred,
                columns=PATTERN_NAMES
            )

        return predictions


    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    def predict_probabilities(self, df):

        X = df[self.feature_columns].copy()

        X_scaled = self.scaler.transform(X)

        probabilities = {}

        for name, model in self.models.items():

            if name == "SVM":

                X_input = X_scaled

            else:

                X_input = X


            probability_list = model.predict_proba(X_input)

            probabilities[name] = pd.DataFrame(
                {
                    PATTERN_NAMES[i]:
                    probability_list[i][:, 1]
                    for i in range(len(PATTERN_NAMES))
                }
            )

        return probabilities


    # ========================================================
    # SAVE MODELS
    # ========================================================

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(path, "wb") as f:

            pickle.dump(
                {
                    "models": self.models,
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                    "patterns": PATTERN_NAMES,
                },
                f,
            )

        print(f"\nModels saved to: {path}")


    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load(self, path):

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.models = data["models"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_pattern_models(
    dataset_path,
    model_path
):

    print(f"\nLoading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    model = CandlePatternMLModels()

    results = model.train(df)

    model.save(model_path)

    return model, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset_path = (
        "data/training/candle_pattern_dataset.csv"
    )

    model_path = (
        "artifacts/models/candle_pattern_models.pkl"
    )

    train_pattern_models(
        dataset_path,
        model_path
    )

