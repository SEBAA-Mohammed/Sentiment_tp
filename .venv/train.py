"""
train.py - Steps 1 to 4: preprocessing + TF-IDF, split, training/saving, evaluation.

Dataset: Sp1786/multiclass-sentiment-analysis-dataset (Hugging Face)
  - already split into train (~31k), validation (~5k) and test (~5k)
  - columns used: "text" and "sentiment" (negative / neutral / positive)

Usage:
    python train.py
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import clean_text

DATASET_NAME = "Sp1786/multiclass-sentiment-analysis-dataset"
MODELS_DIR = Path("models")
LABELS = ["negative", "neutral", "positive"]


def load_splits() -> dict:
    """Download (first run only, then cached) the dataset from Hugging Face
    and return {'train': df, 'validation': df, 'test': df}."""
    from datasets import load_dataset  # imported here so the rest works without it

    ds = load_dataset(DATASET_NAME)
    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()[["text", "sentiment"]].dropna()
        df.columns = ["text", "label"]
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        unknown = set(df["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"Unexpected labels in '{split}': {unknown}")
        frames[split] = df
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None,
                        help="use only N training rows (quick test run)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # STEP 2 - Train / validation / test split
    # The dataset already comes split, so we use its official splits:
    #   train      -> fit the models
    #   validation -> compare / select models
    #   test       -> final, untouched evaluation
    # ------------------------------------------------------------------
    frames = load_splits()
    if args.max_train:
        frames["train"] = frames["train"].sample(args.max_train, random_state=42)

    # ------------------------------------------------------------------
    # STEP 1a - Text processing (cleaning)
    # ------------------------------------------------------------------
    for name, df in frames.items():
        df["clean"] = df["text"].apply(clean_text)
        frames[name] = df[df["clean"].str.len() > 0]   # drop rows left empty
        print(f"{name:<11}: {len(frames[name])} rows")
    print("\nTrain class distribution:\n", frames["train"]["label"].value_counts(), "\n")

    train, val, test = frames["train"], frames["validation"], frames["test"]

    # ------------------------------------------------------------------
    # STEP 1b - TF-IDF feature extraction
    # Fitted on the TRAIN set only (no information leaks from val/test).
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),                 # words + word pairs ("not good")
        token_pattern=r"(?u)\b\w[\w']*\b",  # keeps "don't", "i'm" and 1-letter words
        min_df=2,                           # ignore very rare terms
        max_features=50000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["clean"])
    X_val = vectorizer.transform(val["clean"])
    X_test = vectorizer.transform(test["clean"])
    y_train, y_val, y_test = train["label"], val["label"], test["label"]
    print(f"TF-IDF matrix: {X_train.shape[0]} docs x {X_train.shape[1]} features\n")

    # ------------------------------------------------------------------
    # STEP 3 - Train the 3 models and save them
    # ------------------------------------------------------------------
    models = {
        "svm": LinearSVC(C=0.5, class_weight="balanced", random_state=42),
        "naive_bayes": MultinomialNB(alpha=0.5),
        # min_samples_leaf=3 keeps the forest SMALL (important for free hosting:
        # without it the saved file can be hundreds of MB).
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=42),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib", compress=3)

    results = {}
    for name, model in models.items():
        print(f"=== Training {name} ===")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)

        # --------------------------------------------------------------
        # STEP 4 - Evaluation (validation set, then the final test set)
        # --------------------------------------------------------------
        val_acc = accuracy_score(y_val, model.predict(X_val))
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {"val_accuracy": round(val_acc, 4),
                         "test_accuracy": round(test_acc, 4),
                         "test_macro_f1": round(test_f1, 4),
                         "train_seconds": round(train_time, 2)}
        print(f"Val acc: {val_acc:.4f} | Test acc: {test_acc:.4f} | "
              f"Test macro-F1: {test_f1:.4f} | Time: {train_time:.1f}s")
        print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
        print("Confusion matrix on test (rows=true, cols=pred; order: neg, neu, pos):")
        print(confusion_matrix(y_test, y_pred, labels=LABELS), "\n")

    # Comparison table
    print("=" * 62)
    print(f"{'Model':<15}{'Val acc':>10}{'Test acc':>10}{'Macro-F1':>10}{'Time(s)':>10}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test_accuracy"]):
        print(f"{name:<15}{r['val_accuracy']:>10.4f}{r['test_accuracy']:>10.4f}"
              f"{r['test_macro_f1']:>10.4f}{r['train_seconds']:>10.1f}")
    best = max(results, key=lambda k: results[k]["val_accuracy"])
    print(f"\nBest model (chosen on the validation set): {best}")
    print(f"Its final test accuracy: {results[best]['test_accuracy']:.4f}")

    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # File sizes + version hint (needed for deployment)
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        print(f"{f.name:<24}{f.stat().st_size / 1e6:8.1f} MB")
    import sklearn
    print(f"\nscikit-learn version used: {sklearn.__version__}  "
          f"(pin it in requirements.txt as scikit-learn=={sklearn.__version__})")

    # Optional bar chart for your report
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(results)
        x = range(len(names))
        plt.figure(figsize=(7, 4))
        plt.bar([i - 0.2 for i in x], [results[n]["val_accuracy"] for n in names],
                width=0.4, label="Validation", color="#a5b4fc")
        plt.bar([i + 0.2 for i in x], [results[n]["test_accuracy"] for n in names],
                width=0.4, label="Test", color="#4f46e5")
        plt.xticks(list(x), names)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Model comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "comparison.png", dpi=150)
        print("Chart saved to models/comparison.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
