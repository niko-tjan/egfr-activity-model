"""Evaluate the trained EGFR random forest on the held-out test molecules.

Reads   data/egfr_fingerprints.npz
        data/egfr_split.csv                 (which molecules are test)
        models/egfr_random_forest.joblib
Writes  figures/confusion_matrix.png
        figures/roc_curve.png

Run from the project folder with:
    uv run python scripts/evaluate_model.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save image files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
SPLIT = ROOT / "data" / "egfr_split.csv"
MODEL = ROOT / "models" / "egfr_random_forest.joblib"
FIGURES = ROOT / "figures"

NAMES = ["inactive", "active"]  # label 0, label 1


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    """Return fingerprints and labels for the molecules marked 'test'."""
    data = np.load(FINGERPRINTS, allow_pickle=True)
    split = pd.read_csv(SPLIT).set_index("molecule_chembl_id")["set"]
    is_test = pd.Series(data["molecule_chembl_id"]).map(split).eq("test").to_numpy()
    print(f"Test set: {is_test.sum()} molecules held out from training")
    return data["X"][is_test], data["y"][is_test]


def main() -> None:
    X_test, y_test = load_test_set()
    model = joblib.load(MODEL)

    predicted = model.predict(X_test)
    # Probability the model assigns to "active"; used for ROC-AUC.
    active_probability = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, predicted):.1%}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, active_probability):.3f}\n")
    print(classification_report(y_test, predicted, target_names=NAMES, digits=3))

    matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix (rows = truth, columns = prediction):")
    print(pd.DataFrame(matrix, index=NAMES, columns=NAMES).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("EGFR activity: test-set predictions")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, active_probability, ax=ax, name="Random forest")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {FIGURES / 'confusion_matrix.png'} and {FIGURES / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
