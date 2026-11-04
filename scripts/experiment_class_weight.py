"""Experiment: does class_weight="balanced" fix the weak inactive recall?

The scaffold-split model catches only ~62% of inactive molecules, because
it saw far more actives during training. class_weight="balanced" makes
each inactive count proportionally more when fitting, which should trade
some active recall for inactive recall.

Judged on the scaffold-split test set, the honest one: both models use
the identical split saved by scripts/experiment_scaffold_split.py.

Both models are trained here from the same scaffold split, so this script
stands alone and does not depend on which setting the main pipeline uses.

Reads   data/egfr_fingerprints.npz, data/egfr_scaffold_split.csv
Writes  models/experiments/scaffold_balanced.joblib
        data/class_weight_comparison.csv

Run from the project folder with:
    uv run python scripts/experiment_class_weight.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
SCAFFOLD_SPLIT = ROOT / "data" / "egfr_scaffold_split.csv"
BALANCED_MODEL = ROOT / "models" / "experiments" / "scaffold_balanced.joblib"
OUTPUT = ROOT / "data" / "class_weight_comparison.csv"

SEED, N_TREES = 42, 500


def score(model, X, y) -> dict[str, float]:
    predicted = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
        "recall_active": recall_score(y, predicted, pos_label=1),
        "recall_inactive": recall_score(y, predicted, pos_label=0),
        # Mean of the two recalls: the fair summary when classes are uneven.
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
    }


def main() -> None:
    data = np.load(FINGERPRINTS, allow_pickle=True)
    X, y = data["X"], data["y"]
    ids = pd.Series(data["molecule_chembl_id"])

    split = pd.read_csv(SCAFFOLD_SPLIT).set_index("molecule_chembl_id")["set"]
    assigned = ids.map(split).to_numpy()
    train_mask, test_mask = assigned == "train", assigned == "test"
    print(
        f"Scaffold split: {train_mask.sum()} train "
        f"({y[train_mask].mean():.1%} active), {test_mask.sum()} test "
        f"({y[test_mask].mean():.1%} active)\n"
    )

    models = {}
    for name, weight in (("unweighted (before)", None), ("balanced (after)", "balanced")):
        model = RandomForestClassifier(
            n_estimators=N_TREES,
            n_jobs=-1,
            random_state=SEED,
            class_weight=weight,
        )
        model.fit(X[train_mask], y[train_mask])
        models[name] = model
    joblib.dump(models["balanced (after)"], BALANCED_MODEL, compress=3)
    print(f"Saved balanced model to {BALANCED_MODEL}\n")
    results = pd.DataFrame(
        {name: score(m, X[test_mask], y[test_mask]) for name, m in models.items()}
    ).T
    results.to_csv(OUTPUT)

    shown = results.copy()
    for column in shown.columns.drop("roc_auc"):
        shown[column] = (shown[column] * 100).round(1).astype(str) + "%"
    shown["roc_auc"] = results["roc_auc"].round(3)
    print(shown.to_string())

    print("\nConfusion matrices (rows = truth, columns = prediction):")
    for name, model in models.items():
        matrix = confusion_matrix(y[test_mask], model.predict(X[test_mask]))
        print(f"\n{name}")
        print(
            pd.DataFrame(
                matrix,
                index=["inactive", "active"],
                columns=["inactive", "active"],
            ).to_string()
        )

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
