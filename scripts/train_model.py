"""Train a random forest to predict EGFR activity from Morgan fingerprints.

Reads   data/egfr_fingerprints.npz   (from scripts/make_fingerprints.py)
Writes  models/egfr_random_forest.joblib   the trained model
        data/egfr_split.csv                which molecules are in train / test,
                                           so later evaluation uses the same test set

Run from the project folder with:
    uv run python scripts/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
SPLIT_OUTPUT = ROOT / "data" / "egfr_split.csv"
MODEL_OUTPUT = ROOT / "models" / "egfr_random_forest.joblib"

SEED = 42          # fixed seed: same split and same model every run
TEST_FRACTION = 0.2
N_TREES = 500


def main() -> None:
    data = np.load(FINGERPRINTS, allow_pickle=True)
    X, y, ids = data["X"], data["y"], data["molecule_chembl_id"]
    print(f"Loaded {len(y)} molecules ({y.mean():.1%} active)")

    # 1. Split 80/20, keeping the same active/inactive ratio in both parts.
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(
        f"Train: {len(y_train)} molecules ({y_train.mean():.1%} active) | "
        f"Test: {len(y_test)} molecules ({y_test.mean():.1%} active)"
    )
    pd.DataFrame(
        {
            "molecule_chembl_id": np.concatenate([ids_train, ids_test]),
            "set": ["train"] * len(ids_train) + ["test"] * len(ids_test),
        }
    ).to_csv(SPLIT_OUTPUT, index=False)

    # 2. Train the random forest on the training set only.
    model = RandomForestClassifier(
        n_estimators=N_TREES, n_jobs=-1, random_state=SEED
    )
    model.fit(X_train, y_train)

    # 3. Save the trained model.
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT, compress=3)
    print(f"Saved model to {MODEL_OUTPUT}")

    # 4. Quick check: accuracy on the unseen test set vs always guessing "active".
    accuracy = accuracy_score(y_test, model.predict(X_test))
    baseline = y_test.mean()  # fraction of test molecules that are active
    print(f"Test accuracy:                  {accuracy:.1%}")
    print(f"Always guessing 'active':       {baseline:.1%}")
    print(f"Improvement over the baseline:  {accuracy - baseline:+.1%} points")


if __name__ == "__main__":
    main()
