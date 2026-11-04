"""Experiment: how well does the model do on genuinely new chemistry?

The main model uses a random 80/20 split, so close analogues of a test
molecule are usually sitting in the training set. This splits by
Bemis-Murcko scaffold instead: every molecule sharing a core structure
goes entirely to train or entirely to test, so the test molecules belong
to chemical series the model has never seen.

Largest scaffold groups are assigned to training first, which puts the
common, well-studied cores (quinazolines and friends) in training and
leaves rarer chemistry for the test set. That is deterministic: no seed.

Reads   data/egfr_fingerprints.npz, data/egfr_split.csv
        models/egfr_random_forest.joblib    (main model, for comparison)
Writes  models/experiments/scaffold_split.joblib
        data/egfr_scaffold_split.csv        which molecules are train / test
        data/scaffold_comparison.csv        the side-by-side table

The main model is not touched. Run from the project folder with:
    uv run python scripts/experiment_scaffold_split.py
"""

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
RANDOM_SPLIT = ROOT / "data" / "egfr_split.csv"
MAIN_MODEL = ROOT / "models" / "egfr_random_forest.joblib"
SCAFFOLD_MODEL = ROOT / "models" / "experiments" / "scaffold_split.joblib"
SPLIT_OUTPUT = ROOT / "data" / "egfr_scaffold_split.csv"
COMPARISON_OUTPUT = ROOT / "data" / "scaffold_comparison.csv"

SEED, N_TREES = 42, 500  # identical to scripts/train_model.py
TRAIN_FRACTION = 0.8

RDLogger.DisableLog("rdApp.*")


def scaffold_of(smiles: str) -> str:
    """The molecule's Bemis-Murcko scaffold: its ring system and linkers.

    Side chains are stripped, so molecules that differ only in their
    decoration share a scaffold. Fully acyclic molecules have no scaffold
    and come back as "", which we keep as one group of its own.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""


def scaffold_split(smiles: np.ndarray) -> np.ndarray:
    """Assign each molecule to 'train' or 'test', whole scaffolds at a time."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smi in enumerate(smiles):
        groups[scaffold_of(smi)].append(index)

    # Biggest series first, ties broken by scaffold text so runs are identical.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    assignment = np.empty(len(smiles), dtype=object)
    target_train = int(round(TRAIN_FRACTION * len(smiles)))
    n_train = 0
    for _, indices in ordered:
        where = "train" if n_train < target_train else "test"
        assignment[indices] = where
        if where == "train":
            n_train += len(indices)

    print(
        f"{len(groups)} distinct scaffolds across {len(smiles)} molecules; "
        f"largest group has {len(ordered[0][1])} molecules"
    )
    return assignment


def score(model, X, y) -> dict[str, float]:
    """Accuracy, ROC-AUC and per-class recall on one test set."""
    predicted = model.predict(X)
    return {
        "molecules": len(y),
        "percent_active": y.mean(),
        "accuracy": accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
        "recall_active": recall_score(y, predicted, pos_label=1),
        "recall_inactive": recall_score(y, predicted, pos_label=0),
    }


def main() -> None:
    data = np.load(FINGERPRINTS, allow_pickle=True)
    X, y = data["X"], data["y"]
    ids = pd.Series(data["molecule_chembl_id"])

    assignment = scaffold_split(data["smiles"])
    pd.DataFrame(
        {"molecule_chembl_id": ids, "set": assignment}
    ).to_csv(SPLIT_OUTPUT, index=False)

    train_mask = assignment == "train"
    test_mask = assignment == "test"
    print(
        f"Scaffold split: {train_mask.sum()} train "
        f"({train_mask.sum() / len(y):.1%}), {test_mask.sum()} test "
        f"({y[train_mask].mean():.1%} vs {y[test_mask].mean():.1%} active)\n"
    )

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",  # same configuration as scripts/train_model.py
    )
    model.fit(X[train_mask], y[train_mask])
    SCAFFOLD_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SCAFFOLD_MODEL, compress=3)
    print(f"Saved scaffold model to {SCAFFOLD_MODEL}\n")

    # Each model is judged on its own test set: that contrast is the point.
    random_split = pd.read_csv(RANDOM_SPLIT).set_index("molecule_chembl_id")["set"]
    random_test = (ids.map(random_split) == "test").to_numpy()

    results = pd.DataFrame(
        {
            "random split (main)": score(
                joblib.load(MAIN_MODEL), X[random_test], y[random_test]
            ),
            "scaffold split": score(model, X[test_mask], y[test_mask]),
        }
    ).T
    results["always_guess_active"] = results["percent_active"]
    results.to_csv(COMPARISON_OUTPUT)

    shown = results.copy()
    shown["molecules"] = shown["molecules"].astype(int)
    for column in shown.columns.drop("molecules"):
        shown[column] = (shown[column] * 100).round(1).astype(str) + "%"
    shown.loc[:, "roc_auc"] = results["roc_auc"].round(3).astype(str)
    print(shown.to_string())
    print(f"\nSaved to {COMPARISON_OUTPUT}")


if __name__ == "__main__":
    main()
