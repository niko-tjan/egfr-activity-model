"""Experiment: can the model find the known EGFR drugs if it never saw them?

The main model was trained on data that included gefitinib, erlotinib,
afatinib, osimertinib and lapatinib, so its near-certain scores for them
prove nothing. Here we remove those five molecules from the training set,
retrain with identical settings, and score them again.

For context we also report each drug's closest structural neighbour that
remains in training: a drug with a near-twin still in the data is an easier
case than one whose chemical family was removed with it.

Reads   data/egfr_fingerprints.npz, data/egfr_split.csv
        data/drug_predictions.csv          (which molecules are the 5 drugs)
        models/egfr_random_forest.joblib   (the main model, for "before")
Writes  models/experiments/holdout_egfr_drugs.joblib
        data/holdout_experiment.csv

The main model is not touched. Run from the project folder with:
    uv run python scripts/experiment_holdout_drugs.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
SPLIT = ROOT / "data" / "egfr_split.csv"
DRUGS = ROOT / "data" / "drug_predictions.csv"
MAIN_MODEL = ROOT / "models" / "egfr_random_forest.joblib"
EXPERIMENT_MODEL = ROOT / "models" / "experiments" / "holdout_egfr_drugs.joblib"
OUTPUT = ROOT / "data" / "holdout_experiment.csv"

SEED, N_TREES = 42, 500  # identical to scripts/train_model.py
RADIUS, N_BITS = 2, 2048

RDLogger.DisableLog("rdApp.*")


def closest_neighbour(
    drug_smiles: str, training_smiles: np.ndarray
) -> float:
    """Highest Tanimoto similarity between this drug and any training molecule.

    1.0 means an identical fingerprint, 0.0 means nothing in common.
    Above ~0.7 is usually the same chemical series.
    """
    generator = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    target = generator.GetFingerprint(Chem.MolFromSmiles(drug_smiles))
    others = [
        generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in training_smiles
    ]
    return max(DataStructs.BulkTanimotoSimilarity(target, others))


def main() -> None:
    data = np.load(FINGERPRINTS, allow_pickle=True)
    X, y = data["X"], data["y"]
    ids = pd.Series(data["molecule_chembl_id"])
    smiles = data["smiles"]

    split = pd.read_csv(SPLIT).set_index("molecule_chembl_id")["set"]
    drugs = pd.read_csv(DRUGS)
    drugs = drugs[drugs["expected"] == "EGFR inhibitor"].reset_index(drop=True)
    held_out = set(drugs["chembl_id"])
    print(f"Holding out {len(held_out)} EGFR drugs: {', '.join(drugs['name'])}\n")

    assigned = ids.map(split).to_numpy()
    is_drug = ids.isin(held_out).to_numpy()
    train_mask = (assigned == "train") & ~is_drug
    test_mask = assigned == "test"  # unchanged, so accuracy stays comparable

    print(
        f"Training molecules: {train_mask.sum()} "
        f"({(assigned == 'train').sum()} before removing the drugs)"
    )

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",  # same configuration as scripts/train_model.py
    )
    model.fit(X[train_mask], y[train_mask])
    EXPERIMENT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, EXPERIMENT_MODEL, compress=3)
    print(f"Saved experiment model to {EXPERIMENT_MODEL}\n")

    main_model = joblib.load(MAIN_MODEL)

    # The retrained model should otherwise behave like the main one.
    for name, m in (("main", main_model), ("held-out", model)):
        probabilities = m.predict_proba(X[test_mask])[:, 1]
        print(
            f"Test set, {name:9} model: accuracy "
            f"{accuracy_score(y[test_mask], m.predict(X[test_mask])):.1%}, "
            f"ROC-AUC {roc_auc_score(y[test_mask], probabilities):.3f}"
        )

    rows = []
    training_smiles = smiles[train_mask]
    for _, drug in drugs.iterrows():
        row_index = ids[ids == drug["chembl_id"]].index[0]
        fingerprint = X[row_index].reshape(1, -1)
        rows.append(
            {
                "drug": drug["name"],
                "chembl_id": drug["chembl_id"],
                "p_active_before": round(
                    float(main_model.predict_proba(fingerprint)[0, 1]), 3
                ),
                "p_active_after": round(
                    float(model.predict_proba(fingerprint)[0, 1]), 3
                ),
                "closest_in_training": round(
                    closest_neighbour(smiles[row_index], training_smiles), 2
                ),
            }
        )

    table = pd.DataFrame(rows)
    table["predicted_after"] = np.where(
        table["p_active_after"] >= 0.5, "active", "inactive"
    )
    table["change"] = (
        table["p_active_after"] - table["p_active_before"]
    ).round(3)
    table.to_csv(OUTPUT, index=False)

    print("\nScores for the five drugs (all truly active):")
    print(table.to_string(index=False))
    correct = (table["predicted_after"] == "active").sum()
    print(f"\nStill called active after being held out: {correct} of {len(table)}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
