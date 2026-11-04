"""Sanity-check the trained model on known EGFR drugs and unrelated controls.

Looks each drug up in ChEMBL by name (so no SMILES are typed by hand),
builds the same Morgan fingerprint used in training, and asks the model
how likely each one is to be EGFR-active.

Also reports whether the molecule was already in the training data, which
decides whether a correct answer is a real prediction or just recall.

Reads   models/egfr_random_forest.joblib
        data/egfr_labelled.csv, data/egfr_split.csv
Writes  data/drug_predictions.csv

Run from the project folder with:
    uv run python scripts/predict_drugs.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "egfr_random_forest.joblib"
LABELLED = ROOT / "data" / "egfr_labelled.csv"
SPLIT = ROOT / "data" / "egfr_split.csv"
OUTPUT = ROOT / "data" / "drug_predictions.csv"

RADIUS, N_BITS = 2, 2048  # must match scripts/make_fingerprints.py

# Expected answer, and the names to try in ChEMBL (some differ by region).
DRUGS = [
    ("Gefitinib", "EGFR inhibitor", ["GEFITINIB"]),
    ("Erlotinib", "EGFR inhibitor", ["ERLOTINIB"]),
    ("Afatinib", "EGFR inhibitor", ["AFATINIB"]),
    ("Osimertinib", "EGFR inhibitor", ["OSIMERTINIB"]),
    ("Lapatinib", "EGFR inhibitor", ["LAPATINIB"]),
    ("Aspirin", "control", ["ASPIRIN", "ACETYLSALICYLIC ACID"]),
    ("Paracetamol", "control", ["PARACETAMOL", "ACETAMINOPHEN"]),
    ("Caffeine", "control", ["CAFFEINE"]),
    ("Ibuprofen", "control", ["IBUPROFEN"]),
]

RDLogger.DisableLog("rdApp.*")


def look_up(names: list[str]) -> tuple[str, str] | None:
    """Return (chembl_id, smiles) for the first name ChEMBL recognises."""
    for name in names:
        hits = list(
            new_client.molecule.filter(pref_name__iexact=name).only(
                "molecule_chembl_id", "molecule_structures"
            )
        )
        for hit in hits:
            smiles = (hit.get("molecule_structures") or {}).get("canonical_smiles")
            if smiles:
                return hit["molecule_chembl_id"], smiles
    return None


def main() -> None:
    model = joblib.load(MODEL)
    generator = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

    labelled = pd.read_csv(LABELLED).set_index("molecule_chembl_id")
    split = pd.read_csv(SPLIT).set_index("molecule_chembl_id")["set"]

    rows = []
    for drug, expected, names in DRUGS:
        found = look_up(names)
        if found is None:
            print(f"{drug}: not found in ChEMBL, skipping")
            continue
        chembl_id, smiles = found

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{drug}: RDKit could not read its SMILES, skipping")
            continue

        fingerprint = generator.GetFingerprintAsNumPy(mol).reshape(1, -1)
        probability = model.predict_proba(fingerprint)[0, 1]
        prediction = "active" if probability >= 0.5 else "inactive"

        # Was this exact molecule part of the data the model learnt from?
        if chembl_id in labelled.index:
            known = labelled.loc[chembl_id]
            seen = split.get(chembl_id, "?")
            measured = f"{known['ic50_nM']:.1f} nM ({known['label']})"
        else:
            seen, measured = "not in dataset", "-"

        rows.append(
            {
                "drug": drug,
                "expected": expected,
                "chembl_id": chembl_id,
                "p_active": round(float(probability), 3),
                "predicted": prediction,
                "in_dataset": seen,
                "measured_ic50": measured,
                "smiles": smiles,
            }
        )

    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    print(
        table.drop(columns=["smiles"]).to_string(index=False),
        f"\n\nSaved to {OUTPUT}",
    )

    unseen = table[table["in_dataset"] == "not in dataset"]
    print(
        f"\n{len(table) - len(unseen)} of {len(table)} molecules were already in "
        f"the data; only {len(unseen)} are genuinely new to the model."
    )


if __name__ == "__main__":
    main()
