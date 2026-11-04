"""Predict EGFR activity for any molecule, from the command line.

Examples
--------
Predict one or more SMILES directly:

    uv run python scripts/predict_drugs.py "CC(=O)Oc1ccccc1C(=O)O"
    uv run python scripts/predict_drugs.py "CCO" "c1ccccc1" --output out.csv

Predict a whole CSV (needs a column of SMILES, named "smiles" by default):

    uv run python scripts/predict_drugs.py --csv my_compounds.csv
    uv run python scripts/predict_drugs.py --csv lib.csv --smiles-column structure \\
        --name-column compound_id

With no arguments, re-runs the built-in sanity check on five approved EGFR
drugs and four unrelated controls, looked up in ChEMBL by name:

    uv run python scripts/predict_drugs.py

Every prediction also reports the Tanimoto similarity to the most similar
molecule the model was trained on. A low value means the molecule is unlike
anything the model has seen, so its score should not be trusted: the model
only learnt the chemistry that was in the training set.

Reads   models/egfr_random_forest.joblib
        data/egfr_fingerprints.npz, data/egfr_split.csv, data/egfr_labelled.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "egfr_random_forest.joblib"
FINGERPRINTS = ROOT / "data" / "egfr_fingerprints.npz"
SPLIT = ROOT / "data" / "egfr_split.csv"
LABELLED = ROOT / "data" / "egfr_labelled.csv"
DRUG_PANEL_OUTPUT = ROOT / "data" / "drug_predictions.csv"

RADIUS, N_BITS = 2, 2048  # must match scripts/make_fingerprints.py

# Tanimoto similarity to the nearest training molecule. Rough QSAR convention:
# above ~0.7 is effectively the same chemical series, below ~0.4 is unfamiliar.
SIMILAR = 0.7
UNFAMILIAR = 0.4

# Drugs for the default sanity check: name, expectation, ChEMBL synonyms to try.
DRUG_PANEL = [
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
GENERATOR = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def load_training_set() -> tuple[list, np.ndarray]:
    """Fingerprints and IDs of the molecules the main model was trained on."""
    data = np.load(FINGERPRINTS, allow_pickle=True)
    ids = pd.Series(data["molecule_chembl_id"])
    split = pd.read_csv(SPLIT).set_index("molecule_chembl_id")["set"]
    is_train = ids.map(split).eq("train").to_numpy()

    fingerprints = [
        GENERATOR.GetFingerprint(Chem.MolFromSmiles(s))
        for s in data["smiles"][is_train]
    ]
    return fingerprints, ids[is_train].to_numpy()


def warning_for(similarity: float) -> str:
    """Plain-language reliability note for a nearest-neighbour similarity."""
    if similarity >= 0.99:
        return "IN TRAINING DATA - score reflects memorisation, not prediction"
    if similarity >= SIMILAR:
        return "close analogue in training data - reliable"
    if similarity >= UNFAMILIAR:
        return "related to training data - moderate confidence"
    return "UNLIKE ANYTHING IN TRAINING DATA - do not trust this score"


def collect_inputs(args: argparse.Namespace) -> pd.DataFrame:
    """Build a table of molecules to predict, from SMILES, a CSV, or ChEMBL."""
    if args.smiles:
        return pd.DataFrame({"name": args.smiles, "smiles": args.smiles})

    if args.csv:
        table = pd.read_csv(args.csv)
        column = args.smiles_column
        if column is None:
            matches = [c for c in table.columns if c.strip().lower() == "smiles"]
            if not matches:
                sys.exit(
                    f"No 'smiles' column in {args.csv}. Columns are: "
                    f"{', '.join(table.columns)}. Use --smiles-column to pick one."
                )
            column = matches[0]
        elif column not in table.columns:
            sys.exit(f"Column '{column}' not in {args.csv}.")

        names = table[args.name_column] if args.name_column else table[column]
        result = pd.DataFrame({"name": names, "smiles": table[column]})
        return result.dropna(subset=["smiles"]).reset_index(drop=True)

    # No input given: look the built-in drug panel up in ChEMBL.
    from chembl_webresource_client.new_client import new_client

    rows = []
    for drug, expected, synonyms in DRUG_PANEL:
        for synonym in synonyms:
            hits = list(
                new_client.molecule.filter(pref_name__iexact=synonym).only(
                    "molecule_chembl_id", "molecule_structures"
                )
            )
            found = next(
                (
                    (h["molecule_chembl_id"], h["molecule_structures"]["canonical_smiles"])
                    for h in hits
                    if (h.get("molecule_structures") or {}).get("canonical_smiles")
                ),
                None,
            )
            if found:
                rows.append(
                    {"name": drug, "expected": expected,
                     "chembl_id": found[0], "smiles": found[1]}
                )
                break
        else:
            print(f"{drug}: not found in ChEMBL, skipping", file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict EGFR activity for molecules given as SMILES.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n")[1],
    )
    parser.add_argument("smiles", nargs="*", help="one or more SMILES strings")
    parser.add_argument("--csv", type=Path, help="CSV file containing SMILES")
    parser.add_argument("--smiles-column", help="name of the SMILES column")
    parser.add_argument("--name-column", help="optional column of compound names")
    parser.add_argument("--output", type=Path, help="write results to this CSV")
    parser.add_argument(
        "--warn-below",
        type=float,
        default=UNFAMILIAR,
        help=f"similarity below which to warn (default {UNFAMILIAR})",
    )
    args = parser.parse_args()

    if args.smiles and args.csv:
        sys.exit("Give SMILES arguments or --csv, not both.")

    molecules = collect_inputs(args)
    if molecules.empty:
        sys.exit("Nothing to predict.")

    # Parse structures, keeping track of anything RDKit cannot read.
    molecules["mol"] = molecules["smiles"].map(Chem.MolFromSmiles)
    unreadable = molecules[molecules["mol"].isna()]
    for _, row in unreadable.iterrows():
        print(f"Could not read SMILES, skipping: {row['smiles']}", file=sys.stderr)
    molecules = molecules[molecules["mol"].notna()].reset_index(drop=True)
    if molecules.empty:
        sys.exit("No readable molecules.")

    model = joblib.load(MODEL)
    training_fingerprints, training_ids = load_training_set()
    labelled = pd.read_csv(LABELLED).set_index("molecule_chembl_id")

    X = np.array([GENERATOR.GetFingerprintAsNumPy(m) for m in molecules["mol"]])
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for (_, row), probability in zip(molecules.iterrows(), probabilities):
        similarities = DataStructs.BulkTanimotoSimilarity(
            GENERATOR.GetFingerprint(row["mol"]), training_fingerprints
        )
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        neighbour = training_ids[best]

        record = {
            "name": row["name"],
            "p_active": round(float(probability), 3),
            "predicted": "active" if probability >= 0.5 else "inactive",
            "nearest_similarity": round(similarity, 2),
            "nearest_molecule": neighbour,
            "nearest_ic50_nM": labelled.loc[neighbour, "ic50_nM"],
            "reliability": warning_for(similarity),
            "smiles": row["smiles"],
        }
        for extra in ("expected", "chembl_id"):
            if extra in molecules.columns:
                record[extra] = row[extra]
        results.append(record)

    table = pd.DataFrame(results)

    columns = ["name", "p_active", "predicted", "nearest_similarity", "reliability"]
    if "expected" in table.columns:
        columns.insert(1, "expected")
    print(table[columns].to_string(index=False))

    unfamiliar = table[table["nearest_similarity"] < args.warn_below]
    if not unfamiliar.empty:
        print(
            f"\nWarning: {len(unfamiliar)} of {len(table)} molecules are unlike "
            f"anything in the training data (similarity < {args.warn_below}).",
            file=sys.stderr,
        )
        for _, row in unfamiliar.iterrows():
            print(f"  {row['name']}: {row['nearest_similarity']}", file=sys.stderr)

    destination = args.output
    if destination is None and not args.smiles and not args.csv:
        destination = DRUG_PANEL_OUTPUT  # keep the sanity-check file up to date
    if destination is not None:
        table.to_csv(destination, index=False)
        print(f"\nSaved to {destination}")


if __name__ == "__main__":
    main()
