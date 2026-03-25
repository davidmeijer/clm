#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
from rdkit import Chem


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-standard", required=True, help="Path to gold standard dataset results file from PRISM 4.")
    parser.add_argument("--output", required=True, help="Path to output filtered dataset file.")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel NPClassifier requests.")
    return parser.parse_args()


def smiles_to_mol(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smi}")
    return mol


def canonicalize_smiles(smi: str) -> str | None:
    smi = (smi or "").strip()
    if not smi:
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


NP_API_BASES = [
    "https://npclassifier.ucsd.edu/classify",
    "https://npclassifier.gnps2.org/classify",
]

NP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BioNexus/0.2 (NPClassifier bulk annotate)",
}


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def parse_npclassifier_rows(blob: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """
    Convert raw NPClassifier JSON into rows like:
    [("NPClassifier", "class", "flavonoids"), ...]
    """
    out: list[tuple[str, str, str]] = []
    if not blob or not isinstance(blob, dict):
        return out

    isgly = blob.get("isglycoside", blob.get("isgly", blob.get("is_glycoside")))
    if isgly is not None:
        out.append(("NPClassifier", "is_glycoside", str(bool(isgly)).lower()))

    for key_json, key_out in [
        ("class_results", "class"),
        ("superclass_results", "superclass"),
        ("pathway_results", "pathway"),
    ]:
        for v in (blob.get(key_json) or []):
            out.append(("NPClassifier", key_out, _safe_str(v)))

    return out


def npclassifier_rows_for_smiles(
    smiles: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> list[tuple[str, str, str]]:
    smiles = (smiles or "").strip()
    if not smiles:
        return []

    for base in NP_API_BASES:
        try:
            resp = session.get(
                base,
                params={"smiles": smiles},
                headers=NP_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            blob = resp.json()
            return parse_npclassifier_rows(blob)
        except Exception:
            continue

    return []


def fetch_npclassifier_for_one_smiles(smiles: str) -> tuple[str, list[tuple[str, str, str]]]:
    with requests.Session() as session:
        rows = npclassifier_rows_for_smiles(smiles, session=session)
    return smiles, rows


def summarize_npclassifier_rows(rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    classes = []
    superclasses = []
    pathways = []
    is_glycoside = ""

    for scheme, key, value in rows:
        if scheme != "NPClassifier":
            continue
        if key == "class":
            classes.append(value)
        elif key == "superclass":
            superclasses.append(value)
        elif key == "pathway":
            pathways.append(value)
        elif key == "is_glycoside":
            is_glycoside = value

    classes = sorted(set(classes))
    superclasses = sorted(set(superclasses))
    pathways = sorted(set(pathways))

    return {
        "NPClassifierClass": "; ".join(classes),
        "NPClassifierSuperclass": "; ".join(superclasses),
        "NPClassifierPathway": "; ".join(pathways),
        "NPClassifierIsGlycoside": is_glycoside,
        "IsLipopeptide": "Lipopeptides" in classes,
    }


def main() -> None:
    args = cli()

    df = pd.read_excel(args.gold_standard)

    # Keep only rows with both Cluster and True SMILES
    records_df = (
        df.dropna(subset=["Cluster", "True SMILES"])
        .loc[:, ["Cluster", "True SMILES"]]
        .drop_duplicates(subset=["Cluster"], keep="first")
        .copy()
    )

    print(f"Found {len(records_df)} clusters with true SMILES.")

    # Canonicalize once and drop invalid SMILES
    records_df["CanonicalSMILES"] = records_df["True SMILES"].map(canonicalize_smiles)

    invalid_df = records_df[records_df["CanonicalSMILES"].isna()].copy()
    if len(invalid_df) > 0:
        print(f"Skipping {len(invalid_df)} clusters with invalid SMILES.")

    records_df = records_df.dropna(subset=["CanonicalSMILES"]).copy()

    # Only query unique canonical SMILES once
    unique_smiles = sorted(records_df["CanonicalSMILES"].unique())
    print(f"Querying NPClassifier for {len(unique_smiles)} unique SMILES.")

    smiles_to_rows: dict[str, list[tuple[str, str, str]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_npclassifier_for_one_smiles, smi): smi
            for smi in unique_smiles
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="NPClassifier"):
            smi, rows = future.result()
            smiles_to_rows[smi] = rows

    # Attach NPClassifier results back to all cluster rows
    records_df["NPClassifierRows"] = records_df["CanonicalSMILES"].map(smiles_to_rows)
    summary_df = records_df["NPClassifierRows"].map(summarize_npclassifier_rows).apply(pd.Series)

    out_df = pd.concat(
        [
            records_df.rename(columns={"True SMILES": "SMILES"}),
            summary_df,
        ],
        axis=1,
    )

    # Keep only relevant columns
    out_df = out_df[
        [
            "Cluster",
            "SMILES",
            "CanonicalSMILES",
            "IsLipopeptide",
            "NPClassifierClass",
            "NPClassifierSuperclass",
            "NPClassifierPathway",
            "NPClassifierIsGlycoside",
        ]
    ]

    out_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()