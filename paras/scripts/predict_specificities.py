#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import yaml
import numpy as np
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


PROPERTIES = {
    "-": [0.00, 0.00, 0.00, 1, 8.3, 0.21, 13.59, 145.2, 1.00, 1.03, 0.99, 6.03, 0.06, 0.00, 0.10],
    "A": [0.07, -1.73, 0.09, 0, 8.1, -0.06, 0.00, 90.0, 1.42, 0.83, 0.66, 6.00, 0.06, -0.25, 0.25],
    "C": [0.71, -0.97, 4.13, 0, 5.5, 1.36, 1.48, 103.3, 0.70, 1.19, 1.19, 5.05, -0.56, -0.40, -0.14],
    "D": [3.64, 1.13, 2.36, 1, 13.0, -0.80, 49.70, 117.3, 1.01, 0.54, 1.46, 2.77, 0.97, -0.08, 0.08],
    "E": [3.08, 0.39, -0.07, 1, 12.3, -0.77, 49.90, 142.2, 1.51, 0.37, 0.74, 3.22, 0.85, -0.10, -0.05],
    "F": [-4.92, 1.30, 0.45, 0, 5.2, 1.27, 0.35, 191.9, 1.13, 1.38, 0.60, 5.48, -0.99, 0.18, 0.15],
    "G": [2.23, -5.36, 0.30, 0, 9.0, -0.41, 0.00, 64.9, 0.57, 0.75, 1.56, 5.97, 0.32, -0.32, 0.28],
    "H": [2.41, 1.74, 1.11, 1, 10.4, 0.49, 51.60, 160.0, 1.00, 0.87, 0.95, 7.59, 0.15, -0.03, -0.10],
    "I": [-4.44, -1.68, -1.03, 0, 5.2, 1.31, 0.13, 163.9, 1.08, 1.60, 0.47, 6.02, -1.00, -0.03, 0.10],
    "K": [2.84, 1.41, -3.14, 2, 11.3, -1.18, 49.50, 167.3, 1.16, 0.74, 1.01, 9.74, 1.00, 0.32, 0.11],
    "L": [-4.19, -1.03, -0.98, 0, 4.9, 1.21, 0.13, 164.0, 1.21, 1.30, 0.59, 5.98, -0.83, 0.05, 0.01],
    "M": [-2.49, -0.27, -0.41, 0, 5.7, 1.27, 1.43, 167.0, 1.45, 1.05, 0.60, 5.74, -0.68, -0.01, 0.04],
    "N": [3.22, 1.45, 0.84, 2, 11.6, -0.48, 3.38, 124.7, 0.67, 0.89, 1.56, 5.41, 0.70, -0.06, 0.17],
    "P": [-1.22, 0.88, 2.23, 0, 8.0, 1.1, 1.58, 122.9, 0.57, 0.55, 1.52, 6.30, 0.45, 0.23, 0.41],
    "Q": [2.18, 0.53, -1.14, 2, 10.5, -0.73, 3.53, 149.4, 1.11, 1.10, 0.98, 5.65, 0.71, -0.02, 0.12],
    "R": [2.88, 2.52, -3.44, 4, 10.5, -0.84, 52.00, 194.0, 0.98, 0.93, 0.95, 10.76, 0.80, 0.19, -0.41],
    "S": [1.96, -1.63, 0.57, 1, 9.2, -0.50, 1.67, 95.4, 0.77, 0.75, 1.43, 5.68, 0.48, -0.15, 0.23],
    "T": [0.92, -2.09, -1.40, 1, 8.6, -0.27, 1.66, 121.5, 0.83, 1.19, 0.96, 5.66, 0.38, -0.10, 0.29],
    "V": [-2.69, -2.53, -1.29, 0, 5.9, 1.09, 0.13, 139.0, 1.06, 1.70, 0.50, 5.96, -0.75, -0.19, 0.03],
    "W": [-4.75, 3.65, 0.85, 1, 5.4, 0.88, 2.10, 228.2, 1.08, 1.37, 0.96, 5.89, -0.57, 0.31, 0.34],
    "Y": [-1.39, 2.32, 0.01, 1, 6.2, 0.33, 1.61, 197.0, 0.69, 1.47, 1.14, 5.66, -0.35, 0.40, -0.02],
}


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", type=str, required=True, help="path to signatures.tsv file")
    parser.add_argument("--matching-rules", type=str, required=True, help="path to matching rules file")
    parser.add_argument("--model", type=str, required=True, help="path to trained model file")
    parser.add_argument("--outdir", type=str, required=True, help="path to output directory")
    parser.add_argument("--batch-size", type=int, default=1000, help="number of signatures per prediction batch")
    return parser.parse_args()


def get_domain_features(amino_acid_sequence: str) -> list[float]:
    if len(amino_acid_sequence) != 34:
        raise ValueError(
            f"A-domain extended signature must be 34 amino acids long, got {len(amino_acid_sequence)}."
        )

    features = []
    for amino_acid_id in amino_acid_sequence:
        try:
            features.extend(PROPERTIES[amino_acid_id])
        except KeyError:
            raise ValueError(f"Unknown amino acid character in signature: {amino_acid_id!r}")
    return features


def iter_signature_batches(handle, batch_size: int):
    batch = []

    # skip header
    next(handle)

    for line in handle:
        identifier, signature, extended_signature = line.rstrip("\n").split("\t")
        if len(extended_signature) != 34:
            # log.warning(f"Skipping {identifier} due to invalid extended signature length: {len(extended_signature)}")
            continue
        batch.append((identifier, extended_signature))
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def main() -> None:
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    model: RandomForestClassifier = load(args.model)

    rules = yaml.safe_load(open(args.matching_rules, "r"))
    keys = []
    mapping = defaultdict(list)

    for r in rules:
        if r["props"]["origin"] == "paras":
            key = r["name"]
            paras_names = r["props"]["collapsed"]
            keys.append(key)
            for paras_name in paras_names:
                mapping[paras_name].append(key)

    class_labels = list(model.classes_)

    # Precompute which output keys each model class contributes to
    class_to_key_indices: list[list[int]] = []
    key_to_idx = {key: i for i, key in enumerate(keys)}
    for class_name in class_labels:
        idxs = [key_to_idx[key] for key in mapping.get(class_name, [])]
        class_to_key_indices.append(idxs)

    with open(args.signatures, "r") as f, open(os.path.join(args.outdir, "predictions.tsv"), "w") as out_f:
        out_f.write(f"protein_name\textended_signature\t" + "\t".join(keys) + "\n")

        for batch in tqdm(iter_signature_batches(f, args.batch_size)):
            identifiers = []
            signatures = []
            feature_rows = []

            for identifier, extended_signature in batch:
                identifiers.append(identifier)
                signatures.append(extended_signature)
                feature_rows.append(get_domain_features(extended_signature))

            X = np.asarray(feature_rows, dtype=np.float32)
            prob_matrix = model.predict_proba(X)  # shape: (batch_size, n_classes)

            for row_idx, probs in enumerate(prob_matrix):
                key_probs = np.zeros(len(keys), dtype=np.float64)

                for class_idx, prob in enumerate(probs):
                    for key_idx in class_to_key_indices[class_idx]:
                        key_probs[key_idx] += prob

                key_probs_str = "\t".join(map(str, key_probs))
                out_f.write(f"{identifiers[row_idx]}\t{signatures[row_idx]}\t{key_probs_str}\n")


if __name__ == "__main__":
    main()