#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from collections import Counter

from harvest.loader import prep_clm
from clm.graph_conditional import (
    normalize_condition_graph,
    read_condition_vocab,
    graph_to_condition_graph,
    ConditionGraphCollate,
)

from rdkit import Chem

# new_graph = {
#     "nodes": [
#         {"id": 0, "name": "<UNK>"},
#         {"id": 1, "name": "<UNK>"},
#         {"id": 2, "name": "<UNK>"},
#         {"id": 3, "name": "<UNK>"},
#         {"id": 4, "name": "valine"},
#         {"id": 5, "name": "serine"},
#         {"id": 6, "name": "tyrosine"},
#         {"id": 7, "name": "glycine"},
#         {"id": 8, "name": "glutamine"},
#         {"id": 9, "name": "valine"},
#         {"id": 10, "name": "serine"},
#         {"id": 11, "name": "tyrosine"},
#         {"id": 12, "name": "aspartic_acid"},
#         {"id": 13, "name": "threonine"},
#         {"id": 14, "name": "tyrosine"},
#         {"id": 15, "name": "aspartic_acid"},        
#     ],
#     "links": [
#         {"source": 0, "target": 1},
#         {"source": 1, "target": 0},
#         {"source": 1, "target": 2},
#         {"source": 2, "target": 1},
#         {"source": 2, "target": 3},
#         {"source": 3, "target": 2},
#         {"source": 3, "target": 4},
#         {"source": 4, "target": 3},
#         {"source": 4, "target": 5},
#         {"source": 5, "target": 4},
#         {"source": 5, "target": 6},
#         {"source": 6, "target": 5},
#         {"source": 6, "target": 7},
#         {"source": 7, "target": 6},
#         {"source": 7, "target": 8},
#         {"source": 8, "target": 7},
#         {"source": 8, "target": 9},
#         {"source": 9, "target": 8},
#         {"source": 9, "target": 10},
#         {"source": 10, "target": 9},
#         {"source": 10, "target": 11},
#         {"source": 11, "target": 10},
#         {"source": 11, "target": 12},
#         {"source": 12, "target": 11},
#         {"source": 12, "target": 13},
#         {"source": 13, "target": 12},
#         {"source": 13, "target": 14},
#         {"source": 14, "target": 13},
#         {"source": 14, "target": 15},
#         {"source": 15, "target": 14},
#     ],
# }

new_graph = {
    "nodes": [
        {"id": 0, "name": "<UNK>"},
        {"id": 1, "name": "<UNK>"},
        {"id": 2, "name": "<UNK>"},
        {"id": 3, "name": "<UNK>"},
        {"id": 4, "name": "2,4-diaminobutyric_acid"},
        {"id": 5, "name": "2,4-diaminobutyric_acid"},
        {"id": 6, "name": "2,3-diaminopropionic_acid"},
        {"id": 7, "name": "beta-hydroxytyrosine"},
        {"id": 8, "name": "2,4-diaminobutyric_acid"},
        {"id": 9, "name": "threonine"},
    ],
    "links": [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
        {"source": 8, "target": 9},
    ],
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to model output dir.")
    return parser.parse_args()


def main() -> None:
    args = cli()
    model_cfgs = prep_clm(model_dir=Path(args.data_dir), eval=True)
    cfg = model_cfgs[2]  # or choose the fold/model you want explicitly

    model = cfg.load_model(device=DEVICE)
    model.eval()

    condition_labels = read_condition_vocab(cfg.condition_vocab_path)
    name_to_idx = {name: idx for idx, name in enumerate(condition_labels)}

    normalized_graph = normalize_condition_graph(new_graph)
    for n in normalized_graph['nodes']:
        name = n['name']
        assert name in name_to_idx, f"Node name '{name}' not found in condition vocab!"

    graph_batch = ConditionGraphCollate()(
        [
            graph_to_condition_graph(normalized_graph, name_to_idx)
            for _ in range(N_SAMPLES)
        ]
    )

    smiles = model.sample(
        descriptors=graph_batch,
        n_sequences=N_SAMPLES,
        max_len=250,
    )

    valid_smiles = []
    ik_to_repr_smi = {}
    ik_to_smis = Counter()
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            inchikey = Chem.MolToInchiKey(mol)
            valid_smiles.append(smi)
            ik_to_repr_smi[inchikey] = smi
            ik_to_smis[inchikey] += 1
        except:
            pass

    print(f"Generated {len(smiles)} SMILES, of which {len(valid_smiles)} are valid.")
    # print top 5 most abundant unique molecules with counts and first smiles
    print("Top 5 most abundant unique molecules:")
    for ik, count in ik_to_smis.most_common(5):
        print(f"{ik}: {count} occurrences, example SMILES: {ik_to_repr_smi[ik]}")
        
    


if __name__ == "__main__":
    main()
