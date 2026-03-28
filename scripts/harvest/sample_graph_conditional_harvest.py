#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from collections import Counter
from tqdm import tqdm
import re

from harvest.loader import prep_clm
from harvest.chem import smiles_to_mol, mol_to_morgan_fp, tanimoto
from clm.graph_conditional import (
    normalize_condition_graph,
    read_condition_vocab,
    graph_to_condition_graph,
    ConditionGraphCollate,
)

from rdkit import Chem

new_graph = {
    "nodes": [
        {"id": 4, "name": "valine"},
        {"id": 5, "name": "serine"},
        {"id": 6, "name": "tyrosine"},
        {"id": 7, "name": "glycine"},
        {"id": 8, "name": "glutamine"},
        {"id": 9, "name": "valine"},
        {"id": 10, "name": "serine"},
        {"id": 11, "name": "tyrosine"},
        {"id": 12, "name": "aspartic_acid"},
        {"id": 13, "name": "threonine"},
        {"id": 14, "name": "tyrosine"},
        {"id": 15, "name": "aspartic_acid"},        
    ],
    "links": [
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
        {"source": 8, "target": 9},
        {"source": 9, "target": 10},
        {"source": 10, "target": 11},
        {"source": 11, "target": 12},
        {"source": 12, "target": 13},
        {"source": 13, "target": 14},
        {"source": 14, "target": 15},
    ],
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1_000


target_smi = r"CCC(C)CCCCCCCCC(CC(NC(C(NC(COC1=O)C(NC(CC2=CC=C(O)C=C2)C(NCC(NC(C(NC(C(NC(C(NC(C(NC(CC(O)=O)C(NC(C(O)C)C(NC(CC3=CC=C(O)C=C3)C(NC1CC(O)=O)=O)=O)=O)=O)CC4=CC=C(O)C=C4)=O)CO)=O)C(C)C)=O)CCC(N)=O)=O)=O)=O)=O)C(C)C)=O)O"
target_mol = smiles_to_mol(target_smi)
fp1 = mol_to_morgan_fp(target_mol, radius=2, n_bits=2048)

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

    valid_smiles = []
    ik_to_repr_smi = {}
    ik_to_smis = Counter()
    closest_match_score = 0.0
    closest_match = None

    tyrosine_pattern = Chem.MolFromSmarts("O=CC([NH])Cc1ccc(O)cc1")  # crude way to count tyrosines
    tyrosine_counts = []

    # generate in batches of 1000 to avoid OOM
    num_batches = N_SAMPLES // 1000
    for i in tqdm(range(num_batches)):


        graph_batch = ConditionGraphCollate()(
            [
                graph_to_condition_graph(normalized_graph, name_to_idx)
                for _ in range(1000)
            ]
        )

        smiles = model.sample(
            descriptors=graph_batch,
            n_sequences=1000,
            max_len=250,
        )
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smi}")
                tyrosine_count = len(mol.GetSubstructMatches(tyrosine_pattern))
                print(tyrosine_count)
                tyrosine_counts.append(tyrosine_count)
                inchikey = Chem.MolToInchiKey(mol)
                valid_smiles.append(smi)
                ik_to_repr_smi[inchikey] = smi
                ik_to_smis[inchikey] += 1
                fp2 = mol_to_morgan_fp(mol, radius=2, n_bits=2048)
                score = tanimoto(fp1, fp2)
                if score > closest_match_score:
                    closest_match_score = score
                    closest_match = smi
            except:
                pass

    print(f"Generated {len(smiles)} SMILES, of which {len(valid_smiles)} are valid.")
    # print top 5 most abundant unique molecules with counts and first smiles
    print("Top 5 most abundant unique molecules:")
    for ik, count in ik_to_smis.most_common(5):
        print(f"{ik}: {count} occurrences, example SMILES: {ik_to_repr_smi[ik]}")

    print(closest_match_score)
    print(closest_match)

    # get weight distribution (same?)
    # get kmer distribution (same?)
    # get residue distribution (same?)

    # count tyrosine counts
    print(f"Average tyrosine count: {sum(tyrosine_counts) / len(tyrosine_counts)}")
    # ask for 2, get 1.0
    # ask for 3, get 1.5
    # ask for 4, get 1.9
    # ask for 6, get 2.5



if __name__ == "__main__":
    main()
