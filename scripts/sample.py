#!/usr/bin/env python3

"""Sample trained CLM."""

import argparse 

import torch
import pandas as pd
from clm.datasets import Vocabulary
# from clm.functions import load_dataset, write_to_csv_file
from clm.models import ConditionalRNN, RNN


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model .pt file")
    parser.add_argument("-v", "--vocab", type=str, required=True, help="path to vocabulary file")
    return parser.parse_args()


def main() -> None:
    """Entry point script."""
    args = cli()

    # Load vocab
    vocab = Vocabulary(vocab_file=args.vocab)

    # Instantiate the ConditionalRNN (matching training hyperparams)
    model = ConditionalRNN(
        vocab,
        rnn_type="LSTM",
        embedding_size=128,
        hidden_size=1024,
        n_layers=3,
        dropout=0,
        num_descriptors=44,
        conditional_emb=False,
        conditional_emb_l=True,
        conditional_dec=False,
        conditional_dec_l=True,
        conditional_h=False
    )
    print("> LOADED MODEL")

    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("> LOADED WEIGHTS")

    feature_names = [
        "A","AAD","AIB","ALA","ANT","ARG","ASN","ASP","B","BAL",
        "BHT","C",'CYS',"D","DAB","DHB","DHG","FHO","GLN","GLU",
        "GLY","HIS","HOO","HPG","ILE","LEU","LYS","ORN","PHE","PIP",
        "PRO","SAL","SER","THR","TRP","TYR","VAL","amino_acid","halogenation","methylation",
        "other","oxidation","polyketide","sugar"
    ]
    num_descriptors = len(feature_names)
    num_samples = 10
    descriptors = [0] * num_descriptors
    index_cys = feature_names.index("CYS")
    descriptors[index_cys] = 3  # Set CYS descriptor to 1
    for_sampling = []
    for _ in range(num_samples):
        for_sampling.append(descriptors.copy())
    descriptors = torch.tensor(for_sampling, dtype=torch.float32, device=device)

    with torch.no_grad():
        sampled = model.sample(descriptors=descriptors)
        for idx, smiles in enumerate(sampled):
            print(f"Item {idx}: {smiles}")



if __name__ == "__main__":
    main()
