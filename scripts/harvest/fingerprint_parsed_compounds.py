#!/usr/bin/env python3
"""Fingerprint compounds parsed by RetroMol."""

import argparse
import json
import os
import yaml
import math
import random
import hashlib
from typing import Generator, Any

import numpy as np
from tqdm import tqdm

from retromol.model.reaction_graph import MolNode
from retromol.model.result import Result


FP_EDGES_SIZE = int(os.getenv("FP_EDGES_SIZE", "258"))


def cli() -> argparse.Namespace:
    """
    Command line interface for fingerprinting parsed compounds.

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="directory to save output files")
    parser.add_argument("--jsonl", type=str, required=True, help="path to JSONL file containing RetroMol-parsed compounds")
    parser.add_argument("--matching-rules", type=str, default="matching rules file used for RetroMol parsing")
    parser.add_argument("--fingerprint-edges", action="store_true", help="whether to include edge features in the fingerprint")
    parser.add_argument("--name-dataset", type=str, default="dataset", help="name of the output dataset file (default: dataset)")
    parser.add_argument("--num-samples-edge-features", type=int, default=1, help="number of times to sample edge features for each compound (default: 1)")
    return parser.parse_args()



def iter_jsonl(path: str) -> Generator[dict[str, Any], None, None]:
    """
    Generator that yields JSON objects from a JSONL file.
    
    :param path: path to JSONL file
    :yield: JSON object from each line of the file
    """
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def result_to_monomer_dists(r: Result) -> list[list[tuple[str, float]]]:
    """
    Extract monomer distributions from a Result object.
    
    :param r: Result object containing parsed compound information
    :return: list of monomer distributions, where each distribution is a list of (identity name, probability) tuples
    """
    a = r.linear_readout.assembly_graph
    dists: list[list[tuple[str, float]]] = []
    for n in a.g.nodes:
        ident = a.g.nodes[n].get("identity", None)
        if ident is None:
            continue
        # Currently deterministic, always 1.0 for the identified monomer
        dists.append([(ident.name, 1.0)])
    return dists


def make_fp_residues(r: Result, assignment: list[str]) -> np.ndarray:
    """
    Create residue feature vector for a Result object based on matching rules.
    
    :param r: Result object containing parsed compound information
    :param assignment: list of feature names corresponding to matching rules
    :return: numpy array representing the residue features of the fingerprint
    :raises ValueError: if an identity from the monomer distributions is not found in the assignment list
    """
    fp_residues = np.zeros(len(assignment), dtype=np.float32)
    monomer_dists = result_to_monomer_dists(r)
    for d in monomer_dists:
        for ident, prob in d:
            if ident in assignment:
                idx = assignment.index(ident)
                fp_residues[idx] += prob
            else:
                raise ValueError(f"Identity '{ident}' not found in assignment list!")
    fp_residues /= (np.linalg.norm(fp_residues) + 1e-8)  # normalize to unit vector (avoid division by zero)
    return fp_residues


def pack_fp(fp_residues: np.ndarray, fp_edges: np.ndarray | None, n_ident: int, cov: float) -> np.ndarray:
    """
    Pack fingerprint components into a single numpy array.
    
    :param fp_residues: numpy array representing residue features of the fingerprint
    :param fp_edges: numpy array representing edge features of the fingerprint
    :param n_ident: number of identified monomers in the compound
    :param cov: coverage of the compound by identified monomers
    :return: concatenated and normalized fingerprint as a numpy array
    """
    cov = float(max(0.0, min(1.0, cov)))
    n_ident = float(math.log1p(n_ident))
    if fp_edges is not None:
        return np.concatenate([fp_residues.astype(np.float32), fp_edges.astype(np.float32), np.array([n_ident, cov], dtype=np.float32)])
    else:
        return np.concatenate([fp_residues.astype(np.float32), np.array([n_ident, cov], dtype=np.float32)])
    

def stable_hash(s: str) -> int:
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)


def make_fp_edges(
    r: Result, 
    D: int = FP_EDGES_SIZE,
    ks: tuple[int] = (1, 2, 3),
) -> np.ndarray:
    """
    Create edge feature vector for a Result object based on monomer neighborhoods in assembly graph Result.

    :param r: Result object containing parsed compound information
    :param D: dimensionality of the edge feature vector (default: FP_EDGES_SIZE)
    :param ks: tuple of neighborhood sizes to consider (default: (1, 2, 3))
    :return: numpy array representing the edge features of the fingerprint
    """
    fp = np.zeros(D, dtype=np.float32)

    a = r.linear_readout.assembly_graph
    nodes = a.longest_path()
    
    rng = random.Random(None)
    out: list[list[MolNode]] = []
    i = 0
    n = len(nodes)

    while i < n:
        # only allow k values that still fit in the remaining tail
        remaining = n - i
        valid_ks = [k for k in ks if k <= remaining]
        k = rng.choice(valid_ks)
        out.append(nodes[i:i+k])
        i += k

    # Mine the node sublists for 2-mer edge features
    for sublist in out:
        idents = [n.identity.name if n.identified else None for n in sublist]
        L = len(idents)
        for i in range(1, L):
            r1 = idents[i-1] if idents[i-1] is not None else "UNK"
            r2 = idents[i] if idents[i] is not None else "UNK"
            feat = f"EDGE:{r1}->{r2}"
            w = 1.0 * 1.0
            fp[stable_hash(feat) % D] += w

    # Normalize the edge features to unit vector
    fp /= (np.linalg.norm(fp) + 1e-8)  # avoid division by zero

    return fp


def generate_edge_features(
    r: Result,
    include_edges: bool = False,
    D: int = FP_EDGES_SIZE,
    samples: int = 1,
    ks: tuple[int] = (1, 2, 3),
) -> Generator[np.ndarray | None, None, None]:
    """
    Generator that yields edge feature vectors for a Result object based on monomer neighborhoods in assembly graph Result.
    
    :param r: Result object containing parsed compound information
    :param include_edges: whether to include edge features in the output
    :param D: dimensionality of the edge feature vector (default: FP_EDGES_SIZE)
    :param samples: number of edge feature samples to generate (default: 1)
    :param ks: tuple of neighborhood sizes to consider (default: (1, 2, 3))
    :yield: edge feature vector as a numpy array if include_edges is True, otherwise None
    """
    if not include_edges:
        yield None
    
    else:
        for _ in range(samples):
            yield make_fp_edges(r, D=D, ks=ks)


def fingerprint_result(
    r: Result,
    assignment: list[str],
    include_edges: bool = False,
    num_samples_edge_features: int = 5,
) -> Generator[np.ndarray, None, None]:
    """
    Compute fingerprint for a given Result object based on matching rules.
    
    :param r: Result object containing parsed compound information
    :param assignment: list of feature names corresponding to matching rules
    :param include_edges: whether to include edge features in the fingerprint
    :param sample_edge_features: number of times to sample edge features
    :return: numpy array representing the fingerprint of the compound
    """
    fp_residues = make_fp_residues(r, assignment)
    
    for fp_edges in generate_edge_features(r, include_edges=include_edges, samples=num_samples_edge_features):
        monomer_dists = result_to_monomer_dists(r)
        n_ident = len(monomer_dists)
        cov = r.calculate_coverage()
        yield pack_fp(fp_residues, fp_edges, n_ident, cov)


def main() -> None:
    """
    Main function for fingerprinting parsed compounds.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    # Setup fingerprint design based on matching rules
    with open(args.matching_rules, "r") as f:
        matching_rules = yaml.safe_load(f)
    assignment = [r["name"] for r in matching_rules]

    with open(os.path.join(args.outdir, f"{args.name_dataset}.txt"), "w") as out_f:
        # Write header: smiles, every fingerpint feature, n_ident, cov
        len_fps = len(assignment) + (FP_EDGES_SIZE if args.fingerprint_edges else 0)
        header = ["smiles"] + [f"fp_{i}" for i in range(len_fps)] + ["n_ident", "cov"]
        out_f.write(",".join(header) + "\n")

        for obj_idx, obj in tqdm(enumerate(iter_jsonl(args.jsonl))):
            r = Result.from_dict(obj)
            smiles = r.submission.smiles
            for fp in fingerprint_result(r, assignment, args.fingerprint_edges, num_samples_edge_features=args.num_samples_edge_features):
                fp_str = ",".join([str(x) for x in fp])
                out_f.write(f"{smiles},{fp_str}\n")

                # # TODO: remove this limit after testing
                # if obj_idx >= 1000:  # limit to first 1000 compounds for testing
                #     break


if __name__ == "__main__":
    main()
