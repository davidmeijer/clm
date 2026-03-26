#!/usr/bin/env python3

import argparse
import json
import os

import networkx as nx
from tqdm import tqdm

from retromol.model.result import Result
from retromol.model.reaction_graph import MolNode


def iter_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Path to input JSONL file with RetroMol results.")
    parser.add_argument("-o", required=True, help="Path to output JSONL file for training data.")
    return parser.parse_args()


def node_label(node: MolNode) -> str:
    """Return the conditioning label for a RetroMol node."""
    if node.identified and node.identity is not None and node.identity.name:
        return node.identity.name
    return "<UNK>"


def main() -> None:
    args = cli()
    
    with open(args.o, "w") as f:
        for d in tqdm(iter_jsonl(args.i)):
            result = Result.from_dict(d)
            smiles = result.submission.smiles
            coverage = result.calculate_coverage()
            paths = result.linear_readout.paths
            graph = nx.Graph()
            # Add every path as a disconnected linear subgraph. Unknown nodes are
            # retained explicitly as <UNK> so conditioning is never empty.
            for path in paths:
                prev_node_id = None
                for node in path:
                    curr_node_id = len(graph.nodes)
                    graph.add_node(curr_node_id, name=node_label(node))
                    if prev_node_id is not None:
                        graph.add_edge(prev_node_id, curr_node_id)
                    prev_node_id = curr_node_id

            # If RetroMol produced no usable path nodes, keep a single unknown node.
            if graph.number_of_nodes() == 0:
                graph.add_node(0, name="<UNK>")

            # Write graph-conditioned training rows to JSONL.
            out_dict = {
                "smiles": smiles,
                "coverage": round(coverage, 2),
                "condition_graph": nx.node_link_data(graph),
            }
            f.write(json.dumps(out_dict) + "\n")


if __name__ == "__main__":
    main()
