"""Utilities for graph-conditioned CLMs."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import os

import torch
import torch.nn as nn


def _open_text(path: str, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def load_data_sample(json_line: str) -> dict:
    """
    Load one graph-conditioned sample from a JSONL line.

    Each sample must contain a sequence field under `smiles` and a node-link
    graph dictionary under `condition_graph`.
    """
    obj = json.loads(json_line)
    if "smiles" not in obj:
        raise KeyError("Expected graph-conditioned input to contain a 'smiles' field")
    if "condition_graph" not in obj and "graph" in obj:
        obj["condition_graph"] = obj.pop("graph")
    if "condition_graph" not in obj:
        raise KeyError(
            "Expected graph-conditioned input to contain a 'condition_graph' field"
        )
    obj["condition_graph"] = normalize_condition_graph(obj["condition_graph"])
    return obj


def read_graph_condition_file(path: str, max_lines: int | None = None) -> list[dict]:
    """Read graph-conditioned samples from JSONL."""
    if max_lines == 0:
        max_lines = None
    rows = []
    with _open_text(path, "rt") as handle:
        for idx, line in enumerate(handle):
            if max_lines is not None and idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(load_data_sample(line))
    return rows


def write_graph_condition_file(path: str, rows: list[dict]) -> None:
    """Write graph-conditioned samples as JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def normalize_condition_graph(graph_data: dict) -> dict:
    """
    Normalize graph-conditioned input into the schema expected by training.

    - Empty graphs are replaced with a single unknown node.
    - Nodes without an identified label are mapped to `<UNK>`.
    """
    if not isinstance(graph_data, dict):
        raise TypeError("Condition graph must be a dictionary")

    normalized = dict(graph_data)
    nodes = [dict(node) for node in normalized.get("nodes", [])]
    links = list(normalized.get("links", normalized.get("edges", [])))

    if not nodes:
        normalized["nodes"] = [{"id": 0, "name": "<UNK>"}]
        normalized["links"] = []
        normalized.pop("edges", None)
        return normalized

    normalized_nodes = []
    for idx, node in enumerate(nodes):
        normalized_node = dict(node)
        if "id" not in normalized_node:
            normalized_node["id"] = idx

        name = normalized_node.get("name", None)
        if isinstance(name, str) and name:
            normalized_nodes.append(normalized_node)
            continue

        candidates = [
            {"name": cand["name"], "weight": cand.get("weight", 0.0)}
            for cand in normalized_node.get("name_candidates", [])
            if isinstance(cand, dict) and cand.get("name")
        ]
        if candidates:
            normalized_node["name_candidates"] = candidates
        else:
            normalized_node.pop("name_candidates", None)
            normalized_node["name"] = "<UNK>"

        normalized_nodes.append(normalized_node)

    normalized["nodes"] = normalized_nodes
    normalized["links"] = links
    normalized.pop("edges", None)
    return normalized


def normalize_candidates(cands: list[dict]) -> list[dict]:
    """
    Normalize node candidate substrates.
    """
    if not cands:
        raise ValueError("Expected at least one name candidate for ambiguous node")

    total = sum(max(0.0, c.get("weight", 0.0)) for c in cands)
    if total <= 0:
        weight = 1.0 / len(cands)
        return [{"name": c["name"], "weight": weight} for c in cands]

    return [
        {"name": c["name"], "weight": max(0.0, c.get("weight", 0.0)) / total}
        for c in cands
    ]


def _extract_nodes_and_edges(graph_data: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    nodes = graph_data.get("nodes", [])
    raw_edges = graph_data.get("links", graph_data.get("edges", []))
    if not nodes:
        raise ValueError("Condition graph must contain at least one node")

    node_id_to_idx = {}
    ordered_nodes = []
    for idx, node in enumerate(nodes):
        node_id = node.get("id", idx)
        node_id_to_idx[node_id] = idx
        ordered_nodes.append(node)

    edges = []
    for edge in raw_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        if source not in node_id_to_idx or target not in node_id_to_idx:
            raise ValueError("Condition graph edge references unknown node id")
        edges.append((node_id_to_idx[source], node_id_to_idx[target]))

    return ordered_nodes, edges


def iter_conditioning_names(graph_data: dict):
    """Yield all label names referenced by a condition graph."""
    nodes, _ = _extract_nodes_and_edges(graph_data)
    for node in nodes:
        if "name" in node:
            yield node["name"]
            continue
        for cand in node.get("name_candidates", []):
            if "name" in cand:
                yield cand["name"]


def build_condition_vocab(rows: list[dict]) -> list[str]:
    """Build a condition-label vocabulary from graph-conditioned rows."""
    labels = sorted(
        {
            name
            for row in rows
            for name in iter_conditioning_names(row["condition_graph"])
        }
    )
    return ["<UNK>", *labels]


def write_condition_vocab(path: str, labels: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"labels": labels}, handle)


def read_condition_vocab(path: str) -> list[str]:
    with open(path, "r") as handle:
        payload = json.load(handle)
    labels = payload.get("labels", [])
    if not labels:
        return ["<UNK>"]
    return labels


def make_node_feature(node_attrs, name_to_idx, embedding_weight):
    """
    Create a node feature vector by embedding the node's name or its candidate
    names and appending a small ambiguity summary.
    """
    unk_idx = name_to_idx.get("<UNK>", 0)

    if "name" in node_attrs:
        idx = name_to_idx.get(node_attrs["name"], unk_idx)
        emb = embedding_weight[idx]
        ambiguity = torch.tensor([1.0, 1.0], dtype=emb.dtype, device=emb.device)
        return torch.cat([emb, ambiguity], dim=0)

    cands = normalize_candidates(node_attrs["name_candidates"])
    indices = [name_to_idx.get(c["name"], unk_idx) for c in cands]
    probs = torch.tensor(
        [c["weight"] for c in cands],
        dtype=embedding_weight.dtype,
        device=embedding_weight.device,
    )
    probs /= probs.sum()
    vecs = embedding_weight[indices]
    emb = (probs[:, None] * vecs).sum(dim=0)
    ambiguity = torch.stack(
        [
            probs.max(),
            torch.tensor(float(len(cands)), dtype=probs.dtype, device=probs.device),
        ]
    )
    return torch.cat([emb, ambiguity], dim=0)


@dataclass
class ConditionGraph:
    """Single graph represented as packed tensors."""

    edge_index: torch.Tensor
    candidate_index: torch.Tensor
    candidate_weight: torch.Tensor
    candidate_ptr: torch.Tensor
    ambiguity: torch.Tensor
    num_nodes: int


@dataclass
class ConditionGraphBatch:
    """Batched graphs for graph-conditioned training/sampling."""

    edge_index: torch.Tensor
    candidate_index: torch.Tensor
    candidate_weight: torch.Tensor
    candidate_ptr: torch.Tensor
    ambiguity: torch.Tensor
    batch: torch.Tensor
    num_graphs: int

    def to(self, device):
        return ConditionGraphBatch(
            edge_index=self.edge_index.to(device),
            candidate_index=self.candidate_index.to(device),
            candidate_weight=self.candidate_weight.to(device),
            candidate_ptr=self.candidate_ptr.to(device),
            ambiguity=self.ambiguity.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
        )


def graph_to_condition_graph(graph_data: dict, name_to_idx: dict[str, int]) -> ConditionGraph:
    """Convert a JSON graph into packed tensors."""
    nodes, edges = _extract_nodes_and_edges(graph_data)
    unk_idx = name_to_idx.get("<UNK>", 0)

    candidate_indices = []
    candidate_weights = []
    candidate_ptr = [0]
    ambiguity = []

    for node in nodes:
        if "name" in node:
            candidate_indices.append(name_to_idx.get(node["name"], unk_idx))
            candidate_weights.append(1.0)
            candidate_ptr.append(candidate_ptr[-1] + 1)
            ambiguity.append([1.0, 1.0])
            continue

        cands = normalize_candidates(node.get("name_candidates", []))
        candidate_indices.extend(name_to_idx.get(c["name"], unk_idx) for c in cands)
        candidate_weights.extend(c["weight"] for c in cands)
        candidate_ptr.append(candidate_ptr[-1] + len(cands))
        ambiguity.append([max(c["weight"] for c in cands), float(len(cands))])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return ConditionGraph(
        edge_index=edge_index,
        candidate_index=torch.tensor(candidate_indices, dtype=torch.long),
        candidate_weight=torch.tensor(candidate_weights, dtype=torch.float32),
        candidate_ptr=torch.tensor(candidate_ptr, dtype=torch.long),
        ambiguity=torch.tensor(ambiguity, dtype=torch.float32),
        num_nodes=len(nodes),
    )


class ConditionGraphCollate:
    """Batch packed graph tensors across multiple samples."""

    def __call__(self, graphs: list[ConditionGraph]) -> ConditionGraphBatch:
        edge_chunks = []
        candidate_indices = []
        candidate_weights = []
        candidate_ptr = [0]
        ambiguity = []
        batch = []

        node_offset = 0
        candidate_offset = 0

        for graph_idx, graph in enumerate(graphs):
            if graph.edge_index.numel():
                edge_chunks.append(graph.edge_index + node_offset)

            candidate_indices.append(graph.candidate_index)
            candidate_weights.append(graph.candidate_weight)
            ambiguity.append(graph.ambiguity)
            batch.append(
                torch.full((graph.num_nodes,), graph_idx, dtype=torch.long)
            )

            shifted_ptr = graph.candidate_ptr[1:] + candidate_offset
            candidate_ptr.extend(shifted_ptr.tolist())
            node_offset += graph.num_nodes
            candidate_offset += len(graph.candidate_index)

        if edge_chunks:
            edge_index = torch.cat(edge_chunks, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if candidate_indices:
            all_candidate_indices = torch.cat(candidate_indices, dim=0)
            all_candidate_weights = torch.cat(candidate_weights, dim=0)
            all_ambiguity = torch.cat(ambiguity, dim=0)
            batch_tensor = torch.cat(batch, dim=0)
        else:
            all_candidate_indices = torch.empty((0,), dtype=torch.long)
            all_candidate_weights = torch.empty((0,), dtype=torch.float32)
            all_ambiguity = torch.empty((0, 2), dtype=torch.float32)
            batch_tensor = torch.empty((0,), dtype=torch.long)

        return ConditionGraphBatch(
            edge_index=edge_index,
            candidate_index=all_candidate_indices,
            candidate_weight=all_candidate_weights,
            candidate_ptr=torch.tensor(candidate_ptr, dtype=torch.long),
            ambiguity=all_ambiguity,
            batch=batch_tensor,
            num_graphs=len(graphs),
        )


class GraphConv(nn.Module):
    """A small GCN-style layer implemented with plain PyTorch ops."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if num_nodes == 0:
            return x

        device = x.device
        loops = torch.arange(num_nodes, device=device)
        row = torch.cat([edge_index[0], loops], dim=0)
        col = torch.cat([edge_index[1], loops], dim=0)

        degree = torch.bincount(row, minlength=num_nodes).float()
        degree = degree.clamp_min(1.0)
        norm = degree[row].pow(-0.5) * degree[col].pow(-0.5)

        transformed = self.linear(x)
        out = torch.zeros_like(transformed)
        out.index_add_(0, row, transformed[col] * norm.unsqueeze(1))
        return out


def global_mean_pool(
    x: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Pool node embeddings into one embedding per graph."""
    if num_graphs == 0:
        return torch.empty((0, x.size(1)), dtype=x.dtype, device=x.device)

    pooled = torch.zeros((num_graphs, x.size(1)), dtype=x.dtype, device=x.device)
    pooled.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).unsqueeze(1)
    return pooled / counts


class BiosynthesisGraphEncoder(nn.Module):
    """Encode condition graphs into fixed-width conditioning vectors."""

    def __init__(
        self,
        vocab_size: int,
        label_emb_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(vocab_size, label_emb_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(label_emb_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _node_embeddings(self, data: ConditionGraphBatch) -> torch.Tensor:
        candidate_counts = data.candidate_ptr[1:] - data.candidate_ptr[:-1]
        node_index = torch.repeat_interleave(
            torch.arange(len(candidate_counts), device=data.candidate_index.device),
            candidate_counts,
        )
        candidate_emb = self.label_emb(data.candidate_index)
        weighted = candidate_emb * data.candidate_weight.unsqueeze(1)
        node_emb = torch.zeros(
            (len(candidate_counts), candidate_emb.size(1)),
            dtype=weighted.dtype,
            device=weighted.device,
        )
        node_emb.index_add_(0, node_index, weighted)
        return torch.cat([node_emb, data.ambiguity], dim=1)

    def forward(self, data: ConditionGraphBatch) -> torch.Tensor:
        x = self.node_mlp(self._node_embeddings(data))
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        pooled = global_mean_pool(x, data.batch, data.num_graphs)
        return self.out_proj(pooled)
