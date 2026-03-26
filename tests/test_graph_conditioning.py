import json
from pathlib import Path

import pandas as pd

from clm.commands import create_training_sets, preprocess, sample_molecules_RNN, train_models_RNN
from clm.functions import read_csv_file, set_seed
from clm.graph_conditional import read_condition_vocab, read_graph_condition_file


def _write_graph_dataset(path: Path) -> None:
    rows = [
        {
            "smiles": "CCO",
            "graph": {
                "nodes": [],
                "links": [],
            },
        },
        {
            "smiles": "CCN",
            "condition_graph": {
                "nodes": [{"id": 0, "name": "Val"}],
                "links": [],
            },
        },
        {
            "smiles": "CCCO",
            "condition_graph": {
                "nodes": [{"id": 0, "name": "Leu"}, {"id": 1, "name": "Asp"}],
                "links": [{"source": 0, "target": 1}],
            },
        },
        {
            "smiles": "CC(=O)O",
            "condition_graph": {
                "nodes": [{"id": 0, "name": "Ser"}],
                "links": [],
            },
        },
        {
            "smiles": "C1CCCCC1",
            "condition_graph": {
                "nodes": [{"id": 0, "name": "Phe"}],
                "links": [],
            },
        },
        {
            "smiles": "CCOC",
            "condition_graph": {
                "nodes": [{"id": 0, "name": "Gly"}, {"id": 1, "name": "Ala"}],
                "links": [{"source": 0, "target": 1}],
            },
        },
    ]

    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_graph_conditioned_preprocess_and_split(tmp_path):
    raw_input = tmp_path / "graph_input.jsonl"
    processed = tmp_path / "processed.jsonl"
    _write_graph_dataset(raw_input)

    preprocess.preprocess(
        input_file=raw_input,
        output_file=processed,
        max_input_smiles=0,
        min_heavy_atoms=0,
        neutralise=False,
        valid_atoms=["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"],
        conditioning_type="graph",
    )

    processed_rows = read_graph_condition_file(processed)
    assert len(processed_rows) == 6
    assert all("inchikey" in row for row in processed_rows)
    assert "condition_graph" in processed_rows[0]
    assert "graph" not in processed_rows[0]
    assert processed_rows[0]["condition_graph"]["nodes"] == [{"id": 0, "name": "<UNK>"}]

    set_seed(5831)
    create_training_sets.create_training_sets(
        input_file=processed,
        train0_file=tmp_path / "train0_{fold}.smi",
        train_file=tmp_path / "train_{fold}.jsonl",
        test0_file=tmp_path / "test0_{fold}.smi",
        condition_test_file=tmp_path / "test_condition_{fold}.jsonl",
        condition_vocab_file=tmp_path / "condition_vocab_{fold}.json",
        vocab_file=tmp_path / "train_{fold}.vocabulary",
        folds=3,
        which_fold=0,
        enum_factor=0,
        representation="SMILES",
        conditioning_type="graph",
    )

    train_rows = read_graph_condition_file(tmp_path / "train_0.jsonl")
    heldout_rows = read_graph_condition_file(tmp_path / "test_condition_0.jsonl")
    condition_vocab = read_condition_vocab(tmp_path / "condition_vocab_0.json")
    train0 = pd.read_csv(tmp_path / "train0_0.smi")
    test0 = pd.read_csv(tmp_path / "test0_0.smi")

    assert train_rows
    assert heldout_rows
    assert "<UNK>" in condition_vocab
    assert "Ser" in condition_vocab
    assert set(["smiles", "inchikey"]).issubset(train0.columns)
    assert set(["smiles", "inchikey"]).issubset(test0.columns)


def test_graph_conditioned_train_and_sample(tmp_path):
    raw_input = tmp_path / "graph_input.jsonl"
    processed = tmp_path / "processed.jsonl"
    _write_graph_dataset(raw_input)

    preprocess.preprocess(
        input_file=raw_input,
        output_file=processed,
        max_input_smiles=0,
        min_heavy_atoms=0,
        neutralise=False,
        valid_atoms=["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"],
        conditioning_type="graph",
    )

    set_seed(5831)
    create_training_sets.create_training_sets(
        input_file=processed,
        train0_file=tmp_path / "train0_{fold}.smi",
        train_file=tmp_path / "train_{fold}.jsonl",
        test0_file=tmp_path / "test0_{fold}.smi",
        condition_test_file=tmp_path / "test_condition_{fold}.jsonl",
        condition_vocab_file=tmp_path / "condition_vocab_{fold}.json",
        vocab_file=tmp_path / "train_{fold}.vocabulary",
        folds=3,
        which_fold=0,
        enum_factor=0,
        representation="SMILES",
        conditioning_type="graph",
    )

    model_file = tmp_path / "graph_model.pt"
    loss_file = tmp_path / "graph_loss.csv"
    output_file = tmp_path / "graph_samples.csv"

    train_models_RNN.train_models_RNN(
        representation="SMILES",
        rnn_type="LSTM",
        embedding_size=16,
        hidden_size=32,
        n_layers=1,
        dropout=0,
        batch_size=2,
        learning_rate=0.001,
        max_epochs=1,
        patience=10,
        log_every_steps=10,
        log_every_epochs=1,
        sample_mols=4,
        input_file=tmp_path / "train_0.jsonl",
        vocab_file=tmp_path / "train_0.vocabulary",
        model_file=model_file,
        loss_file=loss_file,
        smiles_file=None,
        conditional=True,
        conditioning_type="graph",
        condition_vocab_file=tmp_path / "condition_vocab_0.json",
        graph_label_emb_dim=8,
        graph_hidden_dim=16,
        graph_out_dim=12,
    )

    sample_molecules_RNN.sample_molecules_RNN(
        representation="SMILES",
        rnn_type="LSTM",
        embedding_size=16,
        hidden_size=32,
        n_layers=1,
        dropout=0,
        batch_size=2,
        sample_mols=6,
        vocab_file=tmp_path / "train_0.vocabulary",
        model_file=model_file,
        output_file=output_file,
        conditional=True,
        conditioning_type="graph",
        heldout_file=tmp_path / "test_condition_0.jsonl",
        condition_vocab_file=tmp_path / "condition_vocab_0.json",
        graph_label_emb_dim=8,
        graph_hidden_dim=16,
        graph_out_dim=12,
    )

    assert len(read_csv_file(output_file)) == 6
