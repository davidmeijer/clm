"""Module for loading CLMs from files."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
import torch
from clm.conditioning import resolve_conditioning_type
from clm.datasets import Vocabulary
from clm.graph_conditional import read_condition_vocab
from clm.models import RNN, ConditionalRNN, GraphConditionalRNN


@dataclass
class ModelConfig:
    """
    Data class to hold configuration for loading a CLM model.

    :param enum_factor: enumeration factor for the model
    :param vocab_path: path to the vocabulary file
    :param model_path: path to the model file
    :param test_path: path to the test dataset file
    :param rnn_type: type of RNN (e.g., LSTM, GRU)
    :param embedding_size: size of the embedding layer
    :param hidden_size: size of the hidden layers
    :param n_layers: number of RNN layers
    :param dropout: dropout rate
    :param num_descriptors: number of descriptors (for conditional models)
    :param cond_enabled: whether the model is conditional
    :param cond_emb: whether conditional embedding is used
    :param cond_emb_l: whether conditional embedding is learned
    :param cond_dec: whether conditional decoder is used
    :param cond_dec_l: whether conditional decoder is learned
    :param cond_h: whether conditional hidden state is used
    """

    enum_factor: int
    vocab_path: Path
    model_path: Path
    test_path: Path

    rnn_type: str
    embedding_size: int
    hidden_size: int
    n_layers: int
    dropout: float
    num_descriptors: int
    cond_enabled: bool
    cond_emb: bool
    cond_emb_l: bool
    cond_dec: bool
    cond_dec_l: bool
    cond_h: bool
    conditioning_type: str = "none"
    condition_vocab_path: Path | None = None
    graph_label_emb_dim: int = 64
    graph_hidden_dim: int = 128
    graph_out_dim: int = 256

    def load_model(self, device: str | torch.device) -> RNN | ConditionalRNN:
        """
        Load the CLM model from file.

        :param device: device to load the model onto
        :return: loaded CLM model
        """ 
        vocab = Vocabulary(vocab_file=self.vocab_path)

        if self.conditioning_type == "graph":
            if self.condition_vocab_path is None:
                raise ValueError("condition_vocab_path is required for graph models")
            condition_labels = read_condition_vocab(self.condition_vocab_path)
            model = GraphConditionalRNN(
                vocabulary=vocab,
                condition_vocab_size=len(condition_labels),
                rnn_type=self.rnn_type,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
                graph_label_emb_dim=self.graph_label_emb_dim,
                graph_hidden_dim=self.graph_hidden_dim,
                graph_out_dim=self.graph_out_dim,
                conditional_emb=self.cond_emb,
                conditional_emb_l=self.cond_emb_l,
                conditional_dec=self.cond_dec,
                conditional_dec_l=self.cond_dec_l,
                conditional_h=self.cond_h,
            )
        elif self.cond_enabled:
            model = ConditionalRNN(
                vocabulary=vocab,
                rnn_type=self.rnn_type,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
                num_descriptors=self.num_descriptors,
                conditional_emb=self.cond_emb,
                conditional_emb_l=self.cond_emb_l,
                conditional_dec=self.cond_dec,
                conditional_dec_l=self.cond_dec_l,
                conditional_h=self.cond_h
            )
        else:
            model = RNN(
                vocabulary=vocab,
                rnn_type=self.rnn_type,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                dropout=self.dropout
            )

        # Check if device is torch.device, if string convert it
        if isinstance(device, str):
            device = torch.device(device)

        state = torch.load(self.model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        
        return model


def prep_clm(
    model_dir: Path | str,
    eval: bool = False
) -> list[ModelConfig]:
    """
    Compile all relevant information for loading a CLM from file.

    :param model_dir: directory containing the model files
    :param eval: whether to prepare for evaluation (default: False)
    :return: list of ModelConfig objects for loading the CLM models
    """
    # Retrieve config_path from the model_dir
    configs = list((Path(model_dir) / "prior" / "raw").glob("*_config.yaml"))
    if len(configs) == 0:
        raise FileNotFoundError("No config file matching '*_config.yaml' found.")
    elif len(configs) > 1:
        raise RuntimeError(f"Multiple config files found: {configs}")
    config_path = configs[0]

    # Check if model dir is Path object, if string convert it
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    # Check if config file and model dir exist
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_dir}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Parse relevant information from config file for loading model
    enum_factors = config.get("enum_factors", None)
    dataset_path = config.get("paths", {}).get("dataset", None)
    model_params = config.get("model_params", None)
    
    if enum_factors is None:
        raise ValueError("'enum_factors' not specified in config file!")
    if dataset_path is None:
        raise ValueError("'dataset' not specified in config file!")
    if model_params is None:
        raise ValueError("'model_params' not specified in config file!")
    
    # Parse model parameters
    rnn_type        = model_params.get("rnn_type", "LSTM")
    embedding_size  = model_params.get("embedding_size", 128)
    hidden_size     = model_params.get("hidden_size", 1024)
    n_layers        = model_params.get("n_layers", 3)
    dropout         = model_params.get("dropout", 0)
    conditional_cfg = model_params.get("conditional", {})
    cond_enabled    = conditional_cfg.get("enabled", False)
    cond_emb        = conditional_cfg.get("emb", False)
    cond_emb_l      = conditional_cfg.get("emb_l", True)
    cond_dec        = conditional_cfg.get("dec", False)
    cond_dec_l      = conditional_cfg.get("dec_l", True)
    cond_h          = conditional_cfg.get("h", False)
    conditioning_type = resolve_conditioning_type(
        conditional=cond_enabled,
        conditioning_type=conditional_cfg.get("type"),
    )
    cond_enabled = conditioning_type != "none"
    graph_cfg = conditional_cfg.get("graph", {})
    graph_label_emb_dim = graph_cfg.get("label_emb_dim", 64)
    graph_hidden_dim = graph_cfg.get("hidden_dim", 128)
    graph_out_dim = graph_cfg.get("out_dim", 256)
    
    # Get file name from dataset_path using Path, with file extension
    dataset_fn = Path(dataset_path).name if dataset_path is not None else None

    if dataset_fn is None:
        raise ValueError("Could not determine dataset file name from dataset path!")
    
    # Check header of dataset file to determine number of descriptors
    keep_duplicates = config.get("preprocess", {}).get("keep_duplicates", False)
    num_descriptors = 0
    if conditioning_type == "graph":
        num_descriptors = graph_out_dim
    elif cond_enabled:
        dataset_full_path = os.path.join(model_dir, "prior", "raw", dataset_fn)
        with open(dataset_full_path, "r") as f:
            header = f.readline().strip()
            num_descriptors = len(header.split(",")) - 1  # assuming first column is SMILES
            # We subtract an additional column since when keep_duplicates is False, an extra InChIKey column is added
            num_descriptors = num_descriptors - (0 if keep_duplicates and not eval else 1)
    
    # Get file name from dataset_path using Path, without file extension
    dataset_name = Path(dataset_path).stem if dataset_path is not None else None

    if dataset_name is None:
        raise ValueError("Could not determine dataset name from dataset path!")
    
    # Compile model configurations
    model_configs = []

    for enum_factor in enum_factors:
        vocab_dir_path = os.path.join(model_dir, f"{enum_factor}", "prior", "inputs")
        model_dir_path = os.path.join(model_dir, f"{enum_factor}", "prior", "models")

        # Check if directories exist
        if not os.path.isdir(vocab_dir_path):
            raise NotADirectoryError(f"Vocabulary directory not found: {vocab_dir_path}")
        if not os.path.isdir(model_dir_path):
            raise NotADirectoryError(f"Model directory not found: {model_dir_path}")

        dataset_name = "synthetic_lipopeptides_graph_conditioning"
        vocab_file_pattern = re.compile(f"train_{dataset_name}_SMILES_\d+.vocabulary")
        # vocab_file_pattern = re.compile(f"train_{dataset_name}_SMILES_\d+.condition_vocab.json|train_{dataset_name}_SMILES_\d+.vocabulary")
        model_file_pattern = re.compile(f"{dataset_name}_SMILES_\d+_\d+_model.pt")

        # Find all vocab files in vocab_dir_path
        vocab_files = [f for f in os.listdir(vocab_dir_path) if vocab_file_pattern.match(f)]
        if not vocab_files:
            raise FileNotFoundError(f"No vocabulary files found in: {vocab_dir_path}")
        
        # Find all model files in model_dir_path
        model_files = [f for f in os.listdir(model_dir_path) if model_file_pattern.match(f)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in: {model_dir_path}")
        
        # print(len(vocab_files))
        # print(len(model_files))
        
        # Assert that we found same number of vocab and model files
        if len(vocab_files) != len(model_files):
            raise ValueError(f"Mismatch in number of vocab and model files for enum factor {enum_factor}!")
        
        # Match vocab files to model files based on the split number
        vocab_split_pattern = re.compile(f"train_{dataset_name}_SMILES_(\d+).vocabulary")
        model_split_pattern = re.compile(f"{dataset_name}_SMILES_(\d+)_\d+_model.pt")

        # Find the split numbers
        found_vocab_splits = {int(vocab_split_pattern.match(f).group(1)): f for f in vocab_files}
        found_model_splits = {int(model_split_pattern.match(f).group(1)): f for f in model_files}

        # Check if keys (split numbers) perfectly overlap
        overlapping_splits = set(found_vocab_splits.keys()).intersection(set(found_model_splits.keys()))
        assert len(overlapping_splits) == len(found_vocab_splits) == len(found_model_splits), "Mismatch in splits between vocab and model files!"

        # Compile model configs
        for split in overlapping_splits:
            vocab_path = Path(vocab_dir_path) / found_vocab_splits[split]
            model_path = Path(model_dir_path) / found_model_splits[split]
            condition_vocab_path = (
                Path(vocab_dir_path)
                / f"train_{dataset_name}_SMILES_{split}.condition_vocab.json"
            )
            if conditioning_type == "graph" and not condition_vocab_path.exists():
                raise FileNotFoundError(
                    f"Condition vocab file not found: {condition_vocab_path}"
                )

            # Find test dataset file
            # test_path = os.path.join(model_dir, str(enum_factor), "prior", "inputs", f"test0_{dataset_name}_SMILES_{split}.smi")
            test_path = os.path.join(model_dir, str(enum_factor), "prior", "inputs", f"test_condition_{dataset_name}_SMILES_{split}.jsonl")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test dataset file not found: {test_path}")

            model_configs.append(ModelConfig(
                enum_factor=enum_factor,
                vocab_path=vocab_path,
                model_path=model_path,
                test_path=test_path,
                rnn_type=rnn_type,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                dropout=dropout,
                num_descriptors=num_descriptors,
                cond_enabled=cond_enabled,
                cond_emb=cond_emb,
                cond_emb_l=cond_emb_l,
                cond_dec=cond_dec,
                cond_dec_l=cond_dec_l,
                cond_h=cond_h,
                conditioning_type=conditioning_type,
                condition_vocab_path=condition_vocab_path if conditioning_type == "graph" else None,
                graph_label_emb_dim=graph_label_emb_dim,
                graph_hidden_dim=graph_hidden_dim,
                graph_out_dim=graph_out_dim,
            ))

    return model_configs