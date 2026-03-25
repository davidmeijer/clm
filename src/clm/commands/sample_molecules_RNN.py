import logging
import os.path

import pandas as pd
import torch
from tqdm import tqdm

from clm.conditioning import resolve_conditioning_type
from clm.datasets import Vocabulary, SelfiesVocabulary
from clm.graph_conditional import read_condition_vocab
from clm.models import RNN, ConditionalRNN, GraphConditionalRNN
from clm.functions import load_dataset, write_to_csv_file

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )

    parser.add_argument(
        "--rnn_type", type=str, help="Type of RNN used (e.g., LSTM, GRU)"
    )

    parser.add_argument(
        "--embedding_size", type=int, help="Size of the embedding layer"
    )

    parser.add_argument("--hidden_size", type=int, help="Size of the hidden layers")

    parser.add_argument("--n_layers", type=int, help="Number of layers in the RNN")

    parser.add_argument("--dropout", type=float, help="Dropout rate for the RNN")

    parser.add_argument(
        "--conditional", action="store_true", help="Activate Conditional RNN model"
    )
    parser.add_argument(
        "--conditional_emb",
        action="store_true",
        help="Add descriptor with the input smiles without passing it through an embedding layer",
    )
    parser.add_argument(
        "--conditional_emb_l",
        action="store_true",
        help="Pass the descriptors through an embedding layer and add descriptor with the input smiles",
    )
    parser.add_argument(
        "--conditional_dec",
        action="store_true",
        help="Add descriptor with the rnn output without passing it through decoder layer",
    )
    parser.add_argument(
        "--conditional_dec_l",
        action="store_true",
        help="Pass the descriptors through a decoder layer and add descriptor with the rnn output",
    )
    parser.add_argument(
        "--conditional_h",
        action="store_true",
        help="Add descriptor in hidden and cell state",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="none",
        help="Conditioning mode (one of: none/descriptors/graph)",
    )
    parser.add_argument(
        "--heldout_file",
        type=str,
        nargs="+",
        help="Testing file in heldout set. Useful for sampling from a Conditional RNN model",
    )

    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    parser.add_argument(
        "--sample_mols", type=int, help="Number of molecules to generate"
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Output path for the vocabulary file ({fold} is populated automatically)",
    )

    parser.add_argument(
        "--model_file", type=str, help="File path to the saved trained model"
    )
    parser.add_argument(
        "--output_file", type=str, help="File path to save the output file"
    )
    parser.add_argument(
        "--condition_vocab_file",
        type=str,
        default=None,
        help="Condition-label vocabulary file used for graph conditioning",
    )
    parser.add_argument(
        "--graph_label_emb_dim",
        type=int,
        default=64,
        help="Label embedding size for graph conditioning",
    )
    parser.add_argument(
        "--graph_hidden_dim",
        type=int,
        default=128,
        help="Hidden size for graph-conditioning GNN layers",
    )
    parser.add_argument(
        "--graph_out_dim",
        type=int,
        default=256,
        help="Output size of the graph-conditioning encoder",
    )

    return parser


def sample_molecules_RNN(
    representation,
    rnn_type,
    embedding_size,
    hidden_size,
    n_layers,
    dropout,
    batch_size,
    sample_mols,
    vocab_file,
    model_file,
    output_file,
    conditional=False,
    conditional_emb=False,
    conditional_emb_l=True,
    conditional_dec=False,
    conditional_dec_l=True,
    conditional_h=False,
    conditioning_type="none",
    heldout_file=None,
    condition_vocab_file=None,
    graph_label_emb_dim=64,
    graph_hidden_dim=128,
    graph_out_dim=256,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    logging.info(f"cuda: {torch.cuda.is_available()}")

    if representation == "SELFIES":
        vocab = SelfiesVocabulary(vocab_file=vocab_file)
    else:
        vocab = Vocabulary(vocab_file=vocab_file)

    conditioning_type = resolve_conditioning_type(
        conditional=conditional,
        conditioning_type=conditioning_type,
    )
    conditional = conditioning_type != "none"

    heldout_dataset = None
    if conditioning_type == "graph":
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for graph-conditioned sampling"
        if condition_vocab_file is None:
            raise ValueError(
                "condition_vocab_file must be provided for graph-conditioned sampling"
            )
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
            conditioning_type=conditioning_type,
            condition_vocab_file=condition_vocab_file,
            training_split=1.0,
        )
        condition_labels = read_condition_vocab(condition_vocab_file)
        model = GraphConditionalRNN(
            vocab,
            condition_vocab_size=len(condition_labels),
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            graph_label_emb_dim=graph_label_emb_dim,
            graph_hidden_dim=graph_hidden_dim,
            graph_out_dim=graph_out_dim,
            conditional_emb=conditional_emb,
            conditional_emb_l=conditional_emb_l,
            conditional_dec=conditional_dec,
            conditional_dec_l=conditional_dec_l,
            conditional_h=conditional_h,
        )
    elif conditional:
        assert (
            heldout_file is not None
        ), "heldout_file must be provided for conditional RNN Model"
        heldout_dataset = load_dataset(
            representation=representation,
            input_file=heldout_file,
            vocab_file=vocab_file,
            conditioning_type=conditioning_type,
            condition_vocab_file=condition_vocab_file,
            training_split=1.0,
        )
        model = ConditionalRNN(
            vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=heldout_dataset.n_descriptors,
            conditional_emb=conditional_emb,
            conditional_emb_l=conditional_emb_l,
            conditional_dec=conditional_dec,
            conditional_dec_l=conditional_dec_l,
            conditional_h=conditional_h,
        )
    else:
        model = RNN(
            vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    logging.info(vocab.dictionary)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

    model.eval()

    # Erase file contents if there are any
    open(output_file, "w").close()

    with tqdm(total=sample_mols) as pbar:
        for i in range(0, sample_mols, batch_size):
            n_sequences = min(batch_size, sample_mols - i)
            descriptors = None
            if heldout_dataset is not None:
                indices = [
                    _i % len(heldout_dataset) for _i in range(i, i + n_sequences)
                ]
                items = [heldout_dataset[_i] for _i in indices]
                _, _, descriptors = heldout_dataset.collate(items)
                descriptors = descriptors.to(model.device)
            sampled_smiles, losses = model.sample(
                descriptors=descriptors, n_sequences=n_sequences, return_losses=True
            )
            df = pd.DataFrame(zip(losses, sampled_smiles), columns=["loss", "smiles"])

            write_to_csv_file(output_file, mode="w" if i == 0 else "a+", info=df)

            pbar.update(batch_size)


def main(args):
    sample_molecules_RNN(
        representation=args.representation,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        sample_mols=args.sample_mols,
        vocab_file=args.vocab_file,
        model_file=args.model_file,
        output_file=args.output_file,
        conditional=args.conditional,
        conditional_emb=args.conditional_emb,
        conditional_emb_l=args.conditional_emb_l,
        conditional_dec=args.conditional_dec,
        conditional_dec_l=args.conditional_dec_l,
        conditional_h=args.conditional_h,
        conditioning_type=args.conditioning_type,
        heldout_file=args.heldout_file,
        condition_vocab_file=args.condition_vocab_file,
        graph_label_emb_dim=args.graph_label_emb_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_out_dim=args.graph_out_dim,
    )
