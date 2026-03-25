import logging
from copy import deepcopy
from rdkit import Chem
from clm.conditioning import resolve_conditioning_type
from clm.functions import (
    read_file,
    clean_mols,
    remove_salts_solvents,
    NeutraliseCharges,
)
from clm.datasets import Vocabulary
from clm.graph_conditional import read_graph_condition_file, write_graph_condition_file

logger = logging.getLogger(__name__)
VALID_ELEMENTS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"]


def add_args(parser):
    parser.add_argument(
        "--input-file", type=str, required=True, help="File path of smiles file"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Output smiles file path"
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=3,
        help="Minimum number of heavy atoms that a valid molecule should have",
    )
    parser.add_argument(
        "--no-neutralise",
        action="store_true",
        help="Do not neutralise charges (default False)",
    )
    parser.add_argument(
        "--valid-atoms",
        nargs="+",
        help="Elements for valid molecules (default %(default)s)",
        default=VALID_ELEMENTS,
    )
    parser.add_argument(
        "--remove-rare",
        action="store_true",
        help="Remove molecules with tokens found in <0.01%% or <10 molecules",
    )
    parser.add_argument(
        "--max-input-smiles",
        type=int,
        default=None,
        help="Maximum input smiles to read (useful for testing)",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Specifies whether to remove duplicate SMILES from the training set, "
        "identifying duplicates by inchikey. (default False)",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="none",
        help="Conditioning mode (one of: none/descriptors/graph)",
    )

    return parser


def preprocess(
    input_file,
    output_file,
    max_input_smiles=None,
    neutralise=True,
    min_heavy_atoms=3,
    valid_atoms=None,
    remove_rare=False,
    keep_duplicates=False,
    conditioning_type="none",
):
    logger.info("reading input SMILES ...")
    conditioning_type = resolve_conditioning_type(
        conditional=False,
        conditioning_type=conditioning_type,
    )

    def preprocess_smile(
        smile, neutralise=True, min_heavy_atoms=3, valid_elements=None
    ):
        mol = clean_mols([smile])[0]
        if min_heavy_atoms > 0:
            mol = remove_salts_solvents(mol, hac=min_heavy_atoms) if mol else None
        if neutralise:
            mol = NeutraliseCharges(mol) if mol else None

        elements = set([atom.GetSymbol() for atom in mol.GetAtoms()] if mol else [])
        valid_elements = set(valid_elements or VALID_ELEMENTS)
        if elements.difference(valid_elements):
            mol = None
        return mol

    if conditioning_type == "graph":
        rows = read_graph_condition_file(input_file, max_lines=max_input_smiles)
        processed_rows = []
        for row in rows:
            mol = preprocess_smile(
                row["smiles"],
                neutralise=neutralise,
                min_heavy_atoms=min_heavy_atoms,
                valid_elements=valid_atoms,
            )
            if mol is None:
                continue
            updated = deepcopy(row)
            updated["inchikey"] = Chem.inchi.MolToInchiKey(mol)
            updated["smiles"] = Chem.MolToSmiles(mol)
            processed_rows.append(updated)

        if not keep_duplicates:
            deduped = {}
            for row in processed_rows:
                deduped[row["inchikey"]] = row
            processed_rows = list(deduped.values())

        if remove_rare and processed_rows:
            vocabulary = Vocabulary(smiles=[row["smiles"] for row in processed_rows])
            n_smiles = len(processed_rows)
            keep_mask = [True] * len(processed_rows)
            for token in vocabulary.characters:
                has_token = [
                    token in vocabulary.tokenize(row["smiles"])
                    for row in processed_rows
                ]
                if sum(has_token) <= 10 or len(has_token) < 0.0001 * n_smiles:
                    keep_mask = [keep and not has for keep, has in zip(keep_mask, has_token)]
            processed_rows = [
                row for row, keep in zip(processed_rows, keep_mask) if keep
            ]

        write_graph_condition_file(output_file, processed_rows)
        return

    data = read_file(
        smiles_file=input_file,
        max_lines=max_input_smiles,
        smile_only=False,
    )

    data["mol"] = data.apply(
        lambda row: preprocess_smile(
            row["smiles"],
            neutralise=neutralise,
            min_heavy_atoms=min_heavy_atoms,
            valid_elements=valid_atoms,
        ),
        axis=1,
    )
    data = data[~data["mol"].isnull()]

    data["inchikey"] = data.apply(
        lambda row: Chem.inchi.MolToInchiKey(row["mol"]) if row["mol"] else None, axis=1
    )

    if not keep_duplicates:
        data.drop_duplicates(subset="inchikey", inplace=True)

    data["smiles"] = data.apply(
        lambda row: Chem.MolToSmiles(row["mol"]) if row["mol"] else None, axis=1
    )

    data = data.drop(["mol"], axis=1)

    if remove_rare:
        vocabulary = Vocabulary(smiles=data["smiles"].tolist())
        n_smiles = len(data)

        for i, token in enumerate(vocabulary.characters):
            has_token = data.apply(
                lambda row: token in vocabulary.tokenize(row["smiles"]), axis=1
            )
            if has_token.sum() <= 10 or len(has_token) < 0.0001 * n_smiles:
                data = data[~has_token]

    data.to_csv(output_file, sep=",", index=False)


def main(args):
    preprocess(
        input_file=args.input_file,
        output_file=args.output_file,
        max_input_smiles=args.max_input_smiles,
        neutralise=not args.no_neutralise,
        min_heavy_atoms=args.min_heavy_atoms,
        valid_atoms=args.valid_atoms,
        remove_rare=args.remove_rare,
        keep_duplicates=args.keep_duplicates,
        conditioning_type=args.conditioning_type,
    )
