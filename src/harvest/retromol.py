"""Module for parsing compounds with RetroMol."""

import json
import os
import yaml
import logging
from tqdm import tqdm

from retromol.model.rules import RuleSet, ReactionRule, MatchingRule
from retromol.io.streaming import run_retromol_stream, stream_table_rows


log = logging.getLogger(__name__)


def load_ruleset(
    reaction_rules_path: str,
    matching_rules_path: str,
) -> RuleSet:
    """
    Load reaction rules and matching rules from YAML files and create a RuleSet object.

    :param reaction_rules_path: path to YAML file containing reaction rules
    :param matching_rules_path: path to YAML file containing matching rules
    :return: RuleSet object containing the loaded reaction rules and matching rules
    """
    with open(reaction_rules_path, "r") as fo:
        reaction_rules_data = yaml.safe_load(fo)
    reaction_rules: list[ReactionRule] = [ReactionRule.from_dict(d) for d in reaction_rules_data]

    with open(matching_rules_path, "r") as fo:
        matching_rules_data = yaml.safe_load(fo)
    matching_rules: list[MatchingRule] = [MatchingRule.from_dict(d) for d in matching_rules_data]

    return RuleSet(
        match_stereochemistry=False,
        reaction_rules=reaction_rules,
        matching_rules=matching_rules,
    )


def cmd_run_retromol(
    data_path: str,
    reaction_rules_path: str,
    matching_rules_path: str,
    out_dir: str,
    *,
    smiles_col: str = "smiles",
    num_workers: int = 1,
    batch_size: int = 2000,
    pool_chunksize: int = 50,
    maxtasksperchild: int = 2000,
) -> None:
    """
    Command to run RetroMol retrosynthesis algorithm on a set of input compounds.

    :param data_path: path to input file containing SMILES strings to run retrosynthesis on
    :param reaction_rules_path: path to file containing reaction rules for RetroMol
    :param matching_rules_path: path to file containing matching rules for RetroMol
    :param out_dir: directory to save output results
    :param smiles_col: name of column in input file containing SMILES strings (default: "smiles")
    :param num_workers: number of worker processes to use for retrosynthesis (default: 1)
    :param batch_size: number of compounds to process in each batch (default: 2000)
    :param pool_chunksize: chunksize for multiprocessing pool (default: 50)
    :param maxtasksperchild: maximum number of tasks to allow each worker process to complete before restarting it (default: 2000)
    .. note:: assumes that input file has one SMILES string per line, without a header
    """
    ruleset = load_ruleset(reaction_rules_path, matching_rules_path)

    # Setup table streamer
    chunksize = 20_000
    source_iter = stream_table_rows(data_path, sep=",", chunksize=chunksize)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retromol_results.jsonl")
    jsonl_fh = open(out_path, "a", buffering=1)

    pbar = tqdm()
    for evt in run_retromol_stream(
        ruleset=ruleset,
        row_iter=source_iter,
        smiles_col=smiles_col,
        workers=num_workers,
        batch_size=batch_size,
        pool_chunksize=pool_chunksize,
        maxtasksperchild=maxtasksperchild,
    ):
        pbar.update(1)

        if evt.result is not None:
            jsonl_fh.write(json.dumps(evt.result) + "\n")

    if jsonl_fh:
        jsonl_fh.close()


