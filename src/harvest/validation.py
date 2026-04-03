from pathlib import Path
import logging
import os
import json
from collections import Counter

from tqdm import tqdm
from rdkit import RDLogger

from harvest.loader import prep_clm
from harvest.chem import smiles_to_mol, mol_to_morgan_fp, tanimoto, mol_to_inchikey_conn
from clm.graph_conditional import (
    normalize_condition_graph,
    read_condition_vocab,
    graph_to_condition_graph,
    ConditionGraphCollate,
)


RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings about invalid SMILES


log = logging.getLogger(__name__)


def iter_jsonl(path: str):
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


def cmd_validate(
    model_dir: str,
    out_dir: str,
    device: str,
    test_size: int = 1000,
    sample_size: int = 100_000,
    batch_size: int = 1000,
    fp_radius: int = 3,
    fp_n_bits: int = 2048,
) -> None:
    """
    Validate a trained CLM using cross-validation splits.

    :param model_dir: Path to the directory containing the trained CLM with cross-validation splits.
    :param out_dir: Path to the directory where validation results will be saved.
    :param device: Device to run validation on (e.g., 'cuda:0' or 'cpu').
    """
    os.makedirs(out_dir, exist_ok=True)

    model_cfgs = prep_clm(model_dir=Path(model_dir), eval=True)
    log.info(f"Found {len(model_cfgs)} model configurations for validation.")

    out_file_path = os.path.join(out_dir, "validation_results.tsv")

    with open(out_file_path, "w") as out_file:
        # write header
        out_file.write("model_cfg_idx\tdata_idx\ttarget_smiles\ttrue_generated\tvalid_percentage\tmost_common_score\tmost_common_count\texample_smiles_most_common\thighest_score\n")

        for model_cfg_idx, model_cfg in enumerate(model_cfgs):
            log.info(f"Validating model configuration {model_cfg_idx + 1}/{len(model_cfgs)}")
            model = model_cfg.load_model(device=device)
            model.eval()  # just in case

            condition_labels = read_condition_vocab(model_cfg.condition_vocab_path)
            name_to_idx = {name: idx for idx, name in enumerate(condition_labels)}

            test_data_file = model_cfg.test_path
            for data_idx, data in enumerate(tqdm(iter_jsonl(test_data_file))):
                if data_idx >= test_size:
                    break

                target_smiles = data.get("smiles")
                if not target_smiles:
                    log.warning(f"No SMILES found in test data at index {data_idx} in file {test_data_file}. Skipping.")
                    continue

                target_mol = smiles_to_mol(target_smiles)
                target_fp = mol_to_morgan_fp(target_mol, radius=fp_radius, n_bits=fp_n_bits)

                condition_graph = data.get("condition_graph")
                if not condition_graph:
                    log.warning(f"No condition graph found in test data at index {data_idx} in file {test_data_file}. Skipping.")
                    continue

                normalized_graph = normalize_condition_graph(condition_graph)

                # check that all node names are in the vocab, if not replace with <UNK> and log a warning
                for n in normalized_graph['nodes']:
                    name = n['name']
                    if name not in name_to_idx:
                        log.warning(f"Node name '{name}' not found in condition vocab")
                        n['name'] = '<UNK>'

                # try again, should be in vocab now
                for n in normalized_graph['nodes']:
                    name = n['name']
                    assert name in name_to_idx, f"Node name '{name}' not found in condition vocab even after replacement!"
                
                valid = 0
                ik_to_frequency = Counter()
                ik_to_repr_smi = {}
                ik_to_repr_score = {}

                num_batches = (sample_size + batch_size - 1) // batch_size
                for i in range(num_batches):
                    to_sample = min(batch_size, sample_size - i * batch_size)

                    graph_batch = ConditionGraphCollate()(
                        [
                            graph_to_condition_graph(normalized_graph, name_to_idx)
                            for _ in range(to_sample)
                        ]
                    )

                    generated_smiles = model.sample(
                        descriptors=graph_batch,
                        n_sequences=to_sample,
                        max_len=250,
                    )

                    for gen_smi in generated_smiles:
                        try: 
                            gen_mol = smiles_to_mol(gen_smi)
                            if gen_mol is None:
                                raise ValueError(f"Invalid SMILES: {gen_smi}")
                            valid += 1
                            gen_fp = mol_to_morgan_fp(gen_mol, radius=fp_radius, n_bits=fp_n_bits)
                            ik_conn = mol_to_inchikey_conn(gen_mol)
                            score = tanimoto(target_fp, gen_fp)
                            ik_to_frequency[ik_conn] += 1
                            if ik_conn not in ik_to_repr_smi:
                                ik_to_repr_smi[ik_conn] = gen_smi
                                ik_to_repr_score[ik_conn] = score
                        except Exception:
                            continue
                    
                # true compound every generated? any score 1.0
                true_generated = any(score == 1.0 for score in ik_to_repr_score.values())
                log.info(f"Test data index {data_idx}: True compound generated: {true_generated}")
                # % valid smiles
                valid_percentage = valid / sample_size
                log.info(f"Test data index {data_idx}: Valid SMILES percentage: {valid_percentage:.2%}")
                # Tc between ground truth structure and most frequently generated; take average if it is a tie  
                most_common_ik, most_common_count = ik_to_frequency.most_common(1)[0] if ik_to_frequency else (None, 0)
                most_common_score = ik_to_repr_score.get(most_common_ik, 0.0) if most_common_ik else 0.0
                most_common_smiles = ik_to_repr_smi.get(most_common_ik, "N/A") if most_common_ik else "N/A"
                log.info(f"Test data index {data_idx}: Most common Tanimoto score: {most_common_score:.4f} (count: {most_common_count})")
                log.info(f"Test data index {data_idx}: Example SMILES for most common score: {most_common_smiles}")
                # highest recorded score
                highest_score = max(ik_to_repr_score.values()) if ik_to_repr_score else 0.0
                log.info(f"Test data index {data_idx}: Highest Tanimoto score: {highest_score:.4f}")

                # write results to file
                out_file.write(f"{model_cfg_idx}\t{data_idx}\t{target_smiles}\t{true_generated}\t{valid_percentage:.4f}\t{most_common_score:.4f}\t{most_common_count}\t{most_common_smiles}\t{highest_score:.4f}\n")

                # flush so results are written incrementally
                out_file.flush()


    # create for every enum the splits
    # - generate 100k samples for every test item; compute metrics from generated samples:
    #     * true compound every generated?
    #     * % valid smiles
    #     * Tc between ground truth structure and most frequently generated; take average if it is a tie
