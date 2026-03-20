#!/usr/bin/env python3
"""Evaluate conditional model on single compound."""

import argparse
import os
import inspect
from collections import Counter

import yaml
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger

from harvest.retromol import load_ruleset
from harvest.chem import smiles_to_mol, mol_to_weight, mol_to_morgan_fp, tanimoto, mol_to_inchikey_conn
from harvest.sample import prep_clm

from retromol.model.submission import Submission
from retromol.pipelines.parsing import run_retromol_with_timeout

from fingerprint_parsed_compounds import fingerprint_result

import matplotlib.pyplot as plt


RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings


RADIUS_MORGAN = 2
FP_SIZE_MORGAN = 2048


PEPTIDE_BOND_SMARTS = "[NX3,NX4+][CH1,CH2][CX3](=[OX1])[O,N]"


def count_peptide_bonds(mol: Chem.Mol) -> int:
    """Count the number of peptide bonds in a molecule."""
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)))


def count_macrocycles(mol: Chem.Mol) -> int:
    """Count the number of macrocycles in a molecule."""
    return sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) >= 8)


def has_serine_aspartic_acid_depsipeptide_bond(mol: Chem.Mol) -> bool:
    """Check if molecule contains a depsipeptide bond between serine and aspartic acid."""
    DEPSIPEPTIDE_BOND_SMARTS = "C(=O)C(N)COC(=O)C(N)CC(=O)O"
    return mol.HasSubstructMatch(Chem.MolFromSmarts(DEPSIPEPTIDE_BOND_SMARTS))


def cli() -> argparse.Namespace:
    """
    Command line interface.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="path to model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--target", type=str, required=True, help="target to evaluate")
    parser.add_argument("--matching-rules", type=str, required=True, help="path to matching rules file")
    parser.add_argument("--reaction-rules", type=str, required=True, help="path to reaction rules file")
    parser.add_argument("--creativity", type=float, default=0.8, help="hallucination level for generation (default: 0.8)")
    parser.add_argument("--include-edges", action="store_true", help="whether to include edge features in the fingerprint")
    parser.add_argument("--num-samples", type=int, default=1000, help="number of compounds to generate (default: 1000)")
    return parser.parse_args()


def _sample_device_kwargs(sample_fn, device: torch.device) -> dict:
    """
    Return device-related kwargs supported by the CLM sample method.

    This guards against version differences where `sample` may accept
    `device`, `use_cuda`, or `cuda`, or nothing at all.
    """
    try:
        params = inspect.signature(sample_fn).parameters
    except (TypeError, ValueError):
        return {}

    if "device" in params:
        return {"device": device}
    if "use_cuda" in params:
        return {"use_cuda": device.type == "cuda"}
    if "cuda" in params:
        return {"cuda": device.type == "cuda"}
    return {}


def _sync_model_device(model: torch.nn.Module, device: torch.device) -> None:
    """
    Best-effort sync for models that track a `device` attribute internally.
    """
    if hasattr(model, "device"):
        try:
            model.device = device
        except Exception:
            pass


def sample_unconditional_clm(
    model,
    num_samples: int,
    *,
    batch_size: int = 1024,
    max_len: int = 250,
    device: str | torch.device
):
    """
    Sample unconditional generative model.
    
    :param model: CLM model to sample from
    :param num_samples: number of samples to generate
    :param batch_size: batch size for sampling
    :param max_len: maximum length of generated sequences
    :param device: device to run sampling on
    :return: generated SMILES strings
    """
    model.eval()  # loading sets to eval, but ensure in eval mode

    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    _sync_model_device(model, device)
    device_kwargs = _sample_device_kwargs(model.sample, device)

    with torch.inference_mode():
        remaining = num_samples

        while remaining > 0:
            this_batch = min(batch_size, remaining)

            batch_smiles = model.sample(
                n_sequences=this_batch,
                max_len=max_len,
                return_smiles=True,
                return_losses=False,
                descriptors=None,
                **device_kwargs,
            )
            
            # Stream generated SMILES strings
            for s in batch_smiles:
                yield s

            remaining -= this_batch


def sample_conditional_clm(model, c, num_samples, device):
    batch_size = 1024
    max_len = 250

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    _sync_model_device(model, device)
    device_kwargs = _sample_device_kwargs(model.sample, device)
    base = torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        remaining = num_samples
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            desc = base.expand(this_batch, -1).contiguous()
            batch_smiles = model.sample(
                descriptors=desc,
                max_len=max_len,
                return_smiles=True,
                return_losses=False,
                **device_kwargs,
            )
            for s in batch_smiles:
                yield s
            remaining -= this_batch


def main() -> None:
    """Main function."""
    args = cli()
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure creativity is between 0 and 1
    creativity = max(0.0, min(args.creativity, 1.0))

    # Parse compound and generate fingerprint
    ruleset = load_ruleset(args.reaction_rules, args.matching_rules)
    print(ruleset)
    submission = Submission(smiles=args.target)
    result = run_retromol_with_timeout(submission, ruleset)
    print(result)

    # Setup fingerprint design based on matching rules
    with open(args.matching_rules, "r") as f:
        matching_rules = yaml.safe_load(f)
    assignment = [r["name"] for r in matching_rules]
    print(f"found {len(assignment)} monomers in rules")
    cs = [c for c in fingerprint_result(result, assignment, include_edges=args.include_edges, num_samples_edge_features=1)]
    c = cs[0]
    print(len(c))

    # featurize target compound
    target_mol = smiles_to_mol(args.target)
    target_fp = mol_to_morgan_fp(target_mol, radius=RADIUS_MORGAN, n_bits=FP_SIZE_MORGAN)

    # Generate compounds for fingerprint
    model_configs = prep_clm(model_dir=args.model_dir, eval=True)
    device = torch.device("cpu")
    num_models = len(model_configs)
    samples_per_model = args.num_samples // num_models
    remainder = args.num_samples % num_models
    models = []
    for model_idx, model_config in enumerate(model_configs):
        model = model_config.load_model(device=device)
        models.append(model)

    valid = 0
    inchikey_to_smiles = {}
    inchikey_to_weight = {}
    inchikey_to_tc = {}
    inchikey_to_counts = Counter()
    inchikey_to_peptide_bonds = {}
    pbar = tqdm(total=args.num_samples, desc="Sampling compounds")
    for model_idx, model in enumerate(models):
        num_samples = samples_per_model + (remainder if model_idx == num_models - 1 else 0)
        for s in sample_conditional_clm(model, c, num_samples, device):

            pbar.update(1)
            
            try:
                mol = smiles_to_mol(s)
                if mol is None:
                    continue
            except:
                continue
            valid += 1

            inchikey = mol_to_inchikey_conn(mol)
            if inchikey not in inchikey_to_smiles:
                inchikey_to_smiles[inchikey] = s
            if inchikey not in inchikey_to_weight:
                weight = mol_to_weight(mol)
                inchikey_to_weight[inchikey] = weight
            if inchikey not in inchikey_to_tc:
                tc = tanimoto(mol_to_morgan_fp(mol, radius=RADIUS_MORGAN, n_bits=FP_SIZE_MORGAN), target_fp)
                inchikey_to_tc[inchikey] = tc
            if inchikey not in inchikey_to_peptide_bonds:
                peptide_bonds = count_peptide_bonds(mol)
                inchikey_to_peptide_bonds[inchikey] = peptide_bonds
            inchikey_to_counts[inchikey] += 1

    perc_valid = valid / args.num_samples * 100
    num_unique = len(inchikey_to_smiles)
    max_tc = max(inchikey_to_tc.values()) if inchikey_to_tc else 0.0
    mean_tc = sum(inchikey_to_tc.values()) / num_unique if num_unique > 0 else 0.0
    # get all weights, if inchikey count is >1 add weight multiple times
    all_weights = []
    for inchikey, weight in inchikey_to_weight.items():
        count = inchikey_to_counts[inchikey]
        all_weights.extend([weight] * count)
    mean_weight = sum(all_weights) / len(all_weights) if all_weights else 0.0
    median_weight = sorted(all_weights)[len(all_weights) // 2] if all_weights else 0.0
    sd_weight = (sum((x - mean_weight) ** 2 for x in all_weights) / len(all_weights)) ** 0.5 if all_weights else 0.0
    # get all num peptide bonds, if inchikey count is >1 add num peptide bonds multiple times
    all_peptide_bonds = []
    for inchikey, peptide_bonds in inchikey_to_peptide_bonds.items():
        count = inchikey_to_counts[inchikey]
        all_peptide_bonds.extend([peptide_bonds] * count)
    mean_peptide_bonds = sum(all_peptide_bonds) / len(all_peptide_bonds) if all_peptide_bonds else 0.0
    median_peptide_bonds = sorted(all_peptide_bonds)[len(all_peptide_bonds) // 2] if all_peptide_bonds else 0.0
    sd_peptide_bonds = (sum((x - mean_peptide_bonds) ** 2 for x in all_peptide_bonds) / len(all_peptide_bonds)) ** 0.5 if all_peptide_bonds else 0.0
    # get all num macrocycles, if inchikey count is >1 add num macrocycles multiple times
    all_macrocycles = []
    for inchikey, weight in inchikey_to_weight.items():
        count = inchikey_to_counts[inchikey]
        mol = smiles_to_mol(inchikey_to_smiles[inchikey])
        num_macrocycles = count_macrocycles(mol)
        all_macrocycles.extend([num_macrocycles] * count)
    mean_macrocycles = sum(all_macrocycles) / len(all_macrocycles) if all_macrocycles else 0.0
    median_macrocycles = sorted(all_macrocycles)[len(all_macrocycles) // 2] if all_macrocycles else 0.0
    sd_macrocycles = (sum((x - mean_macrocycles) ** 2 for x in all_macrocycles) / len(all_macrocycles)) ** 0.5 if all_macrocycles else 0.0
    # check for all inchikeys if they have serine-aspartic acid depsipeptide bond, if inchikey count is >1 add result multiple times
    all_depsipeptide_bonds = []
    for inchikey in inchikey_to_smiles.keys():
        count = inchikey_to_counts[inchikey]
        mol = smiles_to_mol(inchikey_to_smiles[inchikey])
        has_depsipeptide_bond = has_serine_aspartic_acid_depsipeptide_bond(mol)
        all_depsipeptide_bonds.extend([has_depsipeptide_bond] * count)
    num_with_depsipeptide_bond = sum(all_depsipeptide_bonds)
    perc_with_depsipeptide_bond = num_with_depsipeptide_bond / len(all_depsipeptide_bonds) * 100 if all_depsipeptide_bonds else 0.0

    print(f"Valid compounds: {valid}/{args.num_samples} ({perc_valid:.2f}%)")
    print(f"Unique compounds: {num_unique}")
    print(f"Max Tanimoto similarity: {max_tc:.4f}")
    print(f"Mean Tanimoto similarity: {mean_tc:.4f}")
    print(f"Target weight: {mol_to_weight(target_mol):.2f}")
    print(f"Mean weight: {mean_weight:.2f}")
    print(f"Median weight: {median_weight:.2f}")
    print(f"SD weight: {sd_weight:.2f}")
    print(f"Mean peptide bonds: {mean_peptide_bonds:.2f}")
    print(f"Median peptide bonds: {median_peptide_bonds:.2f}")
    print(f"SD peptide bonds: {sd_peptide_bonds:.2f}")
    print(f"Mean macrocycles: {mean_macrocycles:.2f}")
    print(f"Median macrocycles: {median_macrocycles:.2f}")
    print(f"SD macrocycles: {sd_macrocycles:.2f}")

    # Generate evaluation report figure

    # do horizontal line of plots, from left to right:
    # - weight distribution with target weight as vertical line (density plot, fill in plot) (min x-axis weight should be 0)
    # - Tc distribution against target (density plot, fill in plot), highlight max Tc with vertical line, min-max x-axis is 0-1
    # - barplot with two bars: num of valid compounds (percentage), num of unique compounds (percentage)
    # - barplot with two bars for peptide bonds: mean peptide bonds with error bars, mean macrocycles with error bars

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax_weight, ax_tc, ax_counts, ax_struct = axes

    # weight distribution
    target_weight = mol_to_weight(target_mol)
    if all_weights:
        ax_weight.hist(all_weights, bins=30, density=True, alpha=0.6, color="blue")
        ax_weight.axvline(target_weight, color="red", linestyle="--", label="Target weight")
    else:
        ax_weight.text(0.5, 0.5, "No valid compounds", ha="center", va="center", transform=ax_weight.transAxes)
    ax_weight.set_title("Weight Distribution")
    ax_weight.set_xlabel("Molecular Weight")
    ax_weight.set_ylabel("Density")
    ax_weight.set_xlim(left=0)
    if all_weights:
        ax_weight.legend()

    # Tanimoto distribution
    tc_values = list(inchikey_to_tc.values())
    if tc_values:
        ax_tc.hist(tc_values, bins=30, density=True, alpha=0.6)
        ax_tc.axvline(max_tc, linestyle="--", linewidth=2, label=f"Max: {max_tc:.2f}")
    else:
        ax_tc.text(0.5, 0.5, "No valid compounds", ha="center", va="center", transform=ax_tc.transAxes)
    ax_tc.set_title("Tanimoto similarity to target")
    ax_tc.set_xlabel("Tanimoto similarity")
    ax_tc.set_ylabel("Density")
    ax_tc.set_xlim(0, 1)
    if tc_values:
        ax_tc.legend()

    # Valid / unique percentages
    valid_pct = 100.0 * valid / args.num_samples if args.num_samples > 0 else 0.0
    unique_pct = 100.0 * num_unique / args.num_samples if args.num_samples > 0 else 0.0
    depsipeptide_pct = perc_with_depsipeptide_bond
    count_labels = ["Valid", "Unique", "Target cycl."]
    count_values = [valid_pct, unique_pct, depsipeptide_pct]
    bars = ax_counts.bar(count_labels, count_values)
    ax_counts.set_title("Generation success")
    ax_counts.set_ylabel("Percentage of samples (%)")
    ax_counts.set_ylim(0, 100)
    for bar, value in zip(bars, count_values):
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    # Structural summary: peptide bonds and macrocycles
    struct_labels = ["Peptide bonds", "Macrocycles"]
    struct_means = [mean_peptide_bonds, mean_macrocycles]
    struct_sds = [sd_peptide_bonds, sd_macrocycles]
    bars = ax_struct.bar(struct_labels, struct_means, yerr=struct_sds, capsize=6)
    ax_struct.set_title("Structural features")
    ax_struct.set_ylabel("Mean count")
    for bar, mean_val, sd_val in zip(bars, struct_means, struct_sds):
        ax_struct.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + max(sd_val, 0.05),
            f"{mean_val:.2f}±{sd_val:.2f}",
            ha="center",
            va="bottom",
        )

    # Overall title
    fig.suptitle(
        (
            f"Conditional CLM evaluation\n"
            f"Target: {args.target[:60]}{'...' if len(args.target) > 60 else ''}\n"
            f"Valid {valid}/{args.num_samples} ({perc_valid:.1f}%), "
            f"Unique {num_unique}, Mean Tc {mean_tc:.3f}, Max Tc {max_tc:.3f}"
        ),
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_png = os.path.join(args.output_dir, "evaluation_report.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    


if __name__ == "__main__":
    main()
