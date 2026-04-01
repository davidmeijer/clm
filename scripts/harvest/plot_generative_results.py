#!/usr/env python3

import argparse
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from umap import UMAP
from rdkit import Chem

from retromol.model.result import Result
from harvest.chem import smiles_to_mol, mol_to_morgan_fp, tanimoto, mol_to_weight


# Use Helvetica font for plots
plt.rcParams["font.family"] = "Helvetica"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-smiles", type=str, required=True)
    parser.add_argument("--real-lipopeptides", type=str, required=True)
    parser.add_argument("--retromol-parsed", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def iter_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

PEPTIDE_BOND_SMARTS = "[NX3,NX4+][CH1,CH2][CX3](=[OX1])[O,N]"

def count_peptide_bonds(mol: Chem.Mol) -> int:
    """Count the number of peptide bonds in a molecule."""
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)))


def count_macrocycles(mol: Chem.Mol) -> int:
    """Count the number of macrocycles in a molecule."""
    return sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) >= 8)

def main() -> None:
    args = cli()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse target structure
    target_smiles = args.target_smiles
    target_mol = smiles_to_mol(target_smiles)
    target_wt = mol_to_weight(target_mol)
    target_fp = mol_to_morgan_fp(target_mol, radius=2, n_bits=2048, as_array=False)
    target_fp_arr = mol_to_morgan_fp(target_mol, radius=2, n_bits=2048, as_array=True)

    # Parse real lipopeptides and get their fingerprints as arrays
    real_lipopeptides_fp_arrs = []
    real_lipopeptides_wts = []
    with open(args.real_lipopeptides, "r") as f:
        f.readline() # skip header
        for line in tqdm(f):
            smiles = line.strip().split(",")[0]
            mol = smiles_to_mol(smiles)
            wt = mol_to_weight(mol)
            real_lipopeptides_wts.append(wt)
            fp_arr = mol_to_morgan_fp(mol, radius=2, n_bits=2048, as_array=True)
            real_lipopeptides_fp_arrs.append(fp_arr)
    real_lipopeptides_fp_arrs = np.array(real_lipopeptides_fp_arrs)
    print(real_lipopeptides_fp_arrs.shape)

    # Parse generated lipopeptides and get their fingerprints as arrays
    generated_fp_arrs = []
    generated_fp_arrs_high_counts = []
    generated_wts = []
    high_count_threshold = 5
    global_monomer_counts = defaultdict(list) # monomer name -> list of counts across generated molecules
    total_num_graphs = 0
    num_macrocycles = []
    num_peptide_bonds = []
    for i, d in tqdm(enumerate(iter_jsonl(args.retromol_parsed))):
        result = Result.from_dict(d)
        inchikey = result.submission.props['inchikey']
        count = int(result.submission.props['count'])
        smiles = result.submission.props['example_smiles']
        score_tc = result.submission.props['tanimoto_score']
        
        mol = smiles_to_mol(smiles)
        wt = mol_to_weight(mol)
        fp_arr = mol_to_morgan_fp(mol, radius=2, n_bits=2048, as_array=True)
        for _ in range(count):
            generated_fp_arrs.append(fp_arr)
            generated_wts.append(wt)
        if count > high_count_threshold:
            generated_fp_arrs_high_counts.append(fp_arr)

        if count >1:

            # Get monomer counts
            local_monomer_counts = {}
            for n in result.linear_readout.assembly_graph.monomer_nodes():
                if n.identified:
                    monomer_name = n.identity.name
                    if monomer_name not in local_monomer_counts:
                        local_monomer_counts[monomer_name] = 0
                    local_monomer_counts[monomer_name] += 1
            # add counts to global counts
            for monomer_name, count in local_monomer_counts.items():
                global_monomer_counts[monomer_name].append(count)
            total_num_graphs += 1

            num_macrocycles.append(count_macrocycles(mol))
            num_peptide_bonds.append(count_peptide_bonds(mol))

        if i > 1000:
            break # only process the first 1000 generated molecules for now
    generated_fp_arrs = np.array(generated_fp_arrs)
    generated_fp_arrs_high_counts = np.array(generated_fp_arrs_high_counts)
    print(generated_fp_arrs.shape)
    generated_wts = np.array(generated_wts)
    print(generated_wts.shape)
    
    # Plot histogram of weights generated compounds and show weight target compound with a vertical line
    x_min = 0
    x_max = max(max(generated_wts), max(real_lipopeptides_wts), target_wt)
    x = np.linspace(x_min, x_max, 1000)
    gen_kde = gaussian_kde(generated_wts)
    real_kde = gaussian_kde(real_lipopeptides_wts)
    plt.plot(x, gen_kde(x), label=f"Weights generated structures")
    plt.plot(x, real_kde(x), label=f"Weights real lipopeptides")
    plt.axvline(target_wt, color="red", linestyle="--", label=f"Weight target structure ({target_wt:.1f} g/mol)")
    plt.xlim(0, 2500)
    plt.xlabel("Molecular weight (g/mol)")
    plt.ylabel("Density")
    plt.title("Distribution of molecular weights of generated lipopeptides")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.output_dir, "weight_density.png"), dpi=300)
    plt.close()

    # Plot 2D scatter plot of dimensionality reduced fingerprints of generated compounds and real lipopeptides, and show target compound as a star
    all_fps = np.vstack([generated_fp_arrs, real_lipopeptides_fp_arrs, generated_fp_arrs_high_counts, target_fp_arr])
    print(all_fps.shape)
    pca = PCA(n_components=2)
    all_proj = pca.fit_transform(all_fps)
    print(all_proj.shape)
    # Split back out
    n_generated = generated_fp_arrs.shape[0]
    n_real = real_lipopeptides_fp_arrs.shape[0]
    n_high_counts = generated_fp_arrs_high_counts.shape[0]
    generated_proj = all_proj[:n_generated]
    real_proj = all_proj[n_generated:n_generated + n_real]
    high_counts_proj = all_proj[n_generated + n_real:n_generated + n_real + n_high_counts]
    target_proj = all_proj[-1]  # shape (2,)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Main scatter
    ax.scatter(generated_proj[:, 0], generated_proj[:, 1], s=10, color="blue", edgecolor="black", alpha=0.3, label="Generated lipopeptides", zorder=2, marker="s")
    ax.scatter(real_proj[:, 0],real_proj[:, 1], s=10, color="red", alpha=0.5, edgecolor="black", label="Real lipopeptides", zorder=3, marker="o")
    ax.scatter(high_counts_proj[:, 0], high_counts_proj[:, 1], s=50, color="cyan", edgecolor="black", label=f"Generated lipopeptides (count >{high_count_threshold})", zorder=4, marker="D")
    # Plot target last so it is always on top
    ax.scatter(target_proj[0], target_proj[1], marker="*", s=150, color="gold", edgecolor="black", label="Target compound", zorder=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    # ax.set_title("2D density of fingerprint space")
    ax.legend(loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    # Create marginal axes
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=1.0, pad=0.1, sharex=ax)
    ax_right = divider.append_axes("right", size=1.0, pad=0.1, sharey=ax)

    # ----- Top density: PC1 -----
    x_min = min(generated_proj[:, 0].min(), real_proj[:, 0].min(), high_counts_proj[:, 0].min(), target_proj[0])
    x_max = max(generated_proj[:, 0].max(), real_proj[:, 0].max(), high_counts_proj[:, 0].max(), target_proj[0])
    x_grid = np.linspace(x_min, x_max, 500)
    gen_kde_x = gaussian_kde(generated_proj[:, 0])
    real_kde_x = gaussian_kde(real_proj[:, 0])
    gen_density_x = gen_kde_x(x_grid)
    real_density_x = real_kde_x(x_grid)
    ax_top.plot(x_grid, gen_density_x, color="blue", linewidth=1.5)
    ax_top.fill_between(x_grid, gen_density_x, color="blue", alpha=0.25)
    ax_top.plot(x_grid, real_density_x, color="red", linewidth=1.5)
    ax_top.fill_between(x_grid, real_density_x, color="red", alpha=0.25)
    ax_top.axvline(target_proj[0], color="gold", linestyle="--", linewidth=2)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["left"].set_visible(False)

    # ----- Right density: PC2 -----
    y_min = min(generated_proj[:, 1].min(), real_proj[:, 1].min(), high_counts_proj[:, 1].min(), target_proj[1])
    y_max = max(generated_proj[:, 1].max(), real_proj[:, 1].max(), high_counts_proj[:, 1].max(), target_proj[1])
    y_grid = np.linspace(y_min, y_max, 500)
    gen_kde_y = gaussian_kde(generated_proj[:, 1])
    real_kde_y = gaussian_kde(real_proj[:, 1])
    gen_density_y = gen_kde_y(y_grid)
    real_density_y = real_kde_y(y_grid)
    ax_right.plot(gen_density_y, y_grid, color="blue", linewidth=1.5)
    ax_right.fill_betweenx(y_grid, 0, gen_density_y, color="blue", alpha=0.25)
    ax_right.plot(real_density_y, y_grid, color="red", linewidth=1.5)
    ax_right.fill_betweenx(y_grid, 0, real_density_y, color="red", alpha=0.25)
    ax_right.axhline(target_proj[1], color="gold", linestyle="--", linewidth=2)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fingerprint_pca_density.png"), dpi=300)
    plt.close()

    # for every monomer in global_monomer_counts, plot average + std, sort monomers lexographically, plot in barplot
    # for every monomer, check how many counts are in there, list should have length equal to total_num_graphs, if not padd with zeros
    for monomer in global_monomer_counts:
        counts = global_monomer_counts[monomer]
        if len(counts) < total_num_graphs:
            counts.extend([0] * (total_num_graphs - len(counts)))
            global_monomer_counts[monomer] = counts
    monomers = sorted(global_monomer_counts.keys())
    means = [np.mean(global_monomer_counts[m]) for m in monomers]
    stds = [np.std(global_monomer_counts[m]) for m in monomers]
    for monomer, mean, std in zip(monomers, means, stds):
        print(f"{monomer}: {mean:.2f} ± {std:.2f}")
    plt.figure(figsize=(12, 6))
    plt.bar(monomers, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
    # set y lim
    plt.ylim(0, max(means) + 2)
    for i, (mean, std) in enumerate(zip(means, stds)):
        if mean >= 0.05:
            plt.text(
                i,
                mean + std + 0.02 * max(means),   # a bit above the error bar
                f"{mean:.2f} ± {std:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    plt.xticks(rotation=90)
    plt.xlabel("Monomer")
    plt.ylabel("Average count in generated molecules")
    plt.title("Average count of each monomer in generated molecules")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "monomer_counts.png"), dpi=300)
    plt.close()

    # Plot barplot of mean + std of number of macrocycles and peptide bonds in generated molecules
    mean_macrocycles = np.mean(num_macrocycles)
    std_macrocycles = np.std(num_macrocycles)
    mean_peptide_bonds = np.mean(num_peptide_bonds)
    std_peptide_bonds = np.std(num_peptide_bonds)
    print(f"Macrocycles: {mean_macrocycles:.2f} ± {std_macrocycles:.2f}")
    print(f"Peptide bonds: {mean_peptide_bonds:.2f} ± {std_peptide_bonds:.2f}")
    plt.figure(figsize=(6, 6))
    plt.bar(["Macrocycles", "Peptide bonds"], [mean_macrocycles, mean_peptide_bonds], yerr=[std_macrocycles, std_peptide_bonds], capsize=5, color=["lightcoral", "lightseagreen"], edgecolor="black")
    for i, (mean, std) in enumerate([(mean_macrocycles, std_macrocycles), (mean_peptide_bonds, std_peptide_bonds)]):
        plt.text(
            i,
            mean + std + 0.02 * max(mean_macrocycles, mean_peptide_bonds),   # a bit above the error bar
            f"{mean:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylim(0, max(mean_macrocycles + std_macrocycles, mean_peptide_bonds + std_peptide_bonds) + 1)
    # plt.title("Average number of macrocycles and peptide bonds in generated molecules")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "macrocycles_peptide_bonds.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
