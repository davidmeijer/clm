import argparse
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml
from tqdm import tqdm

from retromol.model.result import Result
from retromol.model.rules import RuleSet, ReactionRule, MatchingRule
from retromol.model.reaction_graph import ReactionGraph, MolNode, MolIdentity
from retromol.io.streaming import run_retromol_stream, stream_table_rows



def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True, nargs="+", help="List of files to process")
    parser.add_argument("--matching-rules", required=True, help="Path to YAML file containing matching rules for RetroMol")
    parser.add_argument("--reaction-rules", required=True, help="Path to YAML file containing reaction rules for RetroMol")
    return parser.parse_args()


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


def main() -> None:
    args = cli()

    # load dataframes
    dfs = [pd.read_csv(file, sep="\t") for file in args.files]
    df_labels = [
        "H3-1k",
        "H3-10k",
        "H3-100k",
    ]
    
    # plots: 
    # - distributions of % valid_percentage
    # - barplot of % true_generated == TRUE
    # - distribution of most_common_count
    # - distributions of most_common_score
    # - distributions of highest_score

    # add more space between plots horizontally
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.6))
    

    # 1) valid_percentage
    all_valid = np.concatenate([df["valid_percentage"].dropna().to_numpy() for df in dfs])
    x_valid = np.linspace(all_valid.min(), all_valid.max(), 500)
    for df_idx, df in enumerate(dfs):
        vals = df["valid_percentage"].dropna().to_numpy()
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            axs[0].plot(x_valid, kde(x_valid), label=df_labels[df_idx])
    axs[0].set_yticks([])
    axs[0].set_xlabel("Valid generated (ratio)")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    # 2) % true_generated == TRUE
    for df_idx, df in enumerate(dfs):
        true_generated_percentage = (df["true_generated"] == True).mean() * 100
        axs[1].bar(df_labels[df_idx], true_generated_percentage)
    axs[1].set_xlabel("Dataset")
    axs[1].set_ylabel("Target ever generated (%)")

    # 3) most_common_count (log x-axis)
    all_counts = np.concatenate([df["most_common_count"].dropna().to_numpy() for df in dfs])
    all_counts = all_counts[all_counts > 0]
    x_counts = np.logspace(np.log10(all_counts.min()), np.log10(all_counts.max()), 500)
    log_x_counts = np.log10(x_counts)

    for df_idx, df in enumerate(dfs):
        vals = df["most_common_count"].dropna().to_numpy()
        vals = vals[vals > 0]
        if len(vals) > 1:
            kde = gaussian_kde(np.log10(vals))
            axs[2].plot(x_counts, kde(log_x_counts), label=df_labels[df_idx])
    axs[2].set_yticks([])
    axs[2].set_xscale("log")
    axs[2].set_xlabel("Count most frequent")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    # 4) most_common_score
    all_mcs = np.concatenate([df["most_common_score"].dropna().to_numpy() for df in dfs])
    x_mcs = np.linspace(all_mcs.min(), all_mcs.max(), 500)
    for df_idx, df in enumerate(dfs):
        vals = df["most_common_score"].dropna().to_numpy()
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            axs[3].plot(x_mcs, kde(x_mcs), label=df_labels[df_idx])
    axs[3].set_yticks([])
    axs[3].set_xlabel("Score most frequent (T$_c$)")
    axs[3].set_ylabel("Density")
    axs[3].legend()

    # 5) highest_score
    all_hs = np.concatenate([df["highest_score"].dropna().to_numpy() for df in dfs])
    x_hs = np.linspace(0, all_hs.max(), 500)
    for df_idx, df in enumerate(dfs):
        vals = df["highest_score"].dropna().to_numpy()
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            axs[4].plot(x_hs, kde(x_hs), label=df_labels[df_idx])
    # set x-axis 0-1
    axs[4].set_xlim(0, 1)
    axs[4].set_yticks([])
    axs[4].set_xlabel("Highest found score (T$_c$)")
    axs[4].set_ylabel("Density")
    axs[4].legend()

    plt.tight_layout()
    # save plot to downloads folder
    plt.savefig(os.path.expanduser("~/Downloads/H003_evaluation_plots.png"), dpi=300)

    # loop over the SMILES and do RetroMol round trip coverage analysis
    ruleset = load_ruleset(args.reaction_rules, args.matching_rules)
    for path_idx, path in enumerate(args.files[::-1]):
        pbar = tqdm()

        target_covs: list[float] = []
        target_monomers: list[list[str]] = []

        gen_covs: list[float] = []
        gen_monomers: list[list[str]] = []

        # parse target smiles
        source_iter = stream_table_rows(path, sep="\t", chunksize=20_000)
        for evt in run_retromol_stream(
            ruleset=ruleset,
            row_iter=source_iter,
            smiles_col="target_smiles",
            workers=min(4, os.cpu_count() - 1 or 1),
            batch_size=2000,
            pool_chunksize=50,
            maxtasksperchild=2000,
        ):
            pbar.update(1)
            result: Result = Result.from_dict(evt.result)
            target_covs.append(result.calculate_coverage())
            
            nodes: list[MolNode] = result.linear_readout.assembly_graph.monomer_nodes()
            nids = []
            for n in nodes:
                if n.identified:
                    nid = n.identity.name
                    nids.append(nid)
            target_monomers.append(nids)
        
        # parse generated smiles
        source_iter = stream_table_rows(path, sep="\t", chunksize=20_000)
        for evt in run_retromol_stream(
            ruleset=ruleset,
            row_iter=source_iter,
            smiles_col="example_smiles_most_common",
            workers=min(4, os.cpu_count() - 1 or 1),
            batch_size=2000,
            pool_chunksize=50,
            maxtasksperchild=2000,
        ):
            pbar.update(1)
            result = Result.from_dict(evt.result)
            gen_covs.append(result.calculate_coverage())

            nodes: list[MolNode] = result.linear_readout.assembly_graph.monomer_nodes()
            nids = []
            for n in nodes:
                if n.identified:
                    nid = n.identity.name
                    nids.append(nid)
            gen_monomers.append(nids)

        # calculate correlation between target_covs and gen_covs
        target_covs = np.array(target_covs)
        gen_covs = np.array(gen_covs)
        correlation = np.corrcoef(target_covs, gen_covs)[0, 1]
        print(f"Correlation between target coverage and generated coverage for {df_labels[::-1][path_idx]}: {correlation:.4f}")

        # plot target_covs vs gen_covs in scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(target_covs, gen_covs, alpha=0.5)
        # add line y=x
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("Target coverage")
        plt.ylabel("Generated coverage")
        plt.title(f"Coverage correlation for {df_labels[::-1][path_idx]} (r={correlation:.4f})")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.expanduser(f"~/Downloads/H003_coverage_correlation_{df_labels[::-1][path_idx]}.png"), dpi=300)
        plt.close()

        # calculate overlap between two lists of string (duplicate items in there calculate how many are in common)
        scores = []
        for tm, gm in zip(target_monomers, gen_monomers):
            # take counts of each monomer in target and generated into account
            tm_counts = {}
            for m in tm:
                if m not in tm_counts:
                    tm_counts[m] = 0
                tm_counts[m] += 1
            gm_counts = {}
            for m in gm:
                if m not in gm_counts:
                    gm_counts[m] = 0
                gm_counts[m] += 1
            # calculate as how close the generated monomer counts are to the target monomer counts, using the formula: score = 1 - (sum of absolute differences in counts) / (sum of target counts)
            target_total = sum(tm_counts.values())
            if target_total == 0:
                score = 1.0 if len(gm_counts) == 0 else 0.0
            else:
                abs_diff_sum = 0
                for m, count in tm_counts.items():
                    gm_count = gm_counts.get(m, 0)
                    abs_diff_sum += abs(count - gm_count)
                score = 1 - (abs_diff_sum / target_total)
            scores.append(score)
        # plot distribution of scores
        plt.figure(figsize=(6, 4))
        x_scores = np.linspace(0, 1, 500)
        kde = gaussian_kde(scores)
        plt.plot(x_scores, kde(x_scores))
        plt.xlabel("Monomer overlap score")
        plt.ylabel("Density")
        plt.title(f"Monomer overlap for {df_labels[::-1][path_idx]} (mean={np.mean(scores):.4f})")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.expanduser(f"~/Downloads/H003_monomer_overlap_{df_labels[::-1][path_idx]}.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
