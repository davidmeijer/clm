#!/usr/bin/env python3
import argparse
import json
import os

from retromol.model.result import Result


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism4-jsonl", required=True, help="Path to parsed PRISM4 gold standard file")
    return parser.parse_args()


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


def main() -> None:
    args = cli()

    all_coverages = []
    lipopeptides_coverages = []

    for d in iter_jsonl(args.prism4_jsonl):
        d = Result.from_dict(d)
        cov = d.calculate_coverage()
        if "Lipopeptides" in d.submission.props.get("NPClassifierClass"):
            lipopeptides_coverages.append(cov)
        else:
            all_coverages.append(cov)

    # create density plot of coverages
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    x = np.linspace(0, 1, 500)
    kde_all_cov = gaussian_kde(all_coverages)
    kde_lipo_cov = gaussian_kde(lipopeptides_coverages)
    plt.plot(x, kde_all_cov(x), label=f"Other compounds (n={len(all_coverages)})")
    plt.plot(x, kde_lipo_cov(x), label=f"Lipopeptides (n={len(lipopeptides_coverages)})")
    plt.xlabel("Coverage")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Coverage distribution for PRISM4 gold standard compounds")
    # save to downloads folder
    plt.savefig(os.path.expanduser("~/Downloads/prism4_gold_standard_coverages.png"), dpi=300)
    

    

if __name__ == "__main__":
    main()
