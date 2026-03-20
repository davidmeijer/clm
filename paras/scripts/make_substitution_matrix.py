#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and visualize a substitution matrix from PARAS prediction profiles."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.tsv produced by the PARAS prediction script.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosine",
        choices=["cosine", "correlation", "dot", "log_odds"],
        help=(
            "How to derive pairwise substitution scores from prediction columns. "
            "'cosine' is the recommended default."
        ),
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster rows/columns to improve heatmap readability.",
    )
    parser.add_argument(
        "--min-total-prob",
        type=float,
        default=0.0,
        help="Drop items whose total probability mass across all rows is below this threshold.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(26, 24),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Default is tuned for ~220 labels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PNG.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap.",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors to export per item.",
    )
    return parser.parse_args()


def get_prediction_columns(df: pd.DataFrame) -> list[str]:
    reserved = {"protein_name", "extended_signature"}
    return [c for c in df.columns if c not in reserved]


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0)
    denom = np.outer(norms, norms)
    S = X.T @ X
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.divide(S, denom, out=np.zeros_like(S), where=denom > 0)
    np.fill_diagonal(S, 1.0)
    return S


def correlation_similarity_matrix(X: np.ndarray) -> np.ndarray:
    S = np.corrcoef(X, rowvar=False)
    S = np.nan_to_num(S, nan=0.0)
    np.fill_diagonal(S, 1.0)
    return S


def dot_product_matrix(X: np.ndarray) -> np.ndarray:
    return X.T @ X


def log_odds_matrix(X: np.ndarray, pseudocount: float = 1e-9) -> np.ndarray:
    N = X.shape[0]
    observed = X.T @ X
    marg = X.sum(axis=0)
    expected = np.outer(marg, marg) / max(N, 1)
    S = np.log2((observed + pseudocount) / (expected + pseudocount))
    return S


def cluster_order(S: np.ndarray) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for --cluster")

    D = 1.0 - S
    D = np.clip(D, 0.0, None)
    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return order


def choose_fontsize(n: int) -> float:
    if n <= 50:
        return 9
    if n <= 100:
        return 7
    if n <= 150:
        return 6
    if n <= 220:
        return 5
    return 4


def plot_heatmap(
    S: np.ndarray,
    labels: list[str],
    out_png: str,
    out_pdf: str,
    figsize: tuple[float, float],
    dpi: int,
    cmap: str,
    title: str,
) -> None:
    n = len(labels)
    fontsize = choose_fontsize(n)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        S,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=Normalize(vmin=S.min(), vmax=S.max()),
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.30, right=0.96, top=0.85, bottom=0.06)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def all_off_diagonal_pairs(S: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """
    Export all unique off-diagonal pairs (i < j), sorted by score descending.
    """
    n = len(labels)
    rows = []

    for i in range(n):
        self_i = float(S[i, i])
        for j in range(i + 1, n):
            self_j = float(S[j, j])
            score = float(S[i, j])

            rel_a = score / self_i if self_i != 0 else np.nan
            rel_b = score / self_j if self_j != 0 else np.nan

            rows.append(
                {
                    "item_a": labels[i],
                    "item_b": labels[j],
                    "score": score,
                    "self_score_a": self_i,
                    "self_score_b": self_j,
                    "relative_to_self_a": rel_a,
                    "relative_to_self_b": rel_b,
                    "relative_to_self_mean": np.nanmean([rel_a, rel_b]),
                    "abs_score": abs(score),
                }
            )

    pairs_df = pd.DataFrame(rows)
    pairs_df = pairs_df.sort_values(
        by=["score", "relative_to_self_mean", "item_a", "item_b"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    pairs_df.insert(0, "rank", np.arange(1, len(pairs_df) + 1))
    return pairs_df


def rank_neighbors(S: np.ndarray, labels: list[str], top_k: int) -> pd.DataFrame:
    n = len(labels)
    rows = []

    for i in range(n):
        scores = []
        self_i = float(S[i, i])

        for j in range(n):
            if i == j:
                continue

            score = float(S[i, j])
            rel = score / self_i if self_i != 0 else np.nan
            scores.append((labels[j], score, rel))

        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for neighbor_rank, (neighbor, score, rel) in enumerate(scores[:top_k], start=1):
            rows.append(
                {
                    "item": labels[i],
                    "neighbor_rank": neighbor_rank,
                    "neighbor": neighbor,
                    "score": score,
                    "self_score": self_i,
                    "relative_to_self": rel,
                }
            )

    return pd.DataFrame(rows)


def summarize_prediction_scores(X_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_rows = len(X_df)

    for col in X_df.columns:
        values = X_df[col].to_numpy(dtype=np.float64)

        rows.append(
            {
                "substrate": col,
                "mean_score": float(np.mean(values)),
                "median_score": float(np.median(values)),
                "std_score": float(np.std(values)),
                "max_score": float(np.max(values)),
                "fraction_above_0": float(np.mean(values > 0.0)),
                "fraction_above_0.01": float(np.mean(values > 0.01)),
                "fraction_above_0.05": float(np.mean(values > 0.05)),
                "fraction_above_0.10": float(np.mean(values > 0.10)),
                "sum_score": float(np.sum(values)),
                "n_predictions": int(n_rows),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["mean_score", "fraction_above_0.05", "max_score", "substrate"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    return summary_df


def choose_bar_fontsize(n: int) -> float:
    if n <= 30:
        return 9
    if n <= 60:
        return 8
    if n <= 120:
        return 7
    if n <= 220:
        return 6
    return 5


def plot_average_prediction_scores(
    summary_df: pd.DataFrame,
    out_png: str,
    out_pdf: str,
    dpi: int,
) -> None:
    labels = summary_df["substrate"].tolist()
    values = summary_df["mean_score"].to_numpy(dtype=np.float64)
    n = len(labels)

    width = max(16, n * 0.18)
    height = 8
    fontsize = choose_bar_fontsize(n)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(np.arange(n), values)

    ax.set_title("Average prediction score per substrate", fontsize=14, pad=12)
    ax.set_ylabel("Mean prediction score")
    ax.set_xlabel("Substrate")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.35)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.predictions, sep="\t")
    pred_cols = get_prediction_columns(df)

    if not pred_cols:
        raise ValueError("No prediction columns found in predictions file.")

    X_df = df[pred_cols].copy()

    for col in pred_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.fillna(0.0)

    if args.min_total_prob > 0.0:
        keep_mask = X_df.sum(axis=0) >= args.min_total_prob
        X_df = X_df.loc[:, keep_mask]
        pred_cols = list(X_df.columns)

    if len(pred_cols) == 0:
        raise ValueError("No items remain after filtering.")

    X = X_df.to_numpy(dtype=np.float64)

    if args.method == "cosine":
        S = cosine_similarity_matrix(X)
        title = "Substitution matrix from PARAS predictions (cosine similarity)"
    elif args.method == "correlation":
        S = correlation_similarity_matrix(X)
        title = "Substitution matrix from PARAS predictions (Pearson correlation)"
    elif args.method == "dot":
        S = dot_product_matrix(X)
        title = "Substitution matrix from PARAS predictions (dot product)"
    elif args.method == "log_odds":
        S = log_odds_matrix(X)
        title = "Substitution matrix from PARAS predictions (soft log-odds)"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    labels = pred_cols

    matrix_df_unclustered = pd.DataFrame(S, index=labels, columns=labels)

    stem_base = f"substitution_matrix.{args.method}"

    out_tsv_unclustered = os.path.join(args.outdir, f"{stem_base}.tsv")
    matrix_df_unclustered.to_csv(out_tsv_unclustered, sep="\t", index=True)

    pairs_df = all_off_diagonal_pairs(S, labels)
    out_pairs = os.path.join(args.outdir, f"all_off_diagonal_pairs.{args.method}.tsv")
    pairs_df.to_csv(out_pairs, sep="\t", index=False)

    neighbors_df = rank_neighbors(S, labels, args.top_neighbors)
    out_neighbors = os.path.join(args.outdir, f"nearest_neighbors.{args.method}.tsv")
    neighbors_df.to_csv(out_neighbors, sep="\t", index=False)

    summary_df = summarize_prediction_scores(X_df)
    out_summary = os.path.join(args.outdir, "average_prediction_scores.tsv")
    summary_df.to_csv(out_summary, sep="\t", index=False)

    out_summary_png = os.path.join(args.outdir, "average_prediction_scores.png")
    out_summary_pdf = os.path.join(args.outdir, "average_prediction_scores.pdf")
    plot_average_prediction_scores(
        summary_df=summary_df,
        out_png=out_summary_png,
        out_pdf=out_summary_pdf,
        dpi=args.dpi,
    )

    if args.cluster:
        if args.method == "log_odds":
            S_for_cluster = S.copy()
            S_for_cluster = S_for_cluster - S_for_cluster.min()
            if S_for_cluster.max() > 0:
                S_for_cluster = S_for_cluster / S_for_cluster.max()
            np.fill_diagonal(S_for_cluster, 1.0)
        else:
            S_for_cluster = S

        order = cluster_order(S_for_cluster)
        S_plot = S[order][:, order]
        labels_plot = [labels[i] for i in order]
    else:
        S_plot = S
        labels_plot = labels

    stem_plot = stem_base
    if args.cluster:
        stem_plot += ".clustered"

    out_png = os.path.join(args.outdir, f"{stem_plot}.png")
    out_pdf = os.path.join(args.outdir, f"{stem_plot}.pdf")

    if args.cluster:
        out_tsv_clustered = os.path.join(args.outdir, f"{stem_plot}.tsv")
        matrix_df_clustered = pd.DataFrame(S_plot, index=labels_plot, columns=labels_plot)
        matrix_df_clustered.to_csv(out_tsv_clustered, sep="\t", index=True)

    plot_heatmap(
        S=S_plot,
        labels=labels_plot,
        out_png=out_png,
        out_pdf=out_pdf,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        cmap=args.cmap,
        title=title,
    )

    print(f"Wrote matrix: {out_tsv_unclustered}")
    if args.cluster:
        print(f"Wrote clustered matrix: {os.path.join(args.outdir, f'{stem_plot}.tsv')}")
    print(f"Wrote heatmap PNG: {out_png}")
    print(f"Wrote heatmap PDF: {out_pdf}")
    print(f"Wrote all off-diagonal pairs: {out_pairs}")
    print(f"Wrote nearest neighbors: {out_neighbors}")
    print(f"Wrote average prediction summary: {out_summary}")
    print(f"Wrote average prediction score PNG: {out_summary_png}")
    print(f"Wrote average prediction score PDF: {out_summary_pdf}")


if __name__ == "__main__":
    main()