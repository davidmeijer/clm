"""Command line interface for Harvest."""

import argparse
from cmath import log
import os
import sys
import shlex
import subprocess
import logging

from harvest.retromol import cmd_run_retromol
from harvest.version import __version__
from harvest.sample import cmd_sample_unconditional
from harvest.train import cmd_train_model


_SLURM_FLAGS_WITH_VALUE = {
    "--part",
    "--cpus",
    "--mem",
    "--time",
    "--gres",
    "--job-name",
}
_SLURM_FLAGS_BOOL = {"--slurm"}


def _strip_slurm_flags(argv: list[str]) -> list[str]:
    """
    Return copy of argv with slurm options removed.

    :param argv: list of command line arguments
    :return: cleaned list of command line arguments
    .. note:: this is what is passed to the inner CLI when using slurm submission
    """
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--snakemake-args":
            # Preserve passthrough args verbatim (may include --slurm)
            cleaned.extend(argv[i:])
            break

        if arg in _SLURM_FLAGS_BOOL:
            i += 1
            continue

        if arg in _SLURM_FLAGS_WITH_VALUE:
            # Skip flag + its value
            i += 2
            continue

        cleaned.append(arg)
        i += 1

    return cleaned


def _submit_via_slurm(slurm_args: argparse.Namespace, cli_argv: list[str]) -> None:
    """
    Submit the current Harvest command to Slurm using sbatch.

    :param slurm_args: parsed command line arguments (including slurm options)
    :param cli-argv: list of command line arguments to pass to the inner Harvest CLI
    """
    python = sys.executable

    # Get output directory from CLI args (fallback to cwd for commands without --out-dir)
    output_dir = os.path.abspath(getattr(slurm_args, "out_dir", os.getcwd()))

    # Ensure log directory exists
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Slurm settings (CLI wins over env, env wins over defaults)
    partition   = slurm_args.part       or os.environ.get("PARTITION", "skinniderlab")
    cpus        = slurm_args.cpus       or int(os.environ.get("CPUS", "8"))
    mem         = slurm_args.mem        or os.environ.get("MEM", "16G")
    time        = slurm_args.time       or os.environ.get("TIME", "24:00:00")
    gres        = slurm_args.gres       or os.environ.get("GRES", "gpu:1")
    job_name    = slurm_args.job_name   or os.environ.get("JOB_NAME", "harvest_job")

    # The command that will run on the node: python -m harvest.cli <args...>
    inner_cmd = [python, "-m", "harvest.cli", *cli_argv]
    inner_cmd_str = shlex.join(inner_cmd)

    sbatch_cmd = [
        "sbatch",
        "-J", job_name,
        "-p", partition,
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "-o", os.path.join(output_dir, "logs", "harvest_%x_%j.out"),
        "-e", os.path.join(output_dir, "logs", "harvest_%x_%j.err"),
    ]

    if gres:
        sbatch_cmd.append(f"--gres={gres}")

    # Wrap inner command so wet get some basic info + timing
    wrap_script = f"""set -euo pipefail
echo "Node: $(hostname)"
echo "Using Python: {python}"
echo "CPUs: ${{OMP_NUM_THREADS:-{cpus}}}; Mem limit: {mem}"
/usr/bin/time -v {inner_cmd_str}
"""
    
    sbatch_cmd.extend(["--export", f"ALL,OMP_NUM_THREADS={cpus},MKL_NUM_THREADS={cpus},PYTHONUNBUFFERED=1"])
    sbatch_cmd.extend(["--wrap", wrap_script])

    if slurm_args.dry_run:
        print("[DRY RUN] Would submit Harvest job to slurm with:")
        print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
        print("\n[DRY RUN] Full --wrap script:\n")
        print(wrap_script)
        return
    
    print("Submitting Harvest job to Slurm:")
    print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
    subprocess.run(sbatch_cmd, check=True)


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command line interface for Harvest.

    :param argv: list of command line arguments (defaults to sys.argv[1:])
    :return: the parsed command line arguments
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"Harvest {__version__}", help="show the version number and exit")

    # Global Slurm options (only meaningful when --slurm is set)
    parser.add_argument("--slurm", action="store_true", help="submit this Harvest command as a slurm job instead of running it locally")
    parser.add_argument("--part", default=None, help="slurm partition (queue) name")
    parser.add_argument("--cpus", type=int, default=None, help="number of CPUs per task for slurm")
    parser.add_argument("--mem", default=None, help="memory request for slurm (e.g., 16G)")
    parser.add_argument("--time", type=str, default=None, help="time limit for slurm (e.g., 24:00:00)")
    parser.add_argument("--gres", type=str, default=None, help="slurm generic resources (e.g., gpu:1)")
    parser.add_argument("--job-name", type=str, default=None, help="slurm job name")
    parser.add_argument("--dry-run", action="store_true", help="if set, only print the slurm submission command without actually submitting")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Common arguments; Slurm expects an output directory for logs, so we require it for all commands to simplify the interface when using --slurm
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=str, required=True, help="directory to save output results")

    # Subparser for training the CLM via Snakemake workflow
    # Output directory is passed through to Snakemake config via --configfile, so we don't need to handle it here
    pt = sub.add_parser("train", help="train CLM via Snakemake workflow")
    pt.add_argument("--configfile", type=str, required=True, help="path to Snakemake config YAML")
    pt.add_argument("--workflow-dir", type=str, default=None, help="path to workflow directory (defaults to repo workflow/)")
    pt.add_argument("--snakefile", type=str, default=None, help="path to Snakefile (defaults to <workflow-dir>/Snakefile)")
    pt.add_argument("--jobs", type=int, default=None, help="Snakemake --jobs value")
    pt.add_argument("--latency-wait", type=int, default=None, help="Snakemake --latency-wait value")
    pt.add_argument("--rerun-incomplete", action="store_true", help="Snakemake --rerun-incomplete")
    pt.add_argument("--default-resources", nargs="+", default=None, help="Snakemake --default-resources values (e.g., slurm_partition=skinniderlab)")
    # Must be last; everything after --snakemake-args is passed verbatim to Snakemake
    pt.add_argument("--snakemake-args", nargs=argparse.REMAINDER, help="additional args passed to Snakemake (must be last), e.g. --snakemake-args --slurm")
    pt.set_defaults(func=lambda args: cmd_train_model(
        configfile=args.configfile,
        workflow_dir=args.workflow_dir,
        snakefile=args.snakefile,
        jobs=args.jobs,
        latency_wait=args.latency_wait,
        rerun_incomplete=args.rerun_incomplete,
        default_resources=args.default_resources,
        snakemake_args=args.snakemake_args,
        dry_run=args.dry_run,
    ))

    # Subparser for sampling the CLM unconditionally
    psu = sub.add_parser("sample-unconditional", parents=[common], help="sample unconditional CLM")
    psu.add_argument("--model-dir", type=str, required=True, help="path to dir trained CLM")
    psu.add_argument("--device", type=str, default="cpu", help="device to run sampling on (e.g., 'cuda:0' or 'cpu')")
    psu.add_argument("--num-samples", type=int, default=1000, help="number of samples to take")
    psu.set_defaults(func=lambda args: cmd_sample_unconditional(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        device=args.device,
        nsamples=args.num_samples,
    ))

    # Subparser for parsing compounds with RetroMol's retrosynthesis algorithm
    pr = sub.add_parser("run-retromol", parents=[common], help="run RetroMol retrosynthesis algorithm on a set of input compounds")
    pr.add_argument("--data-path", type=str, required=True, help="path to input file containing SMILES strings to run retrosynthesis on")
    pr.add_argument("--reaction-rules-path", type=str, required=True, help="path to file containing reaction rules for RetroMol")
    pr.add_argument("--matching-rules-path", type=str, required=True, help="path to file containing matching rules for RetroMol")
    pr.add_argument("--smiles-col", type=str, default="smiles", help="name of column in input file containing SMILES strings (default: 'smiles')")
    pr.add_argument("--num-workers", type=int, default=1, help="number of worker processes to use for retrosynthesis (default: 1)")
    pr.add_argument("--batch-size", type=int, default=2000, help="number of compounds to process in each batch (default: 2000)")
    pr.add_argument("--pool-chunksize", type=int, default=50, help="chunksize for multiprocessing pool (default: 50)")
    pr.add_argument("--maxtasksperchild", type=int, default=2000, help="maximum number of tasks to allow each worker process to complete before restarting it (default: 2000)")
    pr.set_defaults(func=lambda args: cmd_run_retromol(
        data_path=args.data_path,
        reaction_rules_path=args.reaction_rules_path,
        matching_rules_path=args.matching_rules_path,
        out_dir=args.out_dir,
        smiles_col=args.smiles_col,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pool_chunksize=args.pool_chunksize,
        maxtasksperchild=args.maxtasksperchild,
    ))

    args = parser.parse_args(argv)

    if getattr(args, "slurm", False):
        # Rebuild CLi argv without the slurm-only flags
        cli_argv = _strip_slurm_flags(argv)
        _submit_via_slurm(args, cli_argv)
    else:
        args.func(args)


def main(argv: list[str] | None = None) -> None:
    """
    Main entry point for the Harvest CLI.
    """
    cli(argv)


if __name__ == "__main__":
    main()
