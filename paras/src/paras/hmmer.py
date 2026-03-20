"""Functionalities for running HMMER."""

import os
import subprocess
import tempfile
from io import StringIO
from importlib.resources import files
from pathlib import Path

from Bio import SearchIO
from Bio.SearchIO._model import HSP

import paras.data


HMM_DB_HMMSCAN = Path(files(paras.data).joinpath("AMP-binding_full.hmm"))
HMM_DB_HMMPFAM2 = Path(files(paras.data).joinpath("AMP-binding_hmmer2.hmm"))


def run_hmmscan(fasta_str: str) -> str:
    """
    Run hmmscan on the given HMM database and FASTA string.

    :param fasta_str: FASTA-formatted string containing sequences to search
    :return: raw output from hmmscan
    """
    fast_fd, fasta_path = tempfile.mkstemp(prefix="hmmscan_", suffix=".fa")
    os.close(fast_fd)

    try:
        with open(fasta_path, "w") as f:
            f.write(fasta_str)
            if not fasta_str.endswith("\n"):
                f.write("\n")

        cmd = ["hmmscan", HMM_DB_HMMSCAN, fasta_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout
    finally:
        # Best-effort cleanup
        try:
            os.remove(fasta_path)
        except Exception:
            pass


def run_hmmpfam2(fasta_str: str) -> str:
    """
    Run hmmpfam2 on the given HMM database and FASTA string.

    :param fasta_str: FASTA-formatted string containing sequences to search
    :return: raw output from hmmpfam2
    """
    fasta_fd, fasta_path = tempfile.mkstemp(prefix="hmmpfam2_", suffix=".fa")
    os.close(fasta_fd)

    fd, out_path = tempfile.mkstemp(prefix="hmmpfam2_", suffix=".out")
    os.close(fd)

    try:
        with open(fasta_path, "w") as f:
            f.write(fasta_str)
            if not fasta_str.endswith("\n"):
                f.write("\n")

        with open(out_path, "w") as out:
            cmd = ["hmmpfam2", HMM_DB_HMMPFAM2, fasta_path]
            subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True, check=True)

        with open(out_path, "r") as f:
            return f.read()
        
    finally:
        # Best-effort cleanup
        for p in (fasta_path, out_path):
            try:
                os.remove(p)
            except Exception:
                pass


def parse_hmm_results(hmm_str: str, hmmer_version: int = 2) -> dict[str, HSP]:
    """
    Parse hmmpfam2 / hmmscan text output (provided as a string) and return
    a dict of header -> HSP for selected domains.

    :param hmm_str: raw output from hmmpfam2 or hmmscan
    :param hmmer_version: version of HMMer (2 or 3) that produced the output file (default: 2)
    :return: dictionary mapping domain identifier to Biopython HSP instance
    """
    if hmmer_version not in (2, 3):
        raise ValueError(f"Unknown HMMer version: {hmmer_version}")

    fmt = {2: "hmmer2-text", 3: "hmmer3-text"}[hmmer_version]
    filtered_hits: dict[str, HSP] = {}

    handle = StringIO(hmm_str)

    for result in SearchIO.parse(handle, fmt):
        for hsp in result.hsps:
            if hsp.bitscore <= 20:
                continue
            if hsp.hit_id not in {"AMP-binding", "AMP-binding_C"}:
                continue

            header = f"{result.id}|{hsp.hit_id}|{hsp.query_start}-{hsp.query_end}"
            filtered_hits[header] = hsp

    return filtered_hits