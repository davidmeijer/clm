#!/usr/bin/env python3

import argparse
import os
import logging

from tqdm import tqdm
from Bio.SeqFeature import SeqFeature

from paras.logging import setup_logging, add_file_handler
from paras.antismash import get_a_domains_from_gbk
from paras.hmmer import run_hmmpfam2, run_hmmscan, parse_hmm_results
from paras.parsing import (
    AdenylationDomain,
    hits_to_domains,
    update_hmmer2_domain_sequences,
    get_hmmer3_unique_domains,
    set_domain_numbers,
)


log = logging.getLogger(__name__)


def cli() -> argparse.Namespace:
    """
    Command line interface for PARAS-fast.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbks", type=str, required=True, help="path to directory containing antiSMASH output GenBank files")
    parser.add_argument("--outdir", type=str, required=True, help="path to output directory")
    parser.add_argument("--loglevel", type=str, default="INFO", help="logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)")
    return parser.parse_args()


def main() -> None:
    """
    Main function for PARAS-fast command line interface.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    setup_logging(level=args.loglevel)
    logfile_path = os.path.join(args.outdir, "log.txt")
    add_file_handler(logfile_path, level=args.loglevel)

    # Greedily loop through .gbk files in gbks without loading them all into memory at once
    total_parsed = 0

    with open(os.path.join(args.outdir, "signatures.tsv"), "w") as out_f:
        # Write header for signatures output file
        out_f.write("protein_name\tsignature\textended_signature\n")

        for entry in tqdm(os.scandir(args.gbks), desc="Processing GenBank files"):
            if not entry.is_file():
                continue
            if not entry.name.endswith(".gbk"):
                continue

            filename = entry.name
            gbk_path = entry.path
            
            log.debug(f"parsing item {total_parsed}: {filename} ...")

            a_domains: list[SeqFeature] = get_a_domains_from_gbk(gbk_path)
            log.debug(f"found {len(a_domains)} A-domains in {filename}")

            if len(a_domains) == 0:
                continue

            fasta_str = ""
            for d in a_domains:
                location = d.location
                locus_tag = d.qualifiers.get("locus_tag", None)
                translation = d.qualifiers.get("translation", None)
                header = f"{filename}|{locus_tag[0] if locus_tag else 'NA'}|{location}"
                seq = translation[0] if translation else ""
                if seq:
                    fasta_str += f">{header}\n{seq}\n"
            
            try:
                # Run HMM search (with HMMER2 and HMMER3) on fasta and parse results
                res_h2 = run_hmmpfam2(fasta_str)
                res_h3 = run_hmmscan(fasta_str)

                # Parse results
                parsed_h2 = parse_hmm_results(res_h2, hmmer_version=2)
                parsed_h3 = parse_hmm_results(res_h3, hmmer_version=3)

                a_domains_h2: list[AdenylationDomain] = hits_to_domains(parsed_h2, fasta_str, hmm_version=2)
                a_domains_h3: list[AdenylationDomain] = hits_to_domains(parsed_h3, fasta_str, hmm_version=3)

                update_hmmer2_domain_sequences(a_domains_h2, a_domains_h3, fasta_str)
                unique_hmmer3_domains = get_hmmer3_unique_domains(a_domains_h2, a_domains_h3)
                a_domains = a_domains_h2 + unique_hmmer3_domains
                a_domains.sort(key=lambda x: (x.protein_name, x.start))
                set_domain_numbers(a_domains)
                
                for d in a_domains:
                    out_f.write(f"{d.protein_name}\t{d.signature}\t{d.extended_signature}\n")
            except Exception as e:
                log.error(f"Error processing {filename}: {e}")
                continue

            total_parsed += 1

            # flush every 10 files to ensure progress is saved even if the program crashes
            if (total_parsed) % 1000 == 0:
                out_f.flush()


if __name__ == "__main__":
    main()
