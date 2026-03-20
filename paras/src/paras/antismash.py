"""Functionalities for parsing antiSMASH GenBank files."""

import logging

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature


log = logging.getLogger(__name__)


def get_a_domains_from_gbk(
    gbk: str,
    domain_identifiers: list[str] = ["aSDomain"],
) -> list[SeqFeature]:
    """
    Extract A-domains from an antiSMASH GenBank file.
    
    :param gbk: path to the antiSMASH GenBank file
    :param domain_identifiers: list of feature types to identify A-domains (default: ["aSDomain"])
    :return: list of SeqFeature objects representing A-domains
    """
    a_domains = []
    for record in SeqIO.parse(gbk, "genbank"):
        all_domains = [
            f for f in record.features 
            if (
                f.type in domain_identifiers
                and "AMP-binding" in f.qualifiers.get("aSDomain", [])
            )
        ]
        a_domains.extend(all_domains)

    return a_domains