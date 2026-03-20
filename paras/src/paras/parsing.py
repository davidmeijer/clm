"""Module for parsing HMM hits and extracting adenylation domains."""

import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import Optional

from Bio.SearchIO._model import HSP

import paras.data


log = logging.getLogger(__name__)


def _read_positions(path_in: str, start_position: int) -> list[int]:
    """Parse positions from a tab-separated file.

    :param path_in: Path to tab-separated input positions file.
    :type path_in: str
    :param start_position: Relative start position to adjust all positions by.
    :type start_position: int
    :return: List of positions, each relative to the start position.
    :rtype: List[int]
    """
    with open(path_in, "r") as fo:
        text = fo.read().strip()

        # split the text by tabs and adjust the positions by the start position
        positions = []
        for position in text.split("\t"):
            positions.append(int(position) - start_position)

    return positions


A_POSITION_FILE_HMM2 = Path(files(paras.data).joinpath("stachelhaus_hmm2.txt"))
A_POSITION_FILE_34_HMM2 = Path(files(paras.data).joinpath("active_site_hmm2.txt"))
A_POSITION_FILE = Path(files(paras.data).joinpath("stachelhaus.txt"))
A_POSITION_FILE_34 = Path(files(paras.data).joinpath("active_site.txt"))

HMM2_POSITIONS_SIGNATURE = _read_positions(A_POSITION_FILE_HMM2, 0)
HMM2_POSITIONS_EXTENDED_SIGNATURE = _read_positions(A_POSITION_FILE_34_HMM2, 0)
POSITIONS_SIGNATURE = _read_positions(A_POSITION_FILE, 66)
POSITIONS_EXTENDED_SIGNATURE = _read_positions(A_POSITION_FILE_34, 66)


def _get_reference_positions_hmm(
    query_sequence: str, reference_sequence: str, reference_positions: list[int]
) -> Optional[list[int]]:
    """Extract the given positions from a query alignment.

    :param query_sequence: The aligned query sequence.
    :type query_sequence: str
    :param reference_sequence: The aligned reference sequence.
    :type reference_sequence: str
    :param reference_positions: The positions of interest in the unaligned reference.
    :type reference_positions: List[int]
    :return: Positions of sequence elements at the adjusted reference,
        or None if the positions could not be extracted.
    :rtype: list[int
    :raises ValueError: If the reference sequence is too short.

    .. note:: Positions are adjusted to account for gaps in the reference sequence.
    .. note:: This function assumes that the reference sequence is shorter
        than the query sequence.
    .. note:: From antiSMASH 7.0.0.
    """
    # check if the reference is too short
    if len(reference_sequence) < len(reference_positions):
        raise ValueError(
            f"reference sequence is too short: {len(reference_sequence)} < {len(reference_positions)}"
        )  # noqa: E501

    # check if the reference sequence is the same length as the query sequence
    # or if the reference sequence is shorter than the query sequence
    if not (
        len(reference_sequence) == len(query_sequence)
        or len(reference_sequence) < len(query_sequence)
    ):
        raise ValueError(
            f"reference sequence is too long: {len(reference_sequence)} != {len(query_sequence)}"
        )

    # adjust position of interest to account for gaps in the ref sequence alignment
    positions = []
    position_skipping_gaps = 0  # position in the reference sequence

    for amino_acid_idx, amino_acid_id in enumerate(reference_sequence):

        if amino_acid_id in "-.":
            continue

        if position_skipping_gaps in reference_positions:
            positions.append(amino_acid_idx)

        # increment the position in the reference sequence
        position_skipping_gaps += 1

    # check if the number of positions extracted is the same as the number of reference
    # positions to extract
    if len(positions) != len(reference_positions):
        return None

    # extract positions from query sequence
    return positions


def _get_gap_adjusted_positions(query: str, positions: list[int], offset: int) -> list[Optional[int]]:
    """Return sequence positions adjusted for gaps

    :param query: query sequence
    :type query: str
    :param positions: positions in the gapped query sequence
    :type positions: list[int]
    :param offset: query start position
    :type offset: int
    :return: list of gap-adjusted positions
    :rtype: list[int]
    """
    adjusted_positions: list[Optional[int]] = []
    position = offset
    for i, char in enumerate(query):
        if i in positions:
            if char != '-':
                adjusted_positions.append(position)
            else:
                adjusted_positions.append(None)
        if char != '-':
            position += 1

    return adjusted_positions


class AdenylationDomain:
    """Class for representing adenylation domains."""

    def __init__(self, protein_name: str, domain_start: int, domain_end: int) -> None:
        """Initialize an AdenylationDomain object.

        :param protein_name: The name of the protein.
        :type protein_name: str
        :param domain_start: The start position of the domain.
        :type domain_start: int
        :param domain_end: The end position of the domain.
        :type domain_end: int
        """
        self.protein_name = protein_name
        self.domain_nr = 0
        self.start = domain_start
        self.end = domain_end

        self.sequence = ""
        self.protein_sequence = ""
        self.signature = ""
        self.extended_signature = ""
        self.signature_positions: list[int] = []
        self.extended_signature_positions: list[int] = []

    def domains_overlap(self, other: "AdenylationDomain", threshold: int = 50) -> bool:
        """
        Check if two domains overlap by at least a certain threshold of base pairs.

        :param other: Other adenylation domain
        :type other: AdenylationDomain
        :param threshold: The number of base pairs the two domains need to overlap by for a match
        :type threshold: int

        :return: bool, True if domains overlap, False otherwise
        """
        if other.start <= self.start <= other.end:
            if other.end - self.start >= threshold:
                return True
        if self.start <= other.start <= self.end:
            if self.end - other.start >= threshold:
                return True
        return False

    def set_domain_number(self, domain_nr: int) -> None:
        """Set the domain number.

        :param domain_nr: The domain number.
        :type domain_nr: int

        .. note:: This function modifies the domain number attribute.
        """
        self.domain_nr = domain_nr

    def set_protein_sequence(self, protein_sequence: str) -> None:
        """Set the protein sequence that contains the domain

        :param protein_sequence: The sequence of the domain.
        :type protein_sequence: str

        .. note:: This function modifies the sequence attribute.
        """
        self.protein_sequence = protein_sequence

    def set_sequence(self, sequence: str) -> None:
        """Set the sequence of the domain.

        :param sequence: The sequence of the domain.
        :type sequence: str

        .. note:: This function modifies the sequence attribute.
        """
        self.sequence = sequence

    def set_domain_signatures_hmm(
        self, hit_n_terminal: HSP, hit_c_terminal: Optional[HSP] = None
    ) -> None:
        """Extract (extended) signatures from adenylation domains using HMM profile.

        :param hit_n_terminal: The hit object for the N-terminal domain.
        :type hit_n_terminal: HSP
        :param hit_c_terminal: The hit object for the C-terminal domain.
        :type hit_c_terminal: Optional[HSP]

        .. note:: This function modifies the signature and extended signature attributes.
        """
        valid = {
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "-",
        }

        signature_positions = HMM2_POSITIONS_SIGNATURE
        extended_signature_positions = HMM2_POSITIONS_EXTENDED_SIGNATURE
        position_k = [36]  # hmm2 position k

        profile = hit_n_terminal.aln[1].seq
        query = hit_n_terminal.aln[0].seq
        offset = hit_n_terminal.hit_start
        query_offset = hit_n_terminal.query_start

        signature_location = _get_reference_positions_hmm(
            query_sequence=query,
            reference_sequence=profile,
            reference_positions=[p - offset for p in signature_positions],
        )

        if signature_location:
            signature = "".join([query[i] for i in signature_location])
            if all([char in valid for char in signature]):
                self.signature = signature
                self.signature_positions = _get_gap_adjusted_positions(query, signature_location, query_offset)

        lysine = None
        lysine_position = None
        query_c = None
        if hit_c_terminal:
            profile_c = hit_c_terminal.aln[1].seq
            query_c = hit_c_terminal.aln[0].seq
            offset_c = hit_c_terminal.hit_start

            lysine_position = _get_reference_positions_hmm(
                query_sequence=query_c,
                reference_sequence=profile_c,
                reference_positions=[p - offset_c for p in position_k],
            )

        if self.signature:
            if lysine_position and query_c:
                lysine = query_c[lysine_position[0]]
            if lysine and lysine in valid and lysine != "-":
                self.signature += lysine
            else:
                self.signature += "K"

        extended_signature_location = _get_reference_positions_hmm(
            query_sequence=query,
            reference_sequence=profile,
            reference_positions=[p - offset for p in extended_signature_positions],
        )
        if extended_signature_location:
            extended_signature = "".join([query[i] for i in extended_signature_location])
            if all([char in valid for char in extended_signature]):
                self.extended_signature = extended_signature
                self.extended_signature_positions = _get_gap_adjusted_positions(query, extended_signature_location, query_offset)

    # def set_domain_signatures_profile(self, path_temp_dir: str) -> None:
    #     """Extract (extended) signatures from adenylation domains using profile alignment.

    #     :param path_temp_dir: The path to the temporary directory.
    #     :type path_temp_dir: str
    #     :raises Exception: If the sequence is not defined.

    #     .. note:: This function modifies the signature and extended signature attributes.
    #     """

    #     if self.sequence is None:
    #         raise Exception("sequence needs to be defined first")

    #     aligned_domain, aligned_reference = _align_adenylation_domain(
    #         domain_name="DOMAIN_TO_QUERY",
    #         domain_sequence=self.sequence,
    #         alignment_file=ALIGNMENT_FILE,
    #         path_temp_dir=path_temp_dir,
    #     )

    #     aligned_positions_signature = _get_reference_positions(
    #         positions=POSITIONS_SIGNATURE, aligned_reference=aligned_reference
    #     )

    #     aligned_positions_extended_signature = _get_reference_positions(
    #         positions=POSITIONS_EXTENDED_SIGNATURE, aligned_reference=aligned_reference
    #     )

    #     signature = []
    #     for position in aligned_positions_signature:
    #         signature.append(aligned_domain[position])

    #     self.signature = "".join(signature)
    #     self.signature_positions = _get_gap_adjusted_positions(aligned_domain, aligned_positions_signature, self.start)

    #     extended_signature = []
    #     for position in aligned_positions_extended_signature:
    #         extended_signature.append(aligned_domain[position])
    #     self.extended_signature = "".join(extended_signature)
    #     self.extended_signature_positions = _get_gap_adjusted_positions(aligned_domain,
    #                                                                     aligned_positions_extended_signature, self.start)


def merge_hits(hits: list[tuple[str, int, int, str]]) -> tuple[str, int, int, str]:
    """
    Merge N-terminal AMP-binding hits

    :param hits: list of AMP-binding HMM hits
    :type hits: list[tuple[str, int, int, str]]
    """

    if hits:
        seq_id, hit_id, _ = hits[0][3].rsplit('|', maxsplit=2)
        for hit in hits:
            seq_id_2, hit_id_2, _ = hit[3].rsplit('|', maxsplit=2)
            if seq_id_2 != seq_id:
                raise ValueError(f"Cannot merge hits from different sequences! {seq_id}, {seq_id_2}")
            if hit_id_2 != hit_id:
                raise ValueError(f"Cannot merge different hit types! {hit_id}, {hit_id_2}")

        hit_start = min([hit[1] for hit in hits])
        hit_end = max([hit[2] for hit in hits])
        hit_key = f"{seq_id}|{hit_id}|{hit_start}-{hit_end}"
        merged_hit = (hit_id, hit_start, hit_end, hit_key)
        return merged_hit
    else:
        raise ValueError("No hits to merge!")


def group_n_terminal_hits(hit_list: list[tuple[str, int, int, str]]) -> list[tuple[str, int, int, str]]:
    """
    Group and merge N-terminal AMP-binding hits within a single protein

    :param hit_list: list of AMP-binding HMM hits
    :type hit_list: list[tuple[str, int, int, str]]
    """
    n_terminal_hits = []
    c_terminal_hits = []
    seq_ids = set()

    for hit in hit_list:

        hit_id, hit_start, hit_end, hit_key = hit
        seq_id = hit_key.split('|')[0]
        seq_ids.add(seq_id)
        if hit_id == "AMP-binding":
            n_terminal_hits.append(hit)
        elif hit_id == "AMP-binding_C":
            c_terminal_hits.append(hit)

    if len(seq_ids) > 1:
        raise ValueError("Cannot group hits from multiple sequences!")

    n_terminal_hits.sort(key=lambda x: x[1])
    c_terminal_hits.sort(key=lambda x: x[1])

    grouped_hits = []
    if n_terminal_hits:
        group = [n_terminal_hits[0]]

        for i, hit_1 in enumerate(n_terminal_hits):
            if i + 1 < len(n_terminal_hits):
                hit_2 = n_terminal_hits[i + 1]
                if hit_2[1] - hit_1[2] < 60:
                    group.append(hit_2)
                else:
                    grouped_hits.append(group[:])
                    group = [hit_2]
            else:
                grouped_hits.append(group[:])
                group = []

    merged_n_terminal = []

    for group in grouped_hits:
        merged_hit = merge_hits(group)
        merged_n_terminal.append(merged_hit)

    return merged_n_terminal + c_terminal_hits


def parse_fasta_str(fasta_str: str) -> dict[str, str]:
    fasta: dict[str, str] = {}
    current_header = None
    current_sequence = ""
    for line in fasta_str.splitlines():
        if line.startswith(">"):
            if current_header is not None:
                fasta[current_header] = current_sequence
            current_header = line[1:].strip()
            current_sequence = ""
        else:
            current_sequence += line.strip()
    if current_header is not None:
        fasta[current_header] = current_sequence
    return fasta


def hits_to_domains(
    id_to_hit: dict[str, HSP],
    fasta_str: str,
    hmm_version: int = 2
) -> list[AdenylationDomain]:
    """
    Extract adenylation domains from HMM hits.

    :param id_to_hit: dictionary of HMM hits
    :param fasta_str: FASTA-formatted string containing sequences to search
    :param hmm_version: version of HMMer that produced the hits (default: 2)
    """
    hits_by_seq_id: dict[str, list[tuple[str, int, int, str]]] = {}
    for hit_key in id_to_hit.keys():

        # parse domain ID
        seq_id, hit_id, hit_location = hit_key.rsplit("|", maxsplit=2)
        hit_start_str, hit_end_str = hit_location.split("-")
        hit_start = int(hit_start_str)
        hit_end = int(hit_end_str)

        if seq_id not in hits_by_seq_id:
            hits_by_seq_id[seq_id] = []

        hits_by_seq_id[seq_id].append((hit_id, hit_start, hit_end, hit_key))

    counter = 0
    seq_id_to_domains: dict[str, list[AdenylationDomain]] = {}
    seq_id_to_hits = {}
    for seq_id, hits in hits_by_seq_id.items():
        seq_id_to_hits[seq_id] = group_n_terminal_hits(hits)

    for seq_id, hits in seq_id_to_hits.items():
        counter += 1

        for hit_id_1, hit_start_1, hit_end_1, hit_key_1 in hits:

            if hit_id_1 == "AMP-binding":
                if seq_id not in seq_id_to_domains:
                    seq_id_to_domains[seq_id] = []

                match_found = False
                for hit_id_2, hit_start_2, hit_end_2, hit_key_2 in hits:

                    if hit_id_2 == "AMP-binding_C":
                        if hit_start_2 > hit_end_1 and (hit_start_2 - hit_end_1) < 200:

                            a_domain = AdenylationDomain(protein_name=seq_id, domain_start=hit_start_1, domain_end=hit_end_2)

                            if hmm_version == 2:
                                a_domain.set_domain_signatures_hmm(hit_n_terminal=id_to_hit[hit_key_1], hit_c_terminal=id_to_hit[hit_key_2])

                            seq_id_to_domains[seq_id].append(a_domain)
                            match_found = True
                            break

                if not match_found:
                    a_domain = AdenylationDomain(protein_name=seq_id, domain_start=hit_start_1, domain_end=hit_end_1)

                    if hmm_version == 2:
                        a_domain.set_domain_signatures_hmm(hit_n_terminal=id_to_hit[hit_key_1], hit_c_terminal=None)

                    seq_id_to_domains[seq_id].append(a_domain)

        if counter % 1000 == 0:
            log.debug(f"processed {counter} proteins ...")

    # sort domains by start position
    for domains in seq_id_to_domains.values():
        domains.sort(key=lambda x: x.start)

    # Parse fasta string into dict, header as key and then the sequence as value
    fasta = parse_fasta_str(fasta_str)

    for seq_id, sequence in fasta.items():
        counter = 1

        if seq_id in seq_id_to_domains:
            for a_domain in seq_id_to_domains[seq_id]:

                if seq_id != a_domain.protein_name:
                    raise ValueError("Protein name mismatch")

                a_domain_sequence = sequence[a_domain.start:a_domain.end]

                if len(a_domain_sequence) > 100:
                    a_domain.set_sequence(a_domain_sequence)
                    a_domain.set_protein_sequence(sequence)
                    a_domain.set_domain_number(counter)

                    counter += 1

    filtered_a_domains = []
    for a_domains in seq_id_to_domains.values():
        for a_domain in a_domains:
            if a_domain.sequence and a_domain.domain_nr:
                filtered_a_domains.append(a_domain)

    filtered_a_domains.sort(key=lambda x: (x.protein_name, x.start))

    return filtered_a_domains


def update_hmmer2_domain_sequences(hmmer2_domains: list[AdenylationDomain],
                                   hmmer3_domains: list[AdenylationDomain],
                                   fasta_str: str) -> None:
    """
    Update hmmer2 domains with hmmer3 domain sequences, such that the longest detected sequence is maintained.

    :param hmmer2_domains: list of [AdenylationDomain, ->], list of adenylation domains detected by HMMer2. These domains also
    contain information on signatures and extended signatures. Typically contain short sequences
    :type hmmer2_domains: list[AdenylationDomain]
    :param hmmer3_domains: list of [AdenylationDomain, ->], list of adenylation domains detected by HMMer2. These domains do
    not contain information on signatures and extended signatures. Typically contain full-length sequences. Use these
    sequences to update
    :type hmmer3_domains: list[AdenylationDomain]
    :param fasta_str: Fasta string containing protein sequences.
    :type fasta_str: str
    """

    fasta = parse_fasta_str(fasta_str)

    for domain_1 in hmmer2_domains:
        for domain_2 in hmmer3_domains:
            if domain_1.protein_name == domain_2.protein_name and domain_1.domains_overlap(domain_2, threshold=50):
                sequence_altered = False
                if len(domain_2.sequence) > len(domain_1.sequence):
                    domain_1.start = domain_2.start
                    domain_1.end = domain_2.end
                    sequence_altered = True

                if sequence_altered:
                    if domain_1.protein_name not in fasta:
                        raise ValueError("Mismatching protein names")
                    domain_1.set_sequence(fasta[domain_1.protein_name][domain_1.start:domain_1.end])


def get_hmmer3_unique_domains(hmmer2_domains: list[AdenylationDomain],
                              hmmer3_domains: list[AdenylationDomain],
                            #   path_temp_dir: str) -> list[AdenylationDomain]:
) -> list[AdenylationDomain]:
    """
    Get domains that were found by hmmer3 but not hmmer 2, and obtain domain signatures through profile alignment

    :param hmmer2_domains: list of [AdenylationDomain, ->], list of adenylation domains detected by HMMer2. These domains also
    contain information on signatures and extended signatures. Typically contain short sequences
    :type hmmer2_domains: list[AdenylationDomain]
    :param hmmer3_domains: list of [AdenylationDomain, ->], list of adenylation domains detected by HMMer2. These domains do
    not contain information on signatures and extended signatures. Typically contain full-length sequences. Use these
    sequences to update
    :type hmmer3_domains: list[AdenylationDomain]
    :param path_temp_dir: Path to temp dir.
    :type path_temp_dir: str
    """
    unique_domains = []
    for domain_1 in hmmer3_domains:
        match_found = False
        for domain_2 in hmmer2_domains:
            if domain_1.protein_name == domain_2.protein_name and domain_1.domains_overlap(domain_2, threshold=50):
                match_found = True
        # if not match_found:
        #     domain_1.set_domain_signatures_profile(path_temp_dir)
        #     unique_domains.append(domain_1)

    return unique_domains


def set_domain_numbers(a_domains: list[AdenylationDomain]) -> None:
    """
    Set domain numbers from list of A domains

    :param a_domains: list of A domains
    :type a_domains: list[AdenylationDomain, ->]
    """

    if not a_domains:
        return

    a_domains.sort(key=lambda x: (x.protein_name, x.start))

    protein_name = a_domains[0].protein_name
    counter = 1
    a_domains[0].set_domain_number(counter)

    for a_domain in a_domains[1:]:
        if a_domain.protein_name == protein_name:
            counter += 1
        else:
            counter = 1
            protein_name = a_domain.protein_name

        a_domain.set_domain_number(counter)
