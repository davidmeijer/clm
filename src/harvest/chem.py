"""Cheminformatics utility functions."""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs import ExplicitBitVect, TanimotoSimilarity


def smiles_to_mol(smiles: str, print_error_on_fail: bool = False) -> Chem.Mol | None:
    """
    Convert a SMILES string to an RDKit Mol object.
    
    :param smiles: SMILES string to convert
    :param print_error_on_fail: whether to print error message if conversion fails (default: False)
    :return: RDKit Mol object, or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        if print_error_on_fail:
            print(f"Error converting SMILES to Mol: {e}")
        return None
    

def mol_to_weight(mol: Chem.Mol) -> float:
    """
    Calculate the molecular weight of a molecule.
    
    :param mol: RDKit Mol object to calculate molecular weight for
    :return: molecular weight of the molecule
    """
    return Descriptors.MolWt(mol)


def mol_to_morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> ExplicitBitVect:
    """
    Calculate the Morgan fingerprint of a molecule.

    :param mol: RDKit Mol object to calculate fingerprint for
    :param radius: radius of the Morgan fingerprint
    :param n_bits: number of bits in the fingerprint
    :return: Morgan fingerprint as an ExplicitBitVect
    """
    fp_gen = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=False)
    fp = fp_gen.GetFingerprint(mol)
    return fp


def mol_to_inchikey_conn(mol: Chem.Mol) -> str:
    """
    Calculate the InChIKey connectivity layer of a molecule.
    
    :param mol: RDKit Mol object to calculate InChIKey for
    :return: InChIKey of the molecule
    """
    return Chem.MolToInchiKey(mol).split("-")[0]


def tanimoto(fp1: ExplicitBitVect, fp2: ExplicitBitVect) -> float:
    """
    Calculate the Tanimoto similarity between two fingerprints.
    
    :param fp1: first fingerprint as an ExplicitBitVect
    :param fp2: second fingerprint as an ExplicitBitVect
    :return: Tanimoto similarity between the two fingerprints
    """
    return TanimotoSimilarity(fp1, fp2)
