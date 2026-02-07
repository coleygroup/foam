""" chem_utils.py """
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from ms_pred import common

# Use for re to parse chem formulae
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

# Use to create vectors of chem formulae
# VALID_ELEMENTS = ["C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br",
#                   "B", "Se", "Fe", "Co", "As"]

VALID_ELEMENTS = [ "C", "N", "P", "O", "S", "Si", "I", "Cl", "F", "Br",
                  "B", "Se", "Fe", "Co", "As"]
NORM_VEC = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2])

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_TO_POSITION = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
ELEMENT_DIM = len(ELEMENT_VECTORS[0])


def has_valid_els(chem_formula: str) -> bool:
    """has_valid_els"""
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol not in VALID_ELEMENTS:
            return False
    return True


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        if chem_symbol in ELEMENT_TO_POSITION:
            one_hot = ELEMENT_TO_POSITION[chem_symbol].reshape(1, -1)
            one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
            total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(ELEMENT_TO_POSITION))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_norm_dense(chem_formula: str) -> np.ndarray:
    """formula_to_norm_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    dense = formula_to_dense(chem_formula) / NORM_VEC[None, :]
    return dense

def vec_to_formula(form_vec):
    """ vec_to_formula. """
    build_str = ""
    for i in np.argwhere(form_vec > 0 ).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str

def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return CalcMolFormula(mol)


def mass_from_smi(smi: str) -> float:
    """mass_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def min_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.min()


def max_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.max()


def atoms_from_smi(smi: str) -> int:
    """atoms_from_smi.

    Args:
        smi (str): smi

    Return:
        int
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return mol.GetNumAtoms()

def contains_metals(formula: str) -> bool:
    """  returns true if formula contains metals"""
    METAL_RE = "(Fe|Co|Zn|Rh|Pt|Li)"
    return len(re.findall(METAL_RE, formula)) > 0



def fp_from_mol(mol):
    """fp_from_mol."""
    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint


def fp_wt_from_mol(mol):
    """fp_from_mol."""
    fp = fp_from_mol(mol)
    wt = ExactMolWt(mol)
    return fp, wt


def fp_from_smi(smiles):
    """ fp_from_smi. """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        return fp_from_mol(mol)

def fp_wt_from_smi(smiles):
    """ fp_from_smi. """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    else:
        return fp_wt_from_mol(mol)


def uncharged_formula(mol, mol_type="mol") -> str:
    """ Compute uncharged formula """
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def tautomerize_smi(smi):
    """tautomerize_smi, as roundtripped through InChi"""
    mol = Chem.MolFromSmiles(smi)
    inchi = Chem.MolToInchi(mol)
    mol = common.chem_utils.canonical_mol_from_inchi(inchi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)