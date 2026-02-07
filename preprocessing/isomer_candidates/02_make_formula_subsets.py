""" 03_make_formula_subsets.py

Process pubchem smiles s.t. we have an easily queryable pickled file

"""
import argparse
import json
import pickle
from rdkit import Chem
from rdkit import RDLogger

from collections import defaultdict

from typing import List, Tuple

from tqdm import tqdm

import multiprocess.context as ctx
ctx._force_start_method('spawn')
from pathos import multiprocessing as mp
from foam.utils.chem_utils import uncharged_formula
from ms_pred import common

cpus = mp.cpu_count()

RDLogger.DisableLog("rdApp.*")

INCHI_SEARCH = "InChI=1S/([A-Z,a-z,0-9,\.]*)/*"
UNCHARGED_SEARCH = r"^([^\+,^\-]*)"
PUBCHEM_FILE = "data/pubchem/cid_smiles.txt"
PUBCHEM_FORMULA = "data/pubchem/pubchem_formulae_inchikey.{}"
PUBCHEM_FORMULA_DEBUG = "data/pubchem/pubchem_formulae_inchikey_debug.{}"


def single_form_from_smi(smi: str) -> Tuple[str, Tuple[str, str]]:
    """Compute single formula + inchi key from a smiles string"""
    try:
        mol = Chem.MolFromSmiles(smi)

        ## Add conversion to InChi
        # mol = Chem.MolFromInchi(Chem.MolToInchi(mol, logLevel=None),
        #                        logLevel=None)

        ####
        if mol is not None:
            form = uncharged_formula(mol)

            # first remove stereochemistry
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            return form, (smi, inchi_key)
        else:
            return "", ("", "")
    except:
        return "", ("", "")


def convert_smi_list_to_forms(smi_list: List[str]) -> List[Tuple[str, str]]:
    out_entries = []
    for j in smi_list:
        out_entry = single_form_from_smi(j)
        out_entries.append(out_entry)
    return out_entries


# delayed_single_form_from_smi = delayed(single_form_from_smi)
def get_formulae_from_smis(smi_list: List[str]) -> list:
    """Process formulae in parallel"""
    pool = mp.Pool(processes=cpus)

    smi_list_len = len(smi_list)
    num_chunks = 100000
    num_chunks = min(smi_list_len, num_chunks)

    chunked_smis = (smi_list[i::num_chunks] for i in range(num_chunks))

    # Will shuffle order!
    mol_list = list(
        tqdm(pool.imap(convert_smi_list_to_forms, chunked_smis), total=num_chunks)
    )

    mol_list = [j for i in mol_list for j in i]
    formulae, new_mols = zip(*mol_list)
    return formulae, new_mols


def calc_smi_to_formula(smi_list: List[str]) -> dict:
    """Map input smiles to their chem formulae"""
    formulae, _ = get_formulae_from_smis(smi_list)
    dict_map = dict(zip(smi_list, formulae))
    return dict_map


def calc_formula_to_moltuples(smi_list: List[str]) -> dict:
    """Map smiles to their formula + inchikey"""
    formulae, mol_tuples = get_formulae_from_smis(smi_list)
    outdict = defaultdict(lambda: set())
    for mol_tuple, formula in tqdm(zip(mol_tuples, formulae)):
        outdict[formula].add(mol_tuple)
    return dict(outdict)


def get_uniq_chem_forms(smi_list: List[str]) -> set:
    mol_forms = set()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_form = uncharged_formula(mol)
        mol_forms.add(mol_form)
    return mol_forms


def get_pubchem_smi_list(pubchem_file, debug=False):
    """Load in pubchem"""
    smi_list = []
    with open(pubchem_file, "r") as fp:
        for index, line in enumerate(fp):
            line = line.strip()
            if line:
                smi = line.split("\t")[1].strip()
                smi_list.append(smi)
            if debug and index > 10000:
                return smi_list
    return smi_list


def create_pubchem_intermediate(pubchem_input_file, pubchem_output_file, debug=False):
    """Load in pubchem"""
    smi_list = get_pubchem_smi_list(pubchem_input_file, debug=debug)
    form_to_mols = calc_formula_to_moltuples(smi_list)
    # Formual to (smi, key) tuples
    h5obj = common.HDF5Dataset(pubchem_output_file.format('hdf5'), 'w') # export m/z < 1500 items to hdf5
    for k, v in tqdm(form_to_mols.items()):
        if len(v) >= 50:
            h5obj.write_str(k, json.dumps(list(v)))
    with open(pubchem_output_file.format('p'), "wb") as f: # export all items to pickle file
        pickle.dump(form_to_mols, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="If true, debug", default=False, action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    debug = args.debug

    # Get all possible pubchem molecules
    pubchem_file = PUBCHEM_FORMULA
    if debug:
        pubchem_file = PUBCHEM_FORMULA_DEBUG

    create_pubchem_intermediate(PUBCHEM_FILE, pubchem_file, debug=debug)

    # Load in pickled mapping 
    # print("Loading in pickled formula map")
    # with open(pubchem_file, "rb") as f:
    #     form_to_smi = pickle.load(f)
    # print("Done loading in pickled formula map")
