""" This script takes as an argument a molecule in the oracle registry (see src/foam/oracles)
and outputs a text file (to data/isomer_candidates) puwith a all isomers for that molecule """
import argparse
import pickle
from foam.utils.chem_utils import uncharged_formula
from foam.oracles import benchmark_mol_pairs
from pathlib import Path

mol_options = {entry['name'] : entry['smiles'] for entry in benchmark_mol_pairs}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecule", 
        help="molecule whose isomers will be generated", 
        default="pindolol", 
        action="store", 
        choices=list(mol_options.keys()) + ['all'],
    )

    parser.add_argument(
        "--pubchem_formula_path",
        help="path to map pubchem formula to isomers",
        default="data/pubchem/pubchem_formulae_inchikey.p",
        action="store"
    )

    return parser.parse_args()

def pubchem_formula_map(pubchem_file):
    print("Loading in pickled formula map (this may take a couple minutes)")
    with open(pubchem_file, "rb") as f:
        form_to_smi = pickle.load(f)
    print("Done loading in pickled formula map")
    return form_to_smi

def export_isomers():
    args = get_args()
    if args.molecule == 'all':
        mols = list(mol_options.keys())
    else:
        mols = [args.molecule]
    
    pubchem_file = Path(args.pubchem_formula_path)
    form_to_smi = pubchem_formula_map(pubchem_file)
    for mol in mols:
        smiles = mol_options[mol]
        formula = uncharged_formula(smiles, mol_type = "smiles")
        isomers = form_to_smi[formula]
        output_file = Path("data") / "isomer_candidates" / f"{formula}.txt"
        full_file_export = "\n".join([f"{smi}\t{ikey}"for smi, ikey in isomers])
        
        with open(output_file, "w") as fp:
            fp.write(full_file_export)

if __name__ == "__main__":
    export_isomers()
    print("Done exporting isomer file")