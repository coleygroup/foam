"""This script accepts a file of isomers and outputs a file with
a subset of isomers for use as seed molecules. Molecules are below 
some threshold similarity to the target."""
import argparse
import numpy as np 
from pathlib import Path
from foam.oracles import benchmark_mol_pairs
from foam.utils.chem_utils import uncharged_formula
from foam.utils import simple_parallel
from rdkit import Chem


mol_options = {entry['name'] : entry['smiles'] for entry in benchmark_mol_pairs}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecule", 
        help="molecule whose isomers will be generated", 
        default="enterocin", 
        action="store", 
        choices=list(mol_options.keys()) + ['all'],
    )
    parser.add_argument(
        "--isomer-lib-dir",
        help="path to isomer files",
        default="data/isomer_candidates",
        action="store"
    )
    parser.add_argument(
        "--max-seed-sim",
        help="maximum Tanimoto similarity to oracle molecule",
        default=0.1,
        action="store",
        type=float
    )
    parser.add_argument(
        "--seed-lib-size",
        help="number of molecules to include in the generated seed library",
        default=200,
        action="store",
        type=int
    )
    parser.add_argument(
        "--num-workers",
        help="number of cpus to use for fingerprint calculation",
        default=8,
        action="store",
        type=int
    )
    return parser.parse_args()

def load_smiles(isomer_lib_file):
    smiles = [line.split()[0].strip() for line in open(isomer_lib_file, "r").readlines()]
    np.random.shuffle(smiles)
    return smiles

def get_morgan_fp(mol: Chem.Mol) -> np.ndarray:
    curr_fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint

def tanimoto_sim(x: np.ndarray, y: np.ndarray):
        # check this, Calculate tanimoto distance with binary fingerprint
        intersect_mat = x[:, None, :] & y[None, :, :]
        union_mat = x[:, None, :] | y[None, :, :]

        intersection = intersect_mat.sum(-1)
        union = union_mat.sum(-1)

        ### I took the reciprocal here so instead of tanimoto sim, it became
        # distance. Could have just made negative but
        # sklearn doesn't accept negative distance matrices
        output = intersection / union
        return output


def calculate_Tani_sims(isomer_smis, oracle_smi, num_workers):
    fp = get_morgan_fp(Chem.MolFromSmiles(oracle_smi))
    fps = simple_parallel(
        isomer_smis,
        lambda x: get_morgan_fp(Chem.MolFromSmiles(x)),
        max_cpu=num_workers,
    )
    y = np.vstack(fps)
    x = fp.reshape(1, -1)
    sims = tanimoto_sim(x, y).squeeze()
    return sims

def export_seeds():
    args = get_args()
    if args.molecule == 'all':
        mols = list(mol_options.keys())
    else:
        mols = [args.molecule]
    
    for mol in mols:
        oracle_smi = mol_options[mol]
        formula = uncharged_formula(oracle_smi, mol_type="smiles")
        isomer_lib_file = Path(args.isomer_lib_dir) / f"{formula}.txt"
        isomer_smis = load_smiles(isomer_lib_file)
        np.random.shuffle(isomer_smis)
        sims = calculate_Tani_sims(isomer_smis, oracle_smi, args.num_workers)
        mask = sims < args.max_seed_sim
        selected_seeds = np.array(isomer_smis)[mask][0:args.seed_lib_size]
        out_file = Path(args.isomer_lib_dir) / f"{mol}_seeds.txt"
        full_file_export = "\n".join(selected_seeds)
        with open(out_file, "w") as fp:
            fp.write(full_file_export)

if __name__ == "__main__":
    export_seeds()
    print("Done exporting seed file")


