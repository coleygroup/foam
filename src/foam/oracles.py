""" oracles.py

File to store all different oracles

"""
import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import List, Union, Tuple
from platformdirs import user_cache_dir
from tqdm import tqdm
import hashlib
import sys, os

import foam.utils as utils
from ms_pred import common
import numpy as np
import torch
import h5py
import multiprocess
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdqueries
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# load SA oracle from rdkit
try: 
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
    from SA_Score import sascorer
except ImportError:
    # rdkit installed with pip
    from rdkit.Contrib.SA_Score import sascorer

import wandb
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from myopic_mces import MCES
import os, sys, contextlib
import pubchempy as pcp

# Define
oracle_registry = {}

# Define smiles from which to build tani oracle
def add_oracle_to_registry(name, cls):
    """add_oracle_to_registry"""
    oracle_registry[name] = cls

def register_oracle(cls):
    """register_model.

    Add an argument to the model_registry.
    Use this as a decorator on classes defined in the rest of the directory.

    """
    add_oracle_to_registry(cls.oracle_name(), cls) 
    return cls


def build_oracle(spec_id, spec_lib_dir, spec_lib_label, criteria, oracle_type = "Cos_", merge_by_precursor_mz_inchi=False, **kwargs):
    """build_oracle.

    Args:
        spec_id: Spectrum identifier
        spec_lib_dir: Spectrum library directory
        spec_lib_label: Spectrum library label file
        Criteria: similarity measure
        iceberg_param: ICEBERG weight parameter
        sa_param: SA weight parameter
        oracle_type: Choice of oracle (Tani_, Cos_, Cos_SA)


    Builds an oracle from a spectrum identifier, using the label and HDF5 files as lookup for information. 

    """
    if ".hdf5" in spec_lib_dir:
        spec_h5 = common.HDF5Dataset(spec_lib_dir) 
        spec_path = spec_h5.read_str(f'{spec_id}.ms').split('\n')
        spec_label_full = pd.read_csv(spec_lib_label, sep='\t')
        spec_label_entry = spec_label_full[spec_label_full['spec'] == spec_id]
        if spec_label_entry.shape[0] == 1:
            adduct = spec_label_entry['ionization'].values[0]
            smiles = spec_label_entry['smiles'].values[0]
            if "instrument" in spec_label_entry:
                instrument = spec_label_entry["instrument"].values[0]
            else:
                instrument = None
            print("smiles: ", smiles)
            precursor_mz = common.mass_from_smi(smiles) + common.ion2mass[adduct]
            
            if merge_by_precursor_mz_inchi:
                inchikey2d = spec_label_entry["inchikey"].values[0]
                precursor_entry = spec_label_entry["precursor"].values[0] # Likely the nicest way of getting "run"-specific. 
                print("computed precursor_mz", precursor_mz, "documented precursor mz", precursor_entry)
                # collect all spec_ids that match same "run" 

                requirements = (spec_label_full["inchikey"] == inchikey2d) & \
                                (spec_label_full["ionization"] == adduct) & \
                                (spec_label_full["precursor"] == precursor_entry)
                if "instrument" in spec_label_full:
                    requirements = requirements & (spec_label_full["instrument"] == instrument)
                spec_ids = spec_label_full[requirements]["spec"].values
                # collect all spec_ids! Oracle instantiation will take care of collating all spectra. 
                spec_path=[spec_h5.read_str(f'{spec_id1}.ms').split('\n') for spec_id1 in spec_ids]
        if instrument is None:
            oracle = dict(name=spec_id, smiles=smiles, adduct=adduct, 
                        spec_path=spec_path, precursor_mz=precursor_mz, merge_by_precursor_mz_inchi=merge_by_precursor_mz_inchi)
        else:
            oracle = dict(name=spec_id, smiles=smiles, adduct=adduct, instrument=instrument, 
                        spec_path=spec_path, precursor_mz=precursor_mz, merge_by_precursor_mz_inchi=merge_by_precursor_mz_inchi)
        
    else:
        if spec_id == "QI8422.MS" or "QI8422" in spec_id:
            oracle = dict(name=spec_id, smiles=None, adduct="[M+H]+", formula="C17H32N4O4", 
                          precursor_mz=357.2492, spec_path=spec_lib_dir + spec_id)
        elif "tripeptide_purified" in spec_id:
            oracle = dict(name=spec_id, smiles=None, adduct="[M-H]-", formula="C17H32N4O4", 
                          precursor_mz=357.2496, spec_path=spec_lib_dir + spec_id)
            
        elif spec_id == "DX_1.MS":
            oracle = dict(name=spec_id, smiles=None, adduct="[M+H]+", 
                          formula="C38H46N2O14", precursor_mz=755.3017, 
                          stepped=True, stepped_evs=[15, 30, 45],
                          spec_path=spec_lib_dir + spec_id)
        elif spec_id == "colibactin.ms":
            oracle = dict(name=spec_id, smiles=r'NC1=C2C(N(CCC3(O)NC(C(/C3=C\C(NCC4=NC(C(O)=O)=CS4)=O)=C5N[C@@H](C)CC\5)=O)C=N1)=NC=N2', adduct="[M+H]+", 
                          formula="C23H25N9O5S", precursor_mz=540.1763, 
                          stepped=True, stepped_evs=[5, 15, 25],
                          spec_path=spec_lib_dir + spec_id)
        elif spec_id == 'da.ms':
            oracle = dict(name=spec_id, smiles='OCC1C(CC(N2C=NC3=C(N=CN=C32)NCNC4=NC5=C(C(N4)=O)N=CN5C(O6)C(O)C(O)C6CO)O1)O', adduct="[M+H]+", 
                          formula="C21H26N10O8", precursor_mz=547.19,
                          spec_path=spec_lib_dir + spec_id)
    
    partial_cls = partial(_TaniOracle, **oracle)
    add_oracle_to_registry(f"Tani_{oracle['name']}", partial_cls)
    oracle["criteria"] = criteria
    partial_cls = partial(_ICEBERGOracle, **oracle)
    add_oracle_to_registry(f"Cos_{oracle['name']}", partial_cls)
    oracle["iceberg_param"] = 0.8
    oracle["sa_param"] = 0.2
    partial_cls = partial(_ICEBERGWithSAOracle, **oracle)
    add_oracle_to_registry(f"Cos_SA_{oracle['name']}", partial_cls)

    partial_cls = partial(_ICEBERGColliEngOracle, **oracle)
    add_oracle_to_registry(f"Cos_SA_ColliEng_{oracle['name']}", partial_cls)
    return f"{oracle_type}{oracle['name']}"

def get_oracle(oracle_name, spec_id=None, **kwargs):
    if spec_id is not None:
        oracle_name = build_oracle(spec_id, **kwargs)
    oracle_class = oracle_registry.get(oracle_name)
    return oracle_class(**kwargs)
    # return oracle_registry.get(oracle_name)(**kwargs)


def morgan_fp(mol, nbits=2048):
    # Copied from self get_morgan_fp definition, but need something unlinked from self to pass to workers
    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)

    return fingerprint


def clusters(colli_engs):
    if len(colli_engs) <= 5:
        return colli_engs
    colli_engs = np.array(list(colli_engs)).astype(int).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(colli_engs)
    s = np.linspace(0, max(colli_engs))
    e = kde.score_samples(s.reshape(-1, 1))
    
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    cluster_centers = set()
    if len(mi) != 0:
        mi = list(mi)
        # add min and maximum
        mi.insert(0, 0)
        mi.append(colli_engs[-1])
        lower = 0 
        upper = mi[0]
        for i_cluster in range(len(mi)):
            upper = mi[i_cluster]
            cluster = colli_engs[(colli_engs > lower) * (colli_engs <= upper)] #  s[mi][i_cluster]
            if len(cluster) == 0:
                continue
            center = cluster[len(cluster) // 2]
            cluster_centers.add(center)
            lower = upper 
        
        while len(cluster_centers) < 5: # pick mi-based cluster randomly, and add a new cluster from ma
            cluster_i = np.random.choice(len(mi) - 1)
            cluster = colli_engs[(colli_engs > mi[cluster_i]) * (colli_engs <= mi[cluster_i + 1])]
            unselected = list(set(cluster).difference(cluster_centers))
            if len(unselected) == 0:
                continue
            cluster_centers.add(np.random.choice(unselected))
    else: # implies one cluster, maximum
        cluster = colli_engs[colli_engs < ma[0]]
        # pick out 5 (?) from this single cluster
        cluster_centers = np.random.choice(cluster, np.min((len(cluster), 5)), replace=False)
    
    cluster_centers = sorted(cluster_centers)
    return [str(k) for k in cluster_centers]

class Oracle(ABC):
    def __init__(self, num_workers: int = 1, **kwargs):
        self.num_workers = num_workers

    def conditional_info(self) -> dict:
        """Return any conditional info to help optimizer (e.g., chem formula)"""
        return {}

    @staticmethod
    @abstractmethod
    def oracle_name():
        raise NotImplementedError()

    @abstractmethod
    def score_batch(self, examples: List[object], **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_seed_smiles(self) -> List[str]:
        raise NotImplementedError()

class MolOracle(Oracle, ABC):
    """Moloracle.

    Molecule oracle where we attempt to score / optimize against a set of
    example molecules

    """

    def __init__(self, mol: Chem.Mol,
                 formula: str = None,
                 seed_lib_dir: str = None,
                 pubchem_seeds: str = None,
                 extra_seeds: str = None,
                 full_seed_file: str = None,
                 spec_path: str = None, spec_lib_dir: str = None, spec_lib_label: str = None,
                 max_seed_sim: float = None, 
                 merge_by_precursor_mz_inchi: bool = False,
                 **kwargs):
        """__init__.

        Args:
            mol (Chem.Mol): Molecule to try to optimize against
            seed_lib_dir (str): Dir containing seed starting mols
            spec_path (str): Path to corresponding spectrum for mol
            spec_lib_dir (str): Dir containing spectrum library
            spec_lib_label (str): Path to spectrum labels
            seed_sim (float): Similarity of seed to target allowed
        """
        super().__init__(**kwargs)
        if seed_lib_dir:
            self.seed_lib_dir = Path(seed_lib_dir)
        if pubchem_seeds:
            self.pubchem_seeds = Path(pubchem_seeds)
        else:
            self.pubchem_seeds = None
        if extra_seeds:
            self.extra_seeds = Path(extra_seeds)
        else: 
            self.extra_seeds = None
        if full_seed_file:
            self.full_seed_file = Path(full_seed_file)
        if spec_path:
            if type(spec_path) != list:
                self.spec_path = Path(spec_path)
            else:
                self.spec_path = spec_path
                # Assume it was from HDF5 and file bits already read in
                pass
        if spec_lib_dir and spec_lib_label:
            self.spec_lib_dir = Path(spec_lib_dir)
            self.spec_lib_label = Path(spec_lib_label)
        if formula:
            self.formula = formula
            self.nbits = 2048
        if mol: 
            inchi = Chem.MolToInchi(mol)
            self.mol = common.chem_utils.canonical_mol_from_inchi(inchi)
            self.mol_smiles = Chem.MolToSmiles(self.mol)
            logging.info(f"Canonicalized smiles: {self.mol_smiles}")
            self.mol_inchikey = Chem.MolToInchiKey(self.mol)

            self.nbits = 2048
            self.get_morgan_fp = partial(morgan_fp, nbits=self.nbits)
            self.morgan_fp = self.get_morgan_fp(self.mol)
            self.max_seed_sim = max_seed_sim
        
        self.merge_by_precursor_mz_inchi = merge_by_precursor_mz_inchi


    def score_batch(self, examples: List[Chem.Mol], **kwargs) -> np.ndarray:
        """ score_batch of mols """
        non_null_inds = []
        non_null_exs = []
        for ind, ex in enumerate(examples):
            if ex is not None:
                non_null_inds.append(ind)
                non_null_exs.append(ex)

        output = np.zeros(len(examples))
        if len(non_null_exs) > 0:
            data = self.score_valid_mols(non_null_exs)
            for i, obj in enumerate(data):
                output[i, non_null_inds] = obj

        return output



    @abstractmethod
    def score_valid_mols(self, examples: List[Chem.Mol], **kwargs) -> np.ndarray: 
        raise NotImplementedError()

    def __call__(self, examples):
        """ score_batch"""
        return self.score_batch(examples)

    def conditional_info(self) -> dict:
        return {"formula": self._get_formula(), "morgan_fp": self._get_fp()}

    def get_seed_smiles(self, *args, **kwargs) -> List[str]:
        if hasattr(self, 'seed_lib_dir') and self.seed_lib_dir and hasattr(self, 'mol_smiles') and self.mol_smiles:
            print('Initializing seed smiles with Tanimoto distance')
            return self.get_seed_smiles_tani(*args, **kwargs)
        if hasattr(self, 'iceberg_model') and self.iceberg_model and hasattr(self, 'seed_lib_dir') and self.seed_lib_dir:
            print('Initializing seed smiles with ICEBERG prediction')
            return self.get_seed_smiles_iceberg(*args, **kwargs)
        elif hasattr(self, 'spec_path') and self.spec_path:
            print('Initializing seed smiles with Molecular networking')
            return self.get_seed_smiles_molnet(*args, **kwargs)
        elif hasattr(self, 'seed_list'): # has msNovelist:
            print('Initializing seed smiles with preproposed list')
            return self.get_seeds_smiles_list(*args, **kwargs)
        else:
            raise ValueError("Please specify a way of initializing seed smiles")

    def get_seed_smiles_iceberg(self, max_possible: int = 200, top_k: int = 20) -> Tuple[List[str], List[float], List[float]]:
        """
        Get seed smiles by ICEBERG prediction

        Args:
            max_possible (int): Maximum number of possible seed starting points

        """
        if not hasattr(self, 'iceberg_model') or not self.iceberg_model:
            raise ValueError("iceberg_model is missing")

        formula = self.conditional_info().get("formula")

        if self.seed_lib_dir.is_file(): # and self.seed_lib_dir.suffix == 'hdf5':
            seed_file = self.seed_lib_dir # a hdf5 file {formula: (smiles, inchikey)}
            h5obj = common.HDF5Dataset(seed_file)
            cand_str = h5obj.read_str(formula)
            # decode into candidates
            smi_inchi_list = json.loads(cand_str)
            # pickle_obj = pickle.load(open(seed_file, 'rb'))
            # smi_inchi_list = pickle_obj[formula]
            smiles = [pair[0] for pair in smi_inchi_list if '.' not in pair[0]] # if statement removes compound mixtures
        elif self.seed_lib_dir.is_dir():
            formula_file = self.seed_lib_dir / f"{formula}.txt"
            seed_file = self.seed_lib_dir / f"{self.name.split('.MS')[0]}_seeds.txt"
            if seed_file.exists():
                seed_file = seed_file
            elif formula_file.exists():
                seed_file = formula_file
                raise NotImplementedError
            else:
                raise ValueError
            smiles = [line.split()[0].strip() for line in open(seed_file, "r").readlines()]
            if self.pubchem_seeds:
                seed_file = self.pubchem_seeds # a hdf5 file {formula: (smiles, inchikey)}
                h5obj = common.HDF5Dataset(seed_file)
                cand_str = h5obj.read_str(formula)
                # decode into candidates
                smi_inchi_list = json.loads(cand_str)
                # pickle_obj = pickle.load(open(seed_file, 'rb'))
                # smi_inchi_list = pickle_obj[formula]
                smiles = smiles + [pair[0] for pair in smi_inchi_list if '.' not in pair[0]]
                
            # tautomerize
            smiles = utils.simple_parallel(
                smiles,
                lambda x: utils.tautomerize_smi(x),
                max_cpu=self.num_workers,
                task_name="Tautomerizing seed smiles"
            )
            # remove stereochemistry
            def remove_stereo(smi):
                mol = Chem.MolFromSmiles(smi)
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol)
            smiles = [remove_stereo(smi) for smi in smiles]
        else:
            raise ValueError

        # predict spectrum by ICEBERG
        # will be ordered based on criteria; criteria scores will come first. 
        sims = self.score_valid_mols(smiles)

        # return top_k highest score mols + random mols
        if len(np.array(sims).shape) >= 2:
            sorted_indices = np.argsort(sims[0])
        else:
            sorted_indices = np.argsort(sims)
        if top_k < len(sorted_indices):
            num_rand = min(max_possible - top_k, len(sorted_indices) - top_k)
            rand_indices = np.random.choice(sorted_indices[:-top_k], num_rand, replace=False)
            selected_indices = np.concatenate((sorted_indices[-top_k:], rand_indices))
        else:
            selected_indices = sorted_indices[-top_k:]
        # selected_indices = rand_indices
        smiles = np.array(smiles)[selected_indices]
        if len(np.array(sims).shape) >= 2:
            sims = np.array(sims)[:, selected_indices]
            return smiles.tolist(), sims[0].tolist(), sims[1].tolist()
        else:
            sims = np.array(sims)[selected_indices]
            return smiles.tolist(), sims.tolist()

    def get_seed_smiles_tani(self, max_possible: int = 200) -> List[str]:
        """get_seed_smiles.

        Args:
            max_possible (int): Maximum number of possible seed starting points

        """
        formula = self.conditional_info().get("formula")
        fp = self.conditional_info().get("morgan_fp")
        using_diffms_seeds = False
        
        if self.seed_lib_dir.is_file(): 
            seed_file = self.seed_lib_dir
            h5obj = common.HDF5Dataset(seed_file)
            if formula in h5obj:
                cand_str = h5obj.read_str(formula)
                smi_inchi_list = json.loads(cand_str) # json.loads(cand_str.decode('utf-8'))
                
            elif self.name in h5obj:
                with h5py.File(self.seed_lib_dir, "r") as h5f:
                    inchis = list(h5f[self.name])
                inchis = [i.decode("utf-8") for i in inchis]
                extra_smis = [Chem.MolToSmiles(Chem.MolFromInchi(inchi), isomericSmiles=False) for inchi in inchis if inchi is not None]
                extra_smis = list(set([utils.tautomerize_smi(x) for x in extra_smis]))
                smiles = []
                smiles_all = smiles + extra_smis
                smiles_all = list(set(smiles_all))
                
                # logging.info(f"Added {len(extra_smis)} extra seeds from {self.extra_seeds} for {self.name}, total now {len(smiles_all)}")
                extra_seeds_mask = np.array([smi in extra_smis for smi in smiles_all])
                smi_inchi_list = [(smi, Chem.MolToInchiKey(Chem.MolFromSmiles(smi))) for smi in smiles_all]
                using_extra_seeds = True

            else:
                # Implies formula not found in compressed fast look up file
                # Should load from the pickle file (takes longer)
                logging.info(f"Uncommon formula: {formula}, need to load full file...")
                print(self.full_seed_file)
                if not self.full_seed_file.is_file():
                    # instead of raising error, should just be logging and sys.exit(0).
                    # ValueError("No fallback seed file is provided; exiting.")
                    logging.error("No fallback seed file is provided; exiting.")
                    sys.exit(0)
                # Add tag
                # if wandb.run is not None:
                #     wandb.run.tags += ('uncommon_formula',)
                # pickle_obj = pickle.load(open(self.full_seed_file, 'rb'))
                # logging.info("Loaded full PubChem file.")
                # smi_inchi_list = pickle_obj[formula]
                try: 
                    compounds = pcp.get_compounds(formula, namespace='formula')
                    logging.info("Loaded matches via PubChem API.")
                    smi_list = list(set([cmpd.canonical_smiles for cmpd in compounds]))
                    if len(smi_list) and smi_list[0] is None: # try utilizing Inchi instead. 
                        inchi_list = [cmpd.inchi for cmpd in compounds]
                        smi_list = list(set([Chem.MolToSmiles(Chem.MolFromInchi(inchi), isomericSmiles=False) for inchi in inchi_list if inchi is not None]))
        
                    smi_list = list(set([utils.tautomerize_smi(x) for x in smi_list]))
                    smi_inchi_list = [(smi, Chem.MolToInchiKey(Chem.MolFromSmiles(smi))) for smi in smi_list]
                except Exception as e:
                    logging.error(f"Error loading matches via PubChem API: {e}")
                    pickle_obj = pickle.load(open(self.full_seed_file, 'rb'))
                    logging.info("Loaded full PubChem file.")
                    smi_inchi_list = pickle_obj[formula]


            smiles = [pair[0] for pair in smi_inchi_list if '.' not in pair[0]]  # if statement removes compound mixtures
        elif self.seed_lib_dir.is_dir():
            formula_file = self.seed_lib_dir / f"{formula}.txt"
            seed_file = self.seed_lib_dir / f"{self.name}_seeds.txt" 
            if seed_file.exists():
                seed_file = seed_file
            elif formula_file.exists():
                seed_file = formula_file
                raise NotImplementedError
            else:
                raise ValueError
            smiles = [line.split()[0].strip() for line in open(seed_file, "r").readlines()]
            if self.pubchem_seeds:
                seed_file = self.pubchem_seeds # a hdf5 file {formula: (smiles, inchikey)}
                h5obj = common.HDF5Dataset(seed_file)
                cand_str = h5obj.read_str(formula)
                # decode into candidates
                smi_inchi_list = json.loads(cand_str)
                # pickle_obj = pickle.load(open(seed_file, 'rb'))
                # smi_inchi_list = pickle_obj[formula]
                smiles = smiles + [pair[0] for pair in smi_inchi_list if '.' not in pair[0]]
        else:
            raise ValueError

        if self.extra_seeds:
            # load from h5 file
            # Only implemented for MassSpecGym: {MassSpecGymID_XXXXXXX :[inchi1, inchi2, ...], ...}
            # Not indexed by smiles only, since these are **spectra-conditioned** derived candidates.
            with h5py.File(self.extra_seeds, "r") as h5f:
                extra_inchis = list(h5f[self.name])
                extra_inchis = [i.decode("utf-8") for i in extra_inchis]
                extra_smis = [Chem.MolToSmiles(Chem.MolFromInchi(inchi), isomericSmiles=False) for inchi in extra_inchis if inchi is not None]
                extra_smis = list(set([utils.tautomerize_smi(x) for x in extra_smis]))
                smiles_all = smiles + extra_smis
                smiles_all = list(set(smiles_all))
                logging.info(f"Added {len(extra_smis)} extra seeds from {self.extra_seeds} for {self.name}, total now {len(smiles_all)}")
                sources = []
                for smi in smiles_all:
                    if smi in extra_smis and smi not in smiles:
                        sources.append('diffms')
                    elif smi in smiles and smi in extra_smis:
                        sources.append('both')
                    else:
                        sources.append('pubchem')

                using_extra_seeds = True
                smiles = smiles_all

        
        # Tautomerize all seeds to ensure deduplication. Set call will reorder list so reconstruct extra_seeds_mask accordingly        
        smiles = list(set([utils.tautomerize_smi(x) for x in smiles]))
        smiles = [smi for smi in smiles if smi is not None] 
        if using_extra_seeds:
            # extra_smis is already tautomerized. 
            extra_seeds_mask = np.array([smi in extra_smis for smi in smiles])
            # sources = np.array(sources)[seed_idx_ordering]

        if wandb.run is not None:
            wandb.run.summary["num_seeds"] = len(smiles)

        # Shuffle index set, so that the indexing can be used for extra seeds. 
        seed_idx_ordering = np.arange(len(smiles))
        np.random.shuffle(seed_idx_ordering)
        smiles = np.array(smiles)[seed_idx_ordering]
        if using_extra_seeds:
            # The mask needs to be shuffled to retain the same order as the smiles. 
            extra_seeds_mask = extra_seeds_mask[seed_idx_ordering]
            # sources = np.array(sources)[seed_idx_ordering]

        if max_possible > 0: # do not downselect if -1 (or multiple of, e.g. -5 from num_islands = 5)
            smiles = smiles[:max_possible]
        if len(smiles) == 0:
            raise ValueError("No seed smiles found in PubChem database")

        # Filter by tani sim
        nbits = self.nbits 

        time_start = time.time()
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits)
        fps = [mfpgen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
        time_end = time.time()
        logging.info(f"Computed {len(smiles)} fingerprints via rdkit generator in {time_end - time_start} seconds.")
    
        y = np.vstack(fps)
        x = fp.reshape(1, -1)
        sims = self._tanimoto_sim(x, y).flatten()
        if self.max_seed_sim != 1:
            mask = sims < self.max_seed_sim
        else: # interpret as only exact match exclusion.
            mask = sims > -1 
        # drop any mols with inchikey matches
        inchis = np.array([Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in smiles])
        
        mask = mask & ((inchis != self.mol_inchikey) & (inchis != self.smi_inchikey))
        # if extra_seeds provided, do not filter those out even if inchikey matches
        if using_extra_seeds:
            mask = mask | extra_seeds_mask
        # log the maximum tanimoto similarity
        if wandb.run is not None:
            if sum(mask):
                wandb.run.summary["max_seed_sim"] = np.max(sims[mask])
                wandb.run.summary["mean_seed_sim"] = np.mean(sims[mask])
                if using_extra_seeds:
                    # See if inchi showed up in the set we're about to pass through. 
                    # Need to mask first to ensure that the true mol doesn't get included here by accident. 
                    seed_inchis = inchis[mask]
                    seed_inchi_check = any((seed_inchis == self.mol_inchikey) | (seed_inchis == self.smi_inchikey) )
                    wandb.run.summary["target_in_diffms_seeds"] = seed_inchi_check
            else: # no seeds :(
                wandb.run.summary["max_seed_sim"] = 0.0
                wandb.run.summary["mean_seed_sim"] = 0.0
                wandb.run_summary["target_in_diffms_seeds"] = None
        
        # macrocycle_mask = np.array([max([len(j) for j in Chem.rdmolops.GetSymmSSSR(Chem.MolFromSmiles(smi))]) for smi in smiles])
        # mask = mask & (macrocycle_mask > 8)
        smiles = np.array(smiles)[mask]
        return smiles.tolist()
        # if self.extra_seeds: -- a bit annoying to propagate :( 
        #     sources = np.array(sources)[mask]
        #     return smiles.tolist(), sources.tolist()
        # else:
        #     return smiles.tolist()

    def get_seed_smiles_molnet(self, max_possible: int = 200) -> List[str]:
        """
        Get seed smiles through molecular network
        """
        meta, tuples = common.parse_spectra(self.spec_path)
        spec = common.process_spec_file(meta, tuples, merge_specs=True)
        binned_spec = common.bin_spectra([spec])[0]
        label_md5 = hashlib.md5(open(self.spec_lib_label).read().encode('utf-8')).hexdigest()
        cache_dir = Path(user_cache_dir('foam'))
        md5_path = cache_dir / 'checksum'
        spec_lib_cache_path = cache_dir / 'spec_lib.npz'
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        if md5_path.exists() and open(md5_path).read() == label_md5:
            spec_cache = np.load(spec_lib_cache_path)
            spec_lib_names, spec_lib_smis, spec_lib_binned_specs, spec_lib_inchikeys = \
                spec_cache['names'], spec_cache['smiles'], spec_cache['binned_specs'], spec_cache['inchikeys']
        else:
            spec_lib_label = pd.read_csv(self.spec_lib_label, sep='\t')
            if self.spec_lib_dir.is_file() and self.spec_lib_dir.suffix == '.hdf5':
                spec_h5 = common.HDF5Dataset(self.spec_lib_dir)
                spec_lib_names, spec_lib_specs, spec_lib_smis, spec_lib_inchikeys = [], [], [], []
                print('Parsing spectrum from library')
                for spec_name, smi, inchikey in tqdm(zip(spec_lib_label['spec'], spec_lib_label['smiles'], spec_lib_label['inchikey']), total=spec_lib_label.shape[0]):
                    meta, tuples = common.parse_spectra(spec_h5.read_str(f'{spec_name}.ms').split(b'\n\n'))
                    spec = common.process_spec_file(meta, tuples, merge_specs=True)
                    spec_lib_names.append(spec_name)
                    spec_lib_specs.append(spec)
                    spec_lib_smis.append(smi)
                    spec_lib_inchikeys.append(inchikey)
                spec_lib_names = np.array(spec_lib_names)
                spec_lib_smis = np.array(spec_lib_smis)
                spec_lib_inchikeys = np.array(spec_lib_inchikeys)
                spec_lib_binned_specs = common.bin_spectra(spec_lib_specs)
                np.savez_compressed(spec_lib_cache_path,
                                    names=spec_lib_names,
                                    smiles=spec_lib_smis,
                                    inchikeys=spec_lib_inchikeys,
                                    binned_specs=spec_lib_binned_specs)
                with open(md5_path, 'w') as f:
                    f.write(label_md5)
            else:
                NotImplementedError()
        mask = Chem.MolToInchiKey(self.mol) != spec_lib_inchikeys # remove the true mol in testing
        spec_lib_smis = spec_lib_smis[mask]
        spec_lib_binned_specs = spec_lib_binned_specs[mask]
        norm_inp = np.linalg.norm(binned_spec, axis=-1)
        norm_lib = np.linalg.norm(spec_lib_binned_specs, axis=-1)
        cos_sims = binned_spec[None, :] @ spec_lib_binned_specs.T / (norm_inp * norm_lib)
        rank_indices = np.argsort(cos_sims, axis=-1)[0, -max_possible:]
        smiles = spec_lib_smis[rank_indices]
        return smiles.tolist()

    def get_seed_smiles_list(self, max_possible: int = 200) -> List[str]:
        with open(self.seed_list, 'r') as f:
            seeds = f.readlines()
        # exclude any seeds that are the same as the target
        if self.exclude_true:
            seeds = [s for s in seeds if s != self.mol_smiles]
        return seeds[:max_possible]
    

    def _get_formula(self) -> str:
        if self.mol is None:
            return self.formula
        return CalcMolFormula(self.mol)

    def _get_fp(self) -> np.ndarray:
        if self.mol is None:
            return None
        return self.get_morgan_fp(self.mol)

    def get_morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        """get_morgan_fp.

        Args:
            Get morgan fingerprint

        """
        curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.nbits)

        fingerprint = np.zeros((0,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
        return fingerprint

    def _tanimoto_sim(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # Calculate tanimoto distance with binary fingerprint
        intersect_mat = x[:, None, :] & y[None, :, :]
        union_mat = x[:, None, :] | y[None, :, :]

        intersection = intersect_mat.sum(-1)
        union = union_mat.sum(-1)
        output = intersection / union
        return output
    

    def mces(self, smi, threshold=15, time_limit=900):
        if self.smi is None or smi is None:
            return np.nan
        
        with contextlib.redirect_stdout(sys.stderr):
            mces = MCES(self.mol_smiles, smi, solver="PULP_CBC_CMD",
                        threshold=threshold, solver_options={"timeLimit": time_limit, "threads": self.num_workers},
                        catch_errors=True)
        if mces[1] == -1:
            return np.nan
        return mces[1] # returns objective value

    def score_sa(self, examples: List[Union[Chem.Mol, str]], **kwargs) -> np.ndarray:
        """ score_sa a batch of Mols """
        non_null_inds = []
        non_null_exs = []
        for ind, ex in enumerate(examples):
            if ex is not None:
                non_null_inds.append(ind)
                non_null_exs.append(ex)

        if len(non_null_exs) > 0 and type(non_null_exs[0]) == str:
            non_null_exs = [Chem.MolFromSmiles(smi) for smi in non_null_exs]
        output = np.zeros(len(examples))

        if len(non_null_exs) > 0:
            sa_oracle = lambda x: [sascorer.calculateScore(mol) for mol in x]
            non_null_scores = np.array(sa_oracle(non_null_exs))
            output[non_null_inds] = non_null_scores

        return output


class _TaniOracle(MolOracle):
    def __init__(self, smiles: str, name: str, **kwargs):
        """__init__.

        Args:
            mol (Chem.Mol): Molecule to try to optimize against
            name:

        """
        mol = Chem.MolFromSmiles(smiles)
        self.name = name
        super().__init__(mol, **kwargs)

    def score_valid_mols(self, examples: List[Chem.Mol], **kwargs) -> List[float]:
        """score_batch."""
        morgan_fps = np.stack(
            [
                self.get_morgan_fp(i) if i is not None else np.zeros(self.nbits)
                for i in examples
            ]
        )
        output_scores = self._tanimoto_sim(self.morgan_fp[None, :], morgan_fps)

        # Compute the score
        return output_scores.reshape(-1)

    @staticmethod
    def oracle_name():
        return "TaniOracle"


class _ICEBERGOracle(MolOracle):
    def __init__(self, smiles: str, name: str,
                 ignore_precursor_peak: bool = False,
                 denoise_specs: bool = False, 
                 gen_model_ckpt: str = "checkpoints/nist20/dag_gen_nist20/best.ckpt",
                 inten_model_ckpt: str = "checkpoints/nist20/dag_inten_nist20/best.ckpt",
                 num_bins: int = 15000,
                 mass_upper_limit: int = 1500,
                 criteria: str = "cosine",
                 ## GPU and multi-node setup
                 gpu_workers: int = 0,
                 batch_size: int = 32, 
                 use_multi_node: bool = False, 
                 server_uri: str = None,
                 ## energy parameters
                 nce: bool = False,
                 limited_evs: list = None,
                 use_clustered_evs: bool = False,
                 stepped: bool = False, 
                 stepped_evs: list = None,
                 # other alternate parameters
                 use_iceberg_spectra: bool = False,
                 **kwargs):
        """__init__.

        Args:
            mol (Chem.Mol): Molecule to try to optimize against
            name:
            model_ckpt: Checkpoint for ms ffn model

        """
        from ms_pred.dag_pred import inten_model, gen_model, joint_model
        self.name = name
        
        if smiles is None:
            self.smi = None
            self.deploy_mode = True
            self.mol = None
            super().__init__(mol=self.mol, **kwargs)
        else:
            self.smi = smiles
            mol = Chem.MolFromSmiles(smiles)
            Chem.RemoveStereochemistry(mol)
            self.smi_inchikey = Chem.MolToInchiKey(mol) 
            super().__init__(mol, **kwargs)
            # removing stereochemistry so that the match occurs correctly for SMILES where the stereochemistry is encoded
            # may be worth adding a flag to keep stereochemistry in the future
            
            # Getting tautomer as made in tautomer call
            self.old_smi = self.smi
            self.smi = self.mol_smiles
            # should already be tautomer saved to mol_inchikey
            #self.mol_inchikey = Chem.MolToInchiKey(mol)

        self.gpu_workers = gpu_workers
        self.batch_size = batch_size

        self.use_multi_node = use_multi_node
        self.server_uri = server_uri
        if self.use_multi_node and self.server_uri:
            import Pyro4
            self.iceberg_model = Pyro4.Proxy(self.server_uri) 
            inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_model_ckpt, strict=False, map_location='cpu')
            self.iceberg_model._pyroSerializer = "pickle"
            # Supply the rest of the info at runtime? 

        elif self.use_multi_node:
            raise ValueError("A server URI was not specified to self.server_uri, yet RPC inference was requested")
        else: # continue with setup on this node
            inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_model_ckpt, strict=False, map_location='cpu')
            gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_model_ckpt, strict=False, map_location='cpu')

            self.iceberg_model = joint_model.JointModel(
                gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
            )
        
        self.ignore_precursor_peak = ignore_precursor_peak
        self.use_iceberg_spectra = use_iceberg_spectra
        self.stepped = stepped
        self.stepped_evs = stepped_evs

        # ICEBERG argument keywords
        iceberg_kws = ["precursor_mz", 
                       "adduct", 
                       "instrument",
                       "threshold", 
                       "device", 
                       "max_nodes"]
        self.iceberg_kwargs = {'adduct_shift': True}
        for kw in iceberg_kws:
            if kw not in kwargs:
                if kw == "instrument":
                    print("Instrument not specified, verify this is intended")
                    continue
                else:
                    raise ValueError(f"ICEBERG argument not specified: {kw}")
            self.iceberg_kwargs[kw] = kwargs[kw]
        self.precursor_mz = kwargs["precursor_mz"]
        self.adduct = kwargs["adduct"]
        wandb.run.summary["adduct"] = self.adduct
        self.instrument =  kwargs.get("instrument", None) # Will also default to None from build_oracle. 
        wandb.run.summary["instrument"] = self.instrument
        if self.iceberg_kwargs["device"] == "gpu":
            self.iceberg_kwargs["device"] = "cuda:0"
        self.num_bins = num_bins
        self.mass_upper_limit = mass_upper_limit
        self.bin_multiplier = num_bins / mass_upper_limit
        self.criteria = criteria
        self.denoise_specs = denoise_specs
        

        # parameters from the model
        self.iceberg_kwargs["binned_out"] = inten_model_obj.binned_targs
        if not self.iceberg_kwargs['binned_out']:
            raise NotImplementedError
        if self.merge_by_precursor_mz_inchi: 
            all_tuples = [] 
            all_spec_paths = self.spec_path
            for entry in all_spec_paths: # this retains only the last meta object
                meta, tuples = common.parse_spectra(entry)
                all_tuples += tuples
            logging.info(f"Retrieved {len(all_tuples)} from library with matching experimental info")
            all_specs = common.process_spec_file(meta, all_tuples, merge_specs=False)
            self.ref_spec = all_specs

        else: 
            meta, tuples = common.parse_spectra(self.spec_path)
            self.ref_spec = common.process_spec_file(meta, tuples, merge_specs=False)

        if wandb.run is not None:
            wandb.run.summary["num_colli_engs"] = len(self.ref_spec)
            wandb.run.summary["imputed_eV"] = any(["imputed" in collieng for collieng in self.ref_spec])
            wandb.run.summary["num_ref_peaks_unclean"] = sum(len(v) for v in self.ref_spec.values())
            #clean spectra
        if self.denoise_specs:
            real_spec = [(k, common.max_inten_spec(v, max_num_inten=20, inten_thresh=0.05)) for k, v in self.ref_spec.items()]
            self.ref_spec = {k: common.combined_electronic_denoising(v) for k, v in real_spec}
            # self.ref_spec = common.denoise_spectra_dict(self.ref_spec, 
            #                                             experimental=True, 
            #                                             smiles=self.smi, adduct=self.adduct, precursor_mz=self.precursor_mz)
        # keep unbinned 
        wandb.run.summary["num_ref_peaks_cleaned"] = sum(len(v) for v in self.ref_spec.values())
        self.ref_spec_unbinned = {
            common.get_collision_energy(k): v for k, v in self.ref_spec.items()}

        print("Number of energies", len(self.ref_spec))
        # collect binned for scoring
        self.ref_spec = {
            common.get_collision_energy(k): common.bin_spectra([v], num_bins, mass_upper_limit)[0]
            for k, v in self.ref_spec.items()}
        self.colli_engs = list(self.ref_spec.keys())

        # Process stepping before nce, since stepped spectra usually passed as {'nan':} dicts already
        if self.stepped:
            self.colli_engs = self.stepped_evs
            self.ref_spec = {"nan": self.ref_spec[k] for k in self.ref_spec.keys()}

        # normalize to NCE if needed. 
        if nce:
            if self.stepped:
                colli_engs = [common.nce_to_ev(k, self.precursor_mz) for k in self.colli_engs]
                self.colli_engs = [f'{float(k):.0f}' for k in colli_engs]
            else:
                self.ref_spec = {common.nce_to_ev(k, self.precursor_mz): v for k, v in self.ref_spec.items()}
                self.ref_spec = {f'{float(k):.0f}': v for k, v in self.ref_spec.items()}
                self.colli_engs = list(self.ref_spec.keys())
            logging.info("Convert energies to run ICEBERG at: " + str(self.colli_engs))
        
        if not self.stepped and not nce:
            self.colli_engs = list(self.ref_spec.keys())

        self.limited_evs = limited_evs
        if self.limited_evs:
            self.ref_spec = {str(k): self.ref_spec[str(k)] for k in self.limited_evs}
            self.colli_engs = self.ref_spec.keys()


        self.selected_evs = None
        if use_clustered_evs:
            self.selected_evs = clusters(self.colli_engs)
            logging.info(f"Selected collision energies: {self.selected_evs}")

        if self.use_iceberg_spectra and self.smi and self.spec_path:
            self.ref_spec = self.pred_unbinned_spec([self.mol_smiles])[0]
            
        if self.smi: # is already tautomerized    
            self.self_iceberg_scores = self.score_valid_mols([self.mol])
            self.sa_score_target = 1 - (self.score_sa([self.mol_smiles]) - 1)/ 9
            self.self_spec_unbinned = self.pred_unbinned_spec([self.mol_smiles])[0]
            
        else:
            self.self_iceberg_scores = None
            self.sa_score_target = None
            self.self_spec_unbinned = None

    def pred_unbinned_spec(self, smiles): # used for eval plots, does this need to be on gpu, overhead here could be slow. 
        if self.use_multi_node:
            single_kwargs = self.iceberg_kwargs.copy()
            if isinstance(single_kwargs.get("device"), torch.device):
                single_kwargs["device"] = str(single_kwargs["device"])
            single_kwargs.update({
                "collision_eng":    self.colli_engs,
                "num_bins":         self.num_bins,
                "mass_upper_limit": self.mass_upper_limit,
                "final_binned":     False,
            })
            all_pred_specs, self.gpu_workers = self.iceberg_model.predict_mol(smiles, 
                                                            gpu_workers=self.gpu_workers, 
                                                            iceberg_kwargs=single_kwargs)
            
            # Convert back to numpy array:
            all_pred_specs = [{k: np.array(v) for k, v in spec.items()} for spec in all_pred_specs]
            
            return all_pred_specs
                    
        else: 
            avail_gpu_num = torch.cuda.device_count()
            def batch_predict_mol(batch):
                torch.set_num_threads(1)
                try: 
                    if avail_gpu_num > 0:
                        if self.gpu_workers > 0:
                            worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                            gpu_id = worker_id % avail_gpu_num
                            device = f"cuda:{gpu_id}"
                        else:
                            device = self.iceberg_kwargs["device"]
                    else:
                        device = "cpu"
                    torch.cuda.set_device(device)
                    self.iceberg_model.to(device)
                    if "instrument" in batch:

                        batched_specs = self.iceberg_model.predict_mol(smi=batch["smiles"],
                                                                        collision_eng=batch["collision_eng"],
                                                                        adduct=batch["adducts"],
                                                                        precursor_mz=batch["precursor_mz"],
                                                                        instrument=batch["instrument"],
                                                                        threshold=self.iceberg_kwargs["threshold"],
                                                                        device=device, # self.iceberg_kwargs["device"],
                                                                        max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                        binned_out=False, #self.iceberg_kwargs["binned_out"],
                                                                        adduct_shift=self.iceberg_kwargs["adduct_shift"],

                        )
                    else:
                        batched_specs = self.iceberg_model.predict_mol(smi=batch["smiles"],
                                                                        collision_eng=batch["collision_eng"],
                                                                        adduct=batch["adducts"],
                                                                        precursor_mz=batch["precursor_mz"],
                                                                        threshold=self.iceberg_kwargs["threshold"],
                                                                        device=device, # self.iceberg_kwargs["device"],
                                                                        max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                        binned_out=False, #self.iceberg_kwargs["binned_out"],
                                                                        adduct_shift=self.iceberg_kwargs["adduct_shift"],
                        )

                    # clean up each spectra, then bin and save.
                    binned = []
                    for spec in batched_specs['spec']:
                        if type(spec) == torch.Tensor:
                            spec = spec.cpu().numpy()
                        # if self.denoise_specs:
                        #     spec = common.denoise_spectrum(spec)
                        spec = spec[spec[:, 1] > 0.03]
                        unique_mz, inv = np.unique(spec[:, 0], return_inverse=True)
                        summed_inten = np.bincount(inv, weights=spec[:, 1])
                        slim_spec = np.column_stack((unique_mz, summed_inten))
                        if len(slim_spec):
                            slim_spec[:, 1] = slim_spec[:, 1] / slim_spec[:, 1].max()
                        
                        binned.append(slim_spec)
                    batched_specs['spec'] = binned
                    torch.cuda.empty_cache()

                except RuntimeError as err: # if an invalid SMILES is encountered 
                    torch.cuda.empty_cache()
                    if "CUDA error: out of memory" in str(err) or "CUDA out of memory" in str(err):
                        print(err)
                        return "cuda" # raise err
                    else:
                        print(err)
                        return "error" 
                        # Will reprocess one by one after batch callds
                except AttributeError as err:
                    print(err)
                    raise
                    return "error"
                    
                except Exception as err:
                    raise
                    if "CUDA" in str(err):
                        print(err)
                        return "cuda"
                    else:
                        # raise
                        raise

                return batched_specs #, binned_batched_specs

            with torch.no_grad():
                self.iceberg_model.eval()
                self.iceberg_model.freeze()
                # self.iceberg_model.to(self.iceberg_kwargs["device"])
                # construct batch: (size of the batch is currently size of smiles, can truncate if need be. )
                collision_engs = [float(a) for a in self.colli_engs]
                if type(self.iceberg_kwargs['adduct']) == str:
                    self.iceberg_kwargs['adduct'] = [self.iceberg_kwargs['adduct']]
                if type(self.iceberg_kwargs.get("instrument", None)) == str:
                    self.iceberg_kwargs['instrument'] = [self.iceberg_kwargs['instrument']]
                # fix to break up list of smiles into batch_size as desired, and make sure batch works
                full = {"smiles": np.repeat(np.array(smiles), len(collision_engs)),
                         "collision_eng": collision_engs * len(smiles),
                         "adducts": self.iceberg_kwargs['adduct'] * len(smiles) * len(collision_engs),
                         "precursor_mz": [self.precursor_mz] * len(smiles) * len(collision_engs),
                } 
                if self.iceberg_kwargs.get("instrument", None):
                    full["instrument"] = self.iceberg_kwargs["instrument"] * len(smiles) * len(collision_engs)
                
                batch_size = self.batch_size
                batches = [{k: v[i:i+batch_size] for k, v in full.items()} for i in range(0, len(full["smiles"]), batch_size)]
                # TODO: could estimate num workers properly
                max_retries = 3
                num_retries = 0
                batches_specs = [''] * len(batches)
                process_indices = list(range(len(batches)))
                while len(process_indices) > 0 and num_retries < max_retries:
                    process_batches = [batches[i] for i in process_indices]

                    if self.iceberg_kwargs["device"] != 'cpu':

                        result = utils.chunked_parallel_retries(process_batches, batch_predict_mol, chunks=4000, max_cpu=self.gpu_workers,
                                                            task_name="Batched ICEBERG scoring")
                        for idx, p_batch in zip(process_indices, result):
                            if type(p_batch) != str:
                                batches_specs[idx] = p_batch
                        
                        process_indices = [idx for idx, p_batch in zip(process_indices, result) if type(p_batch) == str]
                        if len(process_indices):
                            reduce = int(np.ceil(len(process_indices) / len(result) * self.gpu_workers)) # failure rate * num workers is effective gpu worker fail rate. 
                            self.gpu_workers = max(self.gpu_workers - reduce, 1)
                            logging.info(f"Turning num gpu workers from {self.gpu_workers + reduce} to {self.gpu_workers}")
                            num_retries += 1
                            # Resubmit. 
                    else:
                        iceberg_workers = self.num_workers
                        result = utils.chunked_parallel_retries(process_batches, batch_predict_mol, chunks=4000, max_cpu=iceberg_workers,
                                                            task_name="Batched ICEBERG scoring")
                        for idx, p_batch in zip(process_indices, result):
                            if type(p_batch) != str:
                                batches_specs[idx] = p_batch
                        
                        process_indices = [idx for idx, p_batch in zip(process_indices, result) if type(p_batch) == str]
                        if len(process_indices):
                            reduce = int(np.ceil(len(process_indices) / len(result) * iceberg_workers)) # failure rate * num workers is effective gpu worker fail rate. 
                            iceberg_workers = max(iceberg_workers - reduce, 1)
                            logging.info(f"Turning num gpu workers from {iceberg_workers + reduce} to {iceberg_workers}")
                            num_retries += 1
                            # Resubmit. 

                logging.info("Scoring finished with ICEBERG")

            # flatten batches_specs
            all_specs_raw = [spec for batch in batches_specs for spec in batch["spec"]]
            all_pred_specs = []

            for i in range(len(smiles)):
                start = i * len(self.colli_engs)
                end = (i + 1) * len(self.colli_engs)
                if any([s is None for s in all_specs_raw[start:end]]):
                    print(f"{smiles[i]} errored")
                    all_pred_specs.append(None)
                    continue
                spec_dict = {str(ce_eng): spec for ce_eng, spec in zip(self.colli_engs, 
                                                                        all_specs_raw[start:end])}
                all_pred_specs.append(spec_dict)
            return all_pred_specs


    def score_valid_mols(self, examples: List[Union[Chem.Mol, str]], agg=True, **kwargs) -> Tuple[List[float], List[float]]:
        """score_batch."""
        from ms_pred.retrieval.retrieval_benchmark import dist_bin

        if len(examples) > 0 and type(examples[0]) is Chem.Mol:
            smiles = [Chem.MolToSmiles(m) for m in examples]
        else: # type(examples[0]) is SMILES string
            smiles = examples

        def pred_single_smile(smi):
            torch.set_num_threads(1)
            with torch.no_grad():
                pred_specs = {}
                for eng in self.colli_engs:
                    try:
                        full_output = self.iceberg_model.predict_mol(smi,
                                                                     collision_eng=float(eng),
                                                                     **self.iceberg_kwargs)
                    except RuntimeError as err:
                        print(err)
                        return None
                    pred_spec = full_output['spec']
                    pred_specs[eng] = pred_spec
                return pred_specs
            
        def pred_single_tuple(inputs):
            torch.set_num_threads(1)
            
            avail_gpu_num = torch.cuda.device_count()

            if avail_gpu_num >= 0:
                if self.gpu_workers > 0:
                    worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                    gpu_id = worker_id % avail_gpu_num
                else:
                    gpu_id = 0
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
            with torch.no_grad():
                try: 
                    instrument = None
                    if len(inputs) == 5:
                        smi, collision_eng, adduct, precursor_mz, instrument = inputs
                    elif len(inputs) == 4:
                        smi, collision_eng, adduct, precursor_mz = inputs
                    else:
                        raise ValueError("num of exp. parameters not supported", inputs)
                    smi = str(smi) # in case it's a numpy string
                    if instrument is not None:

                        full_output = self.iceberg_model.predict_mol(smi, 
                                                                    collision_eng=collision_eng,
                                                                    adduct=adduct,
                                                                    precursor_mz=precursor_mz,
                                                                    instrument=instrument,
                                                                    threshold=self.iceberg_kwargs["threshold"],
                                                                    device=torch.device(device), #self.iceberg_kwargs["device"],
                                                                    max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                    binned_out=self.iceberg_kwargs["binned_out"],
                                                                    adduct_shift=self.iceberg_kwargs["adduct_shift"]
                        )
                    else:
                        full_output = self.iceberg_model.predict_mol(smi, 
                                                                    collision_eng=collision_eng,
                                                                    adduct=adduct,
                                                                    precursor_mz=precursor_mz,
                                                                    threshold=self.iceberg_kwargs["threshold"],
                                                                    device=torch.device(device), #self.iceberg_kwargs["device"],
                                                                    max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                    binned_out=self.iceberg_kwargs["binned_out"],
                                                                    adduct_shift=self.iceberg_kwargs["adduct_shift"]
                        )
                    torch.cuda.empty_cache()


                except RuntimeError as err: 
                    torch.cuda.empty_cache()
                    print("pred_single_tuple was run, with device = ", self.iceberg_kwargs["device"])
                    print(err)
                    return None
                except AttributeError as err:
                    torch.cuda.empty_cache()
                    print(err)
                    print(inputs)
                    print("smi", smi)
                    return None
                return full_output["spec"]
                
        if self.iceberg_kwargs["device"] == "cpu":
            all_pred_specs = common.chunked_parallel(smiles, pred_single_smile, chunks=500, max_cpu=self.num_workers)
            # all_pred_specs = [pred_single_smile(s) for s in tqdm(smiles)]


        elif self.use_multi_node:
            self.iceberg_kwargs["collision_eng"] = self.colli_engs
            self.iceberg_kwargs["num_bins"] = self.num_bins
            
            self.iceberg_kwargs["mass_upper_limit"] = self.mass_upper_limit
            all_pred_specs, self.gpu_workers = self.iceberg_model.predict_mol(smiles, gpu_workers=self.gpu_workers, iceberg_kwargs=self.iceberg_kwargs)
            
        else:
            with torch.no_grad():
                self.iceberg_model.eval()
                self.iceberg_model.freeze()
                # self.iceberg_model.to(self.iceberg_kwargs["device"])
                # construct batch: (size of the batch is currently size of smiles, can truncate if need be. )
                collision_engs = [float(a) for a in self.colli_engs]
                if type(self.iceberg_kwargs['adduct']) == str:
                    self.iceberg_kwargs['adduct'] = [self.iceberg_kwargs['adduct']]
                if type(self.iceberg_kwargs.get("instrument", None)) == str:
                    self.iceberg_kwargs['instrument'] = [self.iceberg_kwargs['instrument']]
                # fix to break up list of smiles into batch_size as desired, and make sure batch works
                full = {"smiles": np.repeat(np.array(smiles), len(collision_engs)),
                         "collision_eng": collision_engs * len(smiles),
                         "adducts": self.iceberg_kwargs['adduct'] * len(smiles) * len(collision_engs),
                         "precursor_mz": [self.precursor_mz] * len(smiles) * len(collision_engs),
                } 
                if self.iceberg_kwargs.get("instrument", None):
                    full["instrument"] = self.iceberg_kwargs["instrument"] * len(smiles) * len(collision_engs)
                
                avail_gpu_num = torch.cuda.device_count() 

                def batch_predict_mol(batch):
                    torch.set_num_threads(1)
                    try: 
                        if avail_gpu_num > 0:
                            if self.gpu_workers > 0:
                                worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                                gpu_id = worker_id % avail_gpu_num
                                device = f"cuda:{gpu_id}"
                            else:
                                device = self.iceberg_kwargs["device"]
                        else:
                            device = "cpu"
                        torch.cuda.set_device(device)
                        self.iceberg_model.to(device)
                        if "instrument" in batch:

                            batched_specs = self.iceberg_model.predict_mol(smi=batch["smiles"],
                                                                            collision_eng=batch["collision_eng"],
                                                                            adduct=batch["adducts"],
                                                                            precursor_mz=batch["precursor_mz"],
                                                                            instrument=batch["instrument"],
                                                                            threshold=self.iceberg_kwargs["threshold"],
                                                                            device=device, # self.iceberg_kwargs["device"],
                                                                            max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                            binned_out=False, #self.iceberg_kwargs["binned_out"],
                                                                            adduct_shift=self.iceberg_kwargs["adduct_shift"],

                            )
                        else:
                            batched_specs = self.iceberg_model.predict_mol(smi=batch["smiles"],
                                                                            collision_eng=batch["collision_eng"],
                                                                            adduct=batch["adducts"],
                                                                            precursor_mz=batch["precursor_mz"],
                                                                            threshold=self.iceberg_kwargs["threshold"],
                                                                            device=device, # self.iceberg_kwargs["device"],
                                                                            max_nodes=self.iceberg_kwargs["max_nodes"],
                                                                            binned_out=False, #self.iceberg_kwargs["binned_out"],
                                                                            adduct_shift=self.iceberg_kwargs["adduct_shift"],
                            )

                        # clean up each spectra, then bin and save.
                        binned = []
                        for spec in batched_specs["spec"]:
                            if type(spec) == torch.Tensor:
                                spec = spec.cpu().numpy()
                            # if self.denoise_specs:
                            #     spec = common.denoise_spectrum(spec)
                            spec = spec[spec[:, 1] > 0.03]
                            binned.append(common.bin_spectra([spec], self.num_bins, self.mass_upper_limit)[0])
                        batched_specs["spec"] = binned
                        torch.cuda.empty_cache()

                    except RuntimeError as err: # if an invalid SMILES is encountered 
                        torch.cuda.empty_cache()
                        if "CUDA error: out of memory" in str(err) or "CUDA out of memory" in str(err):
                            print(err)
                            return "cuda" # raise err
                        else:
                            raise
                            print(err)
                            return "error" 
                            # Will reprocess one by one after batch callds
                    except AttributeError as err:
                        raise
                        print(err)
                        return "error"

                    return batched_specs #, binned_batched_specs
                
                # make batches and dispatch
                # for supercloud: 64 batch size, chunks = 4000, gpuworkers = 16  
                batch_size = self.batch_size
                batches = [{k: v[i:i+batch_size] for k, v in full.items()} for i in range(0, len(full["smiles"]), batch_size)]
                # TODO: could estimate num workers properly
                max_retries = 3
                num_retries = 0
                batches_specs = [''] * len(batches)
                process_indices = list(range(len(batches)))
                while len(process_indices) > 0 and num_retries < max_retries:
                    process_batches = [batches[i] for i in process_indices]
                    if self.iceberg_kwargs["device"] != 'cpu':

                        result = utils.chunked_parallel_retries(process_batches, batch_predict_mol, chunks=4000, max_cpu=self.gpu_workers,
                                                            task_name="Batched ICEBERG scoring")
                        for idx, p_batch in zip(process_indices, result):
                            if type(p_batch) != str:
                                batches_specs[idx] = p_batch
                        
                        process_indices = [idx for idx, p_batch in zip(process_indices, result) if type(p_batch) == str]
                        if len(process_indices):
                            reduce = int(np.ceil(len(process_indices) / len(result) * self.gpu_workers)) # failure rate * num workers is effective gpu worker fail rate. 
                            self.gpu_workers = max(self.gpu_workers - reduce, 1)
                            logging.info(f"Turning num gpu workers from {self.gpu_workers + reduce} to {self.gpu_workers}")
                            num_retries += 1
                    else:
                        iceberg_workers = self.num_workers
                        result = utils.chunked_parallel_retries(process_batches, batch_predict_mol, chunks=4000, max_cpu=iceberg_workers,
                                                            task_name="Batched ICEBERG scoring")
                        for idx, p_batch in zip(process_indices, result):
                            if type(p_batch) != str:
                                batches_specs[idx] = p_batch
                        
                        process_indices = [idx for idx, p_batch in zip(process_indices, result) if type(p_batch) == str]
                        if len(process_indices):
                            reduce = int(np.ceil(len(process_indices) / len(result) * iceberg_workers)) # failure rate * num workers is effective gpu worker fail rate. 
                            iceberg_workers = max(iceberg_workers - reduce, 1)
                            logging.info(f"Turning num iceberg workers from {iceberg_workers + reduce} to {iceberg_workers}")
                            num_retries += 1
                    
                    # Resubmit. 
                logging.info("Scoring finished with ICEBERG")

                all_specs_raw = [spec for batch in batches_specs for spec in batch["spec"]]
                all_pred_specs = []

                for i in range(len(smiles)):
                    start = i * len(self.colli_engs)
                    end = (i + 1) * len(self.colli_engs)
                    if any([s is None for s in all_specs_raw[start:end]]):
                        print(f"{smiles[i]} errored")
                        all_pred_specs.append(None)
                        continue
                    spec_dict = {str(ce_eng): spec for ce_eng, spec in zip(self.colli_engs, 
                                                                           all_specs_raw[start:end])}
                    all_pred_specs.append(spec_dict)
        all_pred_specs = np.array(all_pred_specs)
        no_err_mask = all_pred_specs != None

        if self.stepped:
            from ms_pred.common.misc_utils import merge_intens
            merged = np.empty(all_pred_specs.shape, dtype=object)
            merged[no_err_mask] = [merge_intens(spec) for spec in all_pred_specs[no_err_mask]]
            all_pred_specs = merged

        if self.criteria == "cosine":
            criteria = "cos"
        if self.criteria == "entropy":
            criteria = "entropy"
        else:
            raise ValueError(f"Unknown criteria: {self.criteria}")

        dists = dist_bin(cand_preds_dict=all_pred_specs[no_err_mask], true_spec_dict=self.ref_spec, sparse=False,
                        ignore_peak=(self.precursor_mz - 1) * self.bin_multiplier if self.ignore_precursor_peak else None,
                        selected_evs=self.selected_evs, func=criteria, agg=agg)
        if agg:
            sims = np.zeros(all_pred_specs.shape[0])
            # Round to avoid floating point errors
            dists = np.round(dists, 6)
            assert min(dists) >= 0, f"Negative entropy value: {min(dists)}, for smiles = {smiles[np.where(dists < 0)]}"
            assert max(dists) <= 1, f"Entropy value exceeds 1: {max(dists)}, for smiles = {smiles[np.where(dists > 1)]}"
            sims[no_err_mask] = 1 - dists

            return sims
        if not agg:
            separate_dists, agg_dists = dists # separate_dists will be shape (len(colli_engs), n)
            separate_sims = np.zeros((all_pred_specs.shape[0], len(self.colli_engs)))
            agg_sims = np.zeros(all_pred_specs.shape[0])

            separate_dists = np.round(separate_dists, 6)
            agg_dists = np.round(agg_dists, 6)

            assert min(agg_dists) >= 0, f"Negative entropy value: {min(agg_dists)}, for smiles = {smiles[np.where(agg_dists < 0)]}"
            assert max(agg_dists) <= 1, f"Entropy value exceeds 1: {max(agg_dists)}, for smiles = {smiles[np.where(agg_dists > 1)]}"

            agg_sims[no_err_mask] = 1 - agg_dists
            separate_sims[no_err_mask, :] = 1 - separate_dists.T
            
            return agg_sims, separate_sims


    def score_batch(self, examples: List[Chem.Mol], **kwargs) -> np.ndarray:
        """ score_batch of mols """
        non_null_inds = []
        non_null_exs = []
        for ind, ex in enumerate(examples):
            if ex is not None:
                non_null_inds.append(ind)
                non_null_exs.append(ex)
        
        output = np.zeros((2, len(examples)))
        if len(non_null_exs) > 0:
            data = self.score_valid_mols(non_null_exs)
            output = np.zeros((len(data), len(examples)))
            for i, obj in enumerate(data):
                output[i, non_null_inds] = obj

        return output


    @staticmethod
    def oracle_name():
        return "ICEBERGOracle"
    

class _ICEBERGWithSAOracle(_ICEBERGOracle):
    def __init__(self, smiles: str, name: str,
                 iceberg_param: int = 0.8,
                 sa_param: int = 0.2,
                 multiobj: bool = False,
                 **kwargs):
        """__init__.

        Args:
            mol (Chem.Mol): Molecule to try to optimize against
            name:
            model_ckpt: Checkpoint for ms ffn model

        """
        self.iceberg_param = iceberg_param
        self.sa_param = sa_param
        self.multiobj = multiobj 
        super().__init__(smiles, name, **kwargs)
        
        

    def score_valid_mols(self, examples: List[Union[Chem.Mol, str]], **kwargs) -> Tuple[List[float], List[float]]:
        """score_batch."""
        # get values using the super class, then modify by collecting SA scores and weighting with params
        primary_scores = super().score_valid_mols(examples, **kwargs)
        sa_scores = 1 - self.score_sa(examples) / 10

        if self.multiobj:
            return primary_scores, sa_scores
        else:
            primary_weighted = self.iceberg_param * np.array(primary_scores) + self.sa_param * sa_scores
            return primary_weighted, None
        
    @staticmethod
    def oracle_name():
        return "ICEBERGWithSAOracle"

class _ICEBERGColliEngOracle(_ICEBERGWithSAOracle):
    def __init__(self, smiles: str, name: str,
                 multiobj: bool = False,
                 **kwargs):
        """__init__.

        Args:
            mol (Chem.Mol): Molecule to try to optimize against
            name:
            model_ckpt: Checkpoint for ms ffn model

        """
        self.multiobj = multiobj 
        super().__init__(smiles, name, multiobj=multiobj, **kwargs)
        
        

    def score_valid_mols(self, examples: List[Union[Chem.Mol, str]], **kwargs) -> Tuple[List[float], List[float]]:
        """score_batch."""
        # get values using the super class, then modify by collecting SA scores and weighting with params
        primary_scores, primary_scores_separated = super(_ICEBERGWithSAOracle, self).score_valid_mols(examples, agg=False, **kwargs)
        assert primary_scores_separated.shape[1] == len(self.colli_engs)

        sa_scores = 1 - self.score_sa(examples) / 10

        if self.multiobj:
            return primary_scores, sa_scores, *primary_scores_separated.T
        
        else:
            primary_weighted = self.iceberg_param * np.array(primary_scores) + self.sa_param * sa_scores
            return primary_weighted, None
        
    @staticmethod
    def oracle_name():
        return "ICEBERGColliEngOracle"
    

    def get_seed_smiles_iceberg(self, max_possible: int = 200, top_k: int = 20) -> Tuple[List[str], List[float], List[float]]:
        """
        Get seed smiles by ICEBERG prediction

        Args:
            max_possible (int): Maximum number of possible seed starting points

        """
        if not hasattr(self, 'iceberg_model') or not self.iceberg_model:
            raise ValueError("iceberg_model is missing")

        formula = self.conditional_info().get("formula")

        if self.seed_lib_dir.is_file(): # and self.seed_lib_dir.suffix == 'hdf5':
            seed_file = self.seed_lib_dir # a hdf5 file {formula: (smiles, inchikey)}
            h5obj = common.HDF5Dataset(seed_file)
            cand_str = h5obj.read_str(formula)
            # decode into candidates
            smi_inchi_list = json.loads(cand_str)
            # pickle_obj = pickle.load(open(seed_file, 'rb'))
            # smi_inchi_list = pickle_obj[formula]
            smiles = [pair[0] for pair in smi_inchi_list if '.' not in pair[0]] # if statement removes compound mixtures
        elif self.seed_lib_dir.is_dir():
            formula_file = self.seed_lib_dir / f"{formula}.txt"
            seed_file = self.seed_lib_dir / f"{self.name.lower().split('.ms')[0]}_seeds.txt"
            if seed_file.exists():
                seed_file = seed_file
            elif formula_file.exists():
                seed_file = formula_file
                raise NotImplementedError
            else:
                raise ValueError
            smiles = [line.split()[0].strip() for line in open(seed_file, "r").readlines()]
            if self.pubchem_seeds:
                seed_file = self.pubchem_seeds # a hdf5 file {formula: (smiles, inchikey)}
                h5obj = common.HDF5Dataset(seed_file)
                cand_str = h5obj.read_str(formula)
                # decode into candidates
                smi_inchi_list = json.loads(cand_str)
                # pickle_obj = pickle.load(open(seed_file, 'rb'))
                # smi_inchi_list = pickle_obj[formula]
                smiles = smiles + [pair[0] for pair in smi_inchi_list if '.' not in pair[0]]
            
            # tautomerize
            smiles = utils.simple_parallel(
                smiles,
                lambda x: utils.tautomerize_smi(x),
                max_cpu=self.num_workers,
                task_name="Tautomerizing seed smiles"
            )
            # remove stereochemistry
            def remove_stereo(smi):
                mol = Chem.MolFromSmiles(smi)
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol)
            smiles = [remove_stereo(smi) for smi in smiles]
        else:
            raise ValueError

        # predict spectrum by ICEBERG
        # will be ordered based on criteria; criteria scores will come first. 
        sims = self.score_valid_mols(smiles)

        # return top_k highest score mols + random mols
        if len(np.array(sims).shape) >= 2:
            sorted_indices = np.argsort(sims[0])
        else:
            sorted_indices = np.argsort(sims)
        if top_k < len(sorted_indices):
            num_rand = min(max_possible - top_k, len(sorted_indices) - top_k)
            rand_indices = np.random.choice(sorted_indices[:-top_k], num_rand, replace=False)
            selected_indices = np.concatenate((sorted_indices[-top_k:], rand_indices))
        else:
            selected_indices = sorted_indices[-top_k:]
        # selected_indices = rand_indices
        smiles = np.array(smiles)[selected_indices]
        if len(np.array(sims).shape) >= 2:
            sims = np.array(sims)[:, selected_indices]
            addl_sims = [s.tolist() for s in sims[2:]]
            return smiles.tolist(), sims[0].tolist(), sims[1].tolist(), *addl_sims
        else:
            sims = np.array(sims)[selected_indices]
            return smiles.tolist(), sims.tolist()


    