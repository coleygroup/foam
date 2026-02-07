""" evaluators.py """

from typing import List, Dict, Union

import foam.oracles as oracles
import foam.utils as utils
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import wandb
import pandas as pd
import plotly.express as px
from rdkit.Chem.rdFMCS import FindMCS
from myopic_mces import MCES
import time

evaluator_registry = {}

def register_eval(cls):
    """register_eval.

    Add an eval method
    Use this as a decorator on classes defined in the rest of the directory.

    """
    evaluator_registry[cls.eval_name()] = cls
    return cls


def get_evaluator_cls(eval_name: str, **kwargs):
    """get_evaluator_cls.

    Args:
        eval_name (str): eval_name
        kwargs:
    """
    return evaluator_registry.get(eval_name)


class BaseEvaluator(object):
    """BaseEvaluator"""

    def __init__(
        self,
        num_workers: int = 1,
        top_k: List[int] = [1, 10, 100, 1000],
        oracle: oracles.MolOracle = None,
        **kwargs,
    ):
        """BaseEvaluator.

        Args:
            num_workers (int): Number of processes to use for computations
            top_k (List[int]): Top k cutoffs to use to calculate this metric

        """
        super().__init__()
        self.num_workers = num_workers
        self.top_k = top_k
        self.oracle = oracle

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "BaseEval"

    def _eval_batch(self, mols: List[Chem.Mol], **kwargs) -> float:
        """_eval_batch.

        Core evaluation function for the evaluator that is applied to a single
        batch of molecules. Called by eval_batch to return a float

        Args:
            mols (List[Chem.Mol]): List of molecules
            kwargs: Slush args

        Return:
            Dictionary mapping evaluator names to a list
        """
        return NotImplementedError()

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # Step 1: Convert to molecules
        if smi_to_mols is None:
            smi_to_mols = {i: Chem.MolFromSmiles(i)
                           for i in mol_buffer.keys()
                           }

        # Step 2: Sort and filter this list
        sorted_dict = sorted(
            mol_buffer.items(), key=lambda x: x[1]["score"], reverse=True
        )

        # Filter to remove None
        sorted_dict = [x
                       for x in sorted_dict 
                       if smi_to_mols.get(x[0]) is not None]

        # Step 3: Score all these and return
        out_dict = {}
        eval_name = self.__class__.eval_name()
        for k in self.top_k:
            mol_objs = list(map(lambda x: smi_to_mols.get(x[0]), sorted_dict))
            mol_objs = mol_objs[:k]
            score = self._eval_batch(mol_objs)
            out_dict[f"{eval_name}@{k}"] = score
        return out_dict


@register_eval
class DiversityEvaluator(BaseEvaluator):
    """DiversityEvaluator."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "DiversityEval"

    def _calc_fp(
        self,
        mol: Chem.Mol,
        radius=2,
        nbits=2048,
    ) -> np.ndarray:
        """_calc_fp.

        Calculate fingerprint

        Args:
            mol (Chem.Mol)

        Return: np.ndarray

        """
        fp_fn = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        fingerprint = fp_fn(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _eval_batch(self, mols: List[Chem.Mol], **kwargs) -> float:
        """_eval_batch.

        Core evaluation function for the evaluator that is applied to a single
        batch of molecules. Called by eval_batch to return a float

        Args:
            mols (List[Chem.Mol]): List of molecules
            kwargs: Slush args

        Return:
            Dictionary mapping evaluator names to a list
        """
        ## Get tanimoto sim
        fps = list(map(self._calc_fp, mols))
        if len(fps) == 0:
            return 0

        fps = np.vstack(fps)

        intersect = fps[None, :, :] * fps[:, None, :]
        union = fps[None, :, :] + fps[:, None, :] - intersect

        tani_grid = intersect.sum(-1) / (union.sum(-1))
        tani_grid[np.isnan(tani_grid)] = 1

        mask = np.triu(np.ones_like(tani_grid)).astype(bool)
        avg_div = 1 - np.mean(tani_grid[mask])
        return float(avg_div)


@register_eval
class TopScore(BaseEvaluator):
    """TopScore."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "TopScore"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        sorted_vals = sorted(
            mol_buffer.values(), key=lambda x: x["score"], reverse=True
        )
        sorted_vals = [i["score"] for i in sorted_vals]

        # Step 3: Score all these and return
        out_dict = {}
        eval_name = self.__class__.eval_name()
        for k in self.top_k:
            if k > len(sorted_vals):
                k = len(sorted_vals)
            score = sorted_vals[k-1]
            out_dict[f"{eval_name}@{k}"] = float(score)
        return out_dict


@register_eval
class TopSecondaryScore(BaseEvaluator):
    """TopSecondaryScore."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "TopSecondaryScore"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # Sort list
        sorted_vals = sorted(
            mol_buffer.values(), key=lambda x: x["secondary"], reverse=True
        )
        sorted_vals = [i["secondary"] for i in sorted_vals]

        # Step 3: Score all these and return
        out_dict = {}
        eval_name = self.__class__.eval_name()
        for k in self.top_k:
            if k > len(sorted_vals):
                k = len(sorted_vals)
            score = sorted_vals[k-1]
            out_dict[f"{eval_name}@{k}"] = float(score)

        return out_dict

@register_eval
class TopNDSScore(BaseEvaluator):
    """TopNDSScore."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "TopNDSScore"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        sorted_vals = list(mol_buffer.values())
        sorted_vals = [i["scores"] for i in sorted_vals]
        if len(sorted_vals) == 0:
            return {}

        # Step 3: Score all these and return
        out_dict = {}
        eval_name = self.__class__.eval_name()
        for k in self.top_k:
            if k > len(sorted_vals):
                k = len(sorted_vals)
            score = sorted_vals[k-1]
            out_dict[f"{eval_name}@{k}"] = [float(i) for i in score]

        return out_dict



@register_eval
class BestMol(BaseEvaluator):
    """BestMol.

    Get the top molecule

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "BestMol"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # All processing for target molecule
        primary = self.oracle.criteria
        secondary = self.oracle.secondary 

        evals = {}

        if self.oracle.self_iceberg_scores is not None:
            evals[f"target_self_{primary}"] = float(self.oracle.self_iceberg_scores[0][0])
            evals[f"target_self_{secondary}"] = float(self.oracle.self_iceberg_scores[1][0])
            

            target_mol = self.oracle.mol
            AllChem.Compute2DCoords(target_mol)
            AllChem.GenerateDepictionMatching2DStructure(target_mol, target_mol, acceptFailure=True)
                
            target_fp = self.oracle._get_fp()[np.newaxis, :]

        else:
            target_mol = None
            target_fp = None

        # Sort list
        sorted_dict = sorted(
            mol_buffer.items(), key=lambda x: x[1]["score"], reverse=True
        )
        topk = min(self.top_k[-1], len(sorted_dict))
        topk_smis = [sorted_dict[i][0] for i in range(topk)]
        matched = [sorted_dict[i][1]["inchikey-match"] for i in range(topk)]
        eval_name = self.__class__.eval_name()

        evals.update({f"{eval_name}@{rnk+1}": mol + (" *" if match else "")
                for rnk, (mol, match) in enumerate(zip(topk_smis, matched))})
        
        topk_mols = [Chem.MolFromSmiles(smi) for smi in topk_smis]


        scoring = {
            #"tani": [self.oracle._tanimoto_sim(target_fp,
            #        self.oracle.get_morgan_fp(mol)[np.newaxis, :]) for mol in topk_mols if target_fp is not None],
            "sa": 1 - self.oracle.score_sa(topk_smis) / 10,
            "primary": [mol_buffer[smi]["score"] for smi in topk_smis],
            "secondary": [mol_buffer[smi]["secondary"] for smi in topk_smis],
            "seed": ['seed' in mol_buffer[smi] for smi in topk_smis], 
            "tani": [mol_buffer[smi]["tanimoto"] for smi in topk_smis],
            "mces": [int(mol_buffer[smi]["mces"]) for smi in topk_smis],

        }
        assert len(scoring["secondary"]) == len(topk_smis), "not enough scores collected?"
            
        if self.oracle.self_iceberg_scores is not None:
            tautomers = [utils.tautomerize_smi(Chem.MolToSmiles(mol)) for mol in topk_mols]
            tautomers = [cand_smi for cand_smi in tautomers if cand_smi is not None]            
            tautomer_fps = [self.oracle.get_morgan_fp(Chem.MolFromSmiles(tautomer)) for tautomer in tautomers]

            scoring["tani"] = [self.oracle._tanimoto_sim(target_fp, 
                                                fp[np.newaxis, :]).item() for fp in tautomer_fps]

            # all tanimoto related metrics, which cannot be computed without the target molecule.
            scoring["tani"] = np.array(scoring["tani"])
            evals["top_k_tani_similarity"] = float(np.mean(scoring["tani"]))

            # % of scores that are higher than the cos/entr score
            evals[f"top_10_{primary}_better_than_target"] = float(np.mean([i > evals[f"target_self_{primary}"] for i in scoring["primary"]]))
            evals[f"top_10_{secondary}_better_than_target"] = float(np.mean([i > evals[f"target_self_{secondary}"] for i in scoring["secondary"]]))

            evals[f"better_{primary}_decoy"] = int(scoring["primary"][0] > evals[f"target_self_{primary}"])
            evals[f"better_{secondary}_decoy"] = int(scoring["secondary"][0] > evals[f"target_self_{secondary}"])

        null_mol = None 
        ## producing the iter=k top 10 plot

        for mol in topk_mols:
            # AllChem.AssignBondOrdersFromTemplate(target_mol, mol)  # Optional
            if target_mol is not None:
                AllChem.GenerateDepictionMatching2DStructure(mol, target_mol, acceptFailure=True)
            
                if self.oracle.oracle_name() == "ICEBERGWithSAOracle":
                    target_label = f"target \n {primary[:4]}* = {evals[f'target_self_{primary}']:.3f}, {secondary[:4]}* = {evals[f'target_self_{secondary}']:.3f}, SA = {self.oracle.sa_score_target[0]:.3f}"
                    legends = ["", "", target_label, "", ""]  
                    legends += [ f"{primary[:4]}* = {pri:.3f}, Tani = {tani:.3f}, \n {secondary[:4]}* = {entr:.3f}, SA = {sa:.3f}" for pri, tani, entr, sa in zip(scoring["primary"], 
                                                                                                                                                                        scoring["tani"], 
                                                                                                                                                                        scoring["secondary"], 
                                                                                                                                                                        scoring["sa"])]

                else:
                    target_label = f"target \n {primary[:4]} = {evals[f'target_self_{primary}']:.3f}, {secondary[:4]} = {evals[f'target_self_{secondary}']:.3f}, SA = {self.oracle.sa_score_target[0]:.3f}"
                    legends = ["", "", target_label, "", ""]
                    legends += [ f"{primary[:4]} = {pri:.3f}, Tani = {tani:.3f}, \n {secondary[:4]} = {entr:.3f}, SA = {sa:.3f}" for pri, tani, entr, sa in zip(scoring["primary"], 
                                                                                                                                                                        scoring["tani"], 
                                                                                                                                                                        scoring["secondary"], 
                                                                                                                                                                        scoring["sa"])]
            else:
                if self.oracle.oracle_name() == "ICEBERGWithSAOracle":
                    target_label = f"target \n formula: {self.oracle.formula}"
                    legends = ["", "", target_label, "", ""]  
                    legends += [ f"{primary[:4]}* = {pri:.3f}, seed = {seed}, \n {secondary[:4]}* = {entr:.3f}, SA = {sa:.3f}" for pri, seed, entr, sa in zip(scoring["primary"],
                                                                                                                                                              scoring["seed"],
                                                                                                                                                              scoring["secondary"],
                                                                                                                                                              scoring["sa"])]

                else:
                    target_label = f"target \n formula: {self.oracle.formula}"
                    legends = ["", "", target_label, "", ""]
                    legends += [ f"{primary[:4]} = {pri:.3f}, seed = {seed}, \n {secondary[:4]} = {entr:.3f}, SA = {sa:.3f}" for pri, seed, entr, sa in zip(scoring["primary"],
                                                                                                                                                            scoring["seed"],
                                                                                                                                                            scoring["secondary"], 
                                                                                                                                                            scoring["sa"])]


        molecule_comp = Draw.MolsToGridImage([null_mol, null_mol, target_mol, null_mol, null_mol] + topk_mols,
                            molsPerRow=5, 
                            subImgSize=(200,200),
                            legends=legends)
        if wandb.run is not None:
            wandb.log({"Top 10 Overview": wandb.Image(molecule_comp, caption="Top 10 proposed molecules")}, commit=False)
        
        
        return evals


@register_eval
class NDSBestMol(BaseEvaluator):
    """BestMol.

    Get the top molecule

    """

    def __init__(self, track_mces=False, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)
        self.max_close_match_pct = 0
        self.max_meaningful_similarity_pct = 0
        self.track_mces = track_mces

        self.best_tani_sim = 0
        self.best_topk_avg_tani_sim = 0
        self.best_topk_max_tani_sim = 0 

        # cache these for final computation. Chosen based on top10 avg/top1
        # VERY important: these are not the global best 10 molecules, or best molecule by tanimoto similarity
        # These are instead the best top 10 ranks or top 1 ranks. 
        self.best_topk_smiles = []
        self.best_topk_num_calls = 0
        self.best_smiles = None
        self.best_num_calls = 0

        self.self_spec_plotted = False


    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "NDSBestMol"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # All processing for target molecule
        evals = {}
        primary = self.oracle.criteria

        if self.oracle.self_iceberg_scores:
            evals[f"target_self_{primary}"] = float(self.oracle.self_iceberg_scores[0][0])
            evals[f"target_self_SA_score"] = float(self.oracle.self_iceberg_scores[1][0])
            
            target_mol = self.oracle.mol
            AllChem.Compute2DCoords(target_mol)
            AllChem.GenerateDepictionMatching2DStructure(target_mol, target_mol, acceptFailure=True)
            
            target_fp = self.oracle._get_fp()[np.newaxis, :]

        else:
            target_fp = None
            target_mol = None

        # Should already be sorted by front and by crowding distance within fronts 
        # TODO: this should be on the population! not the buffer! 
        sorted_dict = list(mol_buffer.items())
        if len(sorted_dict) == 0:
            return evals
        topk = min(self.top_k[-1], len(sorted_dict))        
        #topk_smis = [sorted_dict[i][0] for i in range(topk)]
        #use entropy sim
        entries = sorted(sorted_dict, key=lambda x: x[1]["scores"][0], reverse=True)[:topk]
        topk_smis = [entries[i][0] for i in range(topk)]
        matched = [sorted_dict[i][1]["inchikey-match"] for i in range(topk)]
        eval_name = self.__class__.eval_name()

        evals.update({f"{eval_name}@{rnk+1}": mol + (" *" if match else "")
                for rnk, (mol, match) in enumerate(zip(topk_smis, matched))})

        
        topk_mols = [Chem.MolFromSmiles(smi) for smi in topk_smis]

        scoring = {
            "sa": 1 - self.oracle.score_sa(topk_smis) / 10,
            "objective": np.array([mol_buffer[smi]["scores"] for smi in topk_smis]),
            "seed": ['seed' in mol_buffer[smi] for smi in topk_smis],
            "front": [int(mol_buffer[smi]["front"]) for smi in topk_smis],
            "tani": [mol_buffer[smi]["tanimoto"] for smi in topk_smis],
            # "mces": [int(mol_buffer[smi]["mces"]) for smi in topk_smis],
        }

        if target_fp is not None:
            # Tanimoto metrics
            scoring["tani"] = np.array(scoring["tani"])
            assert len(scoring["objective"]) == len(topk_smis), "not enough scores collected?"
            # top 10 tani similarity, averaged
            # how about top1, top10:
            evals['top_1_tani_similarity'] = float(scoring["tani"][0])
            evals['top_k_avg_tani_similarity'] = float(np.mean(scoring["tani"]))
            evals['top_k_max_tani_similarity'] = float(np.max(scoring["tani"]))

            if evals['top_1_tani_similarity'] > self.best_tani_sim:
                self.best_tani_sim = evals['top_1_tani_similarity']
                self.best_smiles = topk_smis[0]
                self.best_num_calls = len(mol_buffer)
            
            if evals['top_k_avg_tani_similarity'] > self.best_topk_avg_tani_sim:
                self.best_topk_avg_tani_sim = evals['top_k_avg_tani_similarity']
                self.best_topk_smiles = topk_smis
                self.best_topk_num_calls = len(mol_buffer)

            if evals['top_k_max_tani_similarity'] > self.best_topk_max_tani_sim:
                self.best_topk_max_tani_sim = evals['top_k_max_tani_similarity']
                
            

            # MCES metrics
            if self.track_mces: # must already be computed via oracle. 
                scoring["mces"] = np.array(scoring["mces"])
                evals['top_1_mces'] = float(scoring["mces"][0])
                
                evals['top_k_avg_mces'] = float(np.nanmean(scoring["mces"]))
                evals['top_k_min_mces'] = float(np.nanmin(scoring["mces"]))

            # % of scores that are higher than the entr or sa score
            # TODO: this is hardcoded indexing at the end; should be fixed. Does not need to be sorted, because we're checking all of them
            evals[f"top_10_{primary}_better_than_target"] = float(np.mean([i > evals[f"target_self_{primary}"] for i in scoring["objective"][:, 0]]))
            evals[f"better_{primary}_decoy"] = int(np.max(scoring["objective"][:, 0]) > evals[f"target_self_{primary}"])
            evals[f"better_SA_score_decoy"] = int(np.max(scoring["objective"][:, 1]) > evals[f"target_self_SA_score"])

        null_mol = None 
        ## producing the iter=k top 10 plot

        # if target_mol is not None:
        #     # TODO: fix this and make sure the alignment works properly!
        #     AllChem.Compute2DCoords(target_mol)
        #     for mol in topk_mols:
        #         #mcs = FindMCS([target_mol, mol], ringMatchesRingOnly=True)
        #         #mcs_smarts = Chem.MolFromSmarts(mcs.smartsString)
        #         AllChem.GenerateDepictionMatching2DStructure(mol, target_mol, acceptFailure=True)#refPatt=mcs_smarts)

        # check if seen yet:
        # grab key and value
        smi_match, value_match = sorted(mol_buffer.items(), key=lambda x: x[1]["inchikey-match"], reverse=True)[0]
        if value_match["inchikey-match"]:
            seen_yet = f"yes; NDS rank {value_match['front']} (idx {list(mol_buffer.keys()).index(smi_match)})"
            
        else:
            seen_yet = "No"            
        target_label = ""
        if target_mol is not None:
            target_label = f"target \n {primary[:4]} = {evals[f'target_self_{primary}']:.3f} \n SA = {self.oracle.sa_score_target[0]:.3f}"
            target_label += f"\n seen yet: \n {seen_yet}"
        legends = ["", "", target_label, "", ""]
        legends += [ f"{primary[:4]} = {pri:.3f}, \n SA = {sa:.3f},\n Tani = {tani:.3f} \n NDS rank = {front}" for pri, tani, sa, front in zip(scoring["objective"][:, 0],                                  
                                                                                                                        scoring["tani"], 
                                                                                                                        # scoring["mces"],
                                                                                                                        scoring["sa"],
                                                                                                                        scoring["front"])]
    
        opts = Draw.MolDrawOptions()

        avg_heavy_atoms = np.mean([mol.GetNumHeavyAtoms() if mol else 0 for mol in topk_mols])
        # Scale font size based on heavy atoms
        base_font_size = 28
        scale = min(3.0, max(1.0, avg_heavy_atoms / 20))  # 1× if avg ≤25 atoms, up to 2×
        opts = Draw.MolDrawOptions()
        opts.legendFontSize = int(base_font_size * scale)
        opts.legendFraction = 0.25
        molecule_comp = Draw.MolsToGridImage([null_mol, null_mol, target_mol, null_mol, null_mol] + topk_mols,
                            molsPerRow=5, 
                            subImgSize=(400,400),
                            legends=legends, 
                            drawOptions=opts)
        if wandb.run is not None:
            wandb.log({"Top 10 Overview": wandb.Image(molecule_comp, caption="Top 10 proposed molecules")}, commit=False)

        # plot spectrum in plotly.

        if self.oracle.self_iceberg_scores:
            spec_plots = {}
            if not self.self_spec_plotted:
                ref_spec_plot = (utils.plot_spectrum(self.oracle.ref_spec_unbinned, self.oracle.self_spec_unbinned, ent_score=[self.oracle.self_iceberg_scores[0][0]], sa_score=[self.oracle.self_iceberg_scores[1][0]]))
                
                spec_plots["ref vs. iceberg of self"] = ref_spec_plot
                self.self_spec_plotted = True
            # not logging because I think this one takes up the most space. 
            
            # specs_topk = self.oracle.pred_unbinned_spec(topk_smis)
            # spec_plot = utils.plot_spectrum(self.oracle.ref_spec_unbinned, specs_topk, ent_score=scoring["objective"][:, 0], sa_score=scoring["objective"][:, 1], second_smi=topk_smis)
            # spec_plots["all_topk"] = spec_plot
            # wandb.log(spec_plots, commit=False)

        return evals
    

@register_eval
class FormulaDiffEvaluator(BaseEvaluator):
    """FormulaEvaluator.

    Determine similarity to the base chem formula

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

        self.form_vec = None
        if self.oracle is not None:
            self.cond_info = self.oracle.conditional_info()
            self.formula = self.cond_info.get("formula")
            if self.formula is not None:
                self.form_vec = utils.formula_to_dense(self.formula)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "FormulaDiffEval"

    def _eval_batch(self, mols: List[Chem.Mol], **kwargs) -> float:
        """_eval_batch.

        Core evaluation function for the evaluator that is applied to a single
        batch of molecules. Called by eval_batch to return a float

        Args:
            mols (List[Chem.Mol]): List of molecules
            kwargs: Slush args

        Return:
            Dictionary mapping evaluator names to a list
        """
        if self.form_vec is None:
            return 0

        ## Get tanimoto sim
        formulae_1 = list(map(CalcMolFormula, mols))
        formulae_2 = list(map(utils.formula_to_dense, formulae_1))
        avg_form_diffs = np.abs(self.form_vec[None, :] - formulae_2).sum(-1).mean()
        return float(avg_form_diffs)


@register_eval
class TopIsoScore(BaseEvaluator):
    """TopIsomerScore."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)
        if self.oracle is not None:
            self.cond_info = self.oracle.conditional_info()
            self.formula = self.cond_info.get("formula")

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "TopIsoScore"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # Sort list
        # filter
        mol_buffer_filtered = {
            i: j for i, j in mol_buffer.items() if j["formula"] == self.formula
        }
        sorted_vals = sorted(
            mol_buffer_filtered.values(), key=lambda x: x["score"], reverse=True
        )
        sorted_vals = [i["score"] for i in sorted_vals]

        # Step 3: Score all these and return
        out_dict = {}
        eval_name = self.__class__.eval_name()
        for k in self.top_k:
            score = np.mean(sorted_vals[:k])
            out_dict[f"{eval_name}@{k}"] = float(score)
        return out_dict


@register_eval
class BestIsoMol(BaseEvaluator):
    """BestIsomerMol.

    Get the top molecule with isomer

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)
        if self.oracle is not None:
            self.cond_info = self.oracle.conditional_info()
            self.formula = self.cond_info.get("formula")

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "BestIsoMol"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        smi_to_mols: Dict[str, Chem.Mol] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (Dict[str, dict[str, float]]): Buffer of molecules
            smi_to_mols(Dict[str, Chem.Mol]): Mapping from smiles to mol
                objects. Useful if this has already been computed

        Return:
            Returns evaluation at various top ks
        """
        # Sort list
        mol_buffer_filtered = {
            i: j for i, j in mol_buffer.items() if j["formula"] == self.formula
        }

        sorted_dict = sorted(
            mol_buffer_filtered.items(), key=lambda x: x[1]["score"], reverse=True
        )
        mol = sorted_dict[0][0]
        eval_name = self.__class__.eval_name()
        return {f"{eval_name}@1": mol}


@register_eval
class PercentValid(BaseEvaluator):
    """ PercentValid. 
    
    Calculates percent of molecules that are valid 

    """
    def __init__(self, top_k: List[int] = [1000], **kwargs):
        super().__init__(top_k=[1000], **kwargs)

    def eval_batch(self, mol_buffer: Dict[str, Dict[str, float]], **kwargs) -> float:
        smis = list(mol_buffer.keys())
        valid = [1 if Chem.MolFromSmiles(smi) else 0 for smi in smis]
        score = sum(valid)/len(valid)
        return score

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "PercentValid"


@register_eval
class InchiKeyMatch(BaseEvaluator):
    """InchiKeyMatch."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)
        self.any_match = False
        self.any_top1_match = False
        self.any_top10_match = False

        self.curr_top1_match = False
        self.curr_top10_match = False

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "InchiKeyMatch"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules

        Return:
            Returns evaluation at various top ks
        """
        # Sort list
        if len(mol_buffer) == 0:
            return {}
        
        sorted_vals = sorted(
            mol_buffer.values(), key=lambda x: x["inchikey-match"], reverse=True
        )
        eval_name = self.__class__.eval_name()
        eval_dict = {eval_name: sorted_vals[0]["inchikey-match"]}

        if sorted_vals[0]["inchikey-match"]:
            self.any_match = True
            top10vals = list(mol_buffer.values())[:10]
            if any([i["inchikey-match"] for i in top10vals]):
                self.any_top10_match = True
                self.curr_top10_match = True
                if top10vals[0]["inchikey-match"]:
                    self.any_top1_match = True
                    self.curr_top1_match = True
                
            else:
                self.curr_top10_match = False
                self.curr_top1_match = False

            if self.oracle.multiobj:
                eval_dict["all_scores"] = sorted_vals[0]["scores"].tolist()
                eval_dict[self.oracle.criteria + "_score"] = sorted_vals[0]["scores"][0].tolist()
                eval_dict["front"] = sorted_vals[0]["front"].item()
                eval_dict["sample_num"] = sorted_vals[0]["sample_num"]

            else:
                eval_dict[self.criteria + "_score"] = sorted_vals[0]["score"].item()
        
        else:
            self.curr_top10_match = False
            self.curr_top1_match = False

        return eval_dict

@register_eval
class NDSParetoRanking(BaseEvaluator):
    """NDSParetoRanking."""
    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "NDSParetoRanking"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules

        Return:
            Returns evaluation at various top ks
        """
        out_dict = {}

        eval_name = self.__class__.eval_name()
        if len(mol_buffer) == 0:
            return out_dict
        # TODO: this should be on the population! not the buffer! 
        pareto_front = [(smi, v) for smi, v in mol_buffer.items() if v["front"] == 1]

        # 1 and 2 don't apply to sorting within pareto_front; their analogs are 3 and 4 
        # 1: entropy_score only
        # 2: weighted combination only
        # 3: NDS-based fronts + entropy within fronts

        scores = np.array([pareto_front[x][1]["scores"] for x in range(len(pareto_front))])
        entropy_rank = {pareto_front[idx][0]: rank for rank, idx in enumerate(np.argsort(scores[:, 0])[::-1])} # should index by smi, and save index
        # 4: NDS-based fronts + weighted combination within fronts
        weighted = scores[:, 0] * 0.8 + scores[:, 1] * 0.2
        weighted_rank = {pareto_front[idx][0]: rank for rank, idx in enumerate(np.argsort(weighted)[::-1])}
        # 5: NDS-based fronts + crowding distance 
        # Just assume crowding distance from before is ok
        crowding_rank = {smi: i for i, (smi, v) in enumerate(pareto_front)}
        
        num_ranks = len(entropy_rank)

        # sort by stored ranks
        entropy_rank = {k: v for k, v in sorted(entropy_rank.items(), key=lambda item: item[1])}
        weighted_rank = {k: v for k, v in sorted(weighted_rank.items(), key=lambda item: item[1])}
        crowding_rank = {k: v for k, v in sorted(crowding_rank.items(), key=lambda item: item[1])}

        pareto_df = pd.DataFrame([(rank_name, *rank) for (rank_name, rank) in zip(["Entropy Rank", "Weighted Rank", "Crowding Rank"], 
                                                                          [entropy_rank, weighted_rank, crowding_rank])], 
                            columns = ["Rank Type"] + [f"Rank {i}" for i in range(1, num_ranks + 1)])
        pareto_table = wandb.Table(dataframe=pareto_df)
        
        if wandb.run is not None:
            wandb.log({f"Pareto Ranks": pareto_table}, commit=False)

        eval_dict = {smi: {"entropy_rank": int(entropy_rank[smi]) + 1, 
                            "weighted_rank": int(weighted_rank[smi]) + 1,
                            "crowding_rank": int(crowding_rank[smi]) + 1
                            } for smi, _ in pareto_front
        }

        out_dict["target_in_pareto"] = int(any([info["inchikey-match"] == True for _, info in pareto_front]))

        # Draw image of molecules in pareto front
        mols = [Chem.MolFromSmiles(smi) for smi, _ in pareto_front]
        mols = [mol for mol in mols if mol is not None]
        if len(mols) > 0:
            opts = Draw.MolDrawOptions()
            opts.legendFraction = 0.30
            opts.legendFontSize = 28
            # add legends based on eval_dict entries
            # sort based on entropy rank!
            ent_sort_pareto_front = sorted(pareto_front, key=lambda x: eval_dict[x[0]]["entropy_rank"])

            legends = [f"Entropy Rank: {eval_dict[smi]['entropy_rank']} \n" +
                       f"Weighted Rank: {eval_dict[smi]['weighted_rank']} \n" +
                        f"Crowding Rank: {eval_dict[smi]['crowding_rank']}"
                       for smi, _ in ent_sort_pareto_front]
            # TODO: add flag for correct molecule, if present
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=legends, drawOptions=opts)
            # if wandb.run is not None:
            #     wandb.log({f"Pareto Front Molecules": wandb.Image(img, caption="Pareto Front Molecules")}, commit=False)

        return out_dict

@register_eval
class NDSFronts(BaseEvaluator):
    """NDSFronts."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

        if self.oracle.self_iceberg_scores:
            self.std_mol_fp = self.oracle._get_fp()[np.newaxis, :]
            self.target_mol = self.oracle.mol
        else:
            self.std_mol_fp = None
            self.target_mol = None

        self.wandb_table = None

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "NDSFronts"

    def smi_to_pil_image(self, smi, target_mol=None):
        molecule = Chem.MolFromSmiles(smi)
        Chem.AllChem.Compute2DCoords(molecule)
        if target_mol is not None:
            Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, target_mol, acceptFailure=True)
        else:
            Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
        pil_image = Chem.Draw.MolToImage(molecule, size=(300, 300))
        return wandb.Image(pil_image)

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules

        Return:
            Visualization of front (up to three objectives allowed)
        """

        # Plot fronts 
        if len(mol_buffer) == 0:
            return {}
        
        def check_scaffold_match(smi):
            mol = Chem.MolFromSmiles(smi)
            scaffold_smarts ="nnccccccc"
            return mol.HasSubstructMatch(Chem.MolFromSmarts(scaffold_smarts))

        # will do a 3D thing,,, not sure if needed: wandb.Molecule.from_smiles(smi)
        # wandb.Image(smi_to_pil_image(smi)), 
        column_names = []
        score_ex = list(mol_buffer.values())[0]["scores"]
        column_names += [f"{self.oracle.criteria} similarity", "SA Score"]

        if len(score_ex) == 3:
            column_names += ["Formula Difference"]
        elif len(score_ex) > 3: # various collision energies 
            column_names += [f"{self.oracle.criteria}_similarity_{eng}eV" for eng in self.oracle.colli_engs]
        
        entries = ["front", "inchikey-match", "sample_num", "tanimoto"]
        column_names += ["Front", "InchiKey Match", "Sample #", "Tanimoto Similarity"]
        if "formula" in mol_buffer[list(mol_buffer.keys())[0]]:
            entries.append("formula")
            column_names.append("Formula")
        flattened = [
            [smi] + [mol_buffer[smi]["scores"][i] for i in range(len(score_ex))] + [mol_buffer[smi][entry] for entry in entries]
            for smi in mol_buffer
        ]

        table = pd.DataFrame(flattened, columns=["SMILES", *column_names]) #, "MCES"])
        #table["mol_image"] = table["SMILES"].apply(lambda x: self.smi_to_pil_image(x, self.target_mol))
        wandb_table = wandb.Table(dataframe=table)
        # round tani, entropy, sa scores to 3 decimal places
        table = table.round({f"{self.oracle.criteria} similarity": 3, "SA Score": 3, "Tanimoto Similarity": 3})
        # TODO: allow for 3D, down the line optimize for peak-based objectives (e.g. most unmatched peaks at round t)
        # Figure 1: NDS Fronts

        
        
        fig1 = px.scatter(table, x=f"{self.oracle.criteria} similarity", y="SA Score", color="Front", hover_data=table.columns)
        if self.oracle.self_iceberg_scores:
            fig1.add_hline(y=self.oracle.self_iceberg_scores[1][0], line_width=2, line_dash='dash')
            fig1.add_vline(x=self.oracle.self_iceberg_scores[0][0], line_width=2, line_dash='dash')
        fig1.update_xaxes(range=[0, 1])
        fig1.update_yaxes(range=[0, 1])
        fig1.update_traces(marker=dict(size=15))


        data = {}
        
        # TODO: this image needs to go into a results folder
        #print(type(fig1))
        #fig1.write_image(f"buffer_molecules_step_{len(mol_buffer)}.png")

        #if wandb.run is not None:
            #wandb.log({"Buffer Molecules, with buffer-based NDS fronts": fig1}, commit=False)
            #wandb.log({"Buffer Molecule Table": wandb_table}, commit=False)

        if self.std_mol_fp is None: # I don't often look at this plot when I have the true molecule anyway. 
            data["Buffer Molecules, with buffer-based NDS fronts"] = fig1
        
        self.wandb_table = wandb_table
       #wandb.run.summary["Buffer Molecule Table"] = wandb_table # can be overwritten? 
            
        
        # Figure 2: + Tanimoto Similarity
        if self.std_mol_fp is not None:
            fig2 = px.scatter(table, x=f"{self.oracle.criteria} similarity", y ="SA Score",  color="Tanimoto Similarity", range_color=(0,1), hover_data=table.columns)
            fig2.add_hline(y=self.oracle.self_iceberg_scores[1][0], line_width=2, line_dash='dash')
            fig2.add_vline(x=self.oracle.self_iceberg_scores[0][0], line_width=2, line_dash='dash')
            fig2.update_xaxes(range=[0, 1])
            fig2.update_yaxes(range=[0, 1])
            fig2.update_traces(marker=dict(size=15))
            
            #if wandb.run is not None:
            #    wandb.log({"Buffer Molecules with Tanimoto Similarity": fig2}, commit=False)
            data["Buffer Molecules with Tanimoto Similarity"] = fig2
            # TODO: use self.save_dir
            #fig2.write_image(f"buffer_molecules_tani_step_{len(mol_buffer)}.png")
            
            #if wandb.run is not None:
            #    wandb.log({"Buffer Molecules with MCES Distance": fig3}, commit=False)
            # TODO: use self.save_dir
            #fig3.write_image(f"buffer_molecules_mces_step_{len(mol_buffer)}.png")

        
        
        if wandb.run is not None:
            wandb.log(data, commit=False)

        
        return None

@register_eval
class NDSPopulationFronts(NDSFronts):
    """NDSPopulationFronts."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs

        """
        super().__init__(**kwargs)

    @staticmethod
    def eval_name():
        """Return evaluators name"""
        return "NDSPopulationFronts"

    def eval_batch(
        self,
        mol_buffer: Dict[str, Dict[str, float]],
        population: List[List[Union[str, Chem.Mol]]], 
        scores=None,
        new_gen=None,
        **kwargs,
    ) -> Dict[str, float]:
        """eval_batch.

        Args:
            mol_buffer (dict[str, dict[str, float]]): Buffer of molecules (for fast lookup of attributes)
            population List[List[str, Chem.Mol]]: List of islands of molecules in the population. 

        Return:
            Visualization of population fronts 
        """
        if len(population) == 0:
            return {}

        # TODO: how should I plot by island? Maybe makes more sense to plot all islands on one plot, and hide each island as desired
        # TODO: this is basically called traces 

        if new_gen is not None:
            wandb.log(new_gen, commit=False)

        flattened = []
        for i in range(len(population)):
            pop_i = population[i]
            if len(pop_i) == 0:
                continue
            
            if type(pop_i[0]) == Chem.Mol: # handle as smiles
                pop_i = [Chem.MolToSmiles(mol) for mol in pop_i] 

            def check_scaffold_match(smi):
                mol = Chem.MolFromSmiles(smi)
                scaffold_smarts ="nnccccccc"
                return mol.HasSubstructMatch(Chem.MolFromSmarts(scaffold_smarts))

            column_names = []
            score_ex = list(mol_buffer.values())[0]["scores"]
            column_names += [f"{self.oracle.criteria} similarity", "SA Score"]

            if len(score_ex) == 3:
                column_names += ["Formula Difference"]
            elif len(score_ex) > 3: # various collision energies 
                column_names += [f"{self.oracle.criteria}_similarity_{eng}eV" for eng in self.oracle.colli_engs]
            
            entries = ["front", "inchikey-match", "sample_num", "tanimoto"]
            column_names += ["Front", "InchiKey Match", "Sample #", "Tanimoto Similarity"]
            if "formula" in mol_buffer[list(mol_buffer.keys())[0]]:
                entries.append("formula")
                column_names.append("Formula")

            flattened.extend([
                [smi] + [mol_buffer[smi]["scores"][i] for i in range(len(score_ex))] + [mol_buffer[smi][entry] for entry in entries] + [i]
                for smi in pop_i
                ])      

        table = pd.DataFrame(flattened, columns=["SMILES", *column_names, "island"]) #"MCES", "island"])
        # table["mol_image"] = table["SMILES"].apply(lambda x: self.smi_to_pil_image(x, self.target_mol))
        wandb_table = wandb.Table(dataframe=table)
        # round tani, entropy, sa scores to 3 decimal places
        table = table.round({f"{self.oracle.criteria} similarity": 3, "SA Score": 3, "Tanimoto Similarity": 3})
        # TODO: allow for 3D, down the line optimize for peak-based objectives (e.g. most unmatched peaks at round t)
        # Figure 1: NDS Fronts
        fig1 = px.scatter(table, x=f"{self.oracle.criteria} similarity", y="SA Score", color="Front", hover_data=table.columns)
        if self.oracle.self_iceberg_scores:

            fig1.add_hline(y=self.oracle.self_iceberg_scores[1][0], line_width=2, line_dash='dash')
            fig1.add_vline(x=self.oracle.self_iceberg_scores[0][0], line_width=2, line_dash='dash')
        fig1.update_xaxes(range=[0, 1])
        fig1.update_yaxes(range=[0, 1])
        fig1.update_traces(marker=dict(size=15))
        
        data = {}
        #data["Population NDS fronts"] = fig1
        if wandb.run is not None:
            wandb.log({"Population NDS fronts": fig1}, commit=False)
        # TODO: use self.save_dir to move plots lol
        #fig.write_image(f"population_molecules_step_{len(mol_buffer)}.png")

        if self.std_mol_fp is not None:
            # Figure 2: + Tanimoto Similarity
            # table = table.drop(["mol_image"], axis=1) # will not save otherwise, unfortunately. 
            fig2 = px.scatter(table, x=f"{self.oracle.criteria} similarity", y ="SA Score",  color="Tanimoto Similarity", range_color=(0,1), hover_data=table.columns)
            fig2.add_hline(y=self.oracle.self_iceberg_scores[1][0], line_width=2, line_dash='dash')
            fig2.add_vline(x=self.oracle.self_iceberg_scores[0][0], line_width=2, line_dash='dash')
            fig2.update_xaxes(range=[0, 1])
            fig2.update_yaxes(range=[0, 1])
            fig2.update_traces(marker=dict(size=15))
            if wandb.run is not None:
                wandb.log({"Population Molecules with Tanimoto Similarity": fig2}, commit=False)
            
            # TODO: use self.save_dir
            #fig2.write_image(f"population_molecules_tani_step_{len(mol_buffer)}.png")
            data["Population Molecules with Tanimoto Similarity"] = fig2

            #if wandb.run is not None:
            #    wandb.log({"Population Molecules with MCES Distance": fig3}, commit=False)
            # TODO: use self.save_dir
            #fig3.write_image(f"population_molecules_tani_step_{len(mol_buffer)}.png")
        #if wandb.run is not None:
        #    wandb.log({"Population Molecule Table": wandb_table}, commit=True)
        
        # data["Population Molecule Table"] = wandb_table
        self.wandb_table = wandb_table
        if wandb.run is not None:
            wandb.log(data, commit=True)

        
        return None

