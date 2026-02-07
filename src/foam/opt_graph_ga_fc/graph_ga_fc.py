""" graph_ga.py

Use a graph GA algorithm for optimization

"""
import copy
import logging
from typing import List, Tuple, Union

import random

import numpy as np
import wandb
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

import foam.base_opt as base
import foam.evaluators as evaluators
import foam.utils as utils
from ms_pred import common
import yaml
import time

import itertools
from multiprocessing import Pool
from functools import partial

from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


MINIMUM = 1e-10


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int, multiobj=False, crowding_dists=None, tiebreak="entropy_only"):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # Rather, make decision to decide on whether or not multiobj setup 
    # Use fronts first to tiebreak, and then entropy distance

    if len(population_mol) == 1:
        mating_pool = population_mol * offspring_size
        return mating_pool

    if len(population_mol) == 2:
        # append pairs of population_mol
        mating_pool = [population_mol[0], population_mol[1]] * offspring_size
        return mating_pool
   
    if multiobj:
        # binary tournament selection, avoid using population_scores directly
        # TODO: binary tournament selection will make useless pools when there are only 2 seeds!

        mating_pool = []
        for _ in range(offspring_size):
            idx1, idx2 = np.random.choice(len(population_mol), size=2, replace=False)
            scores1, scores2 = population_scores[idx1], population_scores[idx2]
            if np.all(scores1 >= scores2) and np.any(scores1 > scores2):
                mating_pool.append(population_mol[idx1])
            elif np.all(scores2 >= scores1) and np.any(scores2 > scores1):
                mating_pool.append(population_mol[idx2])
            else:
                if tiebreak == "obj_crowding":
                    mating_pool.append(population_mol[idx1]) if crowding_dists[idx1] >= crowding_dists[idx2] else mating_pool.append(population_mol[idx2])
                elif tiebreak == "cand_crowding":
                    mating_pool.append(population_mol[idx1]) if scores1[0] >= scores2[0] else mating_pool.append(population_mol[idx2])
                elif tiebreak == "cand_crowding_weighted":
                    penalized_sim1, penalized_sim2 = scores1[0] / crowding_dists[idx1], scores2[0] / crowding_dists[idx2]
                    mating_pool.append(population_mol[idx1]) if penalized_sim1 >= penalized_sim2 else mating_pool.append(population_mol[idx2])
                else:
                    mating_pool.append(population_mol[idx1]) if np.random.rand() > 0.5 else mating_pool.append(population_mol[idx2]) # just random? lol 

    else:
        # scores -> probs
        # Add lower bound so that they are all positive
        population_scores = np.array(population_scores).reshape(-1, 1)
        lower_b = -min(np.min(population_scores), 0)
        population_scores = population_scores + MINIMUM + lower_b
        sum_scores = np.sum(population_scores)
        population_probs = population_scores / sum_scores
        population_probs = population_probs.flatten()
        mating_pool = np.random.choice(
            population_mol, p=population_probs, size=offspring_size, replace=True
        )
    return mating_pool


def offspring_generator(pool, mut_rate, num_workers, cap=False, max_offspring=None):
    """
    Lazily yields valid children until cap is reached or no more can be produced.
    """
    reproduce_partial = partial(reproduce, mutation_rate=mut_rate)
    # round‐robin over pool forever
    if not cap:
        parent_stream = itertools.cycle(pool)
    else:
        parent_stream = itertools.islice(itertools.cycle(pool), 0, cap) # islice needs start & stop

    with Pool(processes=num_workers) as p:
        count = 0
        attempts = 0
        pbar = tqdm(desc="Generating offspring lazily")
        # imap_unordered pulls `num_workers` tasks into the pool at a time,
        # and yields results as soon as any worker finishes one.
        for mol, *_ in p.imap_unordered(reproduce_partial, parent_stream):
            attempts += 1
            if mol is not None:
                count += 1
                pbar.update(1)
                yield mol
                
                # Stop if we've reached the maximum number of offspring
                if max_offspring and count >= max_offspring:
                    pbar.close()
                    break
            
            # Break if we've tried too many times without success - seems easier than islice?
            if attempts >= cap:
                logging.warning(f"Reached maximum attempts ({cap}) without finding enough valid offspring. Stopping generation.")
                pbar.close()
                break


def mutate_generator(parents, num_workers):
    """
    Lazily yields mutated parents one at a time.
    """
    from . import mutate as mu
    mutate_partial = partial(mu.mutate, mutation_rate=1.0)
    with Pool(processes=num_workers) as p:
        for mutated in p.imap_unordered(mutate_partial, parents):
            yield mutated


def clear_failed_pairs_cache():
    """Clear the cache of failed crossover pairs"""
    if hasattr(reproduce, 'failed_pairs_cache'):
        reproduce.failed_pairs_cache.clear()


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    from . import crossover as co
    from . import mutate as mu

    crossover_fail = 0
    mutation_fail = 0
    mutation_p_fail = 0

    mating_pool_unique_weighted = list(set(mating_pool))
    weights = [mating_pool.count(x) for x in mating_pool_unique_weighted] 
    weights = np.array(weights) / np.sum(weights)
    parent_a, parent_b = np.random.choice(mating_pool_unique_weighted, size=2, replace=False, p=weights)
    
    # Check if this pair has failed before (using InChI for consistent comparison)
    parent_a_inchi = Chem.MolToInchi(parent_a)
    parent_b_inchi = Chem.MolToInchi(parent_b)
    pair_key = tuple(sorted([parent_a_inchi, parent_b_inchi]))  # Sort for consistent key
    
    # Check cache for failed pairs (using global cache)
    if hasattr(reproduce, 'failed_pairs_cache') and pair_key in reproduce.failed_pairs_cache:
        crossover_fail = 1
        return None, crossover_fail, mutation_fail, mutation_p_fail
    
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_mut_child = mu.mutate(new_child, mutation_rate)
        if new_mut_child is None:
            mutation_fail = 1 
        elif new_mut_child == new_child:
            mutation_p_fail = 1 
        new_child = new_mut_child
        
    else:
        # Cache this failed pair
        if not hasattr(reproduce, 'failed_pairs_cache'):
            reproduce.failed_pairs_cache = set()
        
        # Limit cache size to prevent memory issues
        if len(reproduce.failed_pairs_cache) > 10000:  # Arbitrary limit
            reproduce.failed_pairs_cache.clear()
        
        reproduce.failed_pairs_cache.add(pair_key)
        crossover_fail = 1
    return new_child, crossover_fail, mutation_fail, mutation_p_fail



class GraphGAFCOptimizer(base.OptimizerBase):
    def __init__(
        self,
        population_size: int = 100,
        offspring_size: int = 200,
        starting_seed_size: int = 200,
        num_islands: int = 10,
        mutation_rate: int = 0.1,
        criteria: str = "cosine",
        truncate: bool = False,
        mutate_parents: bool = False,
        selection_sorting_type: str = "cand_crowding",
        parent_tiebreak: str = "cand_crowding",
        use_multi_node: bool = False,
        **kwargs
    ):
        """Init screen"""
        super().__init__(**kwargs)
        self.formula = self.c_info.get("formula")
        self.num_objectives = 2 # default
        if self.oracle.oracle_name() == "ICEBERGColliEngOracle":
            self.num_objectives += len(self.oracle.colli_engs)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.starting_seed_size = starting_seed_size
        self.num_islands = num_islands
        self.mutation_rate = mutation_rate
        self.criteria = criteria
        self.truncate = truncate
        self.mutate_parents = mutate_parents
        self.selection_sorting_type = selection_sorting_type
        self.parent_tiebreak = parent_tiebreak
        self.use_multi_node = use_multi_node
        initial_stats = {
            "population_size": self.population_size,
            "offspring_size": self.offspring_size,
            "num_islands": self.num_islands,
            "mutation_rate": self.mutation_rate,
            "formula": self.formula,
            "criteria": self.criteria,
        }

        population_evaluator = evaluators.get_evaluator_cls("NDSPopulationFronts")(**kwargs)
        self.population_evaluator = population_evaluator
        
        #TODO: need some way to better check if benchmark or test mode
        if hasattr(self.oracle, "self_iceberg_scores") and self.oracle.self_iceberg_scores and hasattr(self.oracle, "sa_score_target") and self.oracle.sa_score_target:
        # if self.oracle.self_iceberg_scores and self.oracle.sa_score_target:
            if self.criteria in ["cosine", "entropy", "emd"]:
                initial_stats[f"target_self_{self.criteria}"] = self.oracle.self_iceberg_scores[0]
            else:
                raise ValueError(f"Criteria {self.criteria} not understood")

            initial_stats["target_SA"] = self.oracle.sa_score_target
            initial_stats["target_smiles"] = self.oracle.mol_smiles
            initial_stats["adduct"] = self.oracle.adduct

        if self.wandb_mode != "disable":
            wandb.log(initial_stats)
            if self.wandb_mode == "offline":
                self.trigger_sync()

    @staticmethod
    def opt_name():
        """opt_name."""
        return "GraphGAFC"

    def _optimize(self, **kwargs):
        """_optimize."""

        patience = 0
        buffer_patience = 0
        self.generation_id = 0

        # 1. Define initial pool
        seeds = self.get_seed_smiles(max_possible=self.starting_seed_size * self.num_islands)
        if len(seeds) == 0:
            logging.info("Insufficient seeds found from PubChem with sufficient dissimilarity. Exiting.")
            return
        
        # TODO: process seeds in a different function. 
        
        if type(seeds) is list:
            smiles = seeds
        elif type(seeds) is tuple and len(seeds) == 2:
            smiles, scores = seeds
            for smi, score in zip(smiles, scores): # TODO: currently isn't even being used... 
                self.mol_buffer[smi] = {
                    "scores": float(score),
                    "formula": common.form_from_smi(smi),
                    "sample_num": 0, # TODO: update with actual num
                    "seed": True,
                    "inchikey-match": 1 if (hasattr(self.oracle, "mol_inchikey") and common.inchikey_from_smiles(smi) == self.oracle.mol_inchikey) else 0,
                }
        elif type(seeds) is tuple and len(seeds) == 3:
            if self.oracle.multiobj:
                smiles, objective1, objective2 = seeds
                # TODO: assuming o1 is spectral similarity, o2 is SA score - need to encode parameter order somewhere!
                # Maybe after I try adding another objective
                
                for i, (smi, score1, score2) in enumerate(zip(smiles, objective1, objective2)):
                    tani = 0
                    if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                        target_fp = self.oracle._get_fp()[np.newaxis, :]
                        # TODO: fix
                        fp = self.oracle.get_morgan_fp(Chem.MolFromSmiles(smi))[np.newaxis, :]
                        tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                    
                    self.mol_buffer[smi] = {
                        "scores": [float(score1), float(score2)],
                        "formula": common.form_from_smi(smi),
                        "sample_num": i,
                        "seed": True,
                        "inchikey-match": 1 if (hasattr(self.oracle, "mol_inchikey") and common.inchikey_from_smiles(smi) == self.oracle.mol_inchikey) else 0,
                        "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                        "tanimoto": tani,
                    }
            else:
                smiles, primary_scores = seeds
                for i, (smi, score) in enumerate(zip(smiles, primary_scores)):
                    tani = 0
                    if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                        target_fp = self.oracle._get_fp()[np.newaxis, :]
                        # TODO: fix
                        fp = self.oracle.get_morgan_fp(Chem.MolFromSmiles(smi))[np.newaxis, :]
                        tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                    self.mol_buffer[smi] = {
                        "scores": float(score),
                        "formula": common.form_from_smi(smi),
                        "sample_num": i,
                        "seed": True,
                        "inchikey-match": 1 if (hasattr(self.oracle, "mol_inchikey") and common.inchikey_from_smiles(smi) == self.oracle.mol_inchikey) else 0,
                        "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                        "tanimoto": tani,
                    }        
        elif type(seeds) is tuple and len(seeds) > 3:
            if self.oracle.multiobj:
                smiles = seeds[0]
                scores = seeds[1:]
                
                for i, (smi, *scores_i) in enumerate(zip(smiles, *scores)):
                    tani = 0
                    if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                        target_fp = self.oracle._get_fp()[np.newaxis, :]
                        # TODO: fix
                        fp = self.oracle.get_morgan_fp(Chem.MolFromSmiles(smi))[np.newaxis, :]
                        tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                    
                    self.mol_buffer[smi] = {
                        "scores": list(map(float, scores_i)),
                        "formula": common.form_from_smi(smi),
                        "sample_num": i,
                        "seed": True,
                        "inchikey-match": 1 if (hasattr(self.oracle, "mol_inchikey") and common.inchikey_from_smiles(smi) == self.oracle.mol_inchikey) else 0,
                        "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                        "tanimoto": tani,
                    }
            else:
                smiles, primary_scores = seeds[0], seeds[1] # ignore other scores? 
                for i, (smi, score) in enumerate(zip(smiles, primary_scores)):
                    tani = 0
                    if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                        target_fp = self.oracle._get_fp()[np.newaxis, :]
                        # TODO: fix
                        fp = self.oracle.get_morgan_fp(Chem.MolFromSmiles(smi))[np.newaxis, :]
                        tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                    self.mol_buffer[smi] = {
                        "scores": float(score),
                        "formula": common.form_from_smi(smi),
                        "sample_num": i,
                        "seed": True,
                        "inchikey-match": 1 if (hasattr(self.oracle, "mol_inchikey") and common.inchikey_from_smiles(smi) == self.oracle.mol_inchikey) else 0,
                        "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                        "tanimoto": tani,
                    }        
        else:
            raise ValueError(f"The return of get_seed_smiles is not understood!")

        import time
        s1 = time.time()
        if self.starting_seed_size >= 0:
            population_smiles = np.random.choice(smiles, self.starting_seed_size * self.num_islands)
        else:
            population_smiles = smiles 
        
        population_mols = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_mols = self.sanitize(population_mols) # Deduplicate to spare calls; consider tautomerizing here?
        self.keep_population += len(population_mols) # Allow extra sized buffer. 
        population_mols, population_scores, signal = self.score_mol(population_mols, seeds=True)
        population_mols = [population_mols[i::self.num_islands] for i in range(self.num_islands)]  # now broken up into by-island
        population_scores = [population_scores[i::self.num_islands] for i in range(self.num_islands)]  # now broken up into by-island
        s2 = time.time()
        logging.debug('Given seeds, sanitize and score: %.2fs', s2 - s1)
        s1 = s2 

        if self.truncate:
            population_mols, population_scores = self.truncate_population(population_mols, population_scores)

        self.population_evaluator.eval_batch(population=population_mols, scores=population_scores, mol_buffer=self.mol_buffer)
        s2 = time.time()
        logging.debug('Initial truncation and population evaluation: %.2fs', s2 - s1)
        s1 = s2 
        old_scores = population_scores

        # 2. Enter loop
        while signal == base.OptSignals.CONT: 
            self.generation_id += 1
            s1 = time.time()
            # 2.1 Get all progenitors
            if self.oracle.multiobj:
                mating_pools = []
                
                for i in range(self.num_islands):
                    crowding_dists = None
                    # TODO: could track front count, but note that it will change btwn iterations (and islands)
                    if self.parent_tiebreak == "obj_crowding":
                        crowding_dists = self.obj_crowding_distance(np.array(population_scores[i]))
                    elif self.parent_tiebreak == "cand_crowding_weighted":
                        crowding_dists = self.cand_crowding_distance(population_mols[i])
                    mating_pools.append(make_mating_pool(
                        population_mols[i], population_scores[i], self.offspring_size, multiobj=self.oracle.multiobj, crowding_dists=crowding_dists, tiebreak=self.parent_tiebreak
                    ))
                
            else:    
                mating_pools = [make_mating_pool(
                    population_mols[i], population_scores[i], self.population_size
                ) for i in range(self.num_islands)]
            
            if self.mutate_parents:
                mutate_parents_size = self.offspring_size // 4 # TODO: can change. 
                reproduce_size = self.offspring_size - mutate_parents_size
            # If not & it might be needed:
            elif len(population_mols[0]) == 1:
                logging.info("There are too few seeds to perform crossover successfully, so we are seeding the population with mutated seeds.")
                mutate_parents_size = self.offspring_size 
                reproduce_size = 0
            elif len(population_mols[0]) <= 5: # handle low n seed count:
                logging.info("There are too few seeds to perform crossover successfully, so we are seeding the population with mutated seeds.")
                mutate_parents_size = self.offspring_size // 4
                reproduce_size = self.offspring_size - mutate_parents_size
            else:
                mutate_parents_size = 0 
                reproduce_size = self.offspring_size - mutate_parents_size
            full_pool = mating_pools * reproduce_size
            # s1 = time.time()
            # This is a lazy parallel generator, good for failure-prone generation
            offspring_gen = offspring_generator(full_pool, self.mutation_rate, self.num_workers, cap=(self.offspring_size * 10))
            offspring_mols = list(itertools.islice(offspring_gen, reproduce_size))
            if len(offspring_mols) < reproduce_size:
                logging.info(f"Unable to find total desired num of offspring, {len(offspring_mols)} vs. {reproduce_size}")

            prev_buffer_size = len(self.mol_buffer)
            population_mols = [population_mols[i] + offspring_mols[i::self.num_islands] for i in range(self.num_islands)]

            if self.mutate_parents or len(population_mols[0]) < 5:
                # TODO: this should be island-dependent
                to_mutate = [x for pool in mating_pools for x in np.random.choice(pool, mutate_parents_size)]

                mut_gen = mutate_generator(to_mutate, self.num_workers)
                mutate_mols = list(itertools.islice(mut_gen, len(to_mutate)))

                population_mols = [population_mols[i] + mutate_mols[i::self.num_islands] for i in range(self.num_islands)]
            
            # 2.3 Sanitize
            population_mols = [self.sanitize(_) for _ in population_mols]
            inverse_idx = np.array([i for i, __ in enumerate(population_mols) for _ in __])

            # 2.4 Oracle and score
            before_size = len([_ for __ in population_mols for _ in __])
            gen_pop = time.time()
            logging.debug("Time to generate population (target ~10s): %.2fs", gen_pop - s1)
            
            scoring = time.time()
            population_mols, population_scores, signal = self.score_mol([_ for __ in population_mols for _ in __])

            if population_scores is None:
                continue

            s2 = time.time()
            logging.debug("Overall score time for offspring set (target ~30s): %.2fs", s2 - scoring)

            ee_start = time.time()
            new_buffer_size = len(self.mol_buffer)
            if new_buffer_size - prev_buffer_size < 10: 
                buffer_patience += 1
                if buffer_patience >= 5:
                    break 

            inverse_idx = inverse_idx[:len(population_mols)]
            
            population_tuples = np.array(list(zip(population_scores, population_mols)), dtype=object)

            population_scores, population_mols = [], []
            for i in range(self.num_islands):
                island_tuples = population_tuples[inverse_idx == i]
                if len(island_tuples) == 0:
                    population_scores.append([]); population_mols.append([])
                    continue
                if self.oracle.multiobj:
                    smis_ordering = self.return_sorted_population(tuples=island_tuples, ceiling=self.population_size) # In fact, this will also limit the population size
                    tuples_dict = {x[1]: x for x in island_tuples} # index tuples by smiles
                    island_tuples = [tuples_dict[smi] for smi in smis_ordering]

                else:
                    island_tuples = sorted(
                        island_tuples, key=lambda x: x[0], reverse=True
                    )[: self.population_size]

                island_scores, island_mols = zip(*island_tuples)
                island_scores, island_mols = list(island_scores), list(island_mols)
                population_scores.append(island_scores)
                population_mols.append(island_mols)

            # early stop based on entropy score only
            if np.max([_[0] for __ in population_scores for _ in __]) == np.max([_[0] for __ in old_scores for _ in __]): # np.all(np.max(population_scores, axis=1) == np.max(old_scores, axis=1)):
                patience += 1
                if patience >= self.patience:
                    logging.info("Exceeded patience")
                    break
            else:
                patience = 0

            
            if self.truncate: 
                # if self.generation_id % 5 == 0: # try truncating only every 5
                if self.generation_id % 1 == 0: # Just do every round now
                    population_mols, population_scores = self.truncate_population(population_mols, population_scores)


            # Clear failed pairs cache periodically to prevent memory buildup
            if self.generation_id % 10 == 0:  # Clear every 10 generations
                clear_failed_pairs_cache()
                from . import crossover as co
                co.clear_crossover_cache()
                logging.info(f"Cleared failed pairs and crossover caches at generation {self.generation_id}")
            
            # Evaluate efficacy:
            new_gen_statistics = {
                "num_offspring": self.offspring_size,
                "num_new_molecules": new_buffer_size - prev_buffer_size,
            }

            self.population_evaluator.eval_batch(population = population_mols, mol_buffer = self.mol_buffer, new_gen=new_gen_statistics)
            old_scores = population_scores

            if self.generation_id % 5 == 0:
                island_scores = [np.mean(_) for _ in population_scores]
                best_idx, worst_idx = np.argmax(island_scores), np.argmin(island_scores)
                population_mols[worst_idx] = copy.deepcopy(population_mols[best_idx])
                population_scores[worst_idx] = copy.deepcopy(population_scores[best_idx])
            
            ee_end = time.time()

        # Wrap up run: compute final metrics (e.g. any inchikey match at a point, etc)
        self.compute_final_metrics()

    def compute_final_metrics(self):
        sorted_dict = list(self.mol_buffer.items())
        # TODO: sort final metrics by entropy similarity only 
        topk = min(10, len(sorted_dict))
        import time
        entries = sorted(sorted_dict, key=lambda x: x[1]["scores"][0], reverse=True)[:topk]
        topk_smis = [entries[i][0] for i in range(topk)]
        wandb.run.summary["final_top10_smiles"] = topk_smis
        wandb.run.summary["final_top1_smiles"] = topk_smis[0]
    
        # get inchi matches
        inchi_eval = [e for e in self.evaluators if e.eval_name() == "InchiKeyMatch"][0]
        wandb.run.summary["any_inchikey_match"] = int(inchi_eval.any_match)
        if inchi_eval.any_match:
            wandb.run.summary["any_top1_inchikey_match"] = int(inchi_eval.any_top1_match)
            wandb.run.summary["any_top10_inchikey_match"] = int(inchi_eval.any_top10_match)
        else:
            wandb.run.summary["any_top1_inchikey_match"] = 0
            wandb.run.summary["any_top10_inchikey_match"] = 0

        wandb.run.summary["final_top1_inchikey_match"] = int(inchi_eval.curr_top1_match)
        wandb.run.summary["final_top10_inchikey_match"] = int(inchi_eval.curr_top10_match)
        

        bestmol_eval = [e for e in self.evaluators if e.eval_name() == "NDSBestMol"][0]

        # get max top1 tani, top10 avg tani, top10 max tani
        wandb.run.summary["best_top1_tani"] = bestmol_eval.best_tani_sim
        wandb.run.summary["best_top1_num_calls"] = bestmol_eval.best_num_calls
        wandb.run.summary["best_top1_smiles"] = bestmol_eval.best_smiles
        
        wandb.run.summary["best_top10_avg_tani"] = bestmol_eval.best_topk_avg_tani_sim
        wandb.run.summary["best_top10_max_tani"] = bestmol_eval.best_topk_max_tani_sim
        wandb.run.summary["best_top10_num_calls"] = bestmol_eval.best_topk_num_calls
        wandb.run.summary["best_top10_smiles"] = bestmol_eval.best_topk_smiles # assuming that this is list of strings. 

        nds_buffer_fronts = [e for e in self.evaluators if e.eval_name() == "NDSFronts"][0]
        artifact = wandb.Artifact("final_buffer_table", type="results")
        artifact.add(nds_buffer_fronts.wandb_table, "final_buffer_table_results")
        wandb.log_artifact(artifact)
        
        
        if len([e for e in self.evaluators if e.eval_name() == "NDSPopulationFronts"]):
            nds_population_fronts = [e for e in self.evaluators if e.eval_name() == "NDSPopulationFronts"][0]
            artifact = wandb.Artifact("final_population_table", type="results")
            artifact.add(nds_population_fronts.wandb_table, "final_population_table_results")
            wandb.log_artifact(artifact)
        
    
    def fast_nds(self, buffer_entries):
        nds = NonDominatedSorting()
        # structure
        scores_only = -np.vstack([entry[0] for entry in buffer_entries])
        # need to invert :-)
        fronts, ranks = nds.do(np.array(scores_only), return_rank=True)

        # TODO: can also do ranking and stop early.. not sure where we'd want this. maybe when ceiling is called. 
        return fronts

    def cand_crowding_distance(self, molecules):
        """
        Currently computes a sharing penalty to help sort within fronts
        Points with higher population homogeneity are discounted when considered for ranking along fronts 
        As well as for parent binary tournament selection 
        """
        crowding_dists = np.zeros(len(molecules))
        if type(list(molecules)[0]) == str:
            molecules = [Chem.MolFromSmiles(mol) for mol in molecules]
        # collect fingerprints
        fps = np.array([utils.fp_from_mol(mol) for mol in molecules])
        # compute tanimoto similarity 
        intersection = np.dot(fps, fps.T)
        union = np.sum(fps, axis=1) + np.sum(fps, axis=1)[:, None] - intersection
        similarity = intersection / union
        # compute sharing metric: 
        threshold = 0.25
        similarity[similarity < threshold] = 0 
        similarity /= threshold
        sharing_scores = similarity.sum(axis=0) - 1/0.25 # remove diagonal contributions
        # use these to adjust the entropy scores FOR ranking (but buffer should remain unchanged.) 
        return sharing_scores
    
    def obj_crowding_distance(self, scores):
        crowding_dists = [0 for _ in range(scores.shape[0])]
        for i in range(scores.shape[1]):
            scores_i = scores[:, i]
            sorted_idx = np.argsort(scores_i)
            crowding_dists[sorted_idx[0]], crowding_dists[sorted_idx[-1]] = np.inf, np.inf
            for j in range(1, len(sorted_idx) - 1):
                crowding_dists[sorted_idx[j]] += scores_i[sorted_idx[j + 1]] - scores_i[sorted_idx[j - 1]]
        return np.array(crowding_dists)

    def select(self, fronts, crowding_dists, population_scores, ceiling=None):
        """
        ceiling: maximum number of molecules to select; examples may be either self.keep_population or self.population_size
        """
        selected = []
        if self.selection_sorting_type == "cand_crowding":
            criteria = [scores[0][0] for scores in population_scores]
        if self.selection_sorting_type == "cand_crowding_weighted":
            diversity_penalized_sims = [scores[0][0] / d for scores, d in zip(population_scores, crowding_dists)]
            criteria = diversity_penalized_sims
        elif self.selection_sorting_type == "obj_crowding":
            criteria = crowding_dists
        for front in fronts: 
            if len(selected) + len(front) <= ceiling: 
                # sort front for ease of taking slices for evaluation 
                front_sorted = sorted(front, key=lambda x: criteria[x], reverse=True)
                selected += front_sorted
            else:
                # need front idx, but now sorted by crowding distance
                front_sorted = sorted(front, key=lambda x: criteria[x], reverse=True)
                selected += front_sorted[:ceiling - len(selected)]
                break
        return selected
    
    def return_sorted_population(self, mol_buffer=None, tuples=None, return_fronts=False, ceiling=None):
        # extract scores from mol_buffer
        if mol_buffer is not None:
            population_scores = [(mol_buffer[smi]["scores"], smi) for smi in mol_buffer]
            smis = list(mol_buffer.keys())
        else:
            population_scores = tuples
            smis = [Chem.MolToSmiles(x[1]) for x in tuples if type(x[1]) == Chem.Mol]
        idx_fronts = self.fast_nds(population_scores)
        # TODO: put under different switch
        if self.selection_sorting_type == "obj_crowding":
            crowding_dists = self.obj_crowding_distance(np.array([x[0] for x in population_scores]))

        elif self.selection_sorting_type == "cand_crowding_weighted":
            crowding_dists = self.cand_crowding_distance(smis)

        elif self.selection_sorting_type == "cand_crowding":
            crowding_dists = None
        
        selected = self.select(idx_fronts, crowding_dists, population_scores, ceiling=ceiling)
        # turn into lists 
        smis_selected = [population_scores[idx][1] for idx in selected]
        front_numbers = np.zeros(len(smis_selected))
        ptr = 0
        for i, front in enumerate(idx_fronts):
            front_numbers[ptr:ptr+len(front)] = i + 1
            ptr += len(front)
        if return_fronts:
            return smis_selected, front_numbers
        return smis_selected
    

    def truncate_population(self, population_mols, population_scores):
        """
        Truncate population to remove molecules below a certain similarity threshold once sufficient molecules over another threshold are present
        """
        # Apply mass extinction: if population has some # of molecules above >= 0.4 similarity: kill all molecules with < 0.3 similarity
        if self.oracle.multiobj:
            for i in range(self.num_islands):
                # TODO: make extinction threshold and percentage a kwarg; 
                # basic: set to 0.4 and 0.3 rn
                # maybe have toggle for MSG: 0.25, 0.2
                # for NIST: 0.4, 0.3
                scores = np.array(population_scores[i])[:, 0]
                if sum(scores >= self.truncate) >= 0.2 * self.population_size:
                    death_mask = scores < (self.truncate - 0.05)
                    # drop entries from population
                    population_scores[i] = np.array(population_scores[i])[~death_mask].tolist()
                    population_mols[i] = np.array(population_mols[i])[~death_mask].tolist()
                    logging.info(f'Truncated population at generation {self.generation_id}')

        else:
            for i in range(self.num_islands):
                scores = np.array(population_scores[i])
                if sum(scores >= self.truncate) >= 0.2 * self.population_size:
                    death_mask = scores < (self.truncate - 0.05)
                    population_scores[i] = scores[~death_mask].tolist()
                    population_mols[i] = np.array(population_mols[i])[~death_mask].tolist()
                    logging.info(f'Truncated population at generation {self.generation_id}')

        return population_mols, population_scores

    def sanitize(self, mol_list: List[Chem.Mol]) -> List[Chem.Mol]:
        """sanitize.py"""
        new_mol_list = super().sanitize(mol_list)
        tautomers = set()
        tautomers.update(set(self.mol_buffer.keys()))
        # will also canonicalize molecule for scoring which takes place right after, so canonical smiles will be used 
        # TODO: make sure that the tautomerization used to store in buffer is the SAME
        for mol in new_mol_list:
            smi = utils.tautomerize_smi(Chem.MolToSmiles(mol))
            if smi not in tautomers and smi is not None:
                tautomers.add(smi)
        
        return [Chem.MolFromSmiles(smi) for smi in tautomers]


    # TODO: simplify score_mol and move superwriting into graph_ga file. 
    def score_mol(
        self, mols: List[Union[Chem.Mol, str]], mol_type: str = "Mol", seeds: bool = False, **kwargs
    ) -> Tuple[List[Chem.Mol], List[float], int]:
        """score_mol.

        Score a single molecule with an optimizer. Handles all logging so that
        the optimizer does not have to worry about it.

        Args:
            mols: List of Smiles or rdkit Objs
            mol_type: Type of molecule
            kwargs

        Return: Tuple[mol_objs, float, int]: mols, scores, signal to continue

        """
        calls_remaining = self.max_calls - self.calls_made
        # If above the limit, do not score
        if calls_remaining <= 0:
            return None, None, base.OptSignals.STOP

        import time
        s1 = time.time()

        logging.info("Converting mol list")
        if mol_type == "Mol":
            mols = np.array(mols)
            smis = np.array([Chem.MolToSmiles(mol) for mol in mols])
        elif mol_type == "smiles":
            smis = np.array(mols)
            mols = np.array([Chem.MolFromSmiles(mol) for mol in mols])
        else:
            raise NotImplementedError()

        if self.replace_population: 
            self.mol_buffer = {}
            new_scores = self.oracle(mols)
            new_scores = np.vstack(new_scores).T
            # ^ will break if not self.oracle.multiobj
            if not seeds:
                self.calls_made += len(smis)
                inds = np.arange(self.calls_made - len(smis), self.calls_made)
            for ind_i, objs_i, smi, mol in zip(inds, new_scores, smis, mols):
                    smi = str(smi)
                    chem_formulae = CalcMolFormula(mol) if mol is not None else ""
                    # TODO: Try storing key mapping elsewhere/within oracle, to be accessible with evaluators 
                    
                    tani = 0
                    if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                        target_fp = self.oracle._get_fp()[np.newaxis, :]
                        fp = self.oracle.get_morgan_fp(mol)[np.newaxis, :]
                        tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                    
                    new_entry = {
                        "scores": objs_i,
                        "formula": chem_formulae,
                        "ind": ind_i,
                        "tanimoto": tani,
                        "inchikey-match": 1 if Chem.MolToInchiKey(mol) == self.oracle.mol_inchikey else 0,
                        }
                    self.mol_buffer[smi] = new_entry
                
            # Log output with some frequency
            if np.any(inds % self.log_freq == 0) or seeds:
                if seeds:
                    logging.info("Logging seed statistics below.")
                output_stats = self._get_stats_log()
                output_stats["Meta"] = {"Calls Made": self.calls_made}
                out_str = yaml.dump(output_stats)
                logging.info(f"Batch statistics after {self.calls_made} calls:\n {out_str}")

                # Submit to wandb
                if self.wandb_mode != "disable":
                    wandb.log(output_stats)
                    if self.wandb_mode == "offline":
                        self.trigger_sync()
            
            return None, new_scores, base.OptSignals.CONT
        
        # Filter down to only mols not yet scored
        all_scores = np.array(
        [self.mol_buffer.get(i, {}).get("scores", [None] * self.num_objectives) for i in smis]
        )
        already_sampled = np.array([i in self.mol_buffer for i in smis])
        not_sampled = ~already_sampled

        already_sampled_mols = mols[already_sampled]
        already_sampled_scores = all_scores[already_sampled]
        

        # Subset molecules not yet scored down to only the amount remaining
        not_sampled = (np.cumsum(not_sampled) <= calls_remaining) * not_sampled
        not_sampled_mols = mols[not_sampled]
        not_sampled_smis = np.array(smis)[not_sampled]

        # Compute new scores.         
        new_scores = self.oracle(not_sampled_mols)

        new_scores = np.vstack(new_scores).T
        if new_scores.shape == (0, 2):
            new_scores = np.zeros((0, self.num_objectives))
        # Increment number of calls
        if seeds:
            if self.starting_seed_size == -1:
                logging.info(f'Scoring {len(not_sampled_smis)} seeds')
            inds = np.array([-1] * len(not_sampled_smis))
        else: 
            self.calls_made += len(not_sampled_smis)
            inds = np.arange(self.calls_made - len(not_sampled_smis), self.calls_made)

        if self.oracle.multiobj:
            for call_num, objs_i, smi, mol in zip(
                inds, new_scores, not_sampled_smis, not_sampled_mols
            ):
                smi = str(smi)
                chem_formulae = CalcMolFormula(mol) if mol is not None else ""
                tani = 0
                if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                    target_fp = self.oracle._get_fp()[np.newaxis, :]
                    fp = self.oracle.get_morgan_fp(mol)[np.newaxis, :]
                    tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                new_entry = {
                    "scores": objs_i,
                    "sample_num": int(call_num),
                    "formula": chem_formulae,
                    "inchikey-match": 1 if hasattr(self.oracle, 'mol_inchikey') and Chem.MolToInchiKey(mol) == self.oracle.mol_inchikey else 0,
                    # "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                    "tanimoto": tani,
                }
                self.mol_buffer[smi] = new_entry
                # Delay selection until after all new scores computed
            
            smi_order, front_numbers = self.return_sorted_population(mol_buffer=self.mol_buffer, 
                                                                     return_fronts=True, 
                                                                     ceiling=self.keep_population) # Does do selection to keep buffer lean (but have turned up to max calls)
            self.mol_buffer = {k: dict(self.mol_buffer[k], front=front_idx) for k, front_idx in zip(smi_order, front_numbers)}

        else: 
            # Update buffer and score with all outputs
            for call_num, score, smi, mol in zip(
                inds, new_scores, not_sampled_smis, not_sampled_mols
            ):
                smi = str(smi)
                chem_formulae = CalcMolFormula(mol) if mol is not None else ""
                # Add mol into our mols list and cutoff if below score thresh
                tani = 0
                if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                    target_fp = self.oracle._get_fp()[np.newaxis, :]
                    fp = self.oracle.get_morgan_fp(mol)[np.newaxis, :]
                    tani = self.oracle._tanimoto_sim(target_fp, fp).item()
                new_entry = {
                    "score": float(score),
                    "sample_num": int(call_num),
                    "formula": chem_formulae,
                    "inchikey-match": 1 if hasattr(self.oracle, 'mol_inchikey') and Chem.MolToInchiKey(mol) == self.oracle.mol_inchikey else 0,
                    # "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                    "tanimoto": tani,
                
                }

                if self.min_score is None:
                    self.mol_buffer[smi] = new_entry
                    self.min_score = score
                elif score > self.min_score:
                    # Pop one item from our dictionary
                    self.mol_buffer[smi] = new_entry
                    sorted_dict = sorted(
                        self.mol_buffer.items(), key=lambda x: x[1]["score"], reverse=True
                    )

                    if self.keep_population is not None:
                        sorted_dict = sorted_dict[: self.keep_population]

                    self.mol_buffer = dict(sorted_dict)
                elif (
                    self.keep_population is None
                    or len(self.mol_buffer) < self.keep_population
                ):
                    self.mol_buffer[smi] = new_entry
                else:
                    pass

            # Log output with some frequency
            best_score = np.max([i["score"] for i in self.mol_buffer.values()])
            
        if np.any(inds % self.log_freq == 0) or seeds:
            if seeds:
                logging.info("Logging seed statistics below.")
            output_stats = self._get_stats_log()
            output_stats["Meta"] = {"Calls Made": self.calls_made}
            if self.oracle.smi: # also validation-mode metrics 
                if output_stats["InchiKeyMatch"]["InchiKeyMatch"]:
                    output_stats["%_topk_better_decoys_over_seen_target"] = output_stats["NDSBestMol"][f"top_10_{self.oracle.criteria}_better_than_target"]
                else:
                    output_stats["%_topk_better_decoys_but_target_unseen"] = output_stats["NDSBestMol"][f"top_10_{self.oracle.criteria}_better_than_target"]
                output_stats["better_decoy_seen_already"] = output_stats["NDSBestMol"][f"better_{self.oracle.criteria}_decoy"]
            out_str = yaml.dump(output_stats)
            logging.info(f"Batch statistics after {self.calls_made} calls:\n {out_str}")
            # Submit to wandb
            if self.wandb_mode != "disable":
                # log differently
                wandb.log(output_stats)
                if self.wandb_mode == "offline":
                    self.trigger_sync()

        ret_mols, ret_scores = np.array([None] * mols.shape[0]), np.tile(np.array([None] * self.num_objectives), (all_scores.shape[0], 1))
        ret_mols[already_sampled], ret_mols[not_sampled] = already_sampled_mols, not_sampled_mols
        ret_scores[already_sampled], ret_scores[not_sampled] = already_sampled_scores, new_scores
        ret_mols = [_ for _ in ret_mols.tolist() if _ is not None]
        # TODO: kind of hardcoded, maybe change
        ret_scores = [j for j in ret_scores.tolist() if j[0] is not None]

        return ret_mols, ret_scores, (base.OptSignals.CONT if self.calls_made < self.max_calls else base.OptSignals.STOP)
