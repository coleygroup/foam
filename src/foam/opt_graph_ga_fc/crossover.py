import itertools
import random
import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from functools import partial

rdBase.DisableLog("rdApp.error")

def _expanded_cut(mol, bond_order=1, enumerate=False):
    """Cut non-ring and return the molecule fragments.
       If enumerate is True, return all fragments; if false, randomly choose a fragment."""
    cuts = {
        1: Chem.MolFromSmarts("[*!D1]!@-[*!D1]"), # Chem.MolFromSmarts("[*]-;!@[*]"),
        2: Chem.MolFromSmarts("[*!D1]!@=[*!D1]"), #Chem.MolFromSmarts("[*]=;!@[*]"),
        3: Chem.MolFromSmarts("[*!D1]!@#[*!D1]"),  #Chem.MolFromSmarts("[*]#;!@[*]")
    }

    cut = cuts[bond_order]
    if not mol.HasSubstructMatch(cut):
        return None
    
    all_bis = mol.GetSubstructMatches(cut)
    if not enumerate:
        all_bis = [random.choice(all_bis)]
        
    fragments_list = []
    for bis in all_bis:
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) != 2:
                fragments_list.append(None)
            else:
                fragments_list.append(fragments)
        except ValueError:
            fragments_list.append(None)

    if not enumerate:
        return fragments_list[0]
    else:
        random.shuffle(fragments_list)
        return fragments_list


def _cut_ring(mol, enumerate=False, bond_order=1):
    """Cut ring and return the molecule fragments.
       If enumerate is True, return all fragments; if false, randomly choose a fragment."""
    if random.random() < 0.5:
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")): # TODO: why do we restrict to 4-membered rings or up? - because we need two fragments, I think 
            return None
        all_bis = mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]"))
        all_bis = [(
            (bis[0], bis[1]),
            (bis[2], bis[3]),
        ) for bis in all_bis]
    else:
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
            return None
        all_bis = mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]"))
        all_bis = [(
            (bis[0], bis[1]),
            (bis[1], bis[2]),
        ) for bis in all_bis]

    if not enumerate:
        all_bis = [random.choice(all_bis)]

    fragments_list = []
    for bis in all_bis: # TODO: think about either restricting to bond order.... or... if both systems aromatic, then tagging as aromatic 
        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]
        fragments_mol = Chem.FragmentOnBonds(
            mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)]
        )
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) != 2:
                fragments_list.append(None)
            else:
                fragments_list.append(fragments)
        except ValueError:
            fragments_list.append(None)

    if not enumerate:
        return fragments_list[0]
    else:
        random.shuffle(fragments_list)
        return fragments_list


def cut_same_form(mol1, mol2, ring=False):
    if ring:
        cut_funcs = [partial(_cut_ring, bond_order=1)] # TODO: bond order isnt actually argument so for cut_ring, it will just test both
    else:
        cut_funcs = [partial(_expanded_cut, bond_order=1), partial(_expanded_cut, bond_order=2), partial(_expanded_cut, bond_order=3)] #_cut

    # will explore all order 1 cuts, then 2, then 3; could sample instead? 
    for cut_func in cut_funcs:
        all_fragments1 = cut_func(mol1, enumerate=True)
        all_fragments2 = cut_func(mol2, enumerate=True)
        
        if all_fragments1 and all_fragments2:
            # Rank by formula/sort by formula
            for fragments1, fragments2 in itertools.product(all_fragments1, all_fragments2):
                # find fragments with same formula
                if fragments1 and fragments2:
                    mol1_frag1, mol1_frag2 = fragments1
                    mol2_frag1, mol2_frag2 = fragments2
                    if CalcMolFormula(mol1_frag1) == CalcMolFormula(mol2_frag1):
                        if Chem.MolToSmiles(mol1_frag1) != Chem.MolToSmiles(mol2_frag1) and Chem.MolToSmiles(mol1_frag2) != Chem.MolToSmiles(mol2_frag2):
                            # TODO: if aromatic/basically breaking same bond: should also exclude
                            return (mol1_frag1, mol2_frag2, cut_func.keywords["bond_order"]), (mol1_frag2, mol2_frag1, cut_func.keywords["bond_order"])
                        else: # fragments are the same
                            continue
                    elif CalcMolFormula(mol1_frag1) == CalcMolFormula(mol2_frag2):
                        if Chem.MolToSmiles(mol1_frag1) != Chem.MolToSmiles(mol2_frag2) and Chem.MolToSmiles(mol2_frag1) != Chem.MolToSmiles(mol1_frag2):
                            return (mol1_frag1, mol2_frag1, cut_func.keywords["bond_order"]), (mol1_frag2, mol2_frag2, cut_func.keywords["bond_order"])
                        else: # fragments are the same
                            continue
                    else:
                        pass
                        # most don't match by formula to begin with 

    return None


def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    # TODO: turn off! 
    #macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[r3,r4]=[r3,r4]")
    )

    return not ring_allene and not double_bond_in_small_ring


def mol_ok(mol, parent=None):
    try:
        Chem.SanitizeMol(mol)
        if any([atom.GetNumRadicalElectrons() != 0 for atom in mol.GetAtoms()]):
            return False
        if parent is None or CalcMolFormula(mol) == CalcMolFormula(parent):
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_A, parent_B):
    start_time = time.time()
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_A.HasSubstructMatch(ring_smarts) or not parent_B.HasSubstructMatch(ring_smarts):
        return None
    
    # Check cache for failed fragment generation
    parent_inchis = [Chem.MolToInchi(parent_A), Chem.MolToInchi(parent_B)]
    pair_key = tuple(sorted(parent_inchis))
    if hasattr(crossover_ring, 'failed_fragment_cache') and pair_key in crossover_ring.failed_fragment_cache:
        return None
    
    inchi_time = time.time()

    rxn_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]", # Consider if we need # / triple bonds here...?
    ]
    rxn_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]

    for i in range(10): # TODO: may benefit from some paring...? with random.choice
        fragment_start = time.time()
        fragment_pairs = cut_same_form(parent_A, parent_B, ring=True)
        fragment_time = time.time()
        # print(f"crossover_ring: Fragment generation (attempt {i+1}) took {fragment_time - fragment_start:.3f}s")

        if fragment_pairs is None:
            # Cache this definitive failure (no compatible fragments found)
            if not hasattr(crossover_ring, 'failed_fragment_cache'):
                crossover_ring.failed_fragment_cache = set()
            crossover_ring.failed_fragment_cache.add(pair_key)
            return None
        new_mol_trial = []
        reaction_start = time.time()
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            for fa, fb, bond_order in fragment_pairs:
                new_mol = rxn1.RunReactants((fa, fb))
                if len(new_mol) > 0:
                    new_mol_trial.append(new_mol[0])
        reaction_time = time.time()
        # print(f"crossover_ring: First reaction step took {reaction_time - reaction_start:.3f}s")

        new_mols = []
        reaction2_start = time.time()
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))
        reaction2_time = time.time()
        #print(f"crossover_ring: Second reaction step took {reaction2_time - reaction2_start:.3f}s")

        new_mols2 = []
        validation_start = time.time()
        for m in new_mols:
            m = m[0]
            if mol_ok(m, parent_A) and ring_OK(m) and Chem.MolToInchi(m) not in parent_inchis: # mol_ok should take care of kekule errors
                new_mols2.append(m)
        validation_time = time.time()
        # print(f"crossover_ring: Validation step took {validation_time - validation_start:.3f}s")
        
        if len(new_mols2) > 0:
            total_time = time.time() - start_time
            # print(f"crossover_ring: Total successful run took {total_time:.3f}s")
            return random.choice(new_mols2)

    total_time = time.time() - start_time
    # rint(f"crossover_ring: Total failed run took {total_time:.3f}s")
    return None


def crossover_non_ring(parent_A, parent_B):
    start_time = time.time()
    
    # Check cache for failed fragment generation
    parent_inchis = [Chem.MolToInchi(parent_A), Chem.MolToInchi(parent_B)]
    pair_key = tuple(sorted(parent_inchis))
    if hasattr(crossover_non_ring, 'failed_fragment_cache') and pair_key in crossover_non_ring.failed_fragment_cache:
        # print(f"crossover_non_ring: Early exit - cached failed fragments took {time.time() - start_time:.3f}s")
        return None
    
    #parent_smiles = [Chem.MolToSmiles(m) for m in (parent_A, parent_B)]
    inchi_time = time.time()
    # print(f"crossover_non_ring: InChI generation took {inchi_time - start_time:.3f}s")

    for i in range(10):
        # Could consider cutting this part down...?
        fragment_start = time.time()
        fragment_pairs = cut_same_form(parent_A, parent_B, ring=False)
        fragment_time = time.time()
        # print(f"crossover_non_ring: Fragment generation (attempt {i+1}) took {fragment_time - fragment_start:.3f}s")
        
        if fragment_pairs is None:
            # Cache this definitive failure (no compatible fragments found)
            if not hasattr(crossover_non_ring, 'failed_fragment_cache'):
                crossover_non_ring.failed_fragment_cache = set()
            crossover_non_ring.failed_fragment_cache.add(pair_key)
            return None
        # TODO: determine what bond forming to perform within this function, rather than relying on cut_same_form
        rxns = {1: AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]"),
                2: AllChem.ReactionFromSmarts("[*:1]=[1*].[1*]=[*:2]>>[*:1]=[*:2]"),
                3: AllChem.ReactionFromSmarts("[*:1]#[1*].[1*]#[*:2]>>[*:1]#[*:2]")}
        new_mols = []
        reaction_start = time.time()
        for fa, fb, bond_order in fragment_pairs: # will this produce both?
            # determine bond order: 
            rxn = rxns[bond_order]
            new_mol = rxn.RunReactants((fa, fb))
            if len(new_mol) > 0:
                mol = new_mol[0][0]
                if mol_ok(mol) and Chem.MolToInchi(mol) not in parent_inchis:
                    new_mols.append(mol)
        reaction_time = time.time()
        # print(f"crossover_non_ring: Reaction processing took {reaction_time - reaction_start:.3f}s")

        if len(new_mols) > 0:
            total_time = time.time() - start_time
            #print(f"crossover_non_ring: Total successful run took {total_time:.3f}s")
            return random.choice(new_mols)

    total_time = time.time() - start_time
    # print(f"crossover_non_ring: Total failed run took {total_time:.3f}s")
    return None


def crossover(parent_A, parent_B):
    start_time = time.time()
    parent_inchis = [Chem.MolToInchi(parent_A), Chem.MolToInchi(parent_B)]
    # parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    
    # Early check: if molecules are identical, no point in proceeding
    if parent_inchis[0] == parent_inchis[1]:
        # print(f"crossover: Early exit - identical molecules took {time.time() - start_time:.3f}s")
        return None
    
    # Early check: if molecular formulas are different, no point in proceeding
    mol1_formula = CalcMolFormula(parent_A)
    mol2_formula = CalcMolFormula(parent_B)
    if mol1_formula != mol2_formula:
        # print(f"crossover: Early exit - different formulas took {time.time() - start_time:.3f}s")
        return None
    
    
    try: # will cause issues with heteroatoms, and is what is likely responsible for the SMILES discrepancy
    # What happens if you try to substitute N with
        Chem.Kekulize(parent_A, clearAromaticFlags=True) # Consider try setting to False? not sure why set of ture
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    ring_smarts = Chem.MolFromSmarts("[R]")
    ring_check = parent_A.HasSubstructMatch(ring_smarts) and parent_B.HasSubstructMatch(ring_smarts)
    for i in range(10):

        # if parent A has no ring or parent B has no ring:
        if random.random() <= 0.5 or not ring_check:
            # non-ring crossover
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                new_inchi = Chem.MolToInchi(new_mol)
                if new_smiles is not None and new_inchi not in parent_inchis:
                    total_time = time.time() - start_time
                    #print(f"crossover: Total successful run took {total_time:.3f}s")
                    return new_mol
                else:
                    pass
            else:
                pass
        else:
            # ring crossover
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                new_inchi = Chem.MolToInchi(new_mol)
                if new_smiles is not None and new_inchi not in parent_inchis:
                    total_time = time.time() - start_time
                    # print(f"crossover: Total successful run took {total_time:.3f}s")
                    return new_mol

    total_time = time.time() - start_time
    # print(f"crossover: Total failed run took {total_time:.3f}s")
    return None


def clear_crossover_cache():
    """Clear the cache of failed crossover pairs"""
    if hasattr(crossover, 'failed_crossover_cache'):
        crossover.failed_crossover_cache.clear()
    if hasattr(crossover_ring, 'failed_fragment_cache'):
        crossover_ring.failed_fragment_cache.clear()
    if hasattr(crossover_non_ring, 'failed_fragment_cache'):
        crossover_non_ring.failed_fragment_cache.clear()
