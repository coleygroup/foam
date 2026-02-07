import itertools
import random
from functools import partial
from typing import Optional, List, Callable

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, RWMol, rdmolops

rdBase.DisableLog("rdApp.error")

def run_rxn(mols: List[Chem.Mol], smarts: str) -> List[Chem.Mol]:
    """ run_rxn helper function"""
    rxn = AllChem.ReactionFromSmarts(smarts)
    outs = rxn.RunReactants(mols)
    return [j for i in outs for j in i]

def downgrade_bond(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """Choose a bond of order >1 & not in ring and downgrade it.
        Return:
        
        New molecule with downgraded bond	

    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(mol)
    match_struct = Chem.MolFromSmarts("[*]!-;!@[*]")
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0: 
        return None
    
    # [Zero, Single, Double, Triple]
    bond_prob_vec = np.array([0, 0, 1, 0.5])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type - 1

    # Remove bond
    m_edit.RemoveBond(*bis)
    
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))

    # Update 
    # Add two new hydrogens
    a1 = Chem.Atom("H")
    a2 = Chem.Atom("H")
    
    ind_0 = m_edit.AddAtom(a1)
    ind_1 = m_edit.AddAtom(a2)
    
    m_edit.AddBond(bis[0], ind_0, Chem.BondType(1))
    m_edit.AddBond(bis[1], ind_1, Chem.BondType(1))
    mol = Chem.Mol(m_edit)

    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def upgrade_bond(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """upgrade_bond.

	Choose a bond of order <3, not in ring, and with >=1H to upgrade

	Return:
		New molecule with upgraded bond	

	"""
    Chem.Kekulize(mol, clearAromaticFlags=True)

    m_edit = RWMol(mol)
    query = "[*;!H0]~;!#;!@[*;!H0]"

    # bonded, not triple, and not aromatic
    match_struct = Chem.MolFromSmarts(query)
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0:
        return None
#     matching_inds.extend(m_edit.GetSubstructMatches(match_struct_2))

    # [Zero, Single, Double, Triple]
    # Potentially look into ring vs. non ring bonds if not aromatic...
    # Don't add 
    bond_prob_vec = np.array([0, 1, 0.05, 0.00])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type + 1
    
    # Remove H's in preparation
    m_edit = RWMol(Chem.AddHs(m_edit))
    for idx in bis:
        atom = m_edit.GetAtomWithIdx(idx)
        
        # Remove a neighboring hydrogen
        for a in atom.GetNeighbors():
            if a.GetAtomicNum() == 1:
                m_edit.RemoveAtom(a.GetIdx())
                break

    # Remove bond
    m_edit.RemoveBond(*bis)
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))
    return Chem.RemoveHs(m_edit)



def downgrade_bond_in_ring(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """downgrade_bond_in_ring.
    
    Choose a bond of order >1 & in ring and downgrade it. 
        Return:
        
        New molecule with downgraded bond	

    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(mol)
    match_struct = Chem.MolFromSmarts("[*]!-;!@[*]")
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0: 
        return None
    
    # [Zero, Single, Double, Triple]
    bond_prob_vec = np.array([0, 0, 1, 0.5])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type - 1

    # Remove bond
    m_edit.RemoveBond(*bis)
    
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))

    # Update 
    # Add two new hydrogens
    a1 = Chem.Atom("H")
    a2 = Chem.Atom("H")
    
    ind_0 = m_edit.AddAtom(a1)
    ind_1 = m_edit.AddAtom(a2)
    
    m_edit.AddBond(bis[0], ind_0, Chem.BondType(1))
    m_edit.AddBond(bis[1], ind_1, Chem.BondType(1))
    mol = Chem.Mol(m_edit)

    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def upgrade_bond_in_ring(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """upgrade_bond_in_ring.

	Choose a bond of order <3, not in ring, and with >=1H to upgrade

	Return:
		New molecule with upgraded bond	

	"""
    Chem.Kekulize(mol, clearAromaticFlags=True)

    m_edit = RWMol(mol)
    query = "[*;!H0]~;!#;!@[*;!H0]"

    # bonded, not triple, and not aromatic
    match_struct = Chem.MolFromSmarts(query)
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0:
        return None
#     matching_inds.extend(m_edit.GetSubstructMatches(match_struct_2))

    # [Zero, Single, Double, Triple]
    # Potentially look into ring vs. non ring bonds if not aromatic...
    # Don't add 
    bond_prob_vec = np.array([0, 1, 0.05, 0.00])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type + 1
    
    # Remove H's in preparation
    m_edit = RWMol(Chem.AddHs(m_edit))
    for idx in bis:
        atom = m_edit.GetAtomWithIdx(idx)
        
        # Remove a neighboring hydrogen
        for a in atom.GetNeighbors():
            if a.GetAtomicNum() == 1:
                m_edit.RemoveAtom(a.GetIdx())
                break

    # Remove bond
    m_edit.RemoveBond(*bis)
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))
    return Chem.RemoveHs(m_edit)



def break_ring(mol: Chem.Mol):
    """ break_ring."""
    # Single only!
    Chem.Kekulize(mol, clearAromaticFlags=True)
    cyc_smarts = "[*:1]-;@[*:2]>>([*:1].[*:2])"
    out_l = list(run_rxn([mol], cyc_smarts))
    if len(out_l) == 0:
        return None
    
    new_mol = (random.choice(out_l))
    
    Chem.SanitizeMol(new_mol)
    return new_mol

def make_ring(mol):
    """make_ring."""
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    
    p = [0.05, 0.05, 0.45, 0.45]

    smarts = np.random.choice(choices, p=p)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    out_l = list(run_rxn([mol], smarts))
    if len(out_l) == 0:
        return None
    
    new_mol = (random.choice(out_l))
    Chem.SanitizeMol(new_mol)
    return new_mol

def cut_and_paste(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """cut_and_paste.

    Take section from one part of molecule and rejoin it to another
    """ 

    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(Chem.AddHs(mol))
    
    # Find a single bond
    match_struct = Chem.MolFromSmarts("[*;!#1]-;!@[*;!#1]")
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) ==  0: 
        return None
    
    # Find all possible acceptors
    h_struct = Chem.MolFromSmarts("[!H0]")
    h_match = m_edit.GetSubstructMatches(h_struct)
    h_match = set([i[0] for i in h_match])
    if len(h_match) ==  0: 
        return None
    
    # Choose a bond to break
    bond = list(random.choice(matching_inds))
    random.shuffle(bond)
    orig_src, leaving = bond
    
    # Delete the first bond and pad Hs
    # Remove bond
    m_edit.RemoveBond(orig_src, leaving)
    
    # Create bond from orig_src to new hydrogen
    a1 = Chem.Atom(1)
    new_h = m_edit.AddAtom(a1)
    m_edit.AddBond(orig_src, new_h, Chem.BondType(1))

    # Filter down the list of potential acceptors
    # Find all connected components starting at oric_src to make sure we
    # don't make two fragments
    mol_frags = rdmolops.GetMolFrags(m_edit)
    unreachable = [j for i in mol_frags for j in i if orig_src not in i]
    invalid_els = set()
    invalid_els.update(
        [i.GetIdx() for i in m_edit.GetAtomWithIdx(leaving).GetNeighbors()]
    )
    invalid_els.update([leaving, orig_src])
    invalid_els.update(unreachable)
    
    # Remove leaving and all its neighbors from attach trg options
    h_match.difference_update(invalid_els)    
    
    # Choose trg
    if len(h_match) == 0: 
        return None
    trg = random.choice(list(h_match))
    
    # Construct a bond from leaving to trg
    m_edit.AddBond(leaving, trg, Chem.BondType(1))
    
    # Remove H on trg
    m_edit.GetAtomWithIdx(trg).GetNeighbors()
    for a in m_edit.GetAtomWithIdx(trg).GetNeighbors():
        if a.GetAtomicNum() == 1:
            m_edit.RemoveAtom(a.GetIdx())
            break
        
    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def break_make_ring(mol: Chem.Mol) -> Optional[Chem.Mol]:
    pass

def heteroatom_swap_in_ring(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """heteroatom_swap_in_ring.

    Proposes a swap between a heteroatom and carbon that belong to the same ring,
    preserving non-ring connectivity on both atoms. 

    Should hopefully not result in multi-branched rings... not sure if that's a feature or bug. 
    """
    heteroatom_in_ring = Chem.MolFromSmarts("[!#6;R]")

    ring_heteroatoms = mol.GetSubstructMatches(heteroatom_in_ring)
    if len(ring_heteroatoms) == 0:
        return None
    rings = mol.GetRingInfo()
    atoms_in_rings = rings.AtomRings()
    for elem in ring_heteroatoms:
        idx = elem[0]
        heteroatom_type = mol.GetAtomWithIdx(idx).GetAtomicNum() 
        all_atom_ring_idx = [atom for ring in atoms_in_rings for atom in ring if idx in ring]
        other_atom_ring_idx = [atom for atom in all_atom_ring_idx if mol.GetAtomWithIdx(atom).GetAtomicNum() != heteroatom_type]

    try:    
        swapping_idx = random.choice(other_atom_ring_idx)
    except IndexError:
        return None
    new_mol = Chem.RWMol(Chem.AddHs(mol))

    rings_heteroatom = [atom for ring in atoms_in_rings for atom in ring if idx in ring]
    rings_swapping = [atom for ring in atoms_in_rings for atom in ring if swapping_idx in ring]
    heteroatom = new_mol.GetAtomWithIdx(idx)
    swapping_atom = new_mol.GetAtomWithIdx(swapping_idx)
    swapping_atom_num = swapping_atom.GetAtomicNum()
    
    # swap: identify ring bonds for heteroatom, non-ring bonds for heteroatom stay connected.
    # Get neighbors of atom1 and atom2
    neighbors_heteroatom = [nbr.GetIdx() for nbr in heteroatom.GetNeighbors() if nbr.GetIdx() not in rings_heteroatom]
    neighbors_swapping_atom = [nbr.GetIdx() for nbr in swapping_atom.GetNeighbors() if nbr.GetIdx() not in rings_swapping]
    
    # Do swap here
    swapping_atom.SetAtomicNum(heteroatom.GetAtomicNum()) 
    heteroatom.SetAtomicNum(swapping_atom_num) 

    # restore connectivity of external bonds!
    for nbr in neighbors_heteroatom:
        # get bond order
        bond_order = new_mol.GetBondBetweenAtoms(idx, nbr).GetBondType()
        new_mol.RemoveBond(idx, nbr)
        new_mol.AddBond(swapping_idx, nbr, bond_order)
    for nbr in neighbors_swapping_atom:
        bond_order = new_mol.GetBondBetweenAtoms(swapping_idx, nbr).GetBondType()
        new_mol.RemoveBond(swapping_idx, nbr)
        new_mol.AddBond(idx, nbr, bond_order)

    try:
        new_mol = Chem.RemoveHs(new_mol)
    except ValueError:
        return None
    return new_mol

def ring_fusion(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """ring_fusion.

    Take two rings and merge them into one system
    (will yield a fused ring system, with a carbon product on exterior)
    """
    # step 0: identify both rings and verify not overlapping - 
    smarts = '[R]~!@[R]'
    pattern = Chem.MolFromSmarts(smarts)
    ring_connecting_atoms = mol.GetSubstructMatches(pattern)
    # find bonds that go from one ring to another; ie, that atom is not part of the same ring as the first. 
    if len(ring_connecting_atoms) == 0:
        return None

    # Function to find bonds connecting different rings
    # I don't think this will be comprehensive in some cases; I'd ideally want
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bonds_to_break = set()
    ring_membership = {}
    for atom1, atom2 in ring_connecting_atoms:
        atom1_rings = set(idx for idx, ring in enumerate(atom_rings) if atom1 in ring)
        atom2_rings = set(idx for idx, ring in enumerate(atom_rings) if atom2 in ring)
        if len(atom1_rings.intersection(atom2_rings)) == 0:
            # two different rings
            bonds_to_break.add((atom1, atom2))
            ring_membership[atom1] = [atom_rings[i] for i in atom1_rings]
            ring_membership[atom2] = [atom_rings[i] for i in atom2_rings]
            
    # pick two bonds to break (and two Cs need to disappear somehow.)
    for ring1_atom, ring2_atom in bonds_to_break:
        new_mol = Chem.RWMol(Chem.AddHs(mol))
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
        
        atoms = [ring1_atom, ring2_atom]
        random.shuffle(atoms)
        ring_break, ring_fuse = atoms
        ring_break_atoms = ring_membership[ring_break][0]
        ring_fuse_atoms = ring_membership[ring_fuse][0]

        # TODO: looks only at the one ring, make this is the right ring
        ring_break_neighbors = [nbr.GetIdx() for nbr in new_mol.GetAtomWithIdx(ring_break).GetNeighbors() if nbr.GetIdx() in ring_break_atoms]
        to_break = [ring_break, random.choice(ring_break_neighbors)]
        
        # to fuse: pick an Carbon with a Hydrogen as neighbor, then migrate that H
        
        one_neighbor_down = [nbr for nbr in new_mol.GetAtomWithIdx(to_break[0]).GetNeighbors() if nbr.GetIdx() != to_break[1] and nbr.GetIdx() in ring_break_atoms][0]
        two_neighbors_down = [nbr for nbr in one_neighbor_down.GetNeighbors() if nbr.GetIdx() != to_break[0] and nbr.GetIdx() in ring_break_atoms]
        two_neighbors_down = two_neighbors_down[0].GetIdx()

        ring_break_to_fuse = [to_break[1], two_neighbors_down]
        bond_break_order1 = Chem.BondType.AROMATIC if new_mol.GetAtomWithIdx(ring_break_to_fuse[0]).GetIsAromatic() else Chem.BondType.SINGLE
        bond_break_order2 = Chem.BondType.AROMATIC if new_mol.GetAtomWithIdx(ring_break_to_fuse[1]).GetIsAromatic() else Chem.BondType.SINGLE

        new_mol.RemoveBond(ring_break, ring_fuse)
        new_mol.RemoveBond(ring_break, to_break[1])               

        ring_fuse_neighbors = [nbr.GetIdx() for nbr in new_mol.GetAtomWithIdx(ring_fuse).GetNeighbors() if nbr.GetIdx() in ring_fuse_atoms and nbr.GetAtomicNum() == 6]
        if len(ring_fuse_neighbors) == 0:
            return None
        to_fuse = [ring_fuse, random.choice(ring_fuse_neighbors)]

        # TODO: need not be H, just non-ring connectivity
        # condition is if nbr is not in ring_fuse_atoms, and has a single bond 
        # nbr and to_fuse[1] have a bond order of 1
        try:
            fusing_ring_H_or_group = [nbr for nbr in new_mol.GetAtomWithIdx(to_fuse[1]).GetNeighbors() if 
                                      (nbr.GetIdx() not in ring_fuse_atoms and new_mol.GetBondBetweenAtoms(to_fuse[1], nbr.GetIdx()).GetBondType() == 1)][0].GetIdx()
            breaking_ring_H_or_group = [nbr for nbr in new_mol.GetAtomWithIdx(ring_break_to_fuse[1]).GetNeighbors() if 
                                        (nbr.GetIdx() not in ring_break_atoms and new_mol.GetBondBetweenAtoms(ring_break_to_fuse[1], nbr.GetIdx()).GetBondType() == 1)][0].GetIdx()
        except IndexError as e:
            return None
        
        new_mol.RemoveBond(to_fuse[1], fusing_ring_H_or_group)
        new_mol.RemoveBond(ring_break_to_fuse[1], breaking_ring_H_or_group)
        new_mol.AddBond(to_break[0], fusing_ring_H_or_group, Chem.BondType.SINGLE)
        new_mol.AddBond(to_break[0], breaking_ring_H_or_group, Chem.BondType.SINGLE)
        
        #for i in range(3):
        #    a1 = Chem.Atom(1)
        #    new_h = new_mol.AddAtom(a1)
        #    new_mol.AddBond(to_break[0], new_h, Chem.BondType(1))
        
        
        new_mol.AddBond(ring_break_to_fuse[0], to_fuse[0], bond_break_order1) #TODO: may not always be aromatic
        new_mol.AddBond(ring_break_to_fuse[1], to_fuse[1], bond_break_order2) #TODO: may not always be aromatic
        try:
            new_mol = Chem.RemoveHs(new_mol)
        except ValueError as e:
            continue
        return new_mol

def expanded_cut_and_paste(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """cut_and_paste.

    Take section from one part of molecule and rejoin it to another
    """ 

    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(Chem.AddHs(mol))
    swaps = {
        "single-H": [Chem.MolFromSmarts("[*;!#1]-;!@[*;!#1]"), Chem.MolFromSmarts("[!H0]")], # First bond specifies that neither atom of first bond should just be a H
        "single-single": [Chem.MolFromSmarts("[*;!#1]-;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]-;!@[*;!#1]")], # between two non-H atoms
        "double-double": [Chem.MolFromSmarts("[*;!#1]=;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]=;!@[*;!#1]")],
        "triple-triple": [Chem.MolFromSmarts("[*;!#1]#;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]#;!@[*;!#1]")],
        "single-double": [Chem.MolFromSmarts("[H]-[*;!#1]-;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]=;!@[*;!#1]")],
        "double-triple": [Chem.MolFromSmarts("[H]-[*;!#1]=;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]#;!@[*;!#1]")],
        "single-triple": [Chem.MolFromSmarts("[H2]-[*;!#1]=;!@[*;!#1]"), Chem.MolFromSmarts("[*;!#1]#;!@[*;!#1]")],
        "double-2H": [Chem.MolFromSmarts("[*;!#1]=;!@[*;!#1]"), Chem.MolFromSmarts("[H2,H3]")], # TODO: need further work
        "triple-3H": [Chem.MolFromSmarts("[*;!#1]#;!@[*;!#1]"), Chem.MolFromSmarts("[H3]")], # TODO: need further work
    }
    p = np.array([0.3, 0.4, 0.2, 0.1, 0, 0, 0, 0, 0])
    while True:
        swap_type = np.random.choice(list(swaps.keys()), p=p)
        # zero out changes in bond order for now
        first_struct, second_struct = swaps[swap_type]

        # Find first bond to break
        matching_inds = m_edit.GetSubstructMatches(first_struct)
        # if none: try different swap type - and update p 
        if len(matching_inds) == 0:
            p[list(swaps.keys()).index(swap_type)] = 0
            if p.sum() == 0:
                return None
            p = p / p.sum()
            continue
        
        # Find second bond to break
        matching_inds_2 = m_edit.GetSubstructMatches(second_struct)
        # also update p if the matches end up being the same and <2 -- is useless in that case!
        if len(matching_inds_2) == 0 or len(set(matching_inds).union(set(matching_inds_2))) < 2:
            p[list(swaps.keys()).index(swap_type)] = 0
            if p.sum() == 0:
                return None
            p = p / p.sum()
            continue
        else:
            break
    
    # Choose a bond to break, by first enumerating and checking length
    all_bond_pairs = itertools.product(matching_inds, matching_inds_2)
    all_bond_pairs = [[bond1, bond2] for bond1, bond2 in all_bond_pairs if len(set(bond1 + bond2)) == len(bond1 + bond2)]
    if len(all_bond_pairs) == 0:
        return None
    bond1, bond2 = random.choice(all_bond_pairs)


    if len(bond1) == 2:
        bond1 = list(bond1)
        bond2 = list(bond2)
        random.shuffle(bond1)
        random.shuffle(bond2)
        orig_src1, leaving1 = bond1
        if len(bond2) == 2:
            orig_src2, leaving2 = bond2
        elif len(bond2) == 1: # is an H-bond specified implicitly, get H differently - no need to shuffle which Hs are moved around
            numHs = [a for a in m_edit.GetAtomWithIdx(bond2[0]).GetNeighbors() if a.GetAtomicNum() == 1]
            if swap_type == "single-H":
                H = random.choice(numHs)
                leaving2 = H.GetIdx()
                orig_src2 = bond2[0] 
            elif swap_type == "double-2H":
                H = random.sample(numHs, 2)
                leaving2 = [H[0].GetIdx(), H[1].GetIdx()]
                orig_src2 = bond2[0]
            elif swap_type == "triple-3H":
                leaving2 = [H[0].GetIdx(), H[1].GetIdx(), H[2].GetIdx()]
                orig_src2 = bond2[0]

        # step 1: remove bonds
        multiple_bonds = type(leaving2) == list
        if not multiple_bonds:
            bond_order1 = m_edit.GetBondBetweenAtoms(orig_src1, leaving1).GetBondType()
            bond_order2 = m_edit.GetBondBetweenAtoms(orig_src2, leaving2).GetBondType()

            m_edit.RemoveBond(orig_src1, leaving1)
            m_edit.RemoveBond(orig_src2, leaving2)

            # recreate bonds by finding correct connections; want to make sure one molecule made at end!
            mol_frags = rdmolops.GetMolFrags(m_edit)

            frag_orig_src1 = [i for i in mol_frags if orig_src1 in i][0]
            frag_leaving1 = [i for i in mol_frags if leaving1 in i][0]
            frag_orig_src2 = [i for i in mol_frags if orig_src2 in i][0]
            frag_leaving2 = [i for i in mol_frags if leaving2 in i][0]
            if orig_src2 not in frag_orig_src1 and leaving2 in frag_orig_src1:
                leaving2, orig_src2 = orig_src2, leaving2
            elif frag_leaving1 == frag_orig_src2 or frag_leaving2 == frag_orig_src1: # also swap
                leaving2, orig_src2 = orig_src2, leaving2
            # TODO: this doesn't enable the case where orig_src1 and orig_src2 are in the same fragment 
            # Which is ok: WHEN formed IF there is somewhere a bond order can be downgraded
            # Must catch bonds already present in the fragment, e.g. check not neighbors of each other
            # display("orig_src1", orig_src1, "leaving1", leaving1, "orig_src2", orig_src2, "leaving2", leaving2)
            
            # now repeat for orig_src2 - also 
            #unreachable_orig_src2 = [j for i in mol_frags for j in i if orig_src2 not in i]
            #invalid_els2 = set()
            #invalid_els2.update(
            #    [i.GetIdx() for i in m_edit.GetAtomWithIdx(leaving2).GetNeighbors()]
            #)
            #invalid_els2.update([leaving2, orig_src2])
            #invalid_els2.update(unreachable_orig_src2)

            # TODO: what to do with invalid_els2?
            m_edit.AddBond(orig_src1, leaving2, bond_order2)
            m_edit.AddBond(orig_src2, leaving1, bond_order1)


        else: # doing a 2H or 3H bond swap 
            bond_order1 = m_edit.GetBondBetweenAtoms(orig_src1, leaving1).GetBondType()
            m_edit.RemoveBond(orig_src1, leaving1)
            bond_orders2 = [m_edit.GetBondBetweenAtoms(orig_src2, i).GetBondType() for i in leaving2]
            for i in leaving2:
                m_edit.RemoveBond(orig_src2, i)

            # recreate bonds by finding correct connections; want to make sure one molecule made at end!
            mol_frags = rdmolops.GetMolFrags(m_edit)
            frag_orig_src1 = [i for i in mol_frags if orig_src1 in i][0]
            frag_leaving1 = [i for i in mol_frags if leaving1 in i][0]
            frag_orig_src2 = [i for i in mol_frags if orig_src2 in i][0]
            # not needed for Hs because should be their own frags
            # frag_leaving2 = [i for i in mol_frags if leaving2 in i][0]
            #if orig_src2 in frag_leaving1:
            #    leaving1, orig_src1 = orig_src1, leaving1
            # display("orig_src1", orig_src1, "leaving1", leaving1, "orig_src2", orig_src2, "leaving2", leaving2)

            for order, leaving_atom in zip(bond_orders2, leaving2):
                m_edit.AddBond(orig_src1, leaving_atom, order)
            
            m_edit.AddBond(orig_src2, leaving1, bond_order1)


    elif len(bond1) == 3: # case where H is also migrating as well (bond order change @ same time)
        leavingH, orig_src1, leaving1 = bond1
        orig_src2, leaving2 = bond2

        bond_order1 = m_edit.GetBondBetweenAtoms(orig_src1, leaving1).GetBondType()
        bond_order2 = m_edit.GetBondBetweenAtoms(orig_src2, leaving2).GetBondType()
        m_edit.RemoveBond(orig_src1, leaving1)
        m_edit.RemoveBond(orig_src1, leavingH)
        m_edit.RemoveBond(orig_src2, leaving2)


        # figure out which fragment orig_src1 is in; orig_src2 or leaving2? swap terms if need be
        mol_frags = rdmolops.GetMolFrags(m_edit)
        frag_orig_src1 = [i for i in mol_frags if orig_src1 in i][0]
        frag_leaving1 = [i for i in mol_frags if leaving1 in i][0]
        frag_orig_src2 = [i for i in mol_frags if orig_src2 in i][0]
        frag_leaving2 = [i for i in mol_frags if leaving2 in i][0]

        if orig_src2 not in frag_orig_src1 and leaving2 in frag_orig_src1:
            #display('swapped')
            leaving2, orig_src2 = orig_src2, leaving2

        elif frag_leaving1 == frag_orig_src2 or frag_leaving2 == frag_orig_src1: # also swap
            leaving2, orig_src2 = orig_src2, leaving2
        #display("orig_src1", orig_src1, "leaving1", leaving1, "orig_src2", orig_src2, "leaving2", leaving2)


        m_edit.AddBond(orig_src1, leaving2, bond_order2)
        m_edit.AddBond(orig_src2, leavingH, Chem.BondType(1))
        m_edit.AddBond(orig_src2, leaving1, bond_order1)

    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def mol_ok(mol):
    """ mol_ok. """
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False



def ring_ok(mol):
    """ ring_ok. 

    Change to allow more flexibility 

    """

    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    # The following code breaks for large conjoined rings
    #cycle_list = mol.GetRingInfo().AtomRings()
    #max_cycle_length = max([len(j) for j in cycle_list])
    # macro_cycle = max_cycle_length > 7
    # Fix: 
    # macro_cycle = max([len(j) for j in Chem.rdmolops.GetSymmSSSR(mol)]) > 7

    double_bond_in_small_ring = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[r3,r4]=[r3,r4]")
    )
    return not ring_allene and not double_bond_in_small_ring

def apply_down_up(mol: Chem.Mol, fn_down: Callable, fn_up: Callable):
    """ apply_down_up.

    Partial function that jointly applies a bond downgrade and upgrade

    Args:
        mol:
        fn_down
        fn_up 
    """
    down_out = fn_down(mol)
    if down_out is None: 
        return None

    up_out = fn_up(down_out)
    if up_out is None: 
        return None
    
    return up_out

def mutate(mol, mutation_rate):
    """ mutate."""

    # Create pairs of functions with down and up
    if random.random() > mutation_rate:
        return mol

    transforms = [
		partial(apply_down_up, fn_down = downgrade_bond, fn_up = upgrade_bond),
		partial(apply_down_up, fn_down = downgrade_bond, fn_up = make_ring),
		partial(apply_down_up, fn_down = break_ring, fn_up = upgrade_bond),
		partial(apply_down_up, fn_down = break_ring, fn_up = make_ring),
		# cut_and_paste,
        expanded_cut_and_paste,
        heteroatom_swap_in_ring,
        ring_fusion,
    ]
    transforms_p = np.array([0.07, 0.14, 0.07, 0.16, 0.28, 0.14, 0.14 ])
        # for stress-testing ring fusion: [0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.50])
        #[0.07, 0.14, 0.07, 0.16, 0.28, 0.14, 0.14 ]) # previously:  [0.03, 0.14, 0.03, 0.2 ,0.6]
    ring_check = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]"))
    if not ring_check:
        transforms_p[2] = 0.0
        transforms_p[3] = 0.0
        transforms_p[5] = 0.0
        transforms_p[6] = 0.0
        transforms_p = transforms_p / transforms_p.sum()
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]~!@[R]")):
        transforms_p[6] = 0.0
        transforms_p = transforms_p / transforms_p.sum()

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return None

    for i in range(10):
        transform = np.random.choice(transforms, p=transforms_p)
        new_mol = transform(mol)
        
        if new_mol is None:
            continue

        # smarts = '[n;R][n;R]'
        # pattern = Chem.MolFromSmarts(smarts)
        # two_n_in_ring = new_mol.GetSubstructMatches(pattern)
        # if len(two_n_in_ring) > 0:
        #     print("two_n_in_ring", Chem.MolToSmiles(new_mol))

        if new_mol is None:
            continue
        elif mol_ok(new_mol) and ring_ok(new_mol): 
            return new_mol
        else: 
            continue
    return None
