# Formula constrained graph genetic algorithms

Graph genetic algorithm with formula conditioning

This approach utilizes a given chemical formula to _only_ make chemical modifications that preserve the chemical formula. Specifically each of the following transformations leads to a net change in number of hydrogens:

1. Bond order upgrade: -2 
2. Bond order downgrade: +2
3. Cycle add: -2  
4. Cycle remove: +2
5. Reattach fragment: 0

In our mutation procedure, we utilize RDKit to execute the above changes. We intentionally pair any bond order upgrade with a bond order downgrade such that the net change in hydrogens is 0. 

Crossover and mutation are both operators utilized. 

This method directly utilizes RDKit rather than operating on Bond Electron (BE) matrices directly. 
