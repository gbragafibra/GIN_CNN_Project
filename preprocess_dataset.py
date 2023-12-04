import numpy as np 
from rdkit import Chem

def preprocess_dataset(dataset, task):
    processed_data = []
    for index, molecule in dataset.iterrows():
        mol = Chem.MolFromSmiles(molecule["smiles"]) 
        node_features = []
        for atom in mol.GetAtoms():
            atom_feats = [atom.GetAtomicNum()]
            node_features.append(atom_feats)
        H = np.array(node_features).astype(np.float32)

        A = np.zeros((len(H), len(H)), dtype=float)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            A[i, j] = bond_type
            A[j, i] = bond_type  
        np.fill_diagonal(A, 1)

        task1 = molecule[task]
        

        processed_data.append({"features": {"H": H, "A": A}, "label": task1})


    return processed_data