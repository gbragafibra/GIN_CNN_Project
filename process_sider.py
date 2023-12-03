from rdkit import Chem
import pandas as pd

sider = pd.read_csv("data/sider.csv")

def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    
    return num_atoms


threshold = 8 #Eliminate every molecule
# with less than 8 atoms
# Doing this given that we need consistent mean
# pooling sizes on the first layer
filter_sider = sider[sider["smiles"].apply(count_atoms) >= threshold]

filter_sider.to_csv("data/filtered_sider.csv", index=False)
