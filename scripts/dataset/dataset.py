import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import torch
from tqdm import tqdm


output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv("BBBP.csv")

#转换
def mol_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return list(fp)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), x=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondTypeAsDouble())
    data = from_networkx(G)
    data.x = data.x.view(-1, 1).float()
    return data

#转换并过滤
print("Converting molecules to fingerprints and graphs...")
valid_smiles, fingerprints, graphs, names, labels, ids = [], [], [], [], [], []
invalid_smiles = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    smi = row["smiles"]
    fp = mol_to_fingerprint(smi)
    graph = mol_to_graph(smi)
    if fp is None or graph is None:
        invalid_smiles.append(smi)
        continue
    fingerprints.append(fp)
    graphs.append(graph)
    valid_smiles.append(smi)
    names.append(row["name"])
    labels.append(row["p_np"])
    ids.append(row["num"])

# 转换为 DataFrame 并按 7:2:1 分层切分
df_valid = pd.DataFrame({
    "smiles": valid_smiles,
    "name": names,
    "p_np": labels,
    "num": ids
})

y = df_valid["p_np"]
X_temp, X_train = train_test_split(df_valid, test_size=0.7, stratify=y, random_state=42)
y_temp = X_temp["p_np"]
val_ratio = 2 / 3
X_val, X_test = train_test_split(X_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=42)

splits = {
    "train": X_train.reset_index(drop=True),
    "val": X_val.reset_index(drop=True),
    "test": X_test.reset_index(drop=True)
}

# 指纹和图也按index一起划分
fp_df = pd.DataFrame(fingerprints)
graph_tensor = graphs

index_map = df_valid.reset_index().set_index("num")["index"].to_dict()

for split_name, split_df in splits.items():
    print(f"Processing {split_name} set...")

    split_indices = [index_map[n] for n in split_df["num"]]


    split_fp = fp_df.iloc[split_indices].copy()
    split_fp["label"] = split_df["p_np"].tolist()
    split_fp["name"] = split_df["name"].tolist()
    split_fp["num"] = split_df["num"].tolist()
    split_fp.to_csv(os.path.join(output_dir, f"{split_name}_fingerprints.csv"), index=False)

    split_graphs = [graph_tensor[i] for i in split_indices]
    torch.save(split_graphs, os.path.join(output_dir, f"{split_name}_graphs.pt"))

    print(f"{split_name.capitalize()} saved to: {output_dir}/{split_name}_fingerprints.csv & .pt")
    

if invalid_smiles:
    invalid_df = pd.DataFrame({"invalid_smiles": invalid_smiles})
    invalid_path = os.path.join(output_dir, "invalid_smiles.csv")
    invalid_df.to_csv(invalid_path, index=False, encoding="utf-8-sig")
    print(f"Invalid SMILES saved to: {invalid_path}")
else:
    print("No invalid SMILES found. All molecules were processed successfully.")

