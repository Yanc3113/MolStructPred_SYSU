import os
import pandas as pd
import torch
from collections import Counter
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

input_dir = "dataset"
output_dir = "dataset_smote"
os.makedirs(output_dir, exist_ok=True)

# 只对训练集做 SMOTE
splits = ["train"]

for split_name in splits:
    print(f"Processing {split_name} set for SMOTE oversampling...")

    csv_path = os.path.join(input_dir, f"{split_name}_fingerprints.csv")
    pt_path = os.path.join(input_dir, f"{split_name}_graphs.pt")

    df = pd.read_csv(csv_path)
    labels = df["label"].tolist()
    print(f"Original {split_name} distribution: {Counter(labels)}")

    feature_cols = [c for c in df.columns if c not in ["label", "name", "num"]]
    X = df[feature_cols].values
    y = df["label"].values

    smote = SMOTE(random_state=13)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"New {split_name} distribution after SMOTE: {Counter(y_resampled)}")

    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled["label"] = y_resampled

    original_len = len(df)
    extra_len = len(df_resampled) - original_len
    names_resampled = list(df["name"]) + [f"synthetic_{i}" for i in range(extra_len)]
    nums_resampled = list(df["num"]) + [f"synthetic_{i}" for i in range(extra_len)]
    df_resampled["name"] = names_resampled
    df_resampled["num"] = nums_resampled

    out_csv = os.path.join(output_dir, f"{split_name}_fingerprints_smote.csv")
    df_resampled.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 图数据不做 SMOTE，原样保存
    graphs = torch.load(pt_path, weights_only=False)
    out_pt = os.path.join(output_dir, f"{split_name}_graphs_original.pt")
    torch.save(graphs, out_pt)

    print(f"Saved SMOTE {split_name}: {out_csv} (graphs kept as original {out_pt})")

print("SMOTE done for training set.")
