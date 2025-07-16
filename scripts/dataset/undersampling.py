import os
import pandas as pd
import torch
from collections import Counter
from sklearn.utils import resample
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

input_dir = "dataset"
output_dir = "dataset_undersampled"
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "val", "test"]

for split_name in splits:
    print(f"Processing {split_name} set for undersampling...")

    csv_path = os.path.join(input_dir, f"{split_name}_fingerprints.csv")
    pt_path = os.path.join(input_dir, f"{split_name}_graphs.pt")
    df = pd.read_csv(csv_path)
    graphs = torch.load(pt_path, weights_only=False)

    labels = df["label"].tolist()
    label_counts = Counter(labels)
    print(f"Original {split_name} distribution: {label_counts}")

    min_count = min(label_counts.values())
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]

    df_pos_sampled = resample(df_pos, replace=False, n_samples=min_count, random_state=42)
    df_neg_sampled = resample(df_neg, replace=False, n_samples=min_count, random_state=42)

    df_balanced = pd.concat([df_pos_sampled, df_neg_sampled], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    original_index_map = {num: i for i, num in enumerate(df["num"])}
    selected_indices = [original_index_map[n] for n in df_balanced["num"]]
    graphs_balanced = [graphs[i] for i in selected_indices]

    out_csv = os.path.join(output_dir, f"{split_name}_fingerprints_undersampled.csv")
    out_pt = os.path.join(output_dir, f"{split_name}_graphs_undersampled.pt")

    df_balanced.to_csv(out_csv, index=False, encoding="utf-8-sig")
    torch.save(graphs_balanced, out_pt)

    print(f"Saved undersampled {split_name}: {out_csv}, {out_pt}")
    print(f"New {split_name} distribution: {Counter(df_balanced['label'])}")

print("All splits undersampled successfully.")
