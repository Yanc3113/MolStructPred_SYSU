import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


dataset_configs = [
    ("dataset", ""),  
    ("dataset_smote", "_smote"),
    ("dataset_adasyn", "_adasyn"),
    ("dataset_undersampled", "_undersampled")
]

param_list = [
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 5},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 5},
#      {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
#  {"n_estimators": 500, "max_depth": None, "min_samples_split": 5},
#  {"n_estimators": 700, "max_depth": None, "min_samples_split": 2},
#  {"n_estimators": 1000, "max_depth": None, "min_samples_split": 2},
#  {"n_estimators": 700, "max_depth": None, "min_samples_split": 5},
]

def load_split_csv(base_dir, split, suffix):
    file_name = f"{split}_fingerprints{suffix}.csv"
    path = os.path.join(base_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ["label", "name", "num"]]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y

overall_results = []

for idx, params in enumerate(tqdm(param_list, desc="Hyperparameter sets")):
    print("="*70)
    print(f"Testing hyperparams {idx+1}/{len(param_list)}: {params}")

    all_val_acc = []
    all_test_acc = []

    for base_dir, suffix in dataset_configs:
        X_train, y_train = load_split_csv(base_dir, "train", suffix)
        X_val, y_val = load_split_csv(base_dir, "val", suffix)
        X_test, y_test = load_split_csv(base_dir, "test", suffix)

        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            random_state=66,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        all_val_acc.append(val_acc)

        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        all_test_acc.append(test_acc)

        print(f"[{base_dir}] ValACC={val_acc:.4f} TestACC={test_acc:.4f}")


    # avg_val = sum(all_val_acc) / len(all_val_acc)
    avg_test = sum(all_test_acc) / len(all_test_acc)
    # overall_acc = (avg_val + avg_test) / 2.0
    overall_acc = avg_test
    print(f"Overall Accuracy for params {idx+1}: {overall_acc:.4f}")

    overall_results.append({
        "params": str(params),
        "overall_acc": overall_acc
    })


results_df = pd.DataFrame(overall_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}_rf_iterative.csv"
results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"All done! Results saved to: {output_file}")


labels = [f"Set{i+1}" for i in range(len(param_list))]
scores = [r["overall_acc"] for r in overall_results]

plt.figure(figsize=(8, 5))
plt.bar(labels, scores)
for i, v in enumerate(scores):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom')
plt.title("Overall Accuracy for Different Hyperparameter Sets")
plt.xlabel("Hyperparameter Set")
plt.ylabel("Overall Accuracy")
plt.tight_layout()
plt.savefig(f"overall_acc_{timestamp}.png", dpi=300)
plt.show()
