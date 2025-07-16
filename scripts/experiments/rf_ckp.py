import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from datetime import datetime
import joblib  

# 目录和对应的后缀
dataset_configs = [
    ("dataset", ""),          # 原始
    ("dataset_smote", "_smote"),
    ("dataset_adasyn", "_adasyn"),
    ("dataset_undersampled", "_undersampled")
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

results = []



# ===================== 主循环 =====================
for base_dir, suffix in tqdm(dataset_configs, desc="Datasets"):
    print("=" * 60)
    print(f"Training on dataset: {base_dir}")

    # 加载数据
    X_train, y_train = load_split_csv(base_dir, "train", suffix)
    X_val, y_val = load_split_csv(base_dir, "val", suffix)
    X_test, y_test = load_split_csv(base_dir, "test", suffix)

    clf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auroc = roc_auc_score(y_val, y_val_prob)

    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_prob)

    # 打印结果
    print(f"[Validation] ACC={val_acc:.4f}, F1={val_f1:.4f}, AUROC={val_auroc:.4f}")
    print(f"[Test] ACC={test_acc:.4f}, F1={test_f1:.4f}, AUROC={test_auroc:.4f}")

    # 保存结果
    results.append({
        "dataset": base_dir,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc
    })


    model_file = f"rf_{base_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(clf, model_file)
    print(f"model saved in: {model_file}")


results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}_randomforest.csv"
results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"🎉 All done! Results saved to: {output_file}")
