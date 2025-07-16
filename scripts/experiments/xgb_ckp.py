import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from datetime import datetime

dataset_configs = [
    ("dataset", "",  {
        'colsample_bytree': 0.8, 'learning_rate': 0.05,
        'min_child_weight': 3, 'n_estimators': 400, 'subsample': 0.8
    }),
    ("dataset_smote", "_smote", {
        'colsample_bytree': 0.8, 'learning_rate': 0.1,
        'min_child_weight': 1, 'n_estimators': 700, 'subsample': 0.8
    }),
    ("dataset_adasyn", "_adasyn", {
        'colsample_bytree': 1.0, 'learning_rate': 0.1,
        'min_child_weight': 1, 'n_estimators': 700, 'subsample': 1.0
    }),
    ("dataset_undersampled", "_undersampled", {
        'colsample_bytree': 1.0, 'learning_rate': 0.05,
        'min_child_weight': 3, 'n_estimators': 400, 'subsample': 1.0
    }),
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
os.makedirs("saved_models", exist_ok=True)

for base_dir, suffix, best_params in tqdm(dataset_configs, desc="Datasets"):
    print("\n" + "=" * 60)
    print(f"▶ Training with best_params on dataset: {base_dir}")
    print(f"Params: {best_params}")

    X_train, y_train = load_split_csv(base_dir, "train", suffix)
    X_val, y_val = load_split_csv(base_dir, "val", suffix)
    X_test, y_test = load_split_csv(base_dir, "test", suffix)


    best_clf = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        device="cuda",
        random_state=42
    )


    best_clf.fit(X_train, y_train)


    y_val_pred = best_clf.predict(X_val)
    y_val_prob = best_clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auroc = roc_auc_score(y_val, y_val_prob)

    y_test_pred = best_clf.predict(X_test)
    y_test_prob = best_clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_prob)

    print(f"[Validation] ACC={val_acc:.4f}, F1={val_f1:.4f}, AUROC={val_auroc:.4f}")
    print(f"[Test]       ACC={test_acc:.4f}, F1={test_f1:.4f}, AUROC={test_auroc:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join("checkpoints", f"xgb_{base_dir}_{timestamp}.pkl")
    joblib.dump(best_clf, model_file)
    print(f"saved: {model_file}")

    results.append({
        "dataset": base_dir,
        "params": str(best_params),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "model_file": model_file
    })

results_df = pd.DataFrame(results)
summary_file = f"xgb_best_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
print(f"\n🎉 All done! Summary saved to: {summary_file}")
