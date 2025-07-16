import os
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# ====== 数据集 ======
dataset_configs = [
    ("dataset", ""),
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

# ====== Optuna 调参逻辑 ======
def objective(trial, X_train, y_train, X_val, y_val):
    # 定义连续/离散搜索空间
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)  # 100~1000 每50取一值
    max_depth = trial.suggest_int("max_depth", 5, 50)  # 连续整数 5~50
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)  # 连续整数 2~10

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    return val_acc  # Optuna 会最大化这个值

all_results = []

# ====== 针对每个数据集调参 ======
for base_dir, suffix in dataset_configs:
    print("=" * 60)
    print(f"Dataset: {base_dir}")

    X_train, y_train = load_split_csv(base_dir, "train", suffix)
    X_val, y_val = load_split_csv(base_dir, "val", suffix)
    X_test, y_test = load_split_csv(base_dir, "test", suffix)

    def wrapped_objective(trial):
        return objective(trial, X_train, y_train, X_val, y_val)

    # 创建 Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=20, show_progress_bar=True)  # 试 20 组

    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params

    # 用最佳参数在训练集上重新训练
    best_clf = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    best_clf.fit(X_train, y_train)

    # 评估验证集和测试集
    val_acc = accuracy_score(y_val, best_clf.predict(X_val))
    test_acc = accuracy_score(y_test, best_clf.predict(X_test))

    print(f"[{base_dir}] ValACC={val_acc:.4f}, TestACC={test_acc:.4f}")

    all_results.append({
        "dataset": base_dir,
        "best_params": str(best_params),
        "val_acc": val_acc,
        "test_acc": test_acc
    })

# ====== 保存结果 ======
results_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}_rf_optuna.csv"
results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Results saved to: {output_file}")
