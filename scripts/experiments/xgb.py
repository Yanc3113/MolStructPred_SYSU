import os
import warnings
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from datetime import datetime

# 屏蔽一些不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

# ===================== 数据集配置 =====================
dataset_configs = [
    ("dataset", ""),  
    ("dataset_smote", "_smote"),
    ("dataset_adasyn", "_adasyn"),
    ("dataset_undersampled", "_undersampled")
]

def load_split_csv(base_dir, split, suffix):
    """读取对应目录下 train/val/test 的 csv 文件"""
    file_name = f"{split}_fingerprints{suffix}.csv"
    path = os.path.join(base_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ["label", "name", "num"]]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y


param_grid = {
    "n_estimators": [400, 700],  
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3]
    # "n_estimators": [400,700],  
    # "learning_rate": [0.05,0.01],
    # "subsample": [0.8,1.0],
    # "colsample_bytree": [0.8],
    # "min_child_weight": [1]
}

results = []


for base_dir, suffix in tqdm(dataset_configs, desc="Datasets (overall)", position=0):
    print("\n" + "=" * 60)
    print(f"▶ Processing dataset: {base_dir}")

    X_train, y_train = load_split_csv(base_dir, "train", suffix)
    X_val, y_val = load_split_csv(base_dir, "val", suffix)
    X_test, y_test = load_split_csv(base_dir, "test", suffix)

    # 初始化 GPU 加速的 XGB（新版写法）
    xgb_gpu_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",    # GPU 加速用 hist + device=cuda
        device="cuda",
        random_state=42
    )

    # 用 tqdm 包裹 GridSearch
    print("Running GridSearchCV ...")
    grid = GridSearchCV(
        estimator=xgb_gpu_base,
        param_grid=param_grid,
        scoring="f1",      # 优化指标
        cv=3,
        verbose=1,
        n_jobs=1           # 不要多进程，防止 GPU 冲突
    )

    # 使用 tqdm 包裹 fit 的进度
    with tqdm(total=1, desc=f"GridSearch on {base_dir}", position=1, leave=False) as pbar:
        grid.fit(X_train, y_train)
        pbar.update(1)

    # 最佳参数
    best_params = grid.best_params_
    print(f" Best params for {base_dir}: {best_params}")

    # 用最佳参数重新训练
    best_clf = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        device="cuda",
        random_state=42
    )

    # tqdm 包裹训练过程
    with tqdm(total=1, desc=f"Training best model on {base_dir}", position=1, leave=False) as pbar:
        best_clf.fit(X_train, y_train)
        pbar.update(1)

    # 验证集评估
    y_val_pred = best_clf.predict(X_val)
    y_val_prob = best_clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auroc = roc_auc_score(y_val, y_val_prob)

    # 测试集评估
    y_test_pred = best_clf.predict(X_test)
    y_test_prob = best_clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_prob)

    print(f"[Validation] ACC={val_acc:.4f}, F1={val_f1:.4f}, AUROC={val_auroc:.4f}")
    print(f"[Test]       ACC={test_acc:.4f}, F1={test_f1:.4f}, AUROC={test_auroc:.4f}")

    results.append({
        "dataset": base_dir,
        "best_params": str(best_params),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc
    })

# ===================== 保存结果 =====================
results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}_xgb_gpu_gridsearch.csv"
results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\nResults saved to: {output_file}")
