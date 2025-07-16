import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from datetime import datetime

# 固定随机种子
SEED = 33
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 数据集配置
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

# 定义 MLP 模型
def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

results = []

for base_dir, suffix in tqdm(dataset_configs, desc="Datasets"):
    print("\n" + "=" * 60)
    print(f"Training MLP on dataset: {base_dir}")

    X_train, y_train = load_split_csv(base_dir, "train", suffix)
    X_val, y_val = load_split_csv(base_dir, "val", suffix)
    X_test, y_test = load_split_csv(base_dir, "test", suffix)

    model = build_mlp(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        verbose=2
    )

    y_val_prob = model.predict(X_val, batch_size=256).ravel()
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auroc = roc_auc_score(y_val, y_val_prob)

    # 测试集评估
    y_test_prob = model.predict(X_test, batch_size=256).ravel()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_prob)

    print(f"[Validation] ACC={val_acc:.4f}, F1={val_f1:.4f}, AUROC={val_auroc:.4f}")
    print(f"[Test]       ACC={test_acc:.4f}, F1={test_f1:.4f}, AUROC={test_auroc:.4f}")

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join("checkpoints", f"mlp_{base_dir}_{timestamp}.keras")
    model.save(model_file)
    print(f"Saved model: {model_file}")

    results.append({
        "dataset": base_dir,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "model_file": model_file
    })


results_df = pd.DataFrame(results)
summary_file = f"mlp_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
print(f"Summary saved to: {summary_file}")
