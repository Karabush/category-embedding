# Category Embedding

A Keras-based neural encoder for categorical variables in tabular machine learning.  
It learns dense vector representations (embeddings) for categorical features and outputs a clean numeric DataFrame that integrates seamlessly with gradient-boosted tree models.

This library is designed for ML engineers who want the benefits of deep-learning-based embeddings **without replacing their GBM models**.  
It is sklearn-compatible, deterministic, and production-ready.

---

## 🚀 Features

- **Multi-column categorical embeddings**  
  Each categorical feature receives its own learned embedding matrix.

- **Smart default embedding dimensions**  
  Uses a simple, interpretable rule:  
  - If `n_cat ≤ 10`: `dim = n_cat - 1`  
  - Else: `dim = max(10, n_cat // 2)`  
  - Always capped at **50**.

- **Per-column embedding dimension overrides**  
  Pass a list of integers to control embedding size manually.

- **Hashing for unseen categories**  
  Unseen values at inference time are deterministically mapped into valid embedding indices.

- **Residual MLP architecture**  
  LayerNorm + GELU + Dropout + skip connections for stable training.

- **Supports regression and binary classification**  
  The neural head is used only for training/tuning; GBMs remain the final predictor.

- **Optional external validation set**  
  Enables clean early stopping and stable embedding learning.

- **Sklearn-compatible API**  
  Implements `fit`, `transform`, `predict`, and `get_feature_names_out`.

- **Outputs a pandas DataFrame**  
  Perfect for LightGBM, XGBoost, CatBoost, or any sklearn model.

---

## 📦 Installation

```bash
pip install category-embedding
```

## Requires:

* Python ≥ 3.9
* TensorFlow ≥ 2.12
* scikit-learn ≥ 1.2
* pandas ≥ 1.5

---

## 🔧 Quick Start

```python
import pandas as pd
import lightgbm as lgb
from category_embedding import CategoryEmbedding

df = pd.DataFrame({
    "country": ["DE", "FR", "DE", "US", "US"],
    "device": ["mobile", "desktop", "tablet", "mobile", "desktop"],
    "age": [25, 40, 31, 22, 35],
})
y = [10.5, 20.1, 15.3, 8.7, 18.0]

enc = CategoryEmbedding(
    task="regression",
    categorical_cols=["country", "device"],
    numeric_cols=["age"],
    epochs=20,
    batch_size=32,
)

enc.fit(df, y)
X_emb = enc.transform(df)

train_ds = lgb.Dataset(X_emb, label=y)
params = {"objective": "regression", "metric": "rmse"}

model = lgb.train(params, train_ds, num_boost_round=200)
```

---

## 🧠 Why Use Category Embedding?

Traditional encoders struggle with:

* high-cardinality categorical features
* sparse interactions
* noisy or rare categories

Neural embeddings solve this by learning dense, continuous representations that capture similarity structure between categories.

This library gives you:

* the power of deep learning
* the simplicity and performance of GBMs
* a clean sklearn interface
* deterministic, production-ready behavior

---

## ⚙️ API Overview

CategoryEmbedding(...)

Key parameters:

* task: "regression" or "classification"
* categorical_cols: list of categorical column names
* numeric_cols: list of numeric column names
* embedding_dims: optional list of per-column embedding sizes
* hidden_units: width of each residual block
* n_blocks: number of residual blocks
* dropout_rate: dropout inside residual blocks
* lr: learning rate
* batch_size, epochs
* val_set: optional (X_val, y_val) tuple for early stopping

.fit(X, y)

Trains the embedding model.

.transform(X)

Returns a DataFrame containing all learned embeddings and numeric features.

.predict(X)

Uses the neural head for tuning/evaluation.

.get_feature_names_out()

Returns the names of the output columns.

---

## 📊 Example: Using Embeddings with XGBoost

```python
import xgboost as xgb

X_train_emb = enc.transform(X_train)
X_test_emb = enc.transform(X_test)

dtrain = xgb.DMatrix(X_train_emb, label=y_train)
dtest = xgb.DMatrix(X_test_emb, label=y_test)

params = {"objective": "reg:squarederror"}
model = xgb.train(params, dtrain, num_boost_round=300)
```

---

## 🔑 Keywords
machine learning, deep learning, tabular data, categorical encoding, entity embeddings, category embeddings, neural encoder, keras, tensorflow, scikit-learn, sklearn transformer, feature engineering, gradient boosting, lightgbm, xgboost, embeddings for gbm, high-cardinality features, optuna tuning, ml pipelines

---

## 📄 License

This project is licensed under the MIT License.
See the LICENSE file for details.