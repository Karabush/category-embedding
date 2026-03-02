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
  - Always capped at **30**.

- **Per-column embedding dimension overrides**  
  Pass a list of integers to control embedding size manually.

- **Robust categorical handling**  
  - Missing values (`NaN`/`None`) → mapped to a dedicated `_MISSING_` token with its own trainable embedding.
  - Unseen categories at inference → mapped to a separate `_UNKNOWN_` token with its own trainable embedding.

- **Residual MLP architecture**  
  LayerNorm + GELU + Dropout + skip connections for stable training.

- **Flexible numeric feature output**  
  Numeric columns are always imputed and scaled internally for stable neural training.
  Use `numeric_output` to control what appears in `transform()` output:
  - `'raw'` (default): Return original numeric values unchanged—ideal for GBMs.
  - `'processed'`: Return imputed + scaled values—ideal for linear models.
  - `None`: Return only categorical embeddings.

- **Configurable numeric imputation**  
  `num_imp_mode` chooses between `'mean'` or `'median'` for internal imputation.
  If your data has no missing numeric values, this parameter has no effect.

- **Optional log-scaling of regression targets**  
  For regression tasks, the target can be optionally transformed using `log(y + 1e-6)` during training 
  (and inverse-transformed at prediction time) by setting `log_target=True`.

- **Supports regression and binary classification**  
  The neural head is used only for training/tuning; GBMs remain the final predictor.

- **Optional external validation set**  
  Enables clean early stopping and stable embedding learning.

- **Sklearn-compatible API**  
  Implements `fit`, `transform`, `predict`, and `get_feature_names_out`.

- **Outputs a pandas DataFrame**  
  Perfect for LightGBM, XGBoost, CatBoost, or any sklearn model.

- **Optional raw categorical passthrough**  
  Use `return_raw_categoricals=True` to include original categorical values 
  alongside embeddings. This gives GBMs both exact matching (for frequent 
  categories) and similarity signals (for rare/unseen categories).
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from category_embedding import CategoryEmbedding

df = pd.DataFrame({
    "country": ["DE", "FR", "DE", "US", "US"],
    "device": ["mobile", "desktop", "tablet", "mobile", "desktop"],
    "age": [25, 40, 31, 22, 35],
})
y = [10.5, 20.1, 15.3, 8.7, 18.0]

categorical = ["country", "device"]
numeric = ["age"]

# CategoryEmbedding acts as a transformer inside a sklearn pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("emb", CategoryEmbedding(
            task="regression",
            categorical_cols=categorical,
            numeric_cols=numeric,
            epochs=20,
            batch_size=32,
            numeric_output='raw',      # Return original numeric values for LightGBM
            num_imp_mode='median',     # Internal imputation strategy (always applied)
        ), categorical + numeric)
    ],
    remainder="drop",
)

model = Pipeline([
    ("prep", preprocess),
    ("lgbm", LGBMRegressor(n_estimators=300, learning_rate=0.05)),
])

model.fit(df, y)
preds = model.predict(df)
```

---

## 🧠 Why Use Category Embedding?

Traditional encoders struggle with:

* High-cardinality categorical features
* Sparse interactions
* Noisy or rare categories
* Missing or unseen values at inference

Neural embeddings solve this by learning dense, continuous representations that capture similarity structure between categories.

This library gives you:

* The power of deep learning
* The simplicity and performance of GBMs as final predictors
* A clean sklearn interface
* Deterministic, production-ready behavior
* Automatic handling of missing/unseen categoricals via dedicated trainable tokens
* Flexible numeric output: raw values for trees, scaled for linear models, or embeddings-only

---

## ⚙️ API Overview

### `CategoryEmbedding(...)`

### Parameters

| Parameter            | Type                               | Default      | Description                                                                 |
|----------------------|------------------------------------|--------------|-----------------------------------------------------------------------------|
| `task`                 | `"regression"` or `"classification"`  | `"regression"` | Determines loss function. |
| `log_target`           | `bool`   | `False` | Whether to apply `log(y + 1e-6)` transformation to regression targets during training (inverse applied at prediction). |
| `categorical_cols`     | `list[str]` or `None`                  | `None`         | Names of categorical columns to embed.                                      |
| `numeric_cols`         | `list[str]` or `None`                  | `None`         | Names of numeric columns to include as inputs.                              |
| `embedding_dims`       | `list[int]` or `None`                  | `None`         | Optional list of embedding dimensions for each categorical column (in order). If `None`, uses default rule: `n_cat≤10 → n_cat-1`, else `max(10, n_cat//2)`, capped at 30. |
| `hidden_units`         | `int`                                 | 64           | Width of each residual MLP block.                                           |
| `n_blocks`             | `int`                                 | 2            | Number of residual blocks.                                                  |
| `dropout_rate`         | `float`                               | 0.2          | Dropout rate inside residual blocks and before output head.                 |
| `l2_emb`               | `float`                               | 1e-6         | L2 regularization for embedding weights.                                    |
| `l2_dense`             | `float`                               | 1e-6         | L2 regularization for dense layers.                                         |
| `batch_size`           | `int`                                 | 512          | Training batch size.                                                        |
| `epochs`               | `int`                                 | 30           | Maximum number of training epochs.                                          |
| `lr`                   | `float`                               | 2e-3         | Learning rate for Adam optimizer.                                           |
| `random_state`         | `int`                                 | 42           | TensorFlow random seed.                                                     |
| `verbose`              | `int`                                 | 1            | Verbosity passed to Keras `.fit()`.                                         |
| `patience`             | `int`                                 | 4            | Early stopping patience.                                                    |
| `reduce_lr_factor`     | `float`                               | 0.5          | LR reduction factor when validation loss plateaus.                          |
| `reduce_lr_patience`   | `int`                                 | 2            | Patience before reducing LR.                                                |
| `val_set`              | `tuple(X_val, y_val)` or `None`        | `None`         | Optional external validation set.                                           |
| `num_imp_mode`              | `"mean"` or `"median"`       | `"median"`        | Strategy for imputing missing numeric values internally during model training. Always applied for NN stability. If your data has no missing values, this is a no-op. Does not affect `transform()` output unless `numeric_output='processed'`.                                           |
| `numeric_output`       | `"raw"`, `"processed"`, or `None`                                | `"raw"`         | Controls numeric features in transform() output: `"raw"` (default) - return original numeric values unchanged `"processed"` - return imputed + scaled values (same as used for training). `None` - exclude numeric features,return only categorical embeddings. Regardless of this setting, the internal model is always trained on imputed + scaled numerics for stability.       |
| `return_raw_categoricals` | `bool` | `False` | If `True`, include original categorical column values (unencoded) in `transform()` output with original column names. Allows GBMs to use both embeddings and raw categories. User must configure GBM appropriately. |

### Methods

#### `.fit(X, y)`
Trains the embedding model:
- learns embeddings  
- fits numeric scaler  
- applies log-scaling to regression targets  
- trains the neural model  

#### `.transform(X)`
Returns a DataFrame containing:
- learned embeddings  
- numeric features (scaled or raw depending on `scaled_num_out`)  

#### `.predict(X)`
Uses the neural head for tuning/evaluation.  
For regression: automatically applies inverse log-transform.

#### `.get_feature_names_out()`
Returns the names of all output columns.

---

## 🔑 Keywords
machine learning, deep learning, tabular data, categorical encoding, entity embeddings, category embeddings, neural encoder, keras, tensorflow, scikit-learn, sklearn transformer, feature engineering, gradient boosting, lightgbm, xgboost, embeddings for gbm, high-cardinality features, optuna tuning, ml pipelines, missing value handling, unseen category handling

---

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.