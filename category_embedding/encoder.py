from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Iterable, List, Optional, Sequence, Tuple, Union, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

class CategoryEmbedding(BaseEstimator, TransformerMixin):
    """Neural entity embedding encoder for categorical features.

    This transformer learns dense vector representations (embeddings) for
    categorical features using a small neural network and can optionally
    include numeric features as additional inputs. The learned embeddings
    are intended to be used as inputs to downstream models such as
    gradient-boosted trees (e.g. LightGBM, XGBoost).

    The model supports both regression and binary classification tasks.
    It exposes a `predict` method primarily for hyperparameter tuning
    (e.g. with Optuna), but is not meant to be the final predictor in
    a production pipeline.

    Parameters
    ----------
    task : str, default="regression"
        Task type. Either ``"regression"`` or ``"classification"``.
        Determines the loss and activation of the output head.
    log_target : bool, default=False
        Whether to apply a log transformation to the target variable for regression tasks.
    categorical_cols : Sequence[str], optional
        Names of categorical columns in the input data.
    numeric_cols : Sequence[str], optional
        Names of numeric columns in the input data. These are passed
        through unchanged in the output of `transform` but are
        included as inputs when training the embedding model.
    embedding_dims : Sequence[int], optional
        Optional list of integers specifying the embedding dimension
        for each categorical column, in the same order as
        ``categorical_cols``. If ``None``, a per-column default rule
        is used (see ``_default_embedding_dim``).
    hidden_units : int, default=64
        Width of each residual MLP block.
    n_blocks : int, default=2
        Number of residual MLP blocks applied after concatenating
        all embeddings and numeric features.
    dropout_rate : float, default=0.2
        Dropout rate used inside residual blocks and before the
        output layer.
    l2_emb : float, default=1e-6
        L2 regularization strength applied to embedding weights.
    l2_dense : float, default=1e-6
        L2 regularization strength applied to dense weights in the
        residual blocks and output head.
    batch_size : int, default=512
        Batch size used during training.
    epochs : int, default=30
        Maximum number of training epochs. Training may stop earlier
        due to early stopping.
    lr : float, default=2e-3
        Learning rate for the Adam optimizer.
    random_state : int, default=42
        Random seed used to seed TensorFlow.
    verbose : int, default=1
        Verbosity level passed to Keras ``Model.fit``.
    patience : int, default=4
        Early stopping patience in epochs. Monitors validation loss.
    reduce_lr_factor : float, default=0.5
        Factor by which the learning rate is reduced when validation
        loss plateaus.
    reduce_lr_patience : int, default=2
        Number of epochs with no improvement after which the learning
        rate is reduced.
    val_set : tuple, optional
        Optional external validation set as a tuple ``(X_val, y_val)``.
        If provided, it is used as validation data in ``fit``. Otherwise
        an internal validation split of 0.2 is used.
    num_imp_mode : {'mean', 'median'}, default='median'
        Strategy for imputing missing numeric values *internally* during
        model training. This does NOT affect the output of `transform`
        unless ``numeric_output='processed'``.
        
        Note: If your numeric columns have no missing values (e.g., you
        preprocessed them upstream), this parameter has no effect on the
        data but imputation will still be applied (as a no-op).
    numeric_output : {'raw', 'processed', None}, default='raw'
        Controls which numeric features appear in the output of `transform`.
        
        - 'raw' (default): Return the original numeric values exactly as provided 
          in the input to `transform()`, without imputation or scaling. Useful when 
          downstream models (e.g., gradient-boosted trees) handle raw numerics well.
        - 'processed': Return numeric features after imputation and scaling (the 
          same preprocessing used internally to train the embeddings). Useful for 
          linear models or when feature consistency is desired.
        - None: Do not include any numeric features in the output; return only 
          learned categorical embeddings.
        
        Note: Regardless of this setting, the internal embedding model is always 
        trained on imputed and scaled numeric features for training stability. 
        This parameter only affects the output of `transform()`, not the training 
        of the embeddings themselves.
    return_raw_categoricals : bool, default=False
        If ``True``, include the original categorical column values (unencoded) 
        in the output of `transform` alongside the learned embeddings. 
        
        This allows downstream GBM models to have both:
        - **Embeddings**: Learned similarity signals (useful for rare/unseen categories)
        - **Raw values**: Exact category matching (useful for frequent categories)
        
        Note: The raw categorical values are passed through unchanged—no encoding 
        or imputation is applied. It is the user's responsibility to configure 
        their GBM model appropriately (e.g., set ``categorical_feature`` in LightGBM) 
        or preprocess these columns upstream if needed.
    focal_gamma : float, optional, default=None
        Focusing parameter for Focal Loss, used only when
        ``task="classification"``. When set, Focal Loss replaces the
        standard binary cross-entropy loss during training.

        Focal Loss down-weights easy (majority class) examples and
        concentrates the gradient signal on hard, uncertain ones,
        producing better-calibrated embedding geometry for imbalanced
        datasets.

        - ``None`` (default): standard binary cross-entropy is used.
        - ``> 0``: activates Focal Loss. Typical range is ``[0.5, 3.0]``;
          start with ``1.0`` or ``2.0`` for moderate imbalance (e.g. 5:1–10:1).
          Higher values increase focus on hard examples.

        Raises ``ValueError`` if set alongside ``task="regression"``.

    Attributes
    ----------
    model_ : keras.Model
        Fitted Keras model instance after calling `fit`.
    cat_maps_ : dict[str, dict]
        Dictionary mapping each categorical column name to a dictionary
        of category -> integer index. Index ``n_categories`` is reserved
        for missing values ('_MISSING_'), index ``n_categories + 1`` is
        reserved for unseen values ('_UNKNOWN_').
    n_categories_ : dict[str, int]
        Dictionary mapping each categorical column name to its number
        of *known* categories seen during training (excluding special tokens).
    num_imputer_ : SimpleImputer
        Fitted numeric imputer.
    num_scaler_ : StandardScaler
        Fitted numeric scaler.
    _feature_names_out : list[str], optional
        List of feature names corresponding to columns produced by
        `transform`.
    """

    def __init__(
        self,
        task: str = "regression",
        log_target: bool = False,
        categorical_cols: Optional[Sequence[str]] = None,
        numeric_cols: Optional[Sequence[str]] = None,
        embedding_dims: Optional[Sequence[int]] = None,
        hidden_units: int = 64,
        n_blocks: int = 2,
        dropout_rate: float = 0.2,
        l2_emb: float = 1e-6,
        l2_dense: float = 1e-6,
        batch_size: int = 512,
        epochs: int = 30,
        lr: float = 2e-3,
        random_state: int = 42,
        verbose: int = 1,
        patience: int = 4,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 2,
        val_set: Optional[Tuple[ArrayLike, ArrayLike]] = None,
        num_imp_mode: Literal['mean', 'median'] = 'median',
        numeric_output: Literal[None, 'raw', 'processed'] = 'raw',
        return_raw_categoricals: bool = False,
        focal_gamma: Optional[float] = None,
    ) -> None:
        
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        if num_imp_mode not in ('mean', 'median'):
            raise ValueError("num_imp_mode must be 'mean' or 'median'")
        if numeric_output not in (None, 'raw', 'processed'):
            raise ValueError("numeric_output must be None, 'raw', or 'processed'")
        if not isinstance(return_raw_categoricals, bool):
            raise ValueError("return_raw_categoricals must be a boolean")
        if focal_gamma is not None and task == "regression":
            raise ValueError("focal_gamma is only supported for task='classification'. ")
        if focal_gamma is not None and focal_gamma <= 0:
            raise ValueError("focal_gamma must be a positive float")

        self.task = task
        self.log_target = log_target
        self.categorical_cols = list(categorical_cols or [])
        self.numeric_cols = list(numeric_cols or [])
        self.embedding_dims = list(embedding_dims) if embedding_dims is not None else None
        self.hidden_units = hidden_units
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate
        self.l2_emb = l2_emb
        self.l2_dense = l2_dense
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.patience = patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.val_set = val_set
        self.num_imp_mode = num_imp_mode
        self.numeric_output = numeric_output
        self.return_raw_categoricals = return_raw_categoricals
        self.focal_gamma = focal_gamma

        self.model_: Optional[keras.Model] = None
        self.cat_maps_: dict[str, dict] = {}
        self.n_categories_: dict[str, int] = {}
        self._feature_names_out: Optional[List[str]] = None
        self.num_imputer_: Optional[SimpleImputer] = None
        self.num_scaler_: Optional[StandardScaler] = None
        self._log_eps = 1e-6
        self._missing_token = '_MISSING_'
        self._unknown_token = '_UNKNOWN_'

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _default_embedding_dim(self, n_cat: int) -> int:
        """Compute default embedding dimension given cardinality."""
        if n_cat <= 10:
            dim = max(1, n_cat - 1)
        else:
            dim = max(10, n_cat // 2)
        return min(dim, 30)

    def _fit_category_maps(self, X: pd.DataFrame) -> None:
        """Build category -> index mappings for each categorical column.
        
        Missing values (NaN/None) are mapped to '_MISSING_' token.
        Unseen categories at transform time are mapped to '_UNKNOWN_' token.
        Both special tokens get their own trainable embeddings.
        """
        self.cat_maps_ = {}
        self.n_categories_ = {}

        for col in self.categorical_cols:
            # Get non-null unique categories
            non_null_vals = X[col].dropna().astype(str).unique()
            # Create mapping for known categories
            mapping = {str(cat): i for i, cat in enumerate(sorted(non_null_vals))}
            # Add special tokens - sequential indices
            mapping[self._missing_token] = len(mapping)      # e.g., index 3
            mapping[self._unknown_token] = len(mapping)      # e.g., index 4 (after _MISSING_ added)
            
            self.cat_maps_[col] = mapping
            # n_categories_ stores count of *known* categories only (excluding special tokens)
            self.n_categories_[col] = len(non_null_vals)

        if self.embedding_dims is not None and len(self.embedding_dims) != len(
            self.categorical_cols
        ):
            raise ValueError(
                "embedding_dims length must match number of categorical_cols "
                f"({len(self.categorical_cols)}), got {len(self.embedding_dims)}"
            )

    def _transform_categories_to_indices(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Convert categorical values to integer indices.
        
        - Missing values (NaN/None) → index for '_MISSING_' token
        - Unseen categories → index for '_UNKNOWN_' token
        - Known categories → their mapped index
        """
        out: dict[str, np.ndarray] = {}
        for col in self.categorical_cols:
            mapping = self.cat_maps_[col]
            missing_idx = mapping[self._missing_token]
            unknown_idx = mapping[self._unknown_token]
            
            def _map_value(val):
                # Handle missing values
                if pd.isna(val):
                    return missing_idx
                # Convert to string for consistent lookup
                val_str = str(val)
                # Known category
                if val_str in mapping and val_str not in (self._missing_token, self._unknown_token):
                    return mapping[val_str]
                # Unseen category
                return unknown_idx
            
            out[col] = np.array([_map_value(val) for val in X[col]], dtype="int32")
        return out

    def _focal_loss(self, gamma: float):
        """Return a Focal Loss function for binary classification.

        Focal Loss = -(1 - p_t)^gamma * log(p_t)

        Down-weights easy (majority) examples exponentially, concentrating
        gradient updates on hard, uncertain minority class examples.
        This produces better embedding geometry for imbalanced datasets
        compared to standard binary cross-entropy.

        Parameters
        ----------
        gamma : float
            Focusing parameter. Higher values increase the down-weighting
            of easy examples. gamma=0 recovers standard binary cross-entropy.
        """
        def focal_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

            # Standard BCE per sample
            bce = -(
                y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            )
            # p_t: probability of the true class
            p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

            # Focal factor: (1 - p_t)^gamma suppresses easy examples
            focal_factor = tf.pow(1.0 - p_t, gamma)

            return tf.reshape(tf.reduce_mean(focal_factor * bce), ())

        focal_loss_fn.__name__ = f"focal_loss_gamma_{gamma}"

        return focal_loss_fn

    def _residual_block(self, x: keras.Tensor, units: int, name_prefix: str) -> keras.Tensor:
        """Residual MLP block: LN -> Dense -> GELU -> Dropout -> Dense + skip."""
        input_dim = x.shape[-1]
        h = layers.LayerNormalization(name=f"{name_prefix}_ln")(x)
        h = layers.Dense(
            units,
            activation="gelu",
            kernel_regularizer=regularizers.l2(self.l2_dense),
            name=f"{name_prefix}_dense1",
        )(h)
        h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(h)
        h = layers.Dense(
            input_dim,
            activation=None,
            kernel_regularizer=regularizers.l2(self.l2_dense),
            name=f"{name_prefix}_dense2",
        )(h)
        return layers.Add(name=f"{name_prefix}_add")([x, h])

    def _build_model(self) -> None:
        """Build and compile the Keras model."""
        tf.random.set_seed(self.random_state)

        inputs: list[keras.Input] = []
        features: list[keras.Tensor] = []

        # Embedding inputs
        for i, col in enumerate(self.categorical_cols):
            n_known = self.n_categories_[col]
            # Embedding input_dim = known categories + 2 (missing + unknown tokens)
            input_dim = n_known + 2
            
            if self.embedding_dims is not None:
                emb_dim = self.embedding_dims[i]
            else:
                emb_dim = self._default_embedding_dim(n_known)

            inp = keras.Input(shape=(1,), name=f"{col}_input", dtype="int32")
            emb_layer = layers.Embedding(
                input_dim=input_dim,
                output_dim=emb_dim,
                name=f"{col}_embedding",
                embeddings_regularizer=regularizers.l2(self.l2_emb),
            )
            emb = emb_layer(inp)
            emb = layers.Flatten(name=f"{col}_flatten")(emb)

            inputs.append(inp)
            features.append(emb)

        # Numeric inputs
        if self.numeric_cols:
            num_inp = keras.Input(
                shape=(len(self.numeric_cols),),
                name="numeric_input",
                dtype="float32",
            )
            inputs.append(num_inp)
            features.append(num_inp)

        # Concatenate all feature streams
        if len(features) > 1:
            x = layers.Concatenate(name="concat")(features)
        else:
            x = features[0]

        x = layers.LayerNormalization(name="pre_mlp_ln")(x)

        # Residual blocks
        for i in range(self.n_blocks):
            x = self._residual_block(x, self.hidden_units, name_prefix=f"resblock_{i}")

        x = layers.LayerNormalization(name="final_ln")(x)
        x = layers.Dropout(self.dropout_rate, name="final_drop")(x)

        # Output head
        if self.task == "regression":
            output = layers.Dense(1, activation="linear", name="output")(x)
            loss = "mse"
        else:
            output = layers.Dense(1, activation="sigmoid", name="output")(x)
            # Use Focal Loss if focal_gamma is set, otherwise standard BCE
            loss = (
                self._focal_loss(self.focal_gamma)
                if self.focal_gamma is not None
                else "binary_crossentropy"
            )

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=loss,
            metrics=["binary_crossentropy" if self.task == "classification" else "mse"],
        )
        self.model_ = model

    # ---------------------------------------------------------
    # scikit-learn API
    # ---------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CategoryEmbedding":
        """Fit the embedding encoder on the provided data."""
        X_df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y).astype("float32")

        if len(y_arr.shape) == 1:
            y_arr = y_arr.reshape(-1, 1)

        missing_cat = set(self.categorical_cols) - set(X_df.columns)
        missing_num = set(self.numeric_cols) - set(X_df.columns)
        if missing_cat:
            raise ValueError(f"Missing categorical columns in X: {missing_cat}")
        if missing_num:
            raise ValueError(f"Missing numeric columns in X: {missing_num}")

        # Fit category maps (handles missing categoricals via _MISSING_ token)
        self._fit_category_maps(X_df) 
        
        # Fit numeric imputer - always applied internally for NN stability
        if self.numeric_cols:
            self.num_imputer_ = SimpleImputer(strategy=self.num_imp_mode)
            num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32")
            num_arr_imputed = self.num_imputer_.fit_transform(num_arr)
            
            # Fit numeric scaler
            self.num_scaler_ = StandardScaler()
            num_arr_scaled = self.num_scaler_.fit_transform(num_arr_imputed)
        else:
            num_arr_scaled = None
        
        # Log-scale target for regression 
        if self.task == "regression" and self.log_target:
            y_arr = np.log(y_arr + self._log_eps)
        
        # Build model
        self._build_model()
        assert self.model_ is not None, "Model was not built."

        # Prepare categorical inputs
        cat_idx = self._transform_categories_to_indices(X_df)
        model_inputs: list[np.ndarray] = [cat_idx[col] for col in self.categorical_cols]

        # Add numeric inputs (always scaled+imputed for training)
        if self.numeric_cols:
            model_inputs.append(num_arr_scaled.astype("float32"))

        # Callbacks
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.reduce_lr_factor,
                patience=self.reduce_lr_patience,
                min_lr=1e-6,
            ),
        ]

        # External validation set if provided
        if self.val_set is not None:
            X_val, y_val = self.val_set
            X_val_df = pd.DataFrame(X_val).copy()
            y_val_arr = np.asarray(y_val).astype("float32")

            if len(y_val_arr.shape) == 1:
                y_val_arr = y_arr.reshape(-1, 1)
                
            if self.task == "regression" and self.log_target:
                y_val_arr = np.log(y_val_arr + self._log_eps)

            cat_idx_val = self._transform_categories_to_indices(X_val_df)
            val_inputs = [cat_idx_val[col] for col in self.categorical_cols]
            
            if self.numeric_cols:
                num_val = X_val_df[self.numeric_cols].to_numpy(dtype="float32")
                num_val = self.num_imputer_.transform(num_val)
                num_val_scaled = self.num_scaler_.transform(num_val)
                val_inputs.append(num_val_scaled)

            self.model_.fit(
                model_inputs,
                y_arr,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_data=(val_inputs, y_val_arr),
                callbacks=callbacks,
            )
        else:
            self.model_.fit(
                model_inputs,
                y_arr,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_split=0.2,
                callbacks=callbacks,
            )

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict using the internal neural head (for tuning/evaluation)."""
        if self.model_ is None:
            raise RuntimeError("The encoder must be fitted before calling predict().")

        X_df = pd.DataFrame(X).copy()

        cat_idx = self._transform_categories_to_indices(X_df) 
        model_inputs = [cat_idx[col] for col in self.categorical_cols]

        if self.numeric_cols:
            num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32")
            num_arr = self.num_imputer_.transform(num_arr)
            num_arr_scaled = self.num_scaler_.transform(num_arr)
            model_inputs.append(num_arr_scaled)

        preds = self.model_.predict(model_inputs, verbose=0).ravel()

        if self.task == "regression" and self.log_target:
            preds = np.exp(preds) - self._log_eps

        return preds

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Transform input data into learned embedding space."""
        if self.model_ is None:
            raise RuntimeError("The encoder must be fitted before calling transform().")

        X_df = pd.DataFrame(X).copy()
        cat_idx = self._transform_categories_to_indices(X_df)

        emb_blocks = [] 
        colnames = []

        # Embeddings
        for col in self.categorical_cols:
            idx = cat_idx[col]
            emb_layer = self.model_.get_layer(f"{col}_embedding")
            emb_matrix = emb_layer.get_weights()[0]

            emb_blocks.append(emb_matrix[idx])
            dim = emb_matrix.shape[1]
            colnames.extend([f"{col}_emb_{i}" for i in range(dim)])

        cat_emb = np.concatenate(emb_blocks, axis=1)

        # Handle numeric output based on parameter
        if self.numeric_cols and self.numeric_output is not None:
            if self.numeric_output == 'raw':
                num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32")
            else:  # 'processed'
                num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32")
                num_arr = self.num_imputer_.transform(num_arr)
                num_arr = self.num_scaler_.transform(num_arr)
            
            full = np.concatenate([cat_emb, num_arr], axis=1) 
            colnames.extend(self.numeric_cols)
        else:
            full = cat_emb

        # Add raw categorical columns if requested
        if self.return_raw_categoricals:
            raw_cats = []
            for col in self.categorical_cols:
                raw_cats.append(X_df[[col]].to_numpy())
                colnames.append(col)
            
            raw_cats_arr = np.concatenate(raw_cats, axis=1)
            full = np.concatenate([full, raw_cats_arr], axis=1)

        self._feature_names_out = colnames
        
        return pd.DataFrame(full, columns=colnames)

    def get_feature_names_out(
        self, input_features: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        """Get output feature names for ColumnTransformer compatibility."""
        if self._feature_names_out is None:
            raise RuntimeError(
                "Feature names are not available. Call transform() at least once "
                "before get_feature_names_out()."
            )
        return np.array(self._feature_names_out)