from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Iterable, List, Optional, Sequence, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
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
    task:
        Task type. Either ``"regression"`` or ``"classification"``.
        Determines the loss and activation of the output head.
    log_target : bool, default=False
        Whether to apply a log transformation to the target variable for regression tasks.
    categorical_cols:
        Names of categorical columns in the input data.
    numeric_cols:
        Names of numeric columns in the input data. These are passed
        through unchanged in the output of :meth:`transform` but are
        included as inputs when training the embedding model.
    embedding_dims:
        Optional list of integers specifying the embedding dimension
        for each categorical column, in the same order as
        ``categorical_cols``. If ``None``, a per-column default rule
        is used:

        - if ``n_cat <= 10``: ``dim = n_cat - 1`` (minimum 1)
        - else: ``dim = max(10, n_cat // 2)``
        - in all cases: ``dim <= 30``.

    hidden_units:
        Width of each residual MLP block. A single integer is used
        for all blocks.
    n_blocks:
        Number of residual MLP blocks applied after concatenating
        all embeddings and numeric features.
    dropout_rate:
        Dropout rate used inside residual blocks and before the
        output layer.
    l2_emb:
        L2 regularization strength applied to embedding weights.
    l2_dense:
        L2 regularization strength applied to dense weights in the
        residual blocks and output head.
    batch_size:
        Batch size used during training.
    epochs:
        Maximum number of training epochs. Training may stop earlier
        due to early stopping.
    lr:
        Learning rate for the Adam optimizer.
    random_state:
        Random seed used to seed TensorFlow. For full determinism,
        users should also control NumPy and Python random seeds
        externally.
    verbose:
        Verbosity level passed to Keras ``Model.fit``.
    patience:
        Early stopping patience in epochs. Monitors validation loss.
    reduce_lr_factor:
        Factor by which the learning rate is reduced when validation
        loss plateaus.
    reduce_lr_patience:
        Number of epochs with no improvement after which the learning
        rate is reduced.
    val_set:
        Optional external validation set as a tuple ``(X_val, y_val)``.
        If provided, it is used as validation data in ``fit``. Otherwise
        an internal validation split of 0.2 is used.

    Attributes
    ----------
    model_:
        Fitted Keras model instance after calling :meth:`fit`.
    cat_maps_:
        Dictionary mapping each categorical column name to a dictionary
        of category -> integer index.
    n_categories_:
        Dictionary mapping each categorical column name to its number
        of categories seen during training.
    _feature_names_out:
        List of feature names corresponding to columns produced by
        :meth:`transform`.
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
        scaled_num_out: bool = True,
        ) -> None:
        
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")

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
        self.scaled_num_out = scaled_num_out

        self.model_: Optional[keras.Model] = None
        self.cat_maps_: dict[str, dict] = {}
        self.n_categories_: dict[str, int] = {}
        self._feature_names_out: Optional[List[str]] = None
        self.num_scaler_: Optional[StandardScaler] = None
        self._log_eps = 1e-6

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _default_embedding_dim(self, n_cat: int) -> int:
        """Compute default embedding dimension given cardinality.

        For each categorical feature with ``n_cat`` distinct levels:

        - if ``n_cat <= 10``: ``dim = max(1, n_cat - 1)``
        - else: ``dim = max(10, n_cat // 2)``
        - finally ``dim = min(dim, 30)``.

        This heuristic keeps small-cardinality features compact while
        giving larger ones more capacity without exceeding 30.
        """
        if n_cat <= 10:
            dim = max(1, n_cat - 1)
        else:
            dim = max(10, n_cat // 2)
        return min(dim, 30)

    def _fit_category_maps(self, X: pd.DataFrame) -> None:
        """Build category -> index mappings for each categorical column."""
        self.cat_maps_ = {}
        self.n_categories_ = {}

        for col in self.categorical_cols:
            cats = pd.Series(X[col].astype("category")).cat.categories
            mapping = {cat: i for i, cat in enumerate(cats)}
            self.cat_maps_[col] = mapping
            self.n_categories_[col] = len(mapping)

        if self.embedding_dims is not None and len(self.embedding_dims) != len(
            self.categorical_cols
        ):
            raise ValueError(
                "embedding_dims length must match number of categorical_cols "
                f"({len(self.categorical_cols)}), got {len(self.embedding_dims)}"
            )

    def _hash_unseen(self, value: object, n_cat: int) -> int:
        """Hash unseen categories into a valid index [0, n_cat-1]."""
        return hash(value) % n_cat

    def _transform_categories_to_indices(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Convert categorical values to integer indices, hashing unseen values."""
        out: dict[str, np.ndarray] = {}
        for col in self.categorical_cols:
            mapping = self.cat_maps_[col]
            n_cat = self.n_categories_[col]

            out[col] = np.array(
                [
                    mapping[val] if val in mapping else self._hash_unseen(val, n_cat)
                    for val in X[col]
                ],
                dtype="int32",
            )
        return out

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
            n_cat = self.n_categories_[col]
            if self.embedding_dims is not None:
                emb_dim = self.embedding_dims[i]
            else:
                emb_dim = self._default_embedding_dim(n_cat)

            inp = keras.Input(shape=(1,), name=f"{col}_input", dtype="int32")
            emb_layer = layers.Embedding(
                input_dim=n_cat,
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

        # Output head for training (not used in transform)
        if self.task == "regression":
            output = layers.Dense(1, activation="linear", name="output")(x)
            loss = "mse"
        else:
            output = layers.Dense(1, activation="sigmoid", name="output")(x)
            loss = "binary_crossentropy"

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=loss,
            metrics=[loss],
        )
        self.model_ = model

    # ---------------------------------------------------------
    # scikit-learn API
    # ---------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CategoryEmbedding":
        """Fit the embedding encoder on the provided data.

        Parameters
        ----------
        X:
            Training features as a pandas DataFrame, NumPy array, or
            compatible array-like. Must contain all columns listed in
            ``categorical_cols`` and ``numeric_cols``.
        y:
            Target values for regression or binary classification.
            Will be converted to a NumPy array of dtype float32.

        Returns
        -------
        self:
            Fitted encoder instance.
        """
        X_df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y).astype("float32")

        # Validate presence of required columns
        missing_cat = set(self.categorical_cols) - set(X_df.columns)
        missing_num = set(self.numeric_cols) - set(X_df.columns)
        if missing_cat:
            raise ValueError(f"Missing categorical columns in X: {missing_cat}")
        if missing_num:
            raise ValueError(f"Missing numeric columns in X: {missing_num}")

        # Fit category maps 
        self._fit_category_maps(X_df) 
        
        # Fit numeric scaler 
        if self.numeric_cols: 
            self.num_scaler_ = StandardScaler() 
            num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32") 
            self.num_scaler_.fit(num_arr) 
            num_arr_scaled = self.num_scaler_.transform(num_arr) 
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

        # Add numeric inputs
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

            if self.task == "regression" and self.log_target:
                y_val_arr = np.log(y_val_arr + self._log_eps)

            cat_idx_val = self._transform_categories_to_indices(X_val_df)
            val_inputs= [cat_idx_val[col] for col in self.categorical_cols]
            
            if self.numeric_cols:
                num_val = X_val_df[self.numeric_cols].to_numpy(dtype="float32") 
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
            # Internal validation split
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
        """Predict using the internal neural head (for tuning/evaluation).

        This method is provided to support use cases such as
        hyperparameter tuning of the encoder itself. In a typical
        production setup, the downstream GBM model will be the
        final predictor.

        Parameters
        ----------
        X:
            Input features as a pandas DataFrame, NumPy array, or
            compatible array-like.

        Returns
        -------
        preds:
            Model predictions as a 1D NumPy array.
        """
        if self.model_ is None:
            raise RuntimeError("The encoder must be fitted before calling predict().")

        X_df = pd.DataFrame(X).copy()

        # Categorical 
        cat_idx = self._transform_categories_to_indices(X_df) 
        model_inputs = [cat_idx[col] for col in self.categorical_cols]

        # Numeric (scaled)
        if self.numeric_cols:
            num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32") 
            num_arr_scaled = self.num_scaler_.transform(num_arr) 
            model_inputs.append(num_arr_scaled)

        preds = self.model_.predict(model_inputs, verbose=0).ravel()

        # Inverse log-transform for regression 
        if self.task == "regression" and self.log_target:
            preds = np.exp(preds) - self._log_eps

        return preds

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Transform input data into learned embedding space.

        The output contains, for each categorical column, its learned
        embedding coordinates, plus all numeric features passed
        through unchanged.

        Parameters
        ----------
        X:
            Input features as a pandas DataFrame, NumPy array, or
            compatible array-like.

        Returns
        -------
        embeddings:
            A pandas DataFrame of shape (n_samples, n_features_out)
            containing all concatenated embeddings and numeric
            features, with informative column names.
        """
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

        # Numeric output (scaled or raw)
        if self.numeric_cols:
            if self.scaled_num_out:
                num_arr = self.num_scaler_.transform(
                    X_df[self.numeric_cols].to_numpy(dtype="float32")
                    )
            else:
                num_arr = X_df[self.numeric_cols].to_numpy(dtype="float32")
            
            full = np.concatenate([cat_emb, num_arr], axis=1) 
            colnames.extend(self.numeric_cols)
        else:
            full = cat_emb

        self._feature_names_out = colnames
        
        return pd.DataFrame(full, columns=colnames)

    def get_feature_names_out(
        self, input_features: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        """Get output feature names for ColumnTransformer compatibility.

        Returns
        -------
        names:
            A NumPy array of output feature names corresponding to the
            columns produced by :meth:`transform`.
        """
        if self._feature_names_out is None:
            raise RuntimeError(
                "Feature names are not available. Call transform() at least once "
                "before get_feature_names_out()."
            )
        return np.array(self._feature_names_out)
