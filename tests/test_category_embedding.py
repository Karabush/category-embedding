import numpy as np
import pandas as pd
import pytest

from category_embedding import CategoryEmbedding

# =========================================================
# Fixtures & Helpers
# =========================================================

@pytest.fixture
def sample_data():
    """Create small sample dataset for quick tests."""
    df_train = pd.DataFrame({
        "cat": ["a", "b", "c", "a", "b"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = np.array([0.0, 1.0, 0.5, 0.8, 0.3])
    return df_train, y


@pytest.fixture
def multi_cat_data():
    """Dataset with multiple categorical columns."""
    df = pd.DataFrame({
        "cat1": ["a", "b", "c"],
        "cat2": ["x", "y", "z"],
        "num": [1.0, 2.0, 3.0]
    })
    y = np.array([0.0, 1.0, 0.5])
    return df, y


@pytest.fixture
def classification_data():
    """Imbalanced binary classification dataset."""
    df = pd.DataFrame({
        "cat": ["a", "b", "c", "a", "b", "c", "a", "b"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    # ~25% positive class — moderate imbalance
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1])
    return df, y


# =========================================================
# Categorical Handling Tests
# =========================================================

def test_unseen_category_handling(sample_data):
    """Unseen categories should map to _UNKNOWN_ token without crashing."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "d"], "num": [4.0, 5.0]})
    X_emb = enc.transform(df_test)

    assert len(X_emb) == 2
    assert X_emb.shape[1] >= 1
    assert any("emb" in col for col in X_emb.columns)


def test_missing_categorical_handling(sample_data):
    """Missing categorical values (NaN/None) should map to _MISSING_ token."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", None], "num": [4.0, 5.0]})
    X_emb = enc.transform(df_test)

    assert len(X_emb) == 2


def test_multiple_categorical_columns(multi_cat_data):
    """Should handle multiple categorical columns correctly."""
    df_train, y = multi_cat_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat1", "cat2"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({
        "cat1": ["a", "d"],  # "d" is unseen
        "cat2": ["x", None],  # missing value
        "num": [4.0, 5.0]
    })
    X_emb = enc.transform(df_test)

    assert len(X_emb) == 2
    assert any("cat1_emb" in col for col in X_emb.columns)
    assert any("cat2_emb" in col for col in X_emb.columns)


# =========================================================
# Numeric Output Mode Tests
# =========================================================

def test_numeric_output_raw(sample_data):
    """numeric_output='raw' should return original numeric values."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        numeric_output='raw',
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [10.0, 20.0]})
    X_emb = enc.transform(df_test)

    assert "num" in X_emb.columns
    np.testing.assert_array_almost_equal(X_emb["num"].values, [10.0, 20.0])


def test_numeric_output_processed(sample_data):
    """numeric_output='processed' should return imputed+scaled numeric values."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        numeric_output='processed',
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [10.0, 20.0]})
    X_emb = enc.transform(df_test)

    assert "num" in X_emb.columns
    assert not np.allclose(X_emb["num"].values, [10.0, 20.0])


def test_numeric_output_none(sample_data):
    """numeric_output=None should return only categorical embeddings."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        numeric_output=None,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [10.0, 20.0]})
    X_emb = enc.transform(df_test)

    assert "num" not in X_emb.columns
    assert all("emb" in col for col in X_emb.columns)


# =========================================================
# Numeric Imputation Tests
# =========================================================

def test_numeric_imputation_internal():
    """Missing numeric values should be imputed internally for model training."""
    df_train = pd.DataFrame({
        "cat": ["a", "b", "c", "a", "b"],
        "num": [1.0, np.nan, 3.0, 4.0, 5.0]
    })
    y = np.array([0.0, 1.0, 0.5, 0.8, 0.3])

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        num_imp_mode='median',
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [np.nan, 20.0]})
    X_emb = enc.transform(df_test)

    assert len(X_emb) == 2


# =========================================================
# Raw Categorical Passthrough Tests
# =========================================================

def test_return_raw_categoricals(sample_data):
    """return_raw_categoricals=True should include raw categorical columns."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        return_raw_categoricals=True,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [10.0, 20.0]})
    X_emb = enc.transform(df_test)

    # Raw categorical column present with original name
    assert "cat" in X_emb.columns
    np.testing.assert_array_equal(X_emb["cat"].values, ["a", "b"])
    # Embeddings also present
    assert any("emb" in col for col in X_emb.columns)
    # Verify categorical_cols attribute
    assert enc.categorical_cols == ["cat"]


def test_return_raw_categoricals_false(sample_data):
    """return_raw_categoricals=False (default) should not include raw categoricals."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        return_raw_categoricals=False,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [10.0, 20.0]})
    X_emb = enc.transform(df_test)

    # Raw categorical should NOT be in output (only embeddings)
    assert "cat" not in X_emb.columns
    assert any("cat_emb" in col for col in X_emb.columns)


# =========================================================
# Task Type Tests
# =========================================================

def test_regression_task(sample_data):
    """Regression task should output continuous predictions."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    preds = enc.predict(df_test)

    assert len(preds) == 2
    assert preds.dtype == np.float32


def test_classification_task(sample_data):
    """Classification task should output probabilities (0-1 range)."""
    df_train, _ = sample_data
    y = np.array([0, 1, 0, 1, 0])

    enc = CategoryEmbedding(
        task="classification",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    preds = enc.predict(df_test)

    assert len(preds) == 2
    assert np.all((preds >= 0) & (preds <= 1))


def test_predict_with_log_target(sample_data):
    """Predict should inverse log-transform when log_target=True."""
    df_train, y = sample_data
    y_positive = np.abs(y) + 1.0

    enc = CategoryEmbedding(
        task="regression",
        log_target=True,
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y_positive)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    preds = enc.predict(df_test)

    assert np.all(preds > 0)
    assert len(preds) == 2


# =========================================================
# Focal Loss Tests
# =========================================================

def test_focal_loss_classification_runs(classification_data):
    """focal_gamma should train without errors and produce valid probabilities."""
    df_train, y = classification_data

    enc = CategoryEmbedding(
        task="classification",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=4,
        verbose=0,
        focal_gamma=2.0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    preds = enc.predict(df_test)

    assert len(preds) == 2
    assert np.all((preds >= 0) & (preds <= 1))


def test_focal_loss_transform_shape(classification_data):
    """focal_gamma should not affect the shape or structure of transform() output."""
    df_train, y = classification_data

    enc_bce = CategoryEmbedding(
        task="classification",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=4,
        verbose=0,
        focal_gamma=None,
    )
    enc_focal = CategoryEmbedding(
        task="classification",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=4,
        verbose=0,
        focal_gamma=2.0,
    )

    enc_bce.fit(df_train, y)
    enc_focal.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    out_bce = enc_bce.transform(df_test)
    out_focal = enc_focal.transform(df_test)

    assert out_bce.shape == out_focal.shape
    assert list(out_bce.columns) == list(out_focal.columns)


def test_focal_loss_gamma_range(classification_data):
    """focal_gamma should work across the typical tuning range [0.5, 3.0]."""
    df_train, y = classification_data
    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})

    for gamma in [0.5, 1.0, 2.0, 3.0]:
        enc = CategoryEmbedding(
            task="classification",
            categorical_cols=["cat"],
            numeric_cols=["num"],
            epochs=2,
            batch_size=4,
            verbose=0,
            focal_gamma=gamma,
        )
        enc.fit(df_train, y)
        preds = enc.predict(df_test)
        assert np.all((preds >= 0) & (preds <= 1)), f"Invalid probabilities for gamma={gamma}"


# =========================================================
# Parameter Validation Tests
# =========================================================

def test_invalid_task():
    """Invalid task should raise ValueError."""
    with pytest.raises(ValueError, match="task must be"):
        CategoryEmbedding(task="invalid")


def test_invalid_num_imp_mode():
    """Invalid num_imp_mode should raise ValueError."""
    with pytest.raises(ValueError, match="num_imp_mode must be"):
        CategoryEmbedding(num_imp_mode="invalid")


def test_invalid_numeric_output():
    """Invalid numeric_output should raise ValueError."""
    with pytest.raises(ValueError, match="numeric_output must be"):
        CategoryEmbedding(numeric_output="invalid")


def test_invalid_return_raw_categoricals():
    """Invalid return_raw_categoricals should raise ValueError."""
    with pytest.raises(ValueError, match="return_raw_categoricals must be"):
        CategoryEmbedding(return_raw_categoricals="invalid")


def test_focal_gamma_with_regression_raises():
    """focal_gamma must raise ValueError when task='regression'."""
    with pytest.raises(ValueError, match="focal_gamma is only supported"):
        CategoryEmbedding(task="regression", focal_gamma=2.0)


def test_focal_gamma_zero_raises():
    """focal_gamma <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="focal_gamma must be a positive float"):
        CategoryEmbedding(task="classification", focal_gamma=0.0)


def test_focal_gamma_negative_raises():
    """Negative focal_gamma should raise ValueError."""
    with pytest.raises(ValueError, match="focal_gamma must be a positive float"):
        CategoryEmbedding(task="classification", focal_gamma=-1.0)


def test_focal_gamma_none_regression_ok(sample_data):
    """focal_gamma=None with task='regression' should not raise."""
    df_train, y = sample_data
    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        focal_gamma=None,
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)  # should not raise


def test_embedding_dims_length_mismatch(multi_cat_data):
    """embedding_dims length must match categorical_cols."""
    df_train, y = multi_cat_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat1", "cat2"],
        embedding_dims=[10],  # Wrong length
        epochs=2,
        batch_size=1,
        verbose=0,
    )

    with pytest.raises(ValueError, match="embedding_dims length must match"):
        enc.fit(df_train, y)


def test_missing_columns_in_fit():
    """Missing columns in X should raise ValueError."""
    df = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0]})
    y = np.array([0.0, 1.0])

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat", "missing_cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )

    with pytest.raises(ValueError, match="Missing categorical columns"):
        enc.fit(df, y)


# =========================================================
# API Compatibility Tests
# =========================================================

def test_fit_transform_basic():
    """Basic fit_transform should return DataFrame with correct shape."""
    df = pd.DataFrame({
        "cat1": ["a", "b", "a", "c"],
        "cat2": ["x", "y", "x", "z"],
        "num1": [1.0, 2.0, 3.0, 4.0],
    })
    y = np.array([0.1, 0.2, 0.3, 0.4])

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat1", "cat2"],
        numeric_cols=["num1"],
        epochs=3,
        batch_size=2,
        verbose=0,
    )

    enc.fit(df, y)
    X_emb = enc.transform(df)

    assert isinstance(X_emb, pd.DataFrame)
    assert len(X_emb) == len(df)
    assert X_emb.shape[1] > 0


def test_get_feature_names_out(sample_data):
    """get_feature_names_out should return correct feature names."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
        numeric_output='raw',
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    X_emb = enc.transform(df_test)

    feature_names = enc.get_feature_names_out()

    assert len(feature_names) == X_emb.shape[1]
    assert "num" in feature_names
    assert any("emb" in name for name in feature_names)


def test_get_feature_names_out_before_transform(sample_data):
    """get_feature_names_out should raise error if transform not called."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    with pytest.raises(RuntimeError):
        enc.get_feature_names_out()


def test_predict_before_fit(sample_data):
    """predict should raise error if model not fitted."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
    )

    with pytest.raises(RuntimeError):
        enc.predict(df_train)


def test_transform_before_fit(sample_data):
    """transform should raise error if model not fitted."""
    df_train, y = sample_data

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
    )

    with pytest.raises(RuntimeError):
        enc.transform(df_train)