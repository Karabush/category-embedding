import numpy as np
import pandas as pd

from category_embedding import CategoryEmbedding


def test_fit_transform_basic_regression():
    df = pd.DataFrame(
        {
            "cat1": ["a", "b", "a", "c"],
            "cat2": ["x", "y", "x", "z"],
            "num1": [1.0, 2.0, 3.0, 4.0],
        }
    )
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
