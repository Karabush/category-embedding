import numpy as np
import pandas as pd

from category_embedding import CategoryEmbedding


def test_get_feature_names_out():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "num": [1.0, 2.0, 3.0],
        }
    )
    y = np.array([0.0, 1.0, 0.5])

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=2,
        verbose=0,
    )
    enc.fit(df, y)
    X_emb = enc.transform(df)
    names = enc.get_feature_names_out()

    assert len(names) == X_emb.shape[1]
    assert any(name.startswith("cat_emb_") for name in names)
    assert "num" in names
