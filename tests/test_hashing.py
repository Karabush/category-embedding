import numpy as np
import pandas as pd

from category_embedding import CategoryEmbedding


def test_unseen_category_hashing():
    df_train = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
    y = np.array([0.0, 1.0, 0.5])

    enc = CategoryEmbedding(
        task="regression",
        categorical_cols=["cat"],
        numeric_cols=["num"],
        epochs=2,
        batch_size=1,
        verbose=0,
    )
    enc.fit(df_train, y)

    df_test = pd.DataFrame({"cat": ["a", "d"], "num": [4, 5]})
    X_emb = enc.transform(df_test)

    assert len(X_emb) == 2
