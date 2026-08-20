import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Small synthetic transaction dataset shaped like the real Kaggle
    credit card fraud dataset, so tests don't depend on the (large,
    not-committed) real file.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {f"V{i}": rng.normal(size=n) for i in range(1, 6)}
    )
    df["Time"] = np.arange(n)
    df["Amount"] = rng.uniform(1, 500, size=n)
    # ~10% fraud rate, exaggerated vs. real-world for a tiny test set
    df["Class"] = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    return df

