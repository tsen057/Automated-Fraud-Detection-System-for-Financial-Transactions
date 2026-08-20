import pandas as pd
import pytest

from src.preprocessing import preprocess, split_features_target, train_test_split_data


class TestPreprocess:
    def test_adds_normalized_amount(self, synthetic_transactions):
        result = preprocess(synthetic_transactions)
        assert "normalizedAmount" in result.columns

    def test_drops_time_and_amount(self, synthetic_transactions):
        result = preprocess(synthetic_transactions)
        assert "Time" not in result.columns
        assert "Amount" not in result.columns

    def test_does_not_mutate_input(self, synthetic_transactions):
        original_cols = list(synthetic_transactions.columns)
        preprocess(synthetic_transactions)
        assert list(synthetic_transactions.columns) == original_cols

    def test_raises_without_amount_column(self):
        df = pd.DataFrame({"Class": [0, 1]})
        with pytest.raises(ValueError):
            preprocess(df)


class TestSplitFeaturesTarget:
    def test_splits_correctly(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        X, y = split_features_target(df)
        assert "Class" not in X.columns
        assert y.name == "Class"
        assert len(X) == len(y)

    def test_raises_without_target_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            split_features_target(df)


class TestTrainTestSplit:
    def test_produces_correct_shapes(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

