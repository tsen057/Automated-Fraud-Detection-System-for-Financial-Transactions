from src.preprocessing import preprocess, split_features_target, train_test_split_data
from src.train import build_candidate_models, train_and_select


class TestBuildCandidateModels:
    def test_returns_both_models(self):
        models = build_candidate_models()
        assert "random_forest" in models
        assert "xgboost" in models


class TestTrainAndSelect:
    def test_selects_a_model_and_returns_scores(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        best_name, model, scores = train_and_select(X_train, y_train, X_test, y_test)

        assert best_name in ("random_forest", "xgboost")
        assert set(scores.keys()) == {"random_forest", "xgboost"}
        assert hasattr(model, "predict")

    def test_model_can_predict(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        _, model, _ = train_and_select(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

