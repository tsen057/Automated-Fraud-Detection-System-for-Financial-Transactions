import tempfile
from pathlib import Path

from src.graph_network import build_fraud_network, save_network_plot
from src.preprocessing import preprocess


class TestBuildFraudNetwork:
    def test_builds_graph_with_only_fraud_nodes(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        graph = build_fraud_network(df)
        fraud_count = (df["Class"] == 1).sum()
        assert graph.number_of_nodes() <= min(fraud_count, 50)

    def test_respects_max_nodes(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        graph = build_fraud_network(df, max_nodes=3)
        assert graph.number_of_nodes() <= 3


class TestSaveNetworkPlot:
    def test_saves_image_file(self, synthetic_transactions):
        df = preprocess(synthetic_transactions)
        graph = build_fraud_network(df)
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "graph.png"
            save_network_plot(graph, output_path)
            assert output_path.exists()

