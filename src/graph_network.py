"""Build and visualize a similarity network of fraudulent transactions."""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_fraud_network(
    df: pd.DataFrame,
    class_col: str = "Class",
    similarity_col: str = "V1",
    threshold: float = 0.5,
    max_nodes: int = 50,
) -> nx.Graph:
    """Build a graph connecting fraudulent transactions with similar values
    of `similarity_col` (default V1, a PCA component in the source dataset).

    Limits to `max_nodes` fraud cases for a readable visualization.
    """
    fraud_df = df[df[class_col] == 1].copy()
    fraud_indices = fraud_df.index.tolist()[:max_nodes]

    graph = nx.Graph()
    graph.add_nodes_from(fraud_indices)

    values = fraud_df.loc[fraud_indices, similarity_col]
    for i, node_i in enumerate(fraud_indices):
        for node_j in fraud_indices[i + 1:]:
            if abs(values[node_i] - values[node_j]) < threshold:
                graph.add_edge(node_i, node_j)

    logger.info(
        "Built fraud network: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def save_network_plot(graph: nx.Graph, output_path: str | Path, title: str = "Fraud Transaction Network") -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend for CI / servers
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 12))
    nx.draw(graph, with_labels=True, node_size=100, font_size=8)
    plt.title(title)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved network plot to %s", output_path)

