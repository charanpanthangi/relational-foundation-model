from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from data_connectors.parquet_connector import ParquetConnector
from graph_engine.graph_builder import build_hetero_graph
from models.relational_transformer import RelationalTransformer
from training.evaluator import evaluate_predictions
from training.losses import compute_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(config_path: str = "configs/training.yaml", model_path: str = "configs/model.yaml") -> dict[str, float]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    model_cfg = yaml.safe_load(Path(model_path).read_text())["model"]

    connector = ParquetConnector("./sample_data")
    payload = build_hetero_graph(connector, max_rows_per_table=500)
    data = payload.data
    metadata = data.metadata()

    task_type = "classification"
    model = RelationalTransformer(
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=model_cfg["out_dim"],
        metadata=metadata,
        task_type=task_type,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    anchor = metadata[0][0]
    n = data[anchor].x.shape[0]
    if n == 0:
        raise RuntimeError("No rows found for anchor table")
    y = torch.randint(0, 2, (n,))

    model.train()
    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        pred = model(data.x_dict, data.edge_index_dict, anchor_type=anchor)
        loss = compute_loss(pred, y, task_type=task_type)
        loss.backward()
        optimizer.step()
        logger.info("epoch=%s loss=%.4f", epoch + 1, float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred = model(data.x_dict, data.edge_index_dict, anchor_type=anchor)
    return evaluate_predictions(pred, y, task_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    print(train(args.training_config, args.model_config))
