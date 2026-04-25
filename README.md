# Relational Foundation Model

Relational Foundation Model (RFM) is a production-oriented, KumoRFM-inspired framework for learning directly from enterprise multi-table data and serving predictive outcomes such as churn, fraud, forecasting, recommendations, lifetime value (LTV), and spend-drop risk.

## Why Multi-Table Prediction Matters

Traditional ML pipelines flatten relational systems into single tables, losing critical context:

- Customer behavior spread across CRM, billing, events, and support tables.
- Fraud patterns existing in cross-entity transaction links.
- Time-aware dependencies where leakage can happen if future rows are used.

RFM preserves this structure by converting relational schemas into temporal heterogeneous graphs.

## Architecture

1. **Connectors** (`data_connectors/`): PostgreSQL, parquet, plus Snowflake/Databricks scaffolds.
2. **PQL (Predictive Query Language)** (`pql/`): declarative task specification.
3. **Schema Engine** (`schema_engine/`): introspection + relationship/time metadata detection.
4. **Graph Engine** (`graph_engine/`): relational tables → PyG `HeteroData` graph.
5. **Sampler** (`sampler/`): temporal-safe context sampling and negatives.
6. **Models** (`models/`): HGT baseline and modular relational transformer + prediction heads.
7. **Training** (`training/`): end-to-end train/evaluate loops.
8. **Inference + API** (`inference/`, `api/`): single/batch/online scoring and explanations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional local Postgres:

```bash
docker compose up -d
```

## Example Dataset Layout

Place parquet tables under `sample_data/`:

- `customer.parquet`
- `orders.parquet`
- `transactions.parquet`

Each file maps to one node type in the graph.

## Example PQL

```text
PREDICT customer.churn
FOR customer
AS OF snapshot_time
```

Additional examples are in `pql/examples/`.

## Training Example

```bash
python -m training.train --training-config configs/training.yaml --model-config configs/model.yaml
```

## API Example

Run API:

```bash
uvicorn api.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict request:

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"entity_id": 42}'
```

## Notebook Starters

- `notebooks/01_schema_to_graph.ipynb`
- `notebooks/02_train_churn_model.ipynb`
- `notebooks/03_predict_from_pql.ipynb`

## Roadmap

- Stronger schema semantics with explicit PK/FK introspection by backend.
- Contrastive/self-supervised pretraining over enterprise relational corpora.
- Feature store integration and online embedding cache.
- Model registry + drift monitoring + continuous retraining workflows.
- Richer explainability (path attributions on heterogeneous neighborhoods).
