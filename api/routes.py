from __future__ import annotations

from fastapi import APIRouter

from api.schemas import BatchPredictRequest, ExplainRequest, PredictRequest, TrainRequest
from training.train import train

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/train")
def train_route(req: TrainRequest) -> dict[str, float]:
    return train(req.training_config, req.model_config)


@router.post("/predict")
def predict_route(req: PredictRequest) -> dict[str, float | int | str]:
    return {"entity_id": req.entity_id, "score": 0.5, "message": "Load model artifact for live predictions."}


@router.post("/batch_predict")
def batch_predict_route(req: BatchPredictRequest) -> dict[str, list[dict[str, float | int]]]:
    return {"predictions": [{"entity_id": i, "score": 0.5} for i in req.entity_ids]}


@router.post("/explain")
def explain_route(req: ExplainRequest) -> dict[str, object]:
    return {"entity_id": req.entity_id, "top_features": [{"feature": "recency_days", "contribution": -0.12}]}


@router.get("/schema")
def schema_route() -> dict[str, object]:
    return {"tables": [], "relationships": []}
