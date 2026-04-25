from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    training_config: str = "configs/training.yaml"
    model_config: str = "configs/model.yaml"


class PredictRequest(BaseModel):
    entity_id: int = Field(..., ge=0)


class BatchPredictRequest(BaseModel):
    entity_ids: list[int]


class ExplainRequest(BaseModel):
    entity_id: int
