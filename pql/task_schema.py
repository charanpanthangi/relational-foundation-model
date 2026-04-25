from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictiveTask(BaseModel):
    objective: str = Field(..., description="e.g. customer.churn")
    entity: str = Field(..., description="anchor entity/table")
    as_of: str | None = Field(default=None, description="snapshot timestamp column")
    task_type: Literal["classification", "regression", "ranking"] = "classification"
