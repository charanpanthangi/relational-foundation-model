from __future__ import annotations

from pql.task_schema import PredictiveTask


def validate_task(task: PredictiveTask, schema: dict[str, dict[str, str]]) -> None:
    if task.entity not in schema:
        raise ValueError(f"Unknown entity table: {task.entity}")
    if task.as_of and task.as_of not in schema[task.entity]:
        raise ValueError(f"Unknown AS OF column '{task.as_of}' for table '{task.entity}'")
