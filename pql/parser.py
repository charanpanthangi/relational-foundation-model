from __future__ import annotations

import re

from pql.task_schema import PredictiveTask

PREDICT_RE = re.compile(r"^PREDICT\s+([a-zA-Z0-9_\.]+)$", re.IGNORECASE)
FOR_RE = re.compile(r"^FOR\s+([a-zA-Z0-9_]+)$", re.IGNORECASE)
ASOF_RE = re.compile(r"^AS\s+OF\s+([a-zA-Z0-9_]+)$", re.IGNORECASE)


def parse_pql(text: str) -> PredictiveTask:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("PQL requires at least PREDICT and FOR lines")

    predict = PREDICT_RE.match(lines[0])
    if not predict:
        raise ValueError("First line must be: PREDICT <target>")
    objective = predict.group(1)

    entity = None
    as_of = None
    for line in lines[1:]:
        m_for = FOR_RE.match(line)
        m_as = ASOF_RE.match(line)
        if m_for:
            entity = m_for.group(1)
        elif m_as:
            as_of = m_as.group(1)

    if not entity:
        raise ValueError("PQL missing FOR clause")

    task_type = "classification"
    if any(k in objective.lower() for k in ["ltv", "spend", "forecast"]):
        task_type = "regression"
    elif any(k in objective.lower() for k in ["recommend", "affinity"]):
        task_type = "ranking"

    return PredictiveTask(objective=objective, entity=entity, as_of=as_of, task_type=task_type)
