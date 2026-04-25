from pql.parser import parse_pql


def test_parse_churn() -> None:
    pql = """PREDICT customer.churn
FOR customer
AS OF snapshot_time"""
    task = parse_pql(pql)
    assert task.objective == "customer.churn"
    assert task.entity == "customer"
    assert task.as_of == "snapshot_time"
