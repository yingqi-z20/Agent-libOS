from src.app import add, status


def test_add() -> None:
    assert add(2, 3) == 5


def test_status() -> None:
    assert status() == "ok"
