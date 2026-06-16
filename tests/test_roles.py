import pytest

from spss.roles import primary_role


@pytest.mark.parametrize(
    "pos,expected",
    [
        ("AM (R), ST (C)", "AM"),
        ("GK", "GK"),
        ("D (C)", "DEF"),
        ("D/WB (R)", "DEF"),   # token base é "D"
        ("WB (R)", "WB"),
        ("DM", "DM"),
        ("M (C)", "MID"),
        ("ST (C)", "FW"),
        ("AM (RLC)", "AM"),
        ("Coach", "OTHER"),    # fora do mapa
    ],
)
def test_primary_role(pos, expected):
    assert primary_role(pos) == expected
