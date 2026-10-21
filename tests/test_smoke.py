from advisory.conditions import CONDITIONS


def test_four_conditions_exist():
    assert set(CONDITIONS) == {"none", "self", "peer", "dedicated"}
