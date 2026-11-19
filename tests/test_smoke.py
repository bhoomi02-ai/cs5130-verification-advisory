from advisory.conditions import CONDITIONS


def test_six_conditions_exist():
    assert set(CONDITIONS) == {"none", "self", "peer", "dedicated", "weakened", "cross_family"}
