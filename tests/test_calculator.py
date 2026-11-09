import math

import pytest

from advisory.calculator import CALCULATOR_TOOL, calculate


def ok(expr):
    out = calculate(expr)
    assert "error" not in out, out
    return out["result"]


def err(expr):
    out = calculate(expr)
    assert "result" not in out, out
    return out["error"]


def test_basic_arithmetic_and_precedence():
    assert ok("2 + 3 * 4") == 14
    assert ok("(2 + 3) * 4") == 20
    assert ok("-2 ** 2") == -4.0
    assert ok("7 // 2 + 7 % 2") == 4
    assert ok("1 / 4") == 0.25


def test_functions_and_constants():
    assert ok("max([1, 5, 3])") == 5
    assert ok("sum([20000, 15000, 20000])") == 55000
    assert ok("round(1.23456, 2)") == 1.23
    assert ok("ceil(432.7631 * 100) / 100") == 432.77
    assert ok("sqrt(16)") == 4.0
    assert ok("pi") == math.pi


def test_s04_fund_comparison():
    a = ok("100000 * (1 - 0.01) * ((1 + 0.09) * (1 - 0.02)) ** 5")
    b = ok("100000 * ((1 + 0.08) * (1 - 0.002)) ** 5")
    assert a == pytest.approx(137688.62, abs=0.01)
    assert b == pytest.approx(145469.35, abs=0.01)


def test_s06_financial_functions():
    # Extra monthly payment to close a $199,954.26 gap over 240 months at 0.5%.
    assert ok("pmt(0.005, 240, 0, -199954.26)") == pytest.approx(432.7631, abs=1e-4)
    assert ok("fv(0.005, 12, -100, 0)") == pytest.approx(1233.56, abs=0.01)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "(1).__class__",
        "x + 1",
        "lambda: 1",
        "[i for i in range(3)]",
        "'abc' * 3",
        "True + 1",
        "max(1, key=abs)",
        "1 if 1 else 2",
        "a = 1",
    ],
)
def test_rejects_anything_but_arithmetic(expr):
    err(expr)


def test_guards_against_huge_or_bad_results():
    assert "too large" in err("10 ** 100000")
    assert "OverflowError" in err("10.0 ** 1000")
    assert "ZeroDivisionError" in err("1 / 0")
    assert "too long" in err("1+" * 300 + "1")
    assert err("") and err("   ")
    assert "only allowed directly inside" in err("[1, 2]")
    assert "ValueError" in err("sqrt(-1)")
    assert "syntax" in err("2 +")


def test_tool_spec_shape():
    assert CALCULATOR_TOOL["name"] == "calculate"
    assert CALCULATOR_TOOL["parameters"]["required"] == ["expression"]


# Follow-ups from the #14 review.


def test_lists_only_inside_sum_min_max():
    assert ok("sum([1, 2, 3])") == 6
    assert ok("min([4, 2]) + max([1, 9])") == 11
    assert ok("min(4, 2)") == 2
    for expr in ["[0] * 1000000000", "[1] + [2]", "abs([1])", "sum([[1], [2]])", "-[1]"]:
        assert "only allowed directly inside" in err(expr), expr


def test_complex_results_are_errors():
    assert "complex" in err("(-8) ** (1/3)")
    assert "complex" in err("1 + (-8) ** 0.5")
    assert ok("(-8) ** 2") == 64.0  # integer powers of negatives are fine


def test_financial_functions_accept_keywords():
    assert ok("fv(0.005, 12, -100, 0, when=1)") == ok("fv(0.005, 12, -100, 0, 1)")
    assert ok("pmt(rate=0.005, nper=240, pv=0, fv=-199954.26)") == pytest.approx(432.7631, abs=1e-4)
    assert "TypeError" in err("fv(0.005, 12, -100, 0, bogus=1)")
    assert "positional arguments only" in err("round(1.234, ndigits=1)")
    assert "positional arguments only" in err("max(1, key=abs)")


def test_memory_error_is_returned_not_raised(monkeypatch):
    from advisory import calculator

    def boom(*args):
        raise MemoryError("simulated")

    monkeypatch.setitem(calculator._FUNCTIONS, "sqrt", boom)
    assert "MemoryError" in err("sqrt(4)")


def test_every_result_is_json_serializable():
    import json

    for expr in ["2 ** 0.5", "fv(0.005, 12, -100, 0)", "ceil(2.1)", "sum([1.5, 2])"]:
        json.dumps(calculate(expr))
