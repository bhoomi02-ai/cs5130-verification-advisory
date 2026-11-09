"""Sandboxed calculator tool for the dedicated verifier.

The verifier sends an arithmetic expression as a string, e.g.
``"100000 * 0.99 * (1.09 * 0.98) ** 5"``, and gets back a number or an error
message. Expressions are parsed with ``ast`` and evaluated node by node against
a whitelist: no names other than the listed functions and constants, no
attributes, no imports, no ``eval``/``exec``. Nothing the model sends can run
arbitrary Python.

``calculate`` never raises; errors are returned so the agent can fix its
expression and try again.
"""

from __future__ import annotations

import ast
import math
import operator

import numpy_financial as npf

MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 10_000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: None,  # handled separately to cap the exponent
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    # numpy-financial, with its sign convention: money paid out is negative.
    "fv": lambda rate, nper, pmt, pv, when=0: float(npf.fv(rate, nper, pmt, pv, when)),
    "pv": lambda rate, nper, pmt, fv=0, when=0: float(npf.pv(rate, nper, pmt, fv, when)),
    "pmt": lambda rate, nper, pv, fv=0, when=0: float(npf.pmt(rate, nper, pv, fv, when)),
}

_CONSTANTS = {"pi": math.pi, "e": math.e}
_LIST_FUNCTIONS = {"sum", "min", "max"}
_KEYWORD_FUNCTIONS = {"fv", "pv", "pmt"}

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate an arithmetic expression and return the exact result. Supports "
        "+ - * / // % ** and parentheses, and the functions "
        + ", ".join(sorted(_FUNCTIONS))
        + ". List literals are allowed only inside sum/min/max, e.g. sum([1, 2, 3]). "
        "fv(rate, nper, pmt, pv, when=0), pv(rate, nper, pmt, fv=0, when=0) and "
        "pmt(rate, nper, pv, fv=0, when=0) follow numpy-financial conventions "
        "(rate per period, cash paid out is negative; when=0 end of period, 1 start) "
        "and accept their arguments by position or by keyword. "
        "Use it to check every number in the answer instead of doing arithmetic "
        "in your head. One expression per call; no variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. 100000 * 0.99 * (1.09 * (1 - 0.02)) ** 5",
            }
        },
        "required": ["expression"],
    },
}


class CalculatorError(ValueError):
    pass


def _number(value):
    """Reject anything that is not a real number (complex, list, ...)."""
    if isinstance(value, complex):
        raise CalculatorError("Result is a complex number (e.g. a fractional power of a negative number)")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError(f"Expected a number, got {type(value).__name__}")
    return value


def _eval(node: ast.AST, allow_list: bool = False):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"Only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _number(_eval(node.left)), _number(_eval(node.right))
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise CalculatorError(f"Exponent too large (max {MAX_EXPONENT})")
            return _number(float(left) ** right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_number(_eval(node.operand)))
    if isinstance(node, (ast.List, ast.Tuple)):
        # Lists only as a direct argument of sum/min/max, never in arithmetic,
        # so e.g. [0] * 10**9 cannot allocate memory.
        if not allow_list:
            raise CalculatorError("Lists are only allowed directly inside " + ", ".join(sorted(_LIST_FUNCTIONS)))
        return [_number(_eval(x)) for x in node.elts]
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"Unknown name {node.id!r}; variables are not supported")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CalculatorError("Only these functions are allowed: " + ", ".join(sorted(_FUNCTIONS)))
        name = node.func.id
        if node.keywords and name not in _KEYWORD_FUNCTIONS:
            raise CalculatorError(
                f"{name}() takes positional arguments only; keywords are allowed for "
                + ", ".join(sorted(_KEYWORD_FUNCTIONS))
            )
        if any(k.arg is None for k in node.keywords):
            raise CalculatorError("**kwargs is not supported")
        args = [_eval(a, allow_list=name in _LIST_FUNCTIONS) for a in node.args]
        kwargs = {k.arg: _number(_eval(k.value)) for k in node.keywords}
        return _number(_FUNCTIONS[name](*args, **kwargs))
    raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate `expression`. Returns {"expression", "result"} or {"expression", "error"}."""
    out: dict = {"expression": expression}
    if not isinstance(expression, str) or not expression.strip():
        out["error"] = "Expression must be a non-empty string"
        return out
    if len(expression) > MAX_EXPRESSION_LENGTH:
        out["error"] = f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)"
        return out
    try:
        value = _number(_eval(ast.parse(expression.strip(), mode="eval")))
        if isinstance(value, float) and not math.isfinite(value):
            raise CalculatorError("Result is not a finite number")
        out["result"] = value
    except SyntaxError:
        out["error"] = "Invalid expression syntax"
    except CalculatorError as e:
        out["error"] = str(e)
    except (ArithmeticError, ValueError, TypeError, MemoryError, RecursionError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out
