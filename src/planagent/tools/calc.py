"""Safe arithmetic, so the model never does mental maths.

Expressions are parsed with ``ast`` and evaluated over a whitelist of node
types and functions: no names, attributes, calls to anything else, or
comprehensions can execute.
"""

from __future__ import annotations

import ast
import math
import operator

from . import Tool

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUN = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and not node.keywords:
        return _FUN[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> dict:
    if len(expression) > 300:
        return {"error": "expression too long"}
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return {"error": f"cannot evaluate: {e}"}
    return {"expression": expression, "value": round(float(value), 10)}


CALCULATE = Tool(
    "calculate",
    "Evaluate an arithmetic expression (+ - * / ** %, parentheses, abs, round, min, max, sqrt, exp, log, log10). "
    "Use it for EVERY derived number (differences, ratios, percentage changes); do not compute in your head.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string", "minLength": 1}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    calculate,
)
