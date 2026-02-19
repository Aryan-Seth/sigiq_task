from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional


_SYMBOL_WORDS = {
    "×": "times",
    "÷": "divided by",
    "±": "plus or minus",
    "=": "equals",
    "≈": "approximately equals",
    "≠": "does not equal",
    "≤": "less than or equal to",
    "≥": "greater than or equal to",
    "π": "pi",
}

_UNIT_WORDS = {
    "kg": "kilograms",
    "g": "grams",
    "mg": "milligrams",
    "km": "kilometers",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "ms": "milliseconds",
    "s": "seconds",
    "min": "minutes",
    "h": "hours",
}

_LATEX_COMMAND_WORDS = {
    "times": "times",
    "cdot": "times",
    "pm": "plus or minus",
    "neq": "does not equal",
    "leq": "less than or equal to",
    "geq": "greater than or equal to",
    "approx": "approximately equals",
    "pi": "pi",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "sigma": "sigma",
    "omega": "omega",
}

_FUNCTION_WORDS = {
    "sin": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "log": "log",
    "ln": "natural log",
    "exp": "exponential",
}

_INLINE_OPEN = r"\("
_INLINE_CLOSE = r"\)"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class NormalizedText:
    text: str
    source_indices: List[int]


def math_normalization_enabled() -> bool:
    raw = os.environ.get("TTS_MATH_NORMALIZE", "1").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def has_unclosed_inline_math(text: str) -> bool:
    opens = text.count(_INLINE_OPEN)
    closes = text.count(_INLINE_CLOSE)
    return opens > closes


def normalize_for_tts(
    text: str, latex_expr_to_speech: Optional[Callable[[str], str]] = None
) -> NormalizedText:
    out_chars: List[str] = []
    out_src: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(_INLINE_OPEN, i):
            close_idx = text.find(_INLINE_CLOSE, i + len(_INLINE_OPEN))
            if close_idx != -1:
                expr = text[i + len(_INLINE_OPEN) : close_idx]
                spoken = ""
                if latex_expr_to_speech is not None:
                    try:
                        spoken = str(latex_expr_to_speech(expr))
                    except Exception:
                        spoken = ""
                if not spoken.strip():
                    spoken = _speak_math_expr(expr)
                src_idx = min(i + len(_INLINE_OPEN), n - 1)
                _append_with_src(out_chars, out_src, spoken, src_idx)
                i = close_idx + len(_INLINE_CLOSE)
                continue
        ch = text[i]
        if ch in _SYMBOL_WORDS:
            _append_with_src(out_chars, out_src, f" {_SYMBOL_WORDS[ch]} ", i)
            i += 1
            continue
        if ch == "%" and i > 0 and text[i - 1].isdigit():
            _append_with_src(out_chars, out_src, " percent", i)
            i += 1
            continue
        if ch == "°" and i + 1 < n and text[i + 1] in {"C", "F"}:
            unit_word = "degrees celsius" if text[i + 1] == "C" else "degrees fahrenheit"
            _append_with_src(out_chars, out_src, f" {unit_word}", i)
            i += 2
            continue

        num_match = _NUMBER_RE.match(text, i)
        if num_match:
            num_txt = num_match.group(0)
            num_start = i
            i = num_match.end()
            unit_i = i
            while unit_i < n and text[unit_i].isspace():
                unit_i += 1
            rest_lower = text[unit_i:].lower()
            unit_end = unit_i
            while unit_end < n and text[unit_end].isalpha():
                unit_end += 1
            unit_txt = text[unit_i:unit_end].lower()
            _append_with_src(out_chars, out_src, num_txt, num_start)
            if rest_lower.startswith("m/s^2") or rest_lower.startswith("m/s2"):
                _append_with_src(out_chars, out_src, " meters per second squared", unit_i)
                i = unit_i + (5 if rest_lower.startswith("m/s^2") else 4)
                continue
            if rest_lower.startswith("m/s"):
                _append_with_src(out_chars, out_src, " meters per second", unit_i)
                i = unit_i + 3
                continue
            if rest_lower.startswith("km/h"):
                _append_with_src(out_chars, out_src, " kilometers per hour", unit_i)
                i = unit_i + 4
                continue
            if unit_txt in _UNIT_WORDS:
                _append_with_src(out_chars, out_src, f" {_UNIT_WORDS[unit_txt]}", unit_i)
                i = unit_end
            continue

        out_chars.append(ch)
        out_src.append(i)
        i += 1

    return _collapse_spaces(out_chars, out_src)


def _append_with_src(out_chars: List[str], out_src: List[int], chunk: str, src_idx: int) -> None:
    for c in chunk:
        out_chars.append(c)
        out_src.append(src_idx)


def _collapse_spaces(chars: List[str], src: List[int]) -> NormalizedText:
    out_chars: List[str] = []
    out_src: List[int] = []
    for c, s in zip(chars, src):
        if c.isspace():
            if out_chars and out_chars[-1] == " ":
                continue
            out_chars.append(" ")
            out_src.append(s)
            continue
        out_chars.append(c)
        out_src.append(s)
    while out_chars and out_chars[0] == " ":
        out_chars.pop(0)
        out_src.pop(0)
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_src.pop()
    return NormalizedText(text="".join(out_chars), source_indices=out_src)


class _MathParser:
    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.n = len(expr)
        self.i = 0

    def parse(self, stop: str | None = None) -> str:
        parts: List[str] = []
        while self.i < self.n:
            ch = self.expr[self.i]
            if stop is not None and ch == stop:
                break
            if ch.isspace():
                self.i += 1
                continue
            if self._consume(r"\frac"):
                numer = self._parse_group_or_atom()
                denom = self._parse_group_or_atom()
                parts.append(f"{numer} over {denom}".strip())
                continue
            if self._consume(r"\sqrt"):
                rad = self._parse_group_or_atom()
                parts.append(f"square root of {rad}".strip())
                continue
            if ch == "{":
                self.i += 1
                inner = self.parse(stop="}")
                if self.i < self.n and self.expr[self.i] == "}":
                    self.i += 1
                parts.append(inner)
                continue
            if ch == "^":
                self.i += 1
                exp = self._parse_group_or_atom()
                base = parts.pop() if parts else ""
                parts.append(_format_power(base, exp))
                continue
            if ch in "+-*/=()":
                self.i += 1
                parts.append(_operator_word(ch))
                continue
            if ch == "\\":
                word = self._parse_command_word()
                if word:
                    parts.append(word)
                continue
            token = self._parse_token()
            if token:
                parts.append(_speak_identifier(token))
                continue
            self.i += 1
        return " ".join(p for p in parts if p).strip()

    def _consume(self, prefix: str) -> bool:
        if self.expr.startswith(prefix, self.i):
            self.i += len(prefix)
            return True
        return False

    def _parse_group_or_atom(self) -> str:
        while self.i < self.n and self.expr[self.i].isspace():
            self.i += 1
        if self.i >= self.n:
            return ""
        if self.expr[self.i] == "{":
            self.i += 1
            inner = self.parse(stop="}")
            if self.i < self.n and self.expr[self.i] == "}":
                self.i += 1
            return inner
        if self.expr[self.i] == "\\":
            return self._parse_command_word()
        token = self._parse_token()
        return _speak_identifier(token)

    def _parse_command_word(self) -> str:
        if self.i >= self.n or self.expr[self.i] != "\\":
            return ""
        self.i += 1
        start = self.i
        while self.i < self.n and self.expr[self.i].isalpha():
            self.i += 1
        cmd = self.expr[start:self.i]
        if not cmd:
            return ""
        if cmd in {"left", "right"}:
            return ""
        return _LATEX_COMMAND_WORDS.get(cmd, _speak_identifier(cmd))

    def _parse_token(self) -> str:
        if self.i >= self.n:
            return ""
        ch = self.expr[self.i]
        if ch.isdigit():
            start = self.i
            self.i += 1
            while self.i < self.n and (self.expr[self.i].isdigit() or self.expr[self.i] == "."):
                self.i += 1
            return self.expr[start:self.i]
        if ch.isalpha():
            start = self.i
            self.i += 1
            while self.i < self.n and self.expr[self.i].isalpha():
                self.i += 1
            return self.expr[start:self.i]
        self.i += 1
        return ch


def _operator_word(ch: str) -> str:
    return {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "divided by",
        "=": "equals",
        "(": "open parenthesis",
        ")": "close parenthesis",
    }.get(ch, ch)


def _speak_identifier(token: str) -> str:
    t = token.strip()
    if not t:
        return ""
    low = t.lower()
    if low in _FUNCTION_WORDS:
        return _FUNCTION_WORDS[low]
    if low in _LATEX_COMMAND_WORDS:
        return _LATEX_COMMAND_WORDS[low]
    if t.replace(".", "", 1).isdigit():
        return t
    if low == "dx":
        return "d x"
    if t.isalpha() and len(t) > 1:
        return " ".join(t)
    return t


def _format_power(base: str, exp: str) -> str:
    base = base.strip()
    exp = exp.strip()
    if not base:
        return f"to the power of {exp}".strip()
    if exp == "2":
        return f"{base} squared"
    if exp == "3":
        return f"{base} cubed"
    return f"{base} to the power of {exp}"


def _speak_math_expr(expr: str) -> str:
    parser = _MathParser(expr)
    spoken = parser.parse()
    return spoken or expr
