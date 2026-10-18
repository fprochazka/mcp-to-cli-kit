"""The registry of expanded values that error messages and logs must not show."""

from __future__ import annotations

import logging

from mcp_to_cli_kit import redaction
from mcp_to_cli_kit.redaction import RedactingFilter


def test_registered_values_are_replaced_longest_first() -> None:
    redaction.register("alpha-secret", "${A}")
    redaction.register("alpha-secret-longer", "${B}")

    assert redaction.redact("x alpha-secret-longer y alpha-secret z") == "x ${B} y ${A} z"


def test_short_values_are_not_registered() -> None:
    redaction.register("abc", "${SHORT}")

    assert redaction.redact("abc") == "abc"


def test_the_filter_redacts_the_message_and_the_exception() -> None:
    redaction.register("beta-secret", "***")
    try:
        raise ValueError("failed with beta-secret")
    except ValueError as error:
        record = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "token %s", ("beta-secret",), (type(error), error, None)
        )

    assert RedactingFilter().filter(record)

    assert record.getMessage() == "token ***"
    assert record.exc_text is not None and "beta-secret" not in record.exc_text and "***" in record.exc_text


def test_encoded_forms_are_registered() -> None:
    redaction.register_encoded("a b@c|d", "***")

    assert redaction.redact("a%20b%40c%7Cd a%20b@c|d a+b%40c%7Cd") == "*** *** ***"


def test_clear_forgets_everything() -> None:
    redaction.register("gamma-secret", "***")
    redaction.clear()

    assert redaction.redact("gamma-secret") == "gamma-secret"
