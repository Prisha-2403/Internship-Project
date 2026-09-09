"""CSV import parsing and validation.

Validation is per row: a file with three bad rows should import the rest and say
exactly which three failed and why.
"""

from __future__ import annotations

from datetime import UTC
from decimal import Decimal

import pytest

from app.core.exceptions import ImportError_
from app.services.import_service import (
    MAX_FILE_BYTES,
    REQUIRED_COLUMNS,
    decode_csv,
    parse_amount,
    parse_timestamp,
)

HEADER = "transaction_id,customer_id,amount,timestamp\n"


class TestTimestampParsing:
    @pytest.mark.parametrize(
        "value",
        [
            "2026-08-20T14:21:00Z",
            "2026-08-20T14:21:00+00:00",
            "2026-08-20 14:21:00",
            "2026-08-20 14:21",
            "2026-08-20",
            "20/08/2026 14:21",
            "20/08/2026",
        ],
    )
    def test_accepted_formats(self, value):
        parsed = parse_timestamp(value)
        assert parsed.year == 2026
        assert parsed.month == 8
        assert parsed.day == 20

    def test_naive_timestamps_are_treated_as_utc(self):
        assert parse_timestamp("2026-08-20 14:21:00").tzinfo == UTC

    def test_offsets_are_converted_to_utc(self):
        assert parse_timestamp("2026-08-20T20:21:00+05:30").hour == 14

    @pytest.mark.parametrize("value", ["", "   ", "not-a-date", "2026-13-45", "yesterday"])
    def test_invalid_values_are_rejected(self, value):
        with pytest.raises(ValueError):
            parse_timestamp(value)


class TestAmountParsing:
    def test_plain_decimal(self):
        assert parse_amount("1234.56") == Decimal("1234.56")

    def test_thousands_separators_are_tolerated(self):
        assert parse_amount("1,234.56") == Decimal("1234.56")

    def test_whitespace_is_trimmed(self):
        assert parse_amount("  99.00  ") == Decimal("99.00")

    def test_values_are_quantised_to_two_places(self):
        assert parse_amount("10.999") == Decimal("11.00")

    @pytest.mark.parametrize(
        ("value", "reason"),
        [("0", "zero"), ("-5", "negative"), ("abc", "text"), ("", "blank"), ("  ", "spaces")],
    )
    def test_invalid_values_are_rejected(self, value, reason):
        with pytest.raises(ValueError):
            parse_amount(value)

    def test_absurdly_large_values_are_rejected(self):
        with pytest.raises(ValueError):
            parse_amount("999999999999999")

    def test_infinity_and_nan_are_rejected(self):
        for value in ("Infinity", "NaN", "-Infinity"):
            with pytest.raises(ValueError):
                parse_amount(value)


class TestCsvDecoding:
    def test_valid_file_parses(self):
        rows = decode_csv(
            (HEADER + "TX-1,CUST-1000,100,2026-08-20T10:00:00Z\n").encode()
        )
        assert len(rows) == 1
        assert rows[0]["transaction_id"] == "TX-1"

    def test_headers_are_normalised_to_lowercase(self):
        rows = decode_csv(
            b"Transaction_ID,Customer_ID,Amount,Timestamp\nTX-1,CUST-1,10,2026-08-20\n"
        )
        assert "transaction_id" in rows[0]

    def test_a_utf8_bom_is_handled(self):
        """Excel writes a BOM by default; it must not corrupt the first header."""
        content = ("﻿" + HEADER + "TX-1,CUST-1,10,2026-08-20\n").encode("utf-8")
        assert decode_csv(content)[0]["transaction_id"] == "TX-1"

    def test_empty_file_is_rejected_with_a_readable_message(self):
        with pytest.raises(ImportError_) as caught:
            decode_csv(b"")
        assert "empty" in caught.value.message.lower()

    def test_missing_required_columns_are_named(self):
        with pytest.raises(ImportError_) as caught:
            decode_csv(b"transaction_id,amount\nTX-1,100\n")
        message = caught.value.message
        assert "customer_id" in message and "timestamp" in message

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_every_required_column_is_enforced(self, column):
        headers = [c for c in REQUIRED_COLUMNS if c != column]
        content = (",".join(headers) + "\n" + ",".join("x" for _ in headers) + "\n").encode()
        with pytest.raises(ImportError_) as caught:
            decode_csv(content)
        assert column in caught.value.message

    def test_oversized_files_are_refused(self):
        with pytest.raises(ImportError_) as caught:
            decode_csv(b"x" * (MAX_FILE_BYTES + 1))
        assert "too large" in caught.value.message.lower()

    def test_error_messages_stay_user_facing(self):
        """No stack traces or library names leak into what an operator reads."""
        with pytest.raises(ImportError_) as caught:
            decode_csv(b"nonsense-without-required-columns\n")
        message = caught.value.message
        assert "Traceback" not in message
        assert "csv.Error" not in message

    def test_optional_columns_are_preserved(self):
        rows = decode_csv(
            b"transaction_id,customer_id,amount,timestamp,location,payment_method\n"
            b"TX-1,CUST-1,100,2026-08-20,Delhi,UPI\n"
        )
        assert rows[0]["location"] == "Delhi"
        assert rows[0]["payment_method"] == "UPI"
