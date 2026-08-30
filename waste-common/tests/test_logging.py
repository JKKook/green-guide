from __future__ import annotations

import pytest

from waste_common.logging import fail_open, get_logger


def test_fail_open_logs_and_swallows(caplog: pytest.LogCaptureFixture) -> None:
    log = get_logger("t")
    with caplog.at_level("WARNING"):
        with fail_open(log, "작업"):
            raise ValueError("boom")
    assert "작업 실패" in caplog.text and "ValueError: boom" in caplog.text
