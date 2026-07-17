# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import signal

import pytest

from sglang_omni.pipeline import stage_workers


class _FakeLibc:
    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple] = []

    def prctl(self, *args) -> int:
        self.calls.append(args)
        return self.rc


def test_parent_death_signal_is_noop_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(stage_workers.sys, "platform", "darwin")

    def _fail(*_a, **_k):  # pragma: no cover - must never run off Linux
        raise AssertionError("ctypes must not be touched off Linux")

    import ctypes

    monkeypatch.setattr(ctypes, "CDLL", _fail)
    stage_workers._install_parent_death_signal()


def test_parent_death_signal_requests_sigkill_on_parent_exit(monkeypatch) -> None:
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: 4321)

    stage_workers._install_parent_death_signal()

    assert fake.calls, "prctl was not called"
    option, sig = fake.calls[0][0], fake.calls[0][1]
    assert option == 1  # PR_SET_PDEATHSIG
    assert sig == signal.SIGKILL


def test_parent_death_signal_exits_when_already_reparented(monkeypatch) -> None:
    """If the parent died before prctl took effect, exit immediately."""
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    # First call captures the real parent; second sees the reparent to init.
    ppids = iter([4321, 1])
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: next(ppids))

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(stage_workers.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        stage_workers._install_parent_death_signal()

    assert exit_codes == [1]


def test_parent_death_signal_survives_prctl_failure(monkeypatch) -> None:
    fake = _FakeLibc(rc=-1)
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: 4321)

    # A failing prctl must be swallowed (logged), not raised.
    stage_workers._install_parent_death_signal()
