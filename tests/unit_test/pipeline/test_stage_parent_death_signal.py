# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import signal

import pytest

from sglang_omni.pipeline import stage_workers

EXPECTED_PARENT = 4321


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
    stage_workers._install_parent_death_signal(EXPECTED_PARENT)


def test_parent_death_signal_requests_sigkill_on_parent_exit(monkeypatch) -> None:
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: EXPECTED_PARENT)

    stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert fake.calls, "prctl was not called"
    option, sig = fake.calls[0][0], fake.calls[0][1]
    assert option == 1  # PR_SET_PDEATHSIG
    assert sig == signal.SIGKILL


def test_parent_death_signal_exits_when_parent_died_before_install(monkeypatch) -> None:
    """Parent died during the spawn bootstrap: getppid no longer matches the
    launcher PID, so exit before touching prctl."""
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    # Already reparented away from the expected launcher PID.
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: 1)

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(stage_workers.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert exit_codes == [1]
    assert not fake.calls, "prctl must not run once the parent is already gone"


def test_parent_death_signal_exits_when_parent_died_during_install(monkeypatch) -> None:
    """Parent alive at entry but dies while prctl is being installed: the second
    getppid check catches the reparent and exits."""
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    # First check matches the launcher; second check sees the reparent to init.
    ppids = iter([EXPECTED_PARENT, 1])
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: next(ppids))

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(stage_workers.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert exit_codes == [1]
    assert fake.calls, "prctl should have run before the second check"


def test_parent_death_signal_survives_prctl_failure(monkeypatch) -> None:
    fake = _FakeLibc(rc=-1)
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: EXPECTED_PARENT)

    # A failing prctl must be swallowed (logged), not raised.
    stage_workers._install_parent_death_signal(EXPECTED_PARENT)
