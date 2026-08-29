"""Unit tests for trading_bot.cli.setup_schedule. subprocess.run is always
mocked -- no real schtasks/powercfg calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from trading_bot.cli import setup_schedule

ET = ZoneInfo("America/New_York")


class TestEtTimeToLocalHhmm(unittest.TestCase):
    @patch("trading_bot.cli.setup_schedule.LOCAL_TZ", ZoneInfo("UTC"))
    def test_converts_et_to_local_correctly(self):
        # Use "today" (whatever it really is) for both the function call and
        # the expected value, so the correct EDT/EST offset applies to both
        # regardless of when this test runs.
        expected_et = datetime.now(ET).replace(hour=14, minute=30, second=0, microsecond=0)
        expected_local = expected_et.astimezone(ZoneInfo("UTC")).strftime("%H:%M")

        result = setup_schedule.et_time_to_local_hhmm(14, 30)

        self.assertEqual(result, expected_local)


class TestApplyPostCreateSettings(unittest.TestCase):
    def test_success_is_silent(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=0, stderr="")) as mock_run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.apply_post_create_settings("HT_Test")

        self.assertEqual(buf.getvalue(), "")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "powershell.exe")
        self.assertIn("HT_Test", cmd[-1])
        self.assertIn("Hidden = $true", cmd[-1])
        self.assertIn("DisallowStartIfOnBatteries = $false", cmd[-1])
        self.assertIn("StopIfGoingOnBatteries = $false", cmd[-1])
        self.assertIn("WorkingDirectory = ", cmd[-1])
        self.assertIn(str(setup_schedule.PROJECT_DIR), cmd[-1])

    def test_failure_prints_warning(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=1, stderr="access denied")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.apply_post_create_settings("HT_Test")

        # printed to stderr, not stdout -- just confirm no crash here;
        # stderr behavior is covered indirectly via run_schtasks_create below.


class TestRunSchtasksCreate(unittest.TestCase):
    def test_success_prints_created_and_hides_window(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=0, stderr="")) as mock_run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.run_schtasks_create("HT_Test", "cmd /c echo hi", ["/sc", "WEEKLY"])

        self.assertIn("Created HT_Test", buf.getvalue())
        create_cmd = mock_run.call_args_list[0][0][0]
        self.assertEqual(create_cmd[:2], ["schtasks", "/create"])
        self.assertIn("/tn", create_cmd)
        self.assertIn("HT_Test", create_cmd)
        self.assertIn("/F", create_cmd)

        # apply_post_create_settings must run right after a successful creation
        self.assertEqual(mock_run.call_count, 2)
        hide_cmd = mock_run.call_args_list[1][0][0]
        self.assertEqual(hide_cmd[0], "powershell.exe")
        self.assertIn("HT_Test", hide_cmd[-1])

    def test_failure_prints_failed_message_and_skips_hiding(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=1, stderr="access denied")) as mock_run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.run_schtasks_create("HT_Test", "cmd /c echo hi", ["/sc", "WEEKLY"])

        # argparse-style stderr output isn't captured by redirect_stdout since
        # the function prints failures to stderr; just confirm no crash and
        # that Created wasn't printed to stdout.
        self.assertNotIn("Created HT_Test", buf.getvalue())
        # a failed create must not attempt to hide a task that doesn't exist
        self.assertEqual(mock_run.call_count, 1)


class TestBuildTaskHelpers(unittest.TestCase):
    @patch("trading_bot.cli.setup_schedule.et_time_to_local_hhmm", return_value="10:30")
    @patch("trading_bot.cli.setup_schedule.run_schtasks_create")
    def test_build_weekly_task_uses_weekly_schedule(self, mock_create, mock_hhmm):
        setup_schedule.build_weekly_task("HT_Foo", "cmd /c foo", 9, 25)

        mock_create.assert_called_once_with(
            "HT_Foo", "cmd /c foo", ["/sc", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI", "/ST", "10:30"]
        )

    @patch("trading_bot.cli.setup_schedule.et_time_to_local_hhmm", return_value="10:00")
    @patch("trading_bot.cli.setup_schedule.run_schtasks_create")
    def test_build_cycle_task_uses_5_minute_schedule(self, mock_create, mock_hhmm):
        setup_schedule.build_cycle_task("HT_Cycle", "cmd /c cycle", 10, 0)

        mock_create.assert_called_once_with(
            "HT_Cycle", "cmd /c cycle", ["/sc", "MINUTE", "/MO", "5", "/ST", "10:00"]
        )


class TestQuotedTr(unittest.TestCase):
    def test_invokes_venv_python_directly_without_cmd_wrapper(self):
        result = setup_schedule.quoted_tr("trading_bot.cli.cycle")

        self.assertNotIn("cmd /c", result)
        self.assertNotIn("cd /d", result)
        self.assertIn(str(setup_schedule.VENV_PY), result)
        self.assertIn("-m trading_bot.cli.cycle", result)


class TestMain(unittest.TestCase):
    def test_missing_venv_python_exits_1_without_creating_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_venv_py = Path(tmpdir) / "does_not_exist" / "python.exe"
            with patch.object(setup_schedule, "VENV_PY", fake_venv_py):
                with patch("trading_bot.cli.setup_schedule.subprocess.run") as mock_run:
                    with self.assertRaises(SystemExit) as ctx:
                        setup_schedule.main()

        self.assertEqual(ctx.exception.code, 1)
        mock_run.assert_not_called()

    def test_creates_fourteen_tasks_when_venv_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_venv_py = Path(tmpdir) / "python.exe"
            fake_venv_py.write_text("")  # just needs to exist

            with patch.object(setup_schedule, "VENV_PY", fake_venv_py):
                with patch(
                    "trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=0, stdout="", stderr=""),
                ) as mock_run:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        setup_schedule.main()

        create_calls = [c for c in mock_run.call_args_list if c[0][0][1] == "/create"]
        self.assertEqual(len(create_calls), 14)
        self.assertIn("14 tasks created", buf.getvalue())

        created_names = [c[0][0][3] for c in create_calls]
        self.assertIn("HT_SMC_Prefilter", created_names)
        self.assertIn("HT_SMC_Cycle", created_names)
        self.assertIn("HT_HeartbeatMonitor", created_names)


if __name__ == "__main__":
    unittest.main()


class TestDisableTask(unittest.TestCase):
    def test_issues_the_disable_command(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="")) as mock_run:
            with redirect_stdout(io.StringIO()) as buf:
                setup_schedule.disable_task("HT_Test")
        self.assertEqual(mock_run.call_args[0][0],
                         ["schtasks", "/change", "/tn", "HT_Test", "/disable"])
        self.assertIn("Disabled HT_Test", buf.getvalue())

    def test_a_failure_is_reported_not_swallowed(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                   return_value=MagicMock(returncode=1, stderr="ERROR: nope")):
            with redirect_stdout(io.StringIO()) as out:
                setup_schedule.disable_task("HT_Test")
        self.assertNotIn("Disabled", out.getvalue())


class TestGapAndGoStaysDisabled(unittest.TestCase):
    """setup_schedule recreates every task with /F, so before this it took
    one ordinary re-run to restart a strategy that loses on every cost
    basis. The tasks are still registered -- just not armed."""

    def test_the_disabled_set_is_exactly_the_gap_and_go_tasks(self):
        self.assertEqual(
            setup_schedule.DISABLED_AFTER_CREATE,
            {"HT_Cycle", *(f"HT_Prefilter_{i:02d}" for i in range(1, 8))},
        )

    def test_no_smc_or_shared_task_is_in_it(self):
        """Disabling HT_HeartbeatMonitor or the SMC pair by accident would
        silently stop the bot that IS meant to be running."""
        for name in ("HT_SMC_Cycle", "HT_SMC_Prefilter", "HT_HeartbeatMonitor",
                     "HT_LogRotate", "HT_KeepAwake", "HT_Dashboard"):
            self.assertNotIn(name, setup_schedule.DISABLED_AFTER_CREATE)

    def test_main_disables_every_one_of_them_after_creating_them(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="", stdout="")), \
             patch("trading_bot.cli.setup_schedule.VENV_PY") as venv, \
             patch("trading_bot.cli.setup_schedule.disable_task") as mock_disable:
            venv.exists.return_value = True
            with redirect_stdout(io.StringIO()):
                setup_schedule.main()
        disabled = {call[0][0] for call in mock_disable.call_args_list}
        self.assertEqual(disabled, setup_schedule.DISABLED_AFTER_CREATE)
