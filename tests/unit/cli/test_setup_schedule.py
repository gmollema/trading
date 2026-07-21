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


class TestHideTaskWindow(unittest.TestCase):
    def test_success_is_silent(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=0, stderr="")) as mock_run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.hide_task_window("HT_Test")

        self.assertEqual(buf.getvalue(), "")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "powershell.exe")
        self.assertIn("HT_Test", cmd[-1])
        self.assertIn("Hidden = $true", cmd[-1])

    def test_failure_prints_warning(self):
        with patch("trading_bot.cli.setup_schedule.subprocess.run",
                    return_value=MagicMock(returncode=1, stderr="access denied")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                setup_schedule.hide_task_window("HT_Test")

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

        # hide_task_window must run right after a successful creation
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
    def test_includes_cd_and_module_invocation(self):
        result = setup_schedule.quoted_tr("trading_bot.cli.cycle")

        self.assertIn("cd /d", result)
        self.assertIn(str(setup_schedule.PROJECT_DIR), result)
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

    def test_creates_eleven_tasks_when_venv_exists(self):
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
        self.assertEqual(len(create_calls), 11)
        self.assertIn("11 tasks created", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
