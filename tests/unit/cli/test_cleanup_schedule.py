"""Unit tests for trading_bot.cli.cleanup_schedule. subprocess.run is
always mocked -- no real schtasks/powercfg calls."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from trading_bot.cli import cleanup_schedule


class TestListHtTasks(unittest.TestCase):
    def test_query_failure_returns_empty_list(self):
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="access denied"),
        ):
            result = cleanup_schedule.list_ht_tasks()

        self.assertEqual(result, [])

    def test_filters_and_dedupes_ht_prefixed_tasks(self):
        csv_output = (
            '"\\HT_Cycle","Ready"\r\n'
            '"\\OtherTask","Ready"\r\n'
            '"\\HT_Prefilter_01","Ready"\r\n'
            '"\\HT_Cycle","Ready"\r\n'
        )
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=csv_output, stderr=""),
        ):
            result = cleanup_schedule.list_ht_tasks()

        self.assertEqual(result, ["HT_Cycle", "HT_Prefilter_01"])

    def test_empty_output_returns_empty_list(self):
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        ):
            result = cleanup_schedule.list_ht_tasks()

        self.assertEqual(result, [])


class TestDeleteTask(unittest.TestCase):
    def test_success_returns_true(self):
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ) as mock_run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = cleanup_schedule.delete_task("HT_Cycle")

        self.assertTrue(result)
        self.assertIn("Deleted HT_Cycle", buf.getvalue())
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd, ["schtasks", "/delete", "/tn", "HT_Cycle", "/f"])

    def test_failure_returns_false(self):
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="task not found"),
        ):
            result = cleanup_schedule.delete_task("HT_Ghost")

        self.assertFalse(result)


class TestRestorePowerSettings(unittest.TestCase):
    def test_calls_powercfg_with_default_timeout(self):
        with patch(
            "trading_bot.cli.cleanup_schedule.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ) as mock_run:
            cleanup_schedule.restore_power_settings()

        mock_run.assert_called_once_with(
            ["powercfg", "/change", "standby-timeout-ac", "30"],
            check=False, capture_output=True, text=True,
        )


class TestMain(unittest.TestCase):
    def test_no_tasks_prints_nothing_to_clean_and_exits_0(self):
        with patch("trading_bot.cli.cleanup_schedule.list_ht_tasks", return_value=[]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    cleanup_schedule.main()

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("Nothing to clean.", buf.getvalue())

    def test_deletes_all_tasks_and_restores_power(self):
        with patch("trading_bot.cli.cleanup_schedule.list_ht_tasks", return_value=["HT_Cycle", "HT_Prefilter_01"]):
            with patch("trading_bot.cli.cleanup_schedule.delete_task", return_value=True) as mock_delete:
                with patch("trading_bot.cli.cleanup_schedule.restore_power_settings") as mock_restore:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        cleanup_schedule.main()

        self.assertEqual(mock_delete.call_count, 2)
        mock_restore.assert_called_once()
        self.assertIn("CLEANUP DONE: 2 tasks deleted", buf.getvalue())

    def test_partial_delete_failure_still_reports_correct_count(self):
        with patch("trading_bot.cli.cleanup_schedule.list_ht_tasks", return_value=["HT_Cycle", "HT_Ghost"]):
            with patch("trading_bot.cli.cleanup_schedule.delete_task", side_effect=[True, False]):
                with patch("trading_bot.cli.cleanup_schedule.restore_power_settings"):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        cleanup_schedule.main()

        self.assertIn("CLEANUP DONE: 1 tasks deleted", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
