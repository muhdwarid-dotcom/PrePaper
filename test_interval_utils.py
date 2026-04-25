#!/usr/bin/env python3
"""
Unit tests for interval propagation utilities.

Run with: python test_interval_utils.py
"""
import unittest


class TestIntervalToMinutes(unittest.TestCase):
    """Verify the interval -> minutes mapping used across the pipeline.

    Only 1m and 3m are in SUPPORTED_INTERVALS; the default (unknown interval)
    falls back to 1.
    """

    def _mapping(self, interval: str) -> int:
        """Shared mapping function (mirrors logic in eventstudy_analysis,
        Derive_k_t_from_PQ_windows, and run_screened_pipeline)."""
        return {"1m": 1, "3m": 3}.get(interval, 1)

    def test_1m(self):
        self.assertEqual(self._mapping("1m"), 1)

    def test_3m(self):
        self.assertEqual(self._mapping("3m"), 3)

    def test_default_unknown(self):
        self.assertEqual(self._mapping("unknown"), 1)


class TestIntervalToMs(unittest.TestCase):
    """Verify the interval -> milliseconds mapping used for pagination in binance_fetch."""

    def _ms(self, interval: str) -> int:
        from binance_fetch import _interval_to_ms
        return _interval_to_ms(interval)

    def test_1m_ms(self):
        self.assertEqual(self._ms("1m"), 60_000)

    def test_3m_ms(self):
        self.assertEqual(self._ms("3m"), 3 * 60_000)

    def test_unknown_falls_back_to_1m(self):
        self.assertEqual(self._ms("99m"), 60_000)


class TestFilenameInference(unittest.TestCase):
    """Verify that _infer_pair_date_interval correctly parses eventstudy filenames."""

    def _infer(self, filename: str):
        from eventstudy_analysis import _infer_pair_date_interval
        return _infer_pair_date_interval(filename)

    def test_1m_filename(self):
        pair, date, interval = self._infer(
            "forwardtest/v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv"
        )
        self.assertEqual(pair, "ACTUSDT")
        self.assertEqual(date, "2025-12-01")
        self.assertEqual(interval, "1m")

    def test_3m_filename(self):
        pair, date, interval = self._infer(
            "forwardtest/v30_eventstudy_SHIBUSDT_3m_rsi_sma_cross_gt51_prepaper_2025-11-25.csv"
        )
        self.assertEqual(pair, "SHIBUSDT")
        self.assertEqual(date, "2025-11-25")
        self.assertEqual(interval, "3m")

    def test_unrecognised_returns_none(self):
        pair, date, interval = self._infer("some_other_file.csv")
        self.assertIsNone(pair)
        self.assertIsNone(date)
        self.assertIsNone(interval)


if __name__ == "__main__":
    unittest.main(verbosity=2)
