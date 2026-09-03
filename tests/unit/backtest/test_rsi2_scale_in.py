"""Unit tests for rsi2_signals.find_rsi2_scale_in_trades.

The fixture is shaped like the real strategy rather than hand-minimised,
because the strategy's two conditions fight each other at short SMA
windows: RSI(2) only reaches single digits on a drop roughly 9x the
average daily gain, which at a 21-bar SMA is the same size as the whole
gap between price and its average -- so the dip that fires the signal
also breaks the trend filter. A 200-bar SMA under a long trend leaves
price far above its average, which is what lets small dips fire the
signal at all. SCALE_CLOSES therefore rises for 200 bars, settles into
half-point oscillation to bring the RSI averages down, then dips three
times with a small bounce between each.

Verified against the module's own indicators before being encoded:
signals land on bars 205, 207 and 209 (RSI 6.78 / 7.42 / 7.62, every
close ~130 points above its SMA), then bar 210 closes at RSI 95.18 and
takes the whole stack out.
"""

import unittest

from trading_bot.backtest import rsi2_signals as rsi2


def build_closes():
    c = [100 + 1.5 * k for k in range(200)]
    for d in (0.5, -0.5, 0.5, -0.5, 0.5):
        c.append(c[-1] + d)
    c.append(c[-1] - 5)     # 205: dip 1, RSI 6.78 -> first position
    c.append(c[-1] + 0.5)   # 206: RSI back to 21.05, no exit
    c.append(c[-1] - 3)     # 207: dip 2, RSI 7.42 -> add
    c.append(c[-1] + 0.5)   # 208: RSI 23.86
    c.append(c[-1] - 3)     # 209: dip 3, RSI 7.62 -> add
    c.append(c[-1] + 40)    # 210: RSI 95.18 -> stack exits
    return c


SCALE_CLOSES = build_closes()
SIGNAL_IDXS = [205, 207, 209]
EXIT_IDX = 210
SMA = 200


def make_bars(closes, opens=None, lows=None):
    n = len(closes)
    opens = list(closes) if opens is None else opens
    lows = [min(o, c) for o, c in zip(opens, closes)] if lows is None else lows
    return {"date": list(range(n)), "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": lows, "close": closes}


def scale(closes=None, **kw):
    kw.setdefault("sma_period", SMA)
    return rsi2.find_rsi2_scale_in_trades(make_bars(closes or SCALE_CLOSES), **kw)


class TestFixture(unittest.TestCase):
    def test_signals_land_where_the_docstring_says(self):
        sig = rsi2.rsi2_entry_signals(SCALE_CLOSES, 2, 10.0, SMA)
        self.assertEqual([i for i, s in enumerate(sig) if s], SIGNAL_IDXS)
        rsi = rsi2.wilder_rsi(SCALE_CLOSES, 2)
        self.assertGreater(rsi[EXIT_IDX], 70.0)


class TestScaleIn(unittest.TestCase):
    def test_adds_one_contract_per_fresh_crossing(self):
        out = scale(max_positions=3)
        self.assertEqual([p["entry_idx"] for p in out], SIGNAL_IDXS)
        self.assertEqual([p["position_num"] for p in out], [1, 2, 3])

    def test_max_positions_caps_the_stack(self):
        self.assertEqual([p["entry_idx"] for p in scale(max_positions=1)], SIGNAL_IDXS[:1])
        self.assertEqual([p["entry_idx"] for p in scale(max_positions=2)], SIGNAL_IDXS[:2])
        # Past the number of dips available, nothing changes -- the video's
        # own finding that positions 3 through 6 give identical results.
        for cap in (3, 4, 5, 6):
            self.assertEqual(len(scale(max_positions=cap)), 3)

    def test_whole_stack_exits_together(self):
        out = scale(max_positions=3)
        self.assertEqual({p["exit_idx"] for p in out}, {EXIT_IDX})
        self.assertEqual({p["reason"] for p in out}, {"rsi_exit"})
        self.assertEqual({p["campaign"] for p in out}, {0})

    def test_entry_and_exit_prices_are_the_closes(self):
        out = scale(max_positions=3)
        for pos, idx in zip(out, SIGNAL_IDXS):
            self.assertAlmostEqual(pos["entry_price"], SCALE_CLOSES[idx])
            self.assertAlmostEqual(pos["exit_price"], SCALE_CLOSES[EXIT_IDX])
        # Later entries are cheaper, which is the whole point of scaling in.
        self.assertGreater(out[0]["points"], 0)
        self.assertGreater(out[2]["points"], out[0]["points"])

    def test_consecutive_sub_level_closes_do_not_re_add(self):
        """A run of closes that stays under the level is ONE signal. The
        video is explicit that RSI must cross back above 10 before another
        position is taken, and a strict crossing already encodes that."""
        closes = SCALE_CLOSES[:206] + [SCALE_CLOSES[205] - 1, SCALE_CLOSES[205] - 2]
        out = scale(closes, max_positions=3)
        self.assertEqual(len(out), 1)

    def test_entry_timing_next_open_fills_on_the_following_bar(self):
        opens = [c - 0.25 for c in SCALE_CLOSES]
        bars = make_bars(SCALE_CLOSES, opens=opens)
        out = rsi2.find_rsi2_scale_in_trades(bars, sma_period=SMA, max_positions=3,
                                             entry_timing="next_open")
        self.assertEqual([p["entry_idx"] for p in out], [i + 1 for i in SIGNAL_IDXS])
        self.assertAlmostEqual(out[0]["entry_price"], opens[206])

    def test_exit_timing_next_open_needs_a_bar_to_fill_on(self):
        closes = SCALE_CLOSES + [SCALE_CLOSES[EXIT_IDX] + 1]
        opens = list(closes)
        opens[EXIT_IDX + 1] = 425.0
        bars = make_bars(closes, opens=opens)
        out = rsi2.find_rsi2_scale_in_trades(bars, sma_period=SMA, max_positions=3,
                                             exit_timing="next_open")
        self.assertEqual({p["exit_idx"] for p in out}, {EXIT_IDX + 1})
        self.assertEqual({p["exit_price"] for p in out}, {425.0})

    def test_per_position_stops_close_individually(self):
        # Position 1 enters at 394.0; a 1% stop sits at 390.06, which the
        # dip to 389 on bar 209 takes out while positions 2 and 3 survive.
        out = scale(max_positions=3, stop_pct=1.0)
        stopped = [p for p in out if p["reason"] == "stop_loss"]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["position_num"], 1)
        self.assertTrue(all(p["reason"] == "rsi_exit" for p in out if p is not stopped[0]))

    def test_open_stack_at_end_of_data_is_marked_to_the_last_close(self):
        closes = SCALE_CLOSES[:EXIT_IDX]  # drop the rally that would exit
        out = scale(closes, max_positions=3)
        self.assertEqual({p["reason"] for p in out}, {"end_of_data"})
        self.assertEqual({p["exit_price"] for p in out}, {closes[-1]})

    def test_mae_is_recorded_per_position(self):
        out = scale(max_positions=3)
        # Position 1 rode the two later dips down; position 3 bought the low.
        self.assertGreater(out[0]["mae_points"], out[2]["mae_points"])

    def test_rejects_bad_arguments(self):
        for kw in ({"max_positions": 0}, {"first_dip": 0},
                   {"entry_timing": "nope"}, {"exit_timing": "nope"}):
            with self.assertRaises(ValueError):
                scale(**kw)

    def test_empty_input(self):
        self.assertEqual(rsi2.find_rsi2_scale_in_trades(make_bars([])), [])


class TestFirstDip(unittest.TestCase):
    """The video's closing question: skip the early dips and buy only the
    deeper one. SCALE_CLOSES dips at 205, 207 and 209 within a single
    oversold sequence, so first_dip selects among exactly those."""

    def test_skips_earlier_dips_and_buys_the_nth(self):
        for dip, idx in enumerate(SIGNAL_IDXS, start=1):
            out = scale(max_positions=1, first_dip=dip)
            self.assertEqual([p["entry_idx"] for p in out], [idx],
                             f"first_dip={dip} should buy only bar {idx}")

    def test_a_dip_that_never_arrives_trades_nothing(self):
        self.assertEqual(scale(max_positions=1, first_dip=4), [])

    def test_first_dip_one_is_the_plain_strategy(self):
        self.assertEqual(scale(max_positions=1, first_dip=1),
                         scale(max_positions=1))

    def test_counter_resets_after_an_overbought_close(self):
        """A second oversold sequence starts counting from one again, so
        first_dip=1 buys the first dip of EACH sequence rather than only
        the first of the whole series."""
        # The RSI-95 rally leaves avg_gain near 20, so no single drop can
        # reach 10 again without breaking the trend filter -- three small
        # oscillation bars bring the averages down first, exactly as the
        # base fixture does before its own first dip.
        closes = list(SCALE_CLOSES)
        for j in range(3):
            closes.append(closes[-1] + (0.5 if j % 2 == 0 else -0.5))
        closes.append(closes[-1] - 26)
        second = len(closes) - 1
        rsi = rsi2.wilder_rsi(closes, 2)
        self.assertLess(rsi[second], 10.0)
        self.assertEqual([i for i, s in enumerate(rsi2.rsi2_entry_signals(closes, 2, 10.0, SMA)) if s],
                         SIGNAL_IDXS + [second])
        out = scale(closes, max_positions=1, first_dip=1)
        self.assertEqual([p["entry_idx"] for p in out], [SIGNAL_IDXS[0], second])
        # And the counter really did reset: dip 3 of the FIRST sequence is
        # still reachable, while there is no dip 3 in the second.
        self.assertEqual([p["entry_idx"] for p in scale(closes, max_positions=1, first_dip=3)],
                         [SIGNAL_IDXS[2]])

    def test_deeper_dip_buys_a_lower_price(self):
        firsts = [scale(max_positions=1, first_dip=d)[0]["entry_price"] for d in (1, 2, 3)]
        self.assertEqual(firsts, sorted(firsts, reverse=True))

    def test_first_dip_still_respects_max_positions(self):
        # Start at dip 2 and allow two contracts: dips 2 and 3 fill.
        out = scale(max_positions=2, first_dip=2)
        self.assertEqual([p["entry_idx"] for p in out], SIGNAL_IDXS[1:])
        self.assertEqual([p["position_num"] for p in out], [1, 2])


class TestEquivalenceWithSinglePosition(unittest.TestCase):
    """max_positions=1 must reduce to the already-tested single-position
    walk in RSI-exit mode. This cross-checks the new function against the
    old one rather than against my own reading of the rules."""

    def _walk(self, closes):
        bars = make_bars(closes)
        old = rsi2.find_rsi2_long_trades(
            bars, sma_period=SMA, stop_points=None,
            exit_mode=rsi2.EXIT_MODE_RSI, exit_timing="close")
        new = rsi2.find_rsi2_scale_in_trades(
            bars, sma_period=SMA, max_positions=1,
            entry_timing="next_open", exit_timing="close")
        key = lambda t: (t["entry_idx"], t["exit_idx"], round(t["entry_price"], 6),
                         round(t["exit_price"], 6), t["reason"])
        return [key(t) for t in old], [key(t) for t in new]

    def test_matches_on_the_fixture(self):
        old, new = self._walk(SCALE_CLOSES)
        self.assertEqual(old, new)
        self.assertTrue(old)

    def test_matches_on_a_long_noisy_series(self):
        closes = []
        price, state = 1000.0, 987654
        for _ in range(4000):
            state = (1103515245 * state + 12345) % (2 ** 31)
            price *= 1 + ((state / (2 ** 31)) - 0.49) * 0.02
            closes.append(price)
        old, new = self._walk(closes)
        self.assertGreater(len(old), 20)
        self.assertEqual(old, new)


class TestInvariantsOnNoisySeries(unittest.TestCase):
    @staticmethod
    def _series(n=4000, seed=24680):
        closes, price, state = [], 1000.0, seed
        for _ in range(n):
            state = (1103515245 * state + 12345) % (2 ** 31)
            price *= 1 + ((state / (2 ** 31)) - 0.49) * 0.02
            closes.append(price)
        return closes

    def test_campaign_structure_holds(self):
        out = scale(self._series(), max_positions=4)
        self.assertGreater(len(out), 40)
        by_campaign = {}
        for p in out:
            by_campaign.setdefault(p["campaign"], []).append(p)
        for positions in by_campaign.values():
            self.assertLessEqual(len(positions), 4)
            self.assertEqual([p["position_num"] for p in positions],
                             list(range(1, len(positions) + 1)))
            # No stops here, so one campaign is one shared exit.
            self.assertEqual(len({p["exit_idx"] for p in positions}), 1)
            for p in positions:
                self.assertLessEqual(p["entry_idx"], p["exit_idx"])

    def test_raising_the_cap_never_removes_contracts(self):
        closes = self._series()
        counts = [len(scale(closes, max_positions=k)) for k in range(1, 7)]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], counts[0])


if __name__ == "__main__":
    unittest.main()
