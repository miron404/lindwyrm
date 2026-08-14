"""Cost accounting.

Cache reads and cache writes are not priced like fresh input -- a read is
much cheaper, a write usually dearer -- so they are counted separately or the
number would be fiction.
"""

import dataclasses
import pathlib
import unittest

from helpers import make_config

from lindwyrm.agent import Agent





def agent_with(**prices):
    a = Agent(make_config(**prices))
    a.total_fresh_input_tokens = 1_000_000
    a.total_cache_read_tokens = 1_000_000
    a.total_cache_write_tokens = 1_000_000
    a.total_output_tokens = 1_000_000
    return a


class TestNoPrices(unittest.TestCase):
    def test_cost_is_none_without_prices(self):
        """Nothing is invented: providers change prices, and a wrong number
        is worse than no number."""
        self.assertIsNone(agent_with().session_cost())

    def test_has_prices_is_false(self):
        self.assertFalse(agent_with().has_prices())

    def test_one_price_is_enough_to_switch_it_on(self):
        self.assertTrue(agent_with(price_input=1.0).has_prices())


class TestArithmetic(unittest.TestCase):
    def test_prices_are_per_million_tokens(self):
        a = Agent(make_config(price_input=3.0))
        a.total_fresh_input_tokens = 1_000_000
        self.assertAlmostEqual(a.session_cost(), 3.0)

    def test_each_tier_is_billed_at_its_own_rate(self):
        a = agent_with(price_input=10.0, price_output=20.0,
                       price_cache_read=1.0, price_cache_write=12.0)
        # 10 + 1 + 12 + 20, one million tokens in each bucket.
        self.assertAlmostEqual(a.session_cost(), 43.0)

    def test_cache_read_is_cheaper_than_fresh_input(self):
        cached = Agent(make_config(price_input=10.0, price_cache_read=1.0))
        cached.total_cache_read_tokens = 1_000_000
        fresh = Agent(make_config(price_input=10.0, price_cache_read=1.0))
        fresh.total_fresh_input_tokens = 1_000_000
        self.assertLess(cached.session_cost(), fresh.session_cost())

    def test_unset_cache_prices_fall_back_to_the_input_price(self):
        """Conservative: better to overstate than to bill part of the context
        at zero and under-report what a session really cost."""
        a = agent_with(price_input=10.0)
        self.assertAlmostEqual(a.session_cost(), 30.0)  # fresh + read + write

    def test_zero_usage_costs_nothing(self):
        self.assertAlmostEqual(Agent(make_config(price_input=5.0)).session_cost(), 0.0)


class TestTokenSplit(unittest.TestCase):
    """input_tokens is the whole context; the cached parts are priced apart."""

    class Handler:
        def __init__(self, total, read=0, write=0, out=0):
            self.input_tokens = total
            self.cache_read_tokens = read
            self.cache_write_tokens = write
            self.output_tokens = out
            self.content = [{"type": "text", "text": "done"}]

    def run_once(self, handler):
        a = Agent(make_config(price_input=1.0))
        a._call_model = lambda *args, **kw: handler
        a.add_user("hi")
        a.run_turn(on_text=lambda _: None)
        return a

    def test_fresh_input_excludes_the_cached_parts(self):
        a = self.run_once(self.Handler(total=1000, read=600, write=100, out=50))
        self.assertEqual(a.total_fresh_input_tokens, 300)
        self.assertEqual(a.total_cache_read_tokens, 600)
        self.assertEqual(a.total_cache_write_tokens, 100)

    def test_a_fully_cached_turn_bills_no_fresh_input(self):
        a = self.run_once(self.Handler(total=500, read=500))
        self.assertEqual(a.total_fresh_input_tokens, 0)

    def test_inconsistent_numbers_never_go_negative(self):
        """A provider reporting cache tokens outside input_tokens would
        otherwise produce a negative bill."""
        a = self.run_once(self.Handler(total=100, read=900))
        self.assertEqual(a.total_fresh_input_tokens, 0)

    def test_totals_accumulate_across_turns(self):
        a = Agent(make_config(price_input=1.0))
        a._call_model = lambda *args, **kw: self.Handler(total=100, out=10)
        for _ in range(3):
            a.add_user("hi")
            a.run_turn(on_text=lambda _: None)
        self.assertEqual(a.total_output_tokens, 30)
        self.assertEqual(a.total_fresh_input_tokens, 300)


class TestMoneyFormatting(unittest.TestCase):
    def test_small_amounts_keep_their_digits(self):
        from lindwyrm.cli import _money
        # Rounding a fraction of a cent to two places would show 0.00 for
        # every turn of a cheap model.
        self.assertNotEqual(_money(0.000123), "0.00")
        self.assertEqual(_money(12.3456), "12.35")


if __name__ == "__main__":
    unittest.main()
