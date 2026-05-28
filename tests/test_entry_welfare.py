"""Tests for entry and welfare modules."""
import numpy as np
import pandas as pd
import pytest

from merger_simulator.entry import entry_probability
from merger_simulator.welfare import hhi, consumer_surplus_logit, consumer_surplus_change_approx


class TestEntryProbability:
    def test_zero_profit(self):
        assert entry_probability(0.0, mu=2.9, sigma=0.6) == 0.0

    def test_negative_profit(self):
        assert entry_probability(-100.0, mu=2.9, sigma=0.6) == 0.0

    def test_high_profit(self):
        prob = entry_probability(1e6, mu=2.9, sigma=0.6)
        assert prob > 0.99

    def test_moderate_profit(self):
        prob = entry_probability(np.exp(2.9), mu=2.9, sigma=0.6)
        assert abs(prob - 0.5) < 0.01

    def test_vector_input(self):
        profits = np.array([0, np.exp(2.9), 1e6])
        probs = entry_probability(profits, mu=2.9, sigma=0.6)
        assert probs[0] == 0.0
        assert abs(probs[1] - 0.5) < 0.01
        assert probs[2] > 0.99

    def test_monotonic(self):
        profits = np.linspace(1, 1000, 50)
        probs = entry_probability(profits, mu=2.9, sigma=0.6)
        assert all(probs[i] <= probs[i+1] for i in range(len(probs)-1))


class TestHHI:
    def test_monopoly(self):
        assert hhi(np.array([1.0]), np.array(["A"])) == 10000.0

    def test_symmetric_duopoly(self):
        result = hhi(np.array([0.5, 0.5]), np.array(["A", "B"]))
        assert abs(result - 5000.0) < 1.0

    def test_symmetric_four_firms(self):
        result = hhi(np.array([0.25, 0.25, 0.25, 0.25]),
                     np.array(["A", "B", "C", "D"]))
        assert abs(result - 2500.0) < 1.0

    def test_merger_increases_hhi(self):
        shares = np.array([0.3, 0.3, 0.2, 0.2])
        firms = np.array(["A", "B", "C", "D"])
        hhi_pre = hhi(shares, firms)
        hhi_post = hhi(shares, firms, ownership={"A": "AB", "B": "AB"})
        assert hhi_post > hhi_pre

    def test_merger_hhi_value(self):
        shares = np.array([0.3, 0.3, 0.2, 0.2])
        firms = np.array(["A", "B", "C", "D"])
        hhi_pre = hhi(shares, firms)
        hhi_post = hhi(shares, firms, ownership={"A": "AB", "B": "AB"})
        assert abs(hhi_pre - 2600.0) < 1.0
        assert abs(hhi_post - 4400.0) < 1.0


class TestConsumerSurplus:
    def test_logit_no_change(self):
        s = np.array([0.1, 0.1, 0.05])
        cs = consumer_surplus_logit(s, s, alpha=2.0)
        assert abs(cs) < 1e-10

    def test_logit_price_increase_reduces_cs(self):
        s_pre = np.array([0.15, 0.15, 0.10])
        s_post = np.array([0.10, 0.10, 0.08])
        cs = consumer_surplus_logit(s_pre, s_post, alpha=2.0)
        assert cs < 0

    def test_approx_no_change(self):
        p = np.array([2.0, 2.5, 3.0])
        q = np.array([100, 80, 60])
        cs = consumer_surplus_change_approx(p, p, q, q)
        assert abs(cs) < 1e-10

    def test_approx_price_increase_negative(self):
        p_pre = np.array([2.0, 2.5])
        p_post = np.array([2.5, 3.0])
        q = np.array([100, 80])
        cs = consumer_surplus_change_approx(p_pre, p_post, q, q)
        assert cs < 0
