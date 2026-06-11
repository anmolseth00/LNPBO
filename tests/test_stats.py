"""Tests for benchmarks.stats statistical functions.

Each test verifies against known analytical values or published tables,
not just structural properties of the output.
"""

import numpy as np
import pytest

from LNPBO.benchmarks.stats import (
    acceleration_factor,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d_paired,
    cumulative_regret,
    enhancement_factor,
    hierarchical_bootstrap_ci,
    higgins_heterogeneity,
    paired_wilcoxon,
    post_hoc_power,
    prospective_power,
    rank_biserial,
    simple_regret,
    win_rate_ci,
)


class TestProspectivePower:
    """Verify against noncentral t distribution (paired one-sample t-test).

    Cohen (1988) Table 2.4.1 for paired/one-sample design.
    """

    def test_known_value_n20_d05(self):
        # Paired t-test, n=20, d=0.5: ncp = 0.5*sqrt(20) = 2.236, df=19
        # Expected power ~0.564 (noncentral t)
        result = prospective_power(20, effect_sizes=(0.5,))
        assert result[0.5] == pytest.approx(0.564, abs=0.01)

    def test_large_effect_high_power(self):
        # n=50, d=0.8: ncp = 0.8*sqrt(50) = 5.66 -> power ~1.0
        result = prospective_power(50, effect_sizes=(0.8,))
        assert result[0.8] > 0.99

    def test_monotonicity_in_n(self):
        p10 = prospective_power(10, effect_sizes=(0.5,))[0.5]
        p30 = prospective_power(30, effect_sizes=(0.5,))[0.5]
        p50 = prospective_power(50, effect_sizes=(0.5,))[0.5]
        assert p10 < p30 < p50

    def test_degenerate_cases(self):
        assert prospective_power(1, effect_sizes=(0.5,))[0.5] == 0.0
        assert prospective_power(20, effect_sizes=(0.0,))[0.0] == 0.0


class TestHigginsHeterogeneity:
    """Verify Q, I², and tau² against hand calculations."""

    def test_known_values_with_standard_errors(self):
        # 3 studies: effects [0.2, 0.5, 0.8], SE=0.1 each
        # w_i = 1/0.01 = 100 for all; mu_hat = mean = 0.5
        # Q = sum(100*(y_i - 0.5)^2) = 100*(0.09 + 0 + 0.09) = 18.0
        # df = 2; I2 = max(0, (Q-df)/Q * 100) = (18-2)/18 * 100 = 88.89%
        effects = [0.2, 0.5, 0.8]
        ses = [0.1, 0.1, 0.1]
        I2, tau2, Q, Q_p = higgins_heterogeneity(effects, study_ses=ses)
        assert pytest.approx(18.0, abs=0.01) == Q
        assert pytest.approx(88.89, abs=0.1) == I2
        assert tau2 > 0
        assert Q_p < 0.001  # highly significant heterogeneity

    def test_no_heterogeneity(self):
        # Identical effects with equal weights -> Q=0, I2=0
        I2, _tau2, Q, _Qp = higgins_heterogeneity([1.0, 1.0, 1.0, 1.0])
        assert I2 == 0.0
        assert Q == 0.0

    def test_fewer_than_two_studies(self):
        I2, _tau2, _Q, Q_p = higgins_heterogeneity([0.5])
        assert I2 == 0.0
        assert Q_p == 1.0


class TestBenjaminiHochberg:
    """Verify BH step-up procedure against hand-calculated values."""

    def test_known_adjusted_values(self):
        # p-values: [0.01, 0.03, 0.04, 0.20], m=4 (already sorted)
        # Raw adjusted: p[i] * m / rank[i]
        #   rank 1: 0.01 * 4/1 = 0.04
        #   rank 2: 0.03 * 4/2 = 0.06
        #   rank 3: 0.04 * 4/3 = 0.05333
        #   rank 4: 0.20 * 4/4 = 0.20
        # Enforce monotonicity (bottom-up cumulative min):
        #   [0.04, min(0.06, 0.05333), 0.05333, 0.20] = [0.04, 0.05333, 0.05333, 0.20]
        pvals = [0.01, 0.03, 0.04, 0.20]
        adj, rej = benjamini_hochberg(pvals, alpha=0.05)
        assert adj[0] == pytest.approx(0.04, abs=1e-10)
        assert adj[1] == pytest.approx(4 * 0.04 / 3, abs=1e-10)  # monotonicity
        assert adj[2] == pytest.approx(4 * 0.04 / 3, abs=1e-10)
        assert adj[3] == pytest.approx(0.20, abs=1e-10)
        # At alpha=0.05: first two rejected (adj < 0.05)
        assert rej[0] is np.True_
        assert rej[1] is np.False_  # 0.0533 >= 0.05

    def test_empty_input(self):
        adj, _rej = benjamini_hochberg([])
        assert len(adj) == 0

    def test_all_significant(self):
        _adj, rej = benjamini_hochberg([0.001, 0.002, 0.003])
        assert all(rej)

    def test_preserves_order_invariance(self):
        # Same p-values in different order should give same adjusted values
        # when mapped back to original positions
        p1 = [0.05, 0.01, 0.10]
        p2 = [0.01, 0.10, 0.05]
        adj1, _ = benjamini_hochberg(p1)
        adj2, _ = benjamini_hochberg(p2)
        # adj1[1] (original p=0.01) should equal adj2[0] (original p=0.01)
        assert adj1[1] == pytest.approx(adj2[0])


class TestBootstrapCI:
    """Percentile bootstrap (Efron & Tibshirani 1993, Ch. 13)."""

    def test_constant_values_collapse(self):
        # All-equal input → every resample has the same mean → CI = (c, c).
        lo, hi = bootstrap_ci([4.0, 4.0, 4.0, 4.0])
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(4.0)

    def test_ci_brackets_point_estimate(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(vals)
        assert lo <= np.mean(vals) <= hi
        assert lo < hi

    def test_deterministic(self):
        # Fixed internal seed=42 → reproducible across calls.
        vals = [0.1, 0.3, 0.2, 0.9, 0.5]
        assert bootstrap_ci(vals) == bootstrap_ci(vals)


class TestPairedWilcoxon:
    """Wraps scipy.stats.wilcoxon; verify against exact small-sample p-values."""

    def test_all_equal_returns_one(self):
        assert paired_wilcoxon([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 1.0

    def test_all_positive_diff_exact_p(self):
        # n=8 differences all +1 → W-=0, two-sided p = 2 / 2**8 = 0.0078125.
        a = list(range(1, 9))
        b = list(range(0, 8))
        assert paired_wilcoxon(a, b) == pytest.approx(0.0078125, abs=1e-6)


class TestCohensDPaired:
    """d = mean(diff) / std(diff, ddof=1); Cohen (1988) thresholds."""

    def test_known_value_and_interpretation(self):
        # diff = [1,2,3,4,5]: mean=3, std(ddof=1)=sqrt(2.5) → d≈1.897 (large).
        d, lo, hi, interp = cohens_d_paired([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        assert d == pytest.approx(1.8973666, abs=1e-5)
        assert interp == "large"
        assert lo < d < hi

    def test_zero_variance_diff_is_zero(self):
        # Constant non-zero difference → std=0 → guarded to d=0.
        d, _, _, interp = cohens_d_paired([5, 6, 7], [4, 5, 6])
        assert d == 0.0
        assert interp == "negligible"


class TestRankBiserial:
    """Kerby (2014) simple difference formula; exact at the extremes."""

    def test_all_positive_is_plus_one(self):
        assert rank_biserial([2, 3, 4, 5], [1, 1, 1, 1]) == pytest.approx(1.0)

    def test_all_negative_is_minus_one(self):
        assert rank_biserial([1, 1, 1, 1], [2, 3, 4, 5]) == pytest.approx(-1.0)

    def test_no_difference_is_zero(self):
        assert rank_biserial([1, 2, 3], [1, 2, 3]) == 0.0


class TestPostHocPower:
    """Noncentral-t power approximation for a paired design."""

    def test_degenerate_returns_zero(self):
        # Guarded: zero effect or n < 2 returns 0.0.
        assert post_hoc_power(0.0, 26) == 0.0
        assert post_hoc_power(0.5, 1) == 0.0

    def test_monotonic_in_effect_and_n(self):
        assert post_hoc_power(0.8, 26) > post_hoc_power(0.2, 26)
        assert post_hoc_power(0.5, 50) > post_hoc_power(0.5, 10)

    def test_large_effect_and_n_high_power(self):
        assert post_hoc_power(1.0, 50) > 0.99


class TestRegret:
    """Simple and cumulative regret are exact arithmetic."""

    def test_simple_regret(self):
        np.testing.assert_allclose(simple_regret([1, 2, 3], 5.0), [4, 3, 2])

    def test_cumulative_regret(self):
        # per-round regret [4,3,2] → cumsum [4,7,9].
        np.testing.assert_allclose(cumulative_regret([1, 2, 3], 5.0), [4, 7, 9])


class TestAccelerationEnhancement:
    def test_acceleration_factor_basic(self):
        # Both reach recall 0.5; random needs 40 evals, BO needs 10 → AF=4.
        rand = {10: 0.1, 20: 0.2, 40: 0.5}
        bo = {10: 0.5, 20: 0.7, 40: 0.9}
        assert acceleration_factor(rand, bo, 0.5) == pytest.approx(4.0)

    def test_acceleration_random_never_reaches_is_inf(self):
        rand = {10: 0.1, 20: 0.2}
        bo = {10: 0.9}
        assert acceleration_factor(rand, bo, 0.5) == float("inf")

    def test_acceleration_bo_never_reaches_is_nan(self):
        rand = {10: 0.9}
        bo = {10: 0.1, 20: 0.2}
        assert np.isnan(acceleration_factor(rand, bo, 0.5))

    def test_enhancement_factor(self):
        assert enhancement_factor(2.0, 3.0) == pytest.approx(1.5)
        assert np.isnan(enhancement_factor(0.0, 3.0))


class TestWinRateCI:
    """Fractional-win bootstrap across studies."""

    def test_dominant_family_wins_all(self):
        # A is strictly best in every study → win_rate 1.0 with degenerate CI.
        data = {"A": [0.9, 0.8, 0.95], "B": [0.1, 0.2, 0.3]}
        res = win_rate_ci(data, n_bootstrap=2000)
        assert res["A"][0] == pytest.approx(1.0)
        assert res["B"][0] == pytest.approx(0.0)
        assert res["A"][1] == pytest.approx(1.0) and res["A"][2] == pytest.approx(1.0)

    def test_exact_ties_split_fractionally(self):
        # Identical families tie in every study → 0.5 fractional win each.
        data = {"A": [0.5, 0.6, 0.7], "B": [0.5, 0.6, 0.7]}
        res = win_rate_ci(data, n_bootstrap=2000)
        assert res["A"][0] == pytest.approx(0.5)
        assert res["B"][0] == pytest.approx(0.5)


class TestHierarchicalBootstrapCI:
    """Two-level (study, seed) percentile bootstrap (Davison & Hinkley 1997)."""

    def test_grand_mean_is_mean_of_study_means(self):
        # Study means 1 and 3 → grand mean 2.0 (unweighted by seed count).
        mean, lo, hi = hierarchical_bootstrap_ci([[1.0, 1.0], [3.0, 3.0]], n_bootstrap=2000)
        assert mean == pytest.approx(2.0)
        assert lo <= 2.0 <= hi

    def test_constant_data_collapses(self):
        mean, lo, hi = hierarchical_bootstrap_ci([[7.0, 7.0], [7.0, 7.0, 7.0]], n_bootstrap=1000)
        assert mean == pytest.approx(7.0)
        assert lo == pytest.approx(7.0)
        assert hi == pytest.approx(7.0)
