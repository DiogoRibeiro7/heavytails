"""Compound distributions, aggregate losses and reinsurance pricing.

Where an identity is available the tests assert it exactly rather than
approximately. There are more of those here than the subject usually allows:

- the aggregate mean and variance have closed forms that hold for any
  frequency and severity, so every gridded result can be checked against them;
- a Binomial(1, p) compound *is* the severity mixed with an atom at zero, so
  the recursion has a known answer, not just a plausible one;
- a per-loss severity with the full frequency and a per-payment severity with
  the thinned frequency describe the same aggregate, so the two must agree;
- excess-of-loss layers are additive, so pricing a to b and b to c must give
  the same answer as pricing a to c.

Where only an approximation is available -- the subexponential tail asymptotic,
the discretisation error -- the test asserts the *rate*: that the error falls
with the grid, that the ratio converges towards one. A single tolerance would
pass on a subtly wrong implementation that happened to land inside it.
"""

from __future__ import annotations

from itertools import pairwise
import math
import statistics

import pytest

from heavytails import LogNormal, Pareto, Weibull
from heavytails.actuarial import (
    AggregateLoss,
    Binomial,
    EmpiricalAggregate,
    LayeredSeverity,
    NegativeBinomial,
    Poisson,
    PolicyTerms,
    _numeric_lev,
    compound_moments,
    discretise_severity,
    excess_of_loss_premium,
    limited_expected_value,
    panjer_recursion,
    simulate_aggregate_loss,
)
from heavytails.extra_distributions import GeneralizedPareto
from heavytails.heavy_tails import RNG, ParameterError

FREQUENCIES = [
    Poisson(lam=2.0),
    NegativeBinomial(r=3.0, beta=1.0),
    Binomial(m=10, p=0.3),
]


# --------------------------- Frequency models -------------------------------- #


class TestFrequencyModels:
    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_the_mass_function_sums_to_one(self, frequency: object) -> None:
        total = sum(frequency.pmf(k) for k in range(400))  # type: ignore[attr-defined]
        assert total == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_the_mass_function_reproduces_the_stated_moments(
        self, frequency: object
    ) -> None:
        mass = [frequency.pmf(k) for k in range(400)]  # type: ignore[attr-defined]
        mean = sum(k * p for k, p in enumerate(mass))
        second = sum(k * k * p for k, p in enumerate(mass))
        assert mean == pytest.approx(frequency.mean(), rel=1e-10)  # type: ignore[attr-defined]
        assert second - mean**2 == pytest.approx(frequency.variance(), rel=1e-9)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    @pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 0.9])
    def test_the_generating_function_matches_its_own_series(
        self, frequency: object, s: float
    ) -> None:
        """``pgf(s) = sum s^k p_k``. Panjer starts from ``pgf(f_0)``.

        A wrong pgf would put the whole aggregate distribution on the wrong
        scale while leaving its shape intact, which is hard to see by eye.
        """
        series = sum(s**k * frequency.pmf(k) for k in range(400))  # type: ignore[attr-defined]
        assert frequency.pgf(s) == pytest.approx(series, rel=1e-10)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_the_recursion_pair_reproduces_the_mass_ratio(
        self, frequency: object
    ) -> None:
        """``p_k / p_{k-1} = a + b/k`` is the defining property of the class.

        Everything Panjer does rests on this, so it is asserted directly rather
        than inferred from the recursion working.
        """
        a, b = frequency.panjer_ab()  # type: ignore[attr-defined]
        for k in range(1, 12):
            previous = frequency.pmf(k - 1)  # type: ignore[attr-defined]
            if previous <= 0.0:
                continue
            ratio = frequency.pmf(k) / previous  # type: ignore[attr-defined]
            assert ratio == pytest.approx(a + b / k, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_thinning_scales_the_mean_and_stays_in_the_family(
        self, frequency: object
    ) -> None:
        thinned = frequency.thin(0.25)  # type: ignore[attr-defined]
        assert type(thinned) is type(frequency)
        assert thinned.mean() == pytest.approx(0.25 * frequency.mean())  # type: ignore[attr-defined]

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_sampling_reproduces_the_moments(self, frequency: object) -> None:
        draws = frequency.rvs(60_000, seed=3)  # type: ignore[attr-defined]
        mean = statistics.fmean(draws)
        error = math.sqrt(frequency.variance() / 60_000)  # type: ignore[attr-defined]
        assert abs(mean - frequency.mean()) < 4.0 * error  # type: ignore[attr-defined]
        assert statistics.pvariance(draws) == pytest.approx(
            frequency.variance(),  # type: ignore[attr-defined]
            rel=0.06,
        )

    def test_the_large_rate_poisson_sampler_is_correct(self) -> None:
        """Above 30 the sampler switches to transformed rejection.

        Different code path, so it needs its own check: the multiplication
        method's ``exp(-lam)`` factor underflows there and it would be far too
        slow anyway.
        """
        draws = Poisson(lam=250.0).rvs(40_000, seed=5)
        assert statistics.fmean(draws) == pytest.approx(250.0, rel=0.01)
        assert statistics.pvariance(draws) == pytest.approx(250.0, rel=0.05)

    def test_dispersion_orders_the_three_families(self) -> None:
        """Under-, equi- and over-dispersed, which is why all three exist."""
        assert Binomial(m=10, p=0.3).variance() < Binomial(m=10, p=0.3).mean()
        assert Poisson(lam=3.0).variance() == Poisson(lam=3.0).mean()
        assert NegativeBinomial(r=2.0, beta=1.5).variance() > (
            NegativeBinomial(r=2.0, beta=1.5).mean()
        )

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (Poisson, {"lam": 0.0}),
            (Poisson, {"lam": -1.0}),
            (NegativeBinomial, {"r": 0.0, "beta": 1.0}),
            (NegativeBinomial, {"r": 1.0, "beta": -2.0}),
            (Binomial, {"m": 0, "p": 0.5}),
            (Binomial, {"m": 5, "p": 1.0}),
        ],
    )
    def test_bad_parameters_are_rejected(self, factory: type, kwargs: dict) -> None:
        with pytest.raises(ParameterError):
            factory(**kwargs)

    def test_thinning_probability_is_validated(self) -> None:
        with pytest.raises(ValueError, match="Thinning probability"):
            Poisson(lam=1.0).thin(0.0)


# --------------------------- Limited expected value -------------------------- #


class TestLimitedExpectedValue:
    @pytest.mark.parametrize(
        "severity",
        [
            Pareto(alpha=2.0, xm=1.0),
            Pareto(alpha=1.2, xm=3.0),
            LogNormal(mu=0.5, sigma=1.2),
            Weibull(k=0.7, lam=2.0),
            GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0),
            GeneralizedPareto(xi=0.0, sigma=2.0, mu=1.0),
        ],
    )
    @pytest.mark.parametrize("d", [1.5, 4.0, 20.0])
    def test_the_closed_form_matches_quadrature(
        self, severity: object, d: float
    ) -> None:
        """The closed forms are per-family; the quadrature is not.

        So agreement is a genuine cross-check rather than two routes through the
        same algebra.
        """
        assert limited_expected_value(severity, d) == pytest.approx(
            _numeric_lev(severity, d, 40_000), rel=2e-5
        )

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (Pareto(alpha=2.0, xm=1.0), 2.0),
            (Pareto(alpha=3.0, xm=2.0), 3.0),
            (LogNormal(mu=0.0, sigma=1.0), math.exp(0.5)),
            (GeneralizedPareto(xi=0.25, sigma=1.0, mu=0.0), 4.0 / 3.0),
        ],
    )
    def test_censoring_at_infinity_gives_the_mean(
        self, severity: object, expected: float
    ) -> None:
        assert limited_expected_value(severity, math.inf) == pytest.approx(
            expected, rel=1e-10
        )

    @pytest.mark.parametrize("alpha", [0.5, 0.9, 1.0])
    def test_it_is_infinite_exactly_when_the_mean_is(self, alpha: float) -> None:
        """A Pareto with ``alpha <= 1`` has no mean, and this says so.

        The whole reason layer pricing goes through limited expected values is
        that a *bounded* layer on such a severity is still perfectly finite.
        """
        severity = Pareto(alpha=alpha, xm=1.0)
        assert limited_expected_value(severity, math.inf) == math.inf
        assert math.isfinite(limited_expected_value(severity, 1000.0))

    def test_it_is_bounded_by_the_censoring_point(self) -> None:
        severity = Pareto(alpha=1.5, xm=1.0)
        for d in [1.0, 2.0, 10.0, 1000.0]:
            assert limited_expected_value(severity, d) <= d + 1e-12

    def test_it_increases_with_the_censoring_point(self) -> None:
        severity = LogNormal(mu=0.0, sigma=1.0)
        values = [limited_expected_value(severity, d) for d in [0.5, 1, 2, 5, 20, 100]]
        assert all(b > a for a, b in pairwise(values))

    def test_censoring_below_the_support_returns_the_censoring_point(self) -> None:
        """Every loss exceeds it, so ``min(X, d)`` is ``d`` with probability one."""
        assert limited_expected_value(Pareto(alpha=2.0, xm=5.0), 3.0) == 3.0

    def test_zero_and_negative(self) -> None:
        assert limited_expected_value(Pareto(alpha=2.0, xm=1.0), 0.0) == 0.0
        with pytest.raises(ValueError, match="non-negative"):
            limited_expected_value(Pareto(alpha=2.0, xm=1.0), -1.0)


# --------------------------- Policy terms ------------------------------------ #


class TestPolicyTerms:
    def test_the_payment_is_the_capped_excess(self) -> None:
        terms = PolicyTerms(deductible=100.0, limit=500.0)
        assert terms.payment(50.0) == 0.0
        assert terms.payment(100.0) == 0.0
        assert terms.payment(300.0) == 200.0
        assert terms.payment(600.0) == 500.0
        assert terms.payment(1e9) == 500.0

    def test_coinsurance_scales_the_payment(self) -> None:
        terms = PolicyTerms(deductible=10.0, limit=100.0, coinsurance=0.8)
        assert terms.payment(60.0) == pytest.approx(40.0)
        assert terms.max_payment == pytest.approx(80.0)

    def test_an_unlimited_policy_has_no_cap(self) -> None:
        terms = PolicyTerms(deductible=10.0)
        assert terms.max_payment == math.inf
        assert terms.upper_loss == math.inf
        assert terms.payment(1e9) == pytest.approx(1e9 - 10.0)

    def test_the_limit_caps_the_excess_not_the_loss(self) -> None:
        """ "1M excess of 100k" pays at most 1M, on a 1.1M loss.

        The other convention -- limit as maximum covered loss -- would pay 900k
        here, and getting it backwards misprices every layer.
        """
        terms = PolicyTerms(deductible=100_000.0, limit=1_000_000.0)
        assert terms.payment(1_100_000.0) == pytest.approx(1_000_000.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"deductible": -1.0},
            {"limit": 0.0},
            {"limit": -5.0},
            {"coinsurance": 0.0},
            {"coinsurance": 1.5},
        ],
    )
    def test_bad_parameters_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ParameterError):
            PolicyTerms(**kwargs)


# --------------------------- Layered severity -------------------------------- #


class TestLayeredSeverity:
    SEVERITY = Pareto(alpha=2.0, xm=1.0)
    TERMS = PolicyTerms(deductible=2.0, limit=8.0)

    def test_the_mean_matches_simulation(self) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        draws = self.SEVERITY.rvs(300_000, seed=7)
        payments = [self.TERMS.payment(x) for x in draws]
        error = statistics.stdev(payments) / math.sqrt(len(payments))
        assert abs(layer.mean() - statistics.fmean(payments)) < 4.0 * error

    def test_the_mean_is_the_difference_of_two_limited_expected_values(self) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        expected = limited_expected_value(self.SEVERITY, 10.0) - limited_expected_value(
            self.SEVERITY, 2.0
        )
        assert layer.mean() == pytest.approx(expected, rel=1e-12)

    def test_per_loss_has_an_atom_at_zero(self) -> None:
        """Losses below the deductible produce a payment of exactly zero.

        That atom is why the per-loss basis pairs with the *unthinned*
        frequency: those claims still occur, they just cost nothing.
        """
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        assert layer.cdf(0.0) == pytest.approx(self.SEVERITY.cdf(2.0))
        assert layer.cdf(0.0) > 0.0

    def test_per_payment_has_no_atom_at_zero(self) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS, basis="per-payment")
        assert layer.cdf(0.0) == 0.0

    def test_per_payment_is_per_loss_divided_by_the_exceedance_probability(
        self,
    ) -> None:
        """The relationship that makes the two bases interchangeable.

        Using a per-payment severity with an unthinned frequency overstates the
        expected aggregate by exactly this factor, which for a high deductible
        is large and still looks like a plausible number.
        """
        per_loss = LayeredSeverity(self.SEVERITY, self.TERMS)
        per_payment = LayeredSeverity(self.SEVERITY, self.TERMS, basis="per-payment")
        assert per_payment.mean() == pytest.approx(
            per_loss.mean() / per_loss.exceedance_probability, rel=1e-12
        )

    def test_the_exceedance_probability_is_the_severity_survival(self) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        assert layer.exceedance_probability == pytest.approx(self.SEVERITY.sf(2.0))

    @pytest.mark.parametrize("basis", ["per-loss", "per-payment"])
    @pytest.mark.parametrize("u", [0.05, 0.3, 0.5, 0.8, 0.95])
    def test_the_quantile_inverts_the_distribution_function(
        self, basis: str, u: float
    ) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS, basis=basis)
        y = layer.ppf(u)
        if 0.0 < y < layer.terms.max_payment:
            assert layer.cdf(y) == pytest.approx(u, abs=1e-9)

    @pytest.mark.parametrize("basis", ["per-loss", "per-payment"])
    def test_the_payment_never_exceeds_the_cap(self, basis: str) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS, basis=basis)
        assert max(layer.rvs(5_000, seed=2)) <= 8.0 + 1e-12
        assert layer.cdf(8.0) == 1.0

    @pytest.mark.parametrize("basis", ["per-loss", "per-payment"])
    def test_sampling_reproduces_the_mean(self, basis: str) -> None:
        layer = LayeredSeverity(self.SEVERITY, self.TERMS, basis=basis)
        draws = layer.rvs(200_000, seed=4)
        error = statistics.stdev(draws) / math.sqrt(len(draws))
        assert abs(statistics.fmean(draws) - layer.mean()) < 4.0 * error

    def test_an_unknown_basis_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Available"):
            LayeredSeverity(self.SEVERITY, self.TERMS, basis="per-claim")

    def test_an_unreachable_deductible_is_rejected_per_payment(self) -> None:
        """No loss can exceed it, so conditioning on one is meaningless."""
        bounded = GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0)
        with pytest.raises(ValueError, match="per-payment"):
            LayeredSeverity(bounded, PolicyTerms(deductible=100.0), basis="per-payment")

    def test_no_terms_leaves_the_severity_alone(self) -> None:
        layer = LayeredSeverity(self.SEVERITY)
        assert layer.mean() == pytest.approx(2.0, rel=1e-9)
        assert layer.cdf(3.0) == pytest.approx(self.SEVERITY.cdf(3.0))


# --------------------------- Discretisation ---------------------------------- #


class TestDiscretiseSeverity:
    @pytest.mark.parametrize("method", ["mass", "mean-preserving"])
    def test_the_probabilities_and_lost_mass_account_for_everything(
        self, method: str
    ) -> None:
        probabilities, tail = discretise_severity(
            Pareto(alpha=2.0, xm=1.0), 0.5, 300, method=method
        )
        assert sum(probabilities) + tail == pytest.approx(1.0, abs=1e-9)
        assert all(p >= 0.0 for p in probabilities)

    def test_the_lost_mass_is_the_true_survival_past_the_grid(self) -> None:
        """It is reported so a heavy tail cannot quietly fall off the end."""
        severity = Pareto(alpha=2.0, xm=1.0)
        _, tail = discretise_severity(severity, 0.5, 200, method="mass")
        assert tail == pytest.approx(severity.sf((200 - 0.5) * 0.5), rel=1e-6)

    def test_mean_preserving_matches_the_censored_mean_exactly(self) -> None:
        """It preserves ``E[X ^ (grid end)]``, which is all it can preserve.

        The mass past the end is gone whatever the method, so matching the full
        mean is not available. What *is* available is an exact match to the
        censored mean, and asserting that instead of an approximate match to
        1.5 makes the test far sharper: it holds to machine precision.
        """
        severity = Pareto(alpha=3.0, xm=1.0)
        h, n = 0.05, 4000
        probabilities, _ = discretise_severity(severity, h, n, "mean-preserving")
        grid_mean = sum(j * h * p for j, p in enumerate(probabilities))

        # Telescoping the second differences leaves exactly this, with m the
        # last grid index: the censored mean, less a boundary term from the
        # grid having an end. It vanishes when the severity is capped inside
        # the grid, which is why a layered severity matches to the last bit.
        m = n - 1
        censored = limited_expected_value(severity, m * h)
        boundary = m * (limited_expected_value(severity, (m + 1) * h) - censored)
        assert grid_mean == pytest.approx(censored - boundary, rel=1e-12)
        assert grid_mean == pytest.approx(1.5, rel=1e-4)

    def test_mean_preserving_is_exact_when_the_severity_is_capped(self) -> None:
        """With nothing past the grid the boundary term is zero.

        A policy limit caps the severity, so a layered one discretises with no
        loss of mean at all -- which is the case that matters for pricing.
        """
        layer = LayeredSeverity(
            Pareto(alpha=1.9, xm=10_000.0),
            PolicyTerms(deductible=25_000.0, limit=500_000.0),
        )
        h, n = 2_000.0, 4_000
        probabilities, tail = discretise_severity(layer, h, n, "mean-preserving")
        grid_mean = sum(j * h * p for j, p in enumerate(probabilities))
        assert grid_mean == pytest.approx(layer.mean(), rel=1e-12)
        assert tail == pytest.approx(0.0, abs=1e-12)

    def test_mean_preserving_beats_the_mass_method_on_the_mean(self) -> None:
        severity = Pareto(alpha=3.0, xm=1.0)
        errors = {}
        for method in ["mass", "mean-preserving"]:
            probabilities, _ = discretise_severity(severity, 0.2, 1000, method=method)
            grid_mean = sum(j * 0.2 * p for j, p in enumerate(probabilities))
            errors[method] = abs(grid_mean - 1.5)
        assert errors["mean-preserving"] < errors["mass"]

    def test_a_finer_grid_is_more_accurate(self) -> None:
        """The error must fall with ``h``, not merely be small at one ``h``."""
        severity = Pareto(alpha=3.0, xm=1.0)
        errors = []
        for h in [0.4, 0.2, 0.1, 0.05]:
            probabilities, _ = discretise_severity(severity, h, int(200 / h))
            grid_mean = sum(j * h * p for j, p in enumerate(probabilities))
            errors.append(abs(grid_mean - 1.5))
        assert all(b < a for a, b in pairwise(errors))

    @pytest.mark.parametrize(
        ("h", "n", "method"),
        [
            (0.0, 10, "mass"),
            (-1.0, 10, "mass"),
            (0.5, 1, "mass"),
            (0.5, 10, "midpoint"),
        ],
    )
    def test_bad_arguments_are_rejected(self, h: float, n: int, method: str) -> None:
        with pytest.raises(ValueError):
            discretise_severity(Pareto(alpha=2.0, xm=1.0), h, n, method=method)


# --------------------------- Panjer recursion -------------------------------- #


class TestPanjerRecursion:
    @pytest.mark.parametrize(
        ("frequency", "severity"),
        [
            (Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0)),
            (NegativeBinomial(r=3.0, beta=1.0), Pareto(alpha=4.0, xm=1.0)),
            (Binomial(m=10, p=0.3), Weibull(k=1.5, lam=2.0)),
        ],
    )
    def test_it_reproduces_the_exact_compound_moments(
        self, frequency: object, severity: object
    ) -> None:
        """``E[S] = E[N]E[X]`` and ``Var[S] = E[N]Var[X] + Var[N]E[X]^2``.

        These hold for any frequency and severity and need no grid, so they
        check the recursion against something entirely outside it.
        """
        aggregate = panjer_recursion(
            frequency, severity, h=0.05, n=4000, method="mean-preserving"
        )
        mean, variance = compound_moments(frequency, severity)
        assert aggregate.mean() == pytest.approx(mean, rel=1e-3)
        assert aggregate.variance() == pytest.approx(variance, rel=1e-2)

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 4.0, 8.0])
    def test_a_single_risk_gives_the_severity_mixture_exactly(self, x: float) -> None:
        """Binomial(1, p) has a known answer: ``(1-p) + p*F(x)``.

        A compound with at most one claim is the severity with an atom at zero,
        so the recursion is checked against an identity rather than another
        numerical method.
        """
        p, severity = 0.3, Weibull(k=1.5, lam=2.0)
        aggregate = panjer_recursion(Binomial(m=1, p=p), severity, h=0.005, n=6000)
        assert aggregate.cdf(x) == pytest.approx(
            (1.0 - p) + p * severity.cdf(x), abs=5e-4
        )

    def test_the_error_falls_with_the_grid(self) -> None:
        """First-order in ``h``, so halving ``h`` should halve the error.

        Asserting the rate rather than a tolerance: a subtly wrong recursion
        could sit inside any fixed tolerance while converging to the wrong
        answer or not converging at all.
        """
        p, severity, x = 0.3, Weibull(k=1.5, lam=2.0), 2.0
        exact = (1.0 - p) + p * severity.cdf(x)
        errors = []
        for h in [0.08, 0.04, 0.02, 0.01]:
            aggregate = panjer_recursion(Binomial(m=1, p=p), severity, h, int(30 / h))
            errors.append(abs(aggregate.cdf(x) - exact))
        assert all(b < a for a, b in pairwise(errors))
        for a, b in pairwise(errors):
            assert b / a == pytest.approx(0.5, abs=0.1)

    def test_the_two_policy_bases_give_the_same_aggregate(self) -> None:
        """Per-loss with the full frequency, per-payment with it thinned.

        Both describe the same portfolio, so they must produce the same
        distribution -- and they do here to the last bit, since the thinning
        factor cancels exactly against the atom at zero.
        """
        severity = Pareto(alpha=2.0, xm=1.0)
        terms = PolicyTerms(deductible=2.0, limit=8.0)
        per_loss = LayeredSeverity(severity, terms)
        per_payment = LayeredSeverity(severity, terms, basis="per-payment")
        frequency = Poisson(lam=4.0)

        full = panjer_recursion(frequency, per_loss, h=0.05, n=1200)
        thinned = panjer_recursion(
            frequency.thin(per_loss.exceedance_probability), per_payment, h=0.05, n=1200
        )
        for a, b in zip(full.probabilities, thinned.probabilities, strict=True):
            assert a == pytest.approx(b, abs=1e-14)

    @pytest.mark.parametrize(("lam", "alpha"), [(1.0, 2.5), (0.5, 2.0), (2.0, 3.0)])
    def test_the_tail_converges_to_the_subexponential_asymptotic(
        self, lam: float, alpha: float
    ) -> None:
        """``P(S > x) -> E[N] P(X > x)`` for a subexponential severity.

        The single big jump principle: one enormous claim, not many moderate
        ones, is what makes the total large. The test asserts the ratio moves
        *towards* one as ``x`` grows rather than checking it at one point,
        because at any single ``x`` a wrong implementation could be close by
        accident.
        """
        frequency, severity = Poisson(lam=lam), Pareto(alpha=alpha, xm=1.0)
        aggregate = panjer_recursion(
            frequency, severity, h=0.25, n=8_000, method="mean-preserving"
        )
        ratios = [
            aggregate.sf(x) / (frequency.mean() * severity.sf(x))
            for x in [100.0, 300.0, 800.0, 1500.0]
        ]
        assert all(abs(b - 1.0) < abs(a - 1.0) for a, b in pairwise(ratios))
        assert ratios[-1] == pytest.approx(1.0, abs=0.01)

    def test_a_large_expected_count_is_reported_rather_than_returned(self) -> None:
        """``g_0 = exp(lam(f_0 - 1))`` underflows and takes everything with it.

        Every later probability is a multiple of ``g_0``, so once it is exactly
        zero the output is all zeros: not a distribution, and not obviously
        wrong unless something checks.
        """
        with pytest.raises(ArithmeticError, match="underflow"):
            panjer_recursion(
                Poisson(lam=2000.0), Pareto(alpha=2.0, xm=1.0), h=0.5, n=100
            )

    def test_the_underflow_message_names_the_expected_count(self) -> None:
        with pytest.raises(ArithmeticError, match="2000"):
            panjer_recursion(
                Poisson(lam=2000.0), Pareto(alpha=2.0, xm=1.0), h=0.5, n=100
            )

    def test_it_agrees_with_simulation_on_a_heavy_severity(self) -> None:
        frequency, severity = Poisson(lam=3.0), Pareto(alpha=2.5, xm=1.0)
        aggregate = panjer_recursion(
            frequency, severity, h=0.05, n=8000, method="mean-preserving"
        )
        sample = EmpiricalAggregate(
            simulate_aggregate_loss(frequency, severity, 60_000, seed=11)
        )
        for level in [0.5, 0.9, 0.95]:
            assert aggregate.value_at_risk(level) == pytest.approx(
                sample.value_at_risk(level), rel=0.03
            )

    def test_the_lost_severity_mass_is_carried_through(self) -> None:
        aggregate = panjer_recursion(
            Poisson(lam=1.0), Pareto(alpha=1.5, xm=1.0), h=0.5, n=100
        )
        assert aggregate.severity_tail_mass > 0.0
        assert aggregate.truncated_mass > 0.0


# --------------------------- Compound moments -------------------------------- #


class TestCompoundMoments:
    def test_the_mean_matches_simulation(self) -> None:
        frequency, severity = Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0)
        expected, _ = compound_moments(frequency, severity)
        sample = simulate_aggregate_loss(frequency, severity, 200_000, seed=1)
        error = statistics.stdev(sample) / math.sqrt(len(sample))
        assert abs(statistics.fmean(sample) - expected) < 4.0 * error

    @pytest.mark.parametrize("alpha", [1.2, 1.8, 2.0])
    def test_the_variance_is_infinite_for_a_heavy_severity(self, alpha: float) -> None:
        """``alpha <= 2`` means no second moment, so no variance for the total.

        Every normal or translated-gamma approximation to the aggregate loss
        matches two moments. Neither is available here, so those approximations
        do not apply, and reporting a large finite number instead would let
        someone use one.
        """
        mean, variance = compound_moments(Poisson(lam=2.0), Pareto(alpha=alpha, xm=1.0))
        assert math.isfinite(mean)
        assert variance == math.inf

    @pytest.mark.parametrize("alpha", [0.5, 0.9, 1.0])
    def test_both_are_infinite_when_the_severity_has_no_mean(
        self, alpha: float
    ) -> None:
        assert compound_moments(Poisson(lam=2.0), Pareto(alpha=alpha, xm=1.0)) == (
            math.inf,
            math.inf,
        )

    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_the_variance_decomposition_holds(self, frequency: object) -> None:
        severity = Pareto(alpha=4.0, xm=1.0)
        mean, variance = compound_moments(frequency, severity)
        severity_mean = 4.0 / 3.0  # alpha*xm/(alpha-1)
        severity_variance = 2.0 - severity_mean**2  # E[X^2] = alpha*xm^2/(alpha-2)
        assert mean == pytest.approx(frequency.mean() * severity_mean)  # type: ignore[attr-defined]
        assert variance == pytest.approx(
            frequency.mean() * severity_variance  # type: ignore[attr-defined]
            + frequency.variance() * severity_mean**2  # type: ignore[attr-defined]
        )

    def test_it_works_for_a_family_without_a_closed_form(self) -> None:
        """Falls back to quadrature, and must still land on the right answer."""
        severity = GeneralizedPareto(xi=0.2, sigma=1.0, mu=0.0)
        mean, variance = compound_moments(Poisson(lam=2.0), severity)
        assert mean == pytest.approx(2.0 * 1.25, rel=1e-6)
        assert math.isfinite(variance)


# --------------------------- Aggregate distribution -------------------------- #


class TestAggregateLoss:
    AGGREGATE = panjer_recursion(
        Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0), h=0.05, n=4000
    )

    def test_the_distribution_function_increases(self) -> None:
        values = [self.AGGREGATE.cdf(x) for x in [0.0, 1.0, 3.0, 10.0, 50.0]]
        assert all(b >= a for a, b in pairwise(values))
        assert values[0] >= 0.0
        assert values[-1] <= 1.0

    def test_the_survival_function_complements_it(self) -> None:
        for x in [0.5, 2.0, 7.0]:
            assert self.AGGREGATE.cdf(x) + self.AGGREGATE.sf(x) == pytest.approx(1.0)

    @pytest.mark.parametrize("level", [0.5, 0.9, 0.99])
    def test_the_quantile_reaches_its_level(self, level: float) -> None:
        assert self.AGGREGATE.cdf(self.AGGREGATE.ppf(level)) >= level - 1e-9

    def test_zero_claims_carries_real_probability(self) -> None:
        """``P(S = 0) = P(N = 0)``, which for a Poisson is ``exp(-lam)``."""
        assert self.AGGREGATE.probabilities[0] == pytest.approx(
            math.exp(-2.0), rel=1e-6
        )

    def test_a_quantile_past_the_grid_is_infinite_rather_than_the_last_point(
        self,
    ) -> None:
        """Returning the last grid point would look like an answer.

        The grid holds no information above ``1 - truncated_mass``, and a
        finite-looking value there is worse than none.
        """
        truncated = AggregateLoss(h=1.0, probabilities=[0.5, 0.3], truncated_mass=0.2)
        assert truncated.ppf(0.95) == math.inf

    def test_the_shortfall_is_infinite_when_the_tail_was_truncated(self) -> None:
        """The truncated part lies entirely inside the region being averaged."""
        truncated = AggregateLoss(h=1.0, probabilities=[0.5, 0.3], truncated_mass=0.2)
        assert truncated.expected_shortfall(0.99) == math.inf

    def test_the_shortfall_exceeds_the_value_at_risk(self) -> None:
        for level in [0.5, 0.9, 0.99]:
            assert self.AGGREGATE.expected_shortfall(level) > (
                self.AGGREGATE.value_at_risk(level)
            )

    def test_the_stop_loss_premium_matches_simulation(self) -> None:
        frequency, severity = Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0)
        sample = simulate_aggregate_loss(frequency, severity, 200_000, seed=13)
        for retention in [2.0, 5.0, 8.0]:
            simulated = statistics.fmean(max(s - retention, 0.0) for s in sample)
            assert self.AGGREGATE.stop_loss_premium(retention) == pytest.approx(
                simulated, rel=0.06
            )

    def test_the_stop_loss_premium_falls_with_the_retention(self) -> None:
        premiums = [self.AGGREGATE.stop_loss_premium(r) for r in [0.0, 2.0, 5.0, 8.0]]
        assert all(b < a for a, b in pairwise(premiums))

    def test_a_retention_too_deep_for_the_grid_is_reported(self) -> None:
        """Past some point the truncated mass dominates the answer.

        A heavy tail always truncates something, so the guard is relative: it
        fires when what fell off the grid would move the premium by more than
        the tolerance, not merely when anything fell off at all.
        """
        assert math.isfinite(self.AGGREGATE.stop_loss_premium(5.0))
        assert self.AGGREGATE.stop_loss_premium(60.0) == math.inf
        assert math.isfinite(self.AGGREGATE.stop_loss_premium(60.0, tolerance=1.0))

    def test_a_zero_retention_prices_the_whole_aggregate(self) -> None:
        assert self.AGGREGATE.stop_loss_premium(0.0) == pytest.approx(
            self.AGGREGATE.mean(), rel=1e-9
        )

    def test_the_stop_loss_premium_is_infinite_when_mass_was_truncated(self) -> None:
        truncated = AggregateLoss(h=1.0, probabilities=[0.5, 0.3], truncated_mass=0.2)
        assert truncated.stop_loss_premium(1.0) == math.inf

    def test_the_support_is_the_grid(self) -> None:
        assert self.AGGREGATE.support[:3] == [0.0, 0.05, 0.1]

    def test_invalid_arguments_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="u must be"):
            self.AGGREGATE.ppf(1.0)
        with pytest.raises(ValueError, match="level must be"):
            self.AGGREGATE.expected_shortfall(1.0)
        with pytest.raises(ValueError, match="non-negative"):
            self.AGGREGATE.stop_loss_premium(-1.0)


# --------------------------- Simulation -------------------------------------- #


class TestSimulation:
    @pytest.mark.parametrize("frequency", FREQUENCIES)
    def test_the_sample_mean_matches_the_exact_mean(self, frequency: object) -> None:
        severity = Pareto(alpha=4.0, xm=1.0)
        expected, _ = compound_moments(frequency, severity)
        sample = simulate_aggregate_loss(frequency, severity, 60_000, seed=17)
        error = statistics.stdev(sample) / math.sqrt(len(sample))
        assert abs(statistics.fmean(sample) - expected) < 4.0 * error

    def test_counts_and_claim_sizes_are_independent(self) -> None:
        """Both come from one stream, which is what keeps them so.

        Seeding a separate stream for each with the same seed would replay the
        frequency's uniforms as claim sizes, coupling the number of claims to
        their size. The aggregate mean is the sensitive statistic: it is exactly
        ``E[N]E[X]`` only under independence.
        """
        frequency, severity = Poisson(lam=4.0), Pareto(alpha=3.0, xm=1.0)
        expected, _ = compound_moments(frequency, severity)
        means = [
            statistics.fmean(simulate_aggregate_loss(frequency, severity, 20_000, s))
            for s in range(10)
        ]
        error = statistics.stdev(means) / math.sqrt(len(means))
        assert abs(statistics.fmean(means) - expected) < 4.0 * error

    def test_the_seed_makes_it_reproducible(self) -> None:
        arguments = (Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0), 500)
        assert simulate_aggregate_loss(*arguments, seed=9) == simulate_aggregate_loss(
            *arguments, seed=9
        )

    def test_different_seeds_give_different_samples(self) -> None:
        arguments = (Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0), 500)
        assert simulate_aggregate_loss(*arguments, seed=1) != simulate_aggregate_loss(
            *arguments, seed=2
        )

    def test_it_handles_a_severity_with_no_mean(self) -> None:
        """Panjer would need an unreachable grid; simulation simply runs."""
        sample = simulate_aggregate_loss(
            Poisson(lam=2.0), Pareto(alpha=0.8, xm=1.0), 5_000, seed=21
        )
        assert len(sample) == 5_000
        assert all(math.isfinite(s) for s in sample)

    def test_a_bad_simulation_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            simulate_aggregate_loss(Poisson(lam=1.0), Pareto(alpha=2.0, xm=1.0), 0)

    def test_periods_with_no_claims_give_exactly_zero(self) -> None:
        sample = simulate_aggregate_loss(
            Poisson(lam=0.3), Pareto(alpha=3.0, xm=1.0), 20_000, seed=23
        )
        zeros = sum(1 for s in sample if s == 0.0)
        assert zeros / len(sample) == pytest.approx(math.exp(-0.3), abs=0.02)


class TestEmpiricalAggregate:
    def test_it_reports_the_sample_statistics(self) -> None:
        aggregate = EmpiricalAggregate([1.0, 4.0, 2.0, 9.0])
        assert aggregate.mean() == 4.0
        assert aggregate.ppf(0.5) == 2.0
        assert aggregate.cdf(4.0) == 0.75
        assert aggregate.sf(4.0) == 0.25

    def test_the_quantile_agrees_with_the_gridded_one(self) -> None:
        frequency, severity = Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0)
        sample = EmpiricalAggregate(
            simulate_aggregate_loss(frequency, severity, 100_000, seed=29)
        )
        gridded = panjer_recursion(
            frequency, severity, h=0.05, n=4000, method="mean-preserving"
        )
        for level in [0.5, 0.9, 0.95]:
            assert sample.value_at_risk(level) == pytest.approx(
                gridded.value_at_risk(level), rel=0.03
            )

    def test_the_stop_loss_premium_agrees_with_the_gridded_one(self) -> None:
        frequency, severity = Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0)
        sample = EmpiricalAggregate(
            simulate_aggregate_loss(frequency, severity, 200_000, seed=31)
        )
        gridded = panjer_recursion(
            frequency, severity, h=0.05, n=4000, method="mean-preserving"
        )
        for retention in [2.0, 5.0]:
            assert sample.stop_loss_premium(retention) == pytest.approx(
                gridded.stop_loss_premium(retention), rel=0.06
            )

    def test_the_shortfall_exceeds_the_value_at_risk(self) -> None:
        sample = EmpiricalAggregate(
            simulate_aggregate_loss(
                Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0), 20_000, seed=33
            )
        )
        for level in [0.5, 0.9, 0.99]:
            assert sample.expected_shortfall(level) >= sample.value_at_risk(level)

    def test_it_rejects_an_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            EmpiricalAggregate([])

    def test_invalid_arguments_are_rejected(self) -> None:
        aggregate = EmpiricalAggregate([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="u must be"):
            aggregate.ppf(0.0)
        with pytest.raises(ValueError, match="level must be"):
            aggregate.expected_shortfall(0.0)
        with pytest.raises(ValueError, match="non-negative"):
            aggregate.stop_loss_premium(-1.0)


# --------------------------- Reinsurance pricing ----------------------------- #


class TestExcessOfLossPremium:
    @pytest.mark.parametrize(
        ("retention", "limit"), [(5.0, 15.0), (10.0, None), (2.0, 3.0), (0.0, 5.0)]
    )
    def test_it_matches_simulation(self, retention: float, limit: float | None) -> None:
        frequency, severity = Poisson(lam=10.0), Pareto(alpha=2.0, xm=1.0)
        terms = PolicyTerms(deductible=retention, limit=limit)
        draws = severity.rvs(300_000, seed=37)
        payments = [terms.payment(x) for x in draws]
        simulated = frequency.mean() * statistics.fmean(payments)
        error = frequency.mean() * statistics.stdev(payments) / math.sqrt(len(payments))
        assert (
            abs(
                excess_of_loss_premium(frequency, severity, retention, limit)
                - simulated
            )
            < 4.0 * error
        )

    def test_layers_are_additive(self) -> None:
        """Pricing 5-to-20 and 20-to-50 must equal pricing 5-to-50.

        An exact identity that no simulation is needed to check, and one that a
        sign error in either limited expected value would break.
        """
        frequency, severity = Poisson(lam=10.0), Pareto(alpha=2.0, xm=1.0)
        lower = excess_of_loss_premium(frequency, severity, 5.0, 15.0)
        upper = excess_of_loss_premium(frequency, severity, 20.0, 30.0)
        whole = excess_of_loss_premium(frequency, severity, 5.0, 45.0)
        assert lower + upper == pytest.approx(whole, rel=1e-12)

    def test_an_unlimited_layer_on_a_severity_with_no_mean_is_infinite(self) -> None:
        """And a limited one is not, which is why cat covers carry limits."""
        frequency, severity = Poisson(lam=1.0), Pareto(alpha=0.9, xm=1.0)
        assert excess_of_loss_premium(frequency, severity, 10.0, None) == math.inf
        assert math.isfinite(excess_of_loss_premium(frequency, severity, 10.0, 100.0))

    def test_a_higher_attachment_costs_less(self) -> None:
        frequency, severity = Poisson(lam=10.0), Pareto(alpha=2.0, xm=1.0)
        premiums = [
            excess_of_loss_premium(frequency, severity, r, 10.0)
            for r in [1.0, 5.0, 20.0, 100.0]
        ]
        assert all(b < a for a, b in pairwise(premiums))

    def test_a_wider_layer_costs_more(self) -> None:
        frequency, severity = Poisson(lam=10.0), Pareto(alpha=2.0, xm=1.0)
        premiums = [
            excess_of_loss_premium(frequency, severity, 5.0, limit)
            for limit in [1.0, 5.0, 20.0, 100.0]
        ]
        assert all(b > a for a, b in pairwise(premiums))

    def test_a_ground_up_unlimited_layer_is_the_whole_expected_loss(self) -> None:
        frequency, severity = Poisson(lam=10.0), Pareto(alpha=2.0, xm=1.0)
        assert excess_of_loss_premium(frequency, severity, 0.0, None) == pytest.approx(
            compound_moments(frequency, severity)[0], rel=1e-12
        )

    def test_it_scales_with_the_expected_claim_count(self) -> None:
        severity = Pareto(alpha=2.0, xm=1.0)
        single = excess_of_loss_premium(Poisson(lam=1.0), severity, 5.0, 15.0)
        many = excess_of_loss_premium(Poisson(lam=7.0), severity, 5.0, 15.0)
        assert many == pytest.approx(7.0 * single, rel=1e-12)

    @pytest.mark.parametrize(
        ("retention", "limit"), [(-1.0, 10.0), (5.0, 0.0), (5.0, -3.0)]
    )
    def test_bad_arguments_are_rejected(self, retention: float, limit: float) -> None:
        with pytest.raises(ValueError):
            excess_of_loss_premium(
                Poisson(lam=1.0), Pareto(alpha=2.0, xm=1.0), retention, limit
            )


# --------------------------- End to end -------------------------------------- #


class TestWorkedPortfolio:
    """A portfolio priced end to end, the way the module is meant to be used."""

    FREQUENCY = NegativeBinomial(r=4.0, beta=2.5)
    SEVERITY = Pareto(alpha=1.9, xm=10_000.0)
    TERMS = PolicyTerms(deductible=25_000.0, limit=500_000.0)

    def test_the_priced_layer_agrees_across_all_three_routes(self) -> None:
        """Closed form, Panjer and simulation must give the same expected cost.

        Three independent calculations of one number: limited expected values,
        a recursion on a discretised grid, and a sample. Agreement is real
        evidence; any one of them alone is not.
        """
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        closed = self.FREQUENCY.mean() * layer.mean()

        gridded = panjer_recursion(
            self.FREQUENCY, layer, h=2_000.0, n=4_000, method="mean-preserving"
        )
        sample = simulate_aggregate_loss(self.FREQUENCY, layer, 40_000, seed=41)

        assert gridded.mean() == pytest.approx(closed, rel=0.01)
        error = statistics.stdev(sample) / math.sqrt(len(sample))
        assert abs(statistics.fmean(sample) - closed) < 4.0 * error

    def test_the_aggregate_variance_does_not_exist(self) -> None:
        """``alpha = 1.9`` on the ground-up severity, so no second moment.

        The layer is capped, so *its* variance is finite -- capping is what
        makes the arithmetic work at all. The ungross-up point stands: without a
        limit there would be nothing to approximate with.
        """
        _, ungrossed = compound_moments(self.FREQUENCY, self.SEVERITY)
        assert ungrossed == math.inf

        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        _, capped = compound_moments(self.FREQUENCY, layer)
        assert math.isfinite(capped)

    def test_reinsurance_reduces_the_retained_tail(self) -> None:
        """Aggregate stop-loss above the 90th percentile of the retained loss."""
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        gridded = panjer_recursion(
            self.FREQUENCY, layer, h=2_000.0, n=4_000, method="mean-preserving"
        )
        retention = gridded.value_at_risk(0.9)
        premium = gridded.stop_loss_premium(retention)
        assert 0.0 < premium < gridded.mean()

    def test_the_frequency_used_matters(self) -> None:
        """Same mean, more dispersion, a fatter aggregate tail.

        Choosing Poisson when the portfolio is over-dispersed understates the
        aggregate tail while matching the mean exactly, so a mean check will not
        catch it.
        """
        layer = LayeredSeverity(self.SEVERITY, self.TERMS)
        equivalent = Poisson(lam=self.FREQUENCY.mean())
        over = panjer_recursion(self.FREQUENCY, layer, h=2_000.0, n=4_000)
        equi = panjer_recursion(equivalent, layer, h=2_000.0, n=4_000)
        assert over.mean() == pytest.approx(equi.mean(), rel=0.01)
        assert over.value_at_risk(0.99) > equi.value_at_risk(0.99)


def test_the_rng_is_shared_with_the_rest_of_the_library() -> None:
    """The frequency samplers draw from the library's own generator.

    So a seed set anywhere behaves the same way everywhere, and simulation
    results reproduce across the whole package rather than per module.
    """
    rng = RNG(0)
    assert isinstance(Poisson(lam=2.0).draw(rng), int)
    assert isinstance(NegativeBinomial(r=2.0, beta=1.0).draw(rng), int)
    assert isinstance(Binomial(m=5, p=0.5).draw(rng), int)
