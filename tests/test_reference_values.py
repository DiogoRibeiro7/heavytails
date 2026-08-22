"""Every family checked against values computed independently of this library.

Most of the suite asserts that the library is *self-consistent*: the quantile
inverts the distribution function, the survival function complements it, a
sample has the moments it should. Those catch a great deal and they cannot
catch a formula that is simply wrong, because a consistently wrong
implementation is still perfectly self-consistent.

That is not hypothetical here. ``InverseGamma.cdf`` was wrong by factors of two
to seventeen across its lower tail from 0.1.0 onwards, and every property test
passed the entire time: the values were monotone, inside [0, 1], and
complementary with the survival function. They were simply not the right
values.

``tests/reference_values.json`` holds values computed by mpmath at 50 decimal
digits from the mathematical definitions, by
``scripts/generate_reference_values.py``. Reading them needs no mpmath and no
arithmetic beyond a comparison, so this runs in milliseconds.

**A failure here means the library changed, not that the file is stale.**
Regenerating the table to make this pass would remove the only independent
opinion in the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import heavytails.extra_distributions as extra
import heavytails.heavy_tails as core

DATABASE = Path(__file__).parent / "reference_values.json"

# Double precision itself, and how many of them to allow for the handful of
# operations between the input and the answer.
EPSILON = 2.220446049250313e-16
SLACK = 40.0

# The floor, for quantities that are perfectly conditioned.
FLOOR = 1e-13


def _load() -> dict[str, Any]:
    return json.loads(DATABASE.read_text(encoding="utf-8"))


def _construct(spec: dict[str, Any]) -> Any:
    module_name, class_name = spec["cls"].split(".")
    module = {"heavy_tails": core, "extra_distributions": extra}[module_name]
    return getattr(module, class_name)(**spec["kwargs"])


# Below this the reference is zero as far as double precision is concerned and
# a relative comparison is meaningless -- the Student-t median is the case, and
# its reference carries a bisection residue of 5e-26 rather than a clean zero.
NEGLIGIBLE = 1e-20


def _relative(got: float, exact: float) -> float:
    """Relative error, falling back to absolute where relative has no meaning."""
    if abs(exact) < NEGLIGIBLE:
        return abs(got) if abs(got) > NEGLIGIBLE else 0.0
    return abs(got - exact) / abs(exact)


def _identify(spec: dict[str, Any]) -> str:
    arguments = ", ".join(f"{k}={v}" for k, v in spec["kwargs"].items())
    return f"{spec['cls'].split('.')[1]}({arguments})"


DATA = _load()
CASES = [
    pytest.param(entry, point, id=f"{_identify(entry)}-u{point['u']:g}")
    for entry in DATA["entries"]
    for point in entry["points"]
]


class TestTheDatabaseItself:
    def test_it_records_where_it_came_from(self) -> None:
        """Without provenance a reference table is just more assertions.

        A reader has to be able to tell what produced these numbers and at what
        precision, or there is no reason to believe them over the code they are
        checking.
        """
        assert DATA["source"] == "mpmath"
        assert DATA["decimal_digits"] >= 30
        assert DATA["mpmath_version"]
        assert DATA["generated"]

    def test_it_covers_every_continuous_family(self) -> None:
        """A family missing from the table is a family with no independent check."""
        covered = {entry["cls"].split(".")[1] for entry in DATA["entries"]}
        expected = {
            "Pareto",
            "Cauchy",
            "StudentT",
            "LogNormal",
            "Weibull",
            "Frechet",
            "GEV_Frechet",
            "GeneralizedPareto",
            "BurrXII",
            "LogLogistic",
            "InverseGamma",
            "BetaPrime",
        }
        assert expected <= covered, f"no reference values for {expected - covered}"

    def test_it_reaches_into_both_tails(self) -> None:
        """The middle of a distribution is where errors are easiest to see."""
        grid = DATA["quantile_grid"]
        assert min(grid) <= 1e-9
        assert max(grid) >= 1 - 1e-6


class TestAgainstTheReference:
    @pytest.mark.parametrize(("entry", "point"), CASES)
    def test_the_quantile_matches(
        self, entry: dict[str, Any], point: dict[str, Any]
    ) -> None:
        dist = _construct(entry)
        assert _relative(dist.ppf(point["u"]), float(point["x"])) < _budget(
            point, "ppf"
        )

    @pytest.mark.parametrize(("entry", "point"), CASES)
    def test_the_distribution_function_matches(
        self, entry: dict[str, Any], point: dict[str, Any]
    ) -> None:
        dist = _construct(entry)
        got = dist.cdf(float(point["x"]))
        assert _relative(got, float(point["cdf"])) < _budget(point, "cdf")

    @pytest.mark.parametrize(("entry", "point"), CASES)
    def test_the_survival_function_matches(
        self, entry: dict[str, Any], point: dict[str, Any]
    ) -> None:
        dist = _construct(entry)
        got = dist.sf(float(point["x"]))
        assert _relative(got, float(point["sf"])) < _budget(point, "sf")

    @pytest.mark.parametrize(("entry", "point"), CASES)
    def test_the_density_matches(
        self, entry: dict[str, Any], point: dict[str, Any]
    ) -> None:
        dist = _construct(entry)
        got = dist.pdf(float(point["x"]))
        assert _relative(got, float(point["pdf"])) < _budget(point, "pdf")


def _budget(point: dict[str, Any], quantity: str) -> float:
    """How much relative error the arithmetic is entitled to at this point.

    A single tolerance across a distribution cannot be right. Near a support
    boundary the *input* pins the answer down only so far: the double nearest
    to the Pareto 1e-9 quantile is 1 + 4e-10, whose distance from the boundary
    carries about seven relative digits, so no formula evaluates the
    distribution function there to more than that. A flat tolerance would have
    to be loose enough to admit that, and would then be blind to a genuine
    eighth-digit error somewhere well-conditioned.

    So the reference stores the condition number at every point, and the
    tolerance follows from it. That is what separated the five real defects
    found here -- GPD, Burr XII, Weibull, Cauchy and LogNormal all computing a
    small probability by subtracting from one -- from Pareto's, which was the
    arithmetic doing as well as it can.
    """
    condition = float(point[f"cond_{quantity}"])
    return max(FLOOR, SLACK * condition * EPSILON)
