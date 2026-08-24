"""One place that knows which name means which distribution.

The library resolved names to classes in at least four ways: a table in
``cli.py``, ``getattr(heavytails, name)`` in ``performance.py``, and if/elif
chains in ``validation.py`` and ``utilities.py`` -- the chain repeated four
times in ``validation.py`` alone. They did not agree. The CLI accepted
``student-t`` and ``gpd``; the chains accepted ``studentt`` and nothing for the
generalized Pareto at all; ``getattr`` accepted the class name and nothing
else. Adding a family meant finding every one of them, and the mypy errors that
suppression in ``pyproject.toml`` was hiding were mostly the chains rebinding
one variable to five different classes.

So the mapping lives here, once, with the aliases each caller used before, and
they all go through it.

    >>> from heavytails.registry import resolve, create
    >>> resolve("student-t").__name__
    'StudentT'
    >>> resolve("studentt").__name__
    'StudentT'
    >>> create("pareto", alpha=2.0, xm=1.0).cdf(2.0)
    0.75
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from heavytails.heavy_tails import (
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Family", "create", "families", "names", "resolve"]


@dataclass(frozen=True)
class Family:
    """What is known about one distribution family, by name.

    Attributes:
        name: Canonical name, lowercase.
        cls: The class itself.
        parameters: Constructor parameter names, in order.
        support: Human-readable description of where the density lives.
        aliases: Other names that resolve here, including every name a caller
            in this library used before the registry existed.
    """

    name: str
    cls: type
    parameters: tuple[str, ...]
    support: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def all_names(self) -> Iterator[str]:
        """Every name that resolves to this family."""
        yield self.name
        yield from self.aliases


# Adding a family is one entry here and nothing else.
_FAMILIES: tuple[Family, ...] = (
    Family("pareto", Pareto, ("alpha", "xm"), "x >= xm"),
    Family("cauchy", Cauchy, ("x0", "gamma"), "the whole real line"),
    Family(
        "studentt",
        StudentT,
        ("nu",),
        "the whole real line",
        aliases=("student-t", "student_t", "t"),
    ),
    Family("lognormal", LogNormal, ("mu", "sigma"), "x > 0", aliases=("log-normal",)),
    Family("weibull", Weibull, ("k", "lam"), "x >= 0"),
    Family("frechet", Frechet, ("alpha", "s", "m"), "x > m"),
    Family(
        "gev_frechet",
        GEV_Frechet,
        ("xi", "mu", "sigma"),
        "x > mu - sigma / xi",
        aliases=("gev", "gevfrechet", "gev-frechet"),
    ),
    Family(
        "generalizedpareto",
        GeneralizedPareto,
        ("xi", "sigma", "mu"),
        "x >= mu, and bounded above when xi < 0",
        aliases=("gpd", "generalized-pareto", "genpareto"),
    ),
    Family("burrxii", BurrXII, ("c", "k", "s"), "x > 0", aliases=("burr", "burr-xii")),
    Family(
        "loglogistic",
        LogLogistic,
        ("kappa", "lam"),
        "x > 0",
        aliases=("log-logistic", "fisk"),
    ),
    Family(
        "inversegamma",
        InverseGamma,
        ("alpha", "beta"),
        "x > 0",
        aliases=("invgamma", "inverse-gamma"),
    ),
    Family(
        "betaprime",
        BetaPrime,
        ("a", "b", "s"),
        "x > 0",
        aliases=("beta-prime", "betaprime2"),
    ),
)

# Built once. Class names resolve too, because `getattr(heavytails, name)` was
# one of the lookups this replaces and callers passing "Pareto" should not
# break.
_BY_NAME: dict[str, Family] = {}
for _family in _FAMILIES:
    for _key in _family.all_names():
        _BY_NAME[_key] = _family
    _BY_NAME[_family.cls.__name__.lower()] = _family


def families() -> tuple[Family, ...]:
    """Every registered family, in a stable order."""
    return _FAMILIES


def names() -> tuple[str, ...]:
    """Canonical names, for error messages and help text."""
    return tuple(family.name for family in _FAMILIES)


def resolve(name: str) -> type:
    """The class for ``name``, case-insensitively and allowing aliases.

    Args:
        name: A canonical name, an alias, or the class name.

    Returns:
        The distribution class.

    Raises:
        ValueError: If nothing is registered under that name. The message
            lists what is, because a caller who mistyped one name cannot
            otherwise discover the right one. It opens with "Unknown
            distribution" because that is the wording the lookups this
            replaces used, and callers match on it.

    Examples:
        >>> resolve("GPD").__name__
        'GeneralizedPareto'
        >>> try:
        ...     resolve("nonesuch")
        ... except ValueError as error:
        ...     str(error).startswith("Unknown distribution 'nonesuch'; known: ")
        True
    """
    family = _BY_NAME.get(str(name).strip().lower())
    if family is None:
        known = ", ".join(sorted(names()))
        raise ValueError(f"Unknown distribution {name!r}; known: {known}")
    return family.cls


def create(name: str, **parameters: Any) -> Any:
    """Construct a distribution by name.

    Args:
        name: As :func:`resolve` accepts.
        **parameters: Passed to the constructor unchanged, so the family's own
            validation reports a bad value rather than this function
            second-guessing it.

    Returns:
        The constructed distribution.

    Raises:
        ValueError: If the name is unknown, or the family rejects the
            parameters.

    Examples:
        >>> create("burr", c=2.0, k=1.5).sf(1.0) < 1.0
        True
    """
    return resolve(name)(**parameters)
