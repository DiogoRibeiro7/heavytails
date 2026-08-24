"""The registry is the single place that maps a name to a distribution.

Before it there were at least four: a table in the CLI, ``getattr`` on the
package, and if/elif chains in ``validation`` and ``utilities`` -- the chain
repeated five times in ``validation`` alone. They disagreed about which names
were acceptable, and adding a family meant finding every one of them.
"""

from __future__ import annotations

import inspect

import pytest

from heavytails import registry
from heavytails.cli import DISTRIBUTIONS


class TestResolving:
    @pytest.mark.parametrize("name", [f.name for f in registry.families()])
    def test_every_canonical_name_resolves(self, name: str) -> None:
        assert registry.resolve(name) is not None

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("pareto", "Pareto"),
            ("PARETO", "Pareto"),
            ("  pareto  ", "Pareto"),
            ("student-t", "StudentT"),
            ("studentt", "StudentT"),
            ("t", "StudentT"),
            ("gpd", "GeneralizedPareto"),
            ("gev", "GEV_Frechet"),
            ("burr", "BurrXII"),
            ("invgamma", "InverseGamma"),
        ],
    )
    def test_aliases_and_case(self, name: str, expected: str) -> None:
        assert registry.resolve(name).__name__ == expected

    def test_class_names_resolve_too(self) -> None:
        """``getattr(heavytails, name)`` was one of the lookups replaced.

        Callers passing the class name must keep working.
        """
        for family in registry.families():
            assert registry.resolve(family.cls.__name__) is family.cls

    def test_an_unknown_name_says_what_is_known(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution") as info:
            registry.resolve("nonesuch")
        message = str(info.value)
        # A caller who mistyped one name cannot otherwise discover the right
        # one, so the message has to carry the list.
        for name in registry.names():
            assert name in message

    def test_the_message_keeps_the_wording_callers_match_on(self) -> None:
        with pytest.raises(ValueError, match=r"^Unknown distribution"):
            registry.resolve("nope")


class TestConstructing:
    def test_create_passes_parameters_through(self) -> None:
        assert registry.create("pareto", alpha=2.0, xm=1.0).cdf(2.0) == pytest.approx(
            0.75
        )

    def test_the_family_validates_its_own_parameters(self) -> None:
        """Not the registry, which would only duplicate the constraint."""
        with pytest.raises(ValueError):
            registry.create("pareto", alpha=-1.0, xm=1.0)

    @pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.name)
    def test_every_family_lists_its_constructor_parameters(self, family) -> None:
        signature = inspect.signature(family.cls.__init__)
        actual = [p for p in signature.parameters if p != "self"]
        assert list(family.parameters) == actual, (
            f"{family.name} lists {family.parameters}, constructor takes {actual}"
        )


class TestTheConsumersAgree:
    def test_the_cli_table_is_a_view_of_the_registry(self) -> None:
        """Adding a family to the registry must add it to the CLI."""
        for name, cls in DISTRIBUTIONS.items():
            assert registry.resolve(name) is cls
        for family in registry.families():
            assert family.name in DISTRIBUTIONS

    def test_the_names_the_cli_always_accepted_still_work(self) -> None:
        """These were the CLI's own keys before the registry existed."""
        for name in ("student-t", "gev", "gpd", "invgamma", "betaprime", "burr"):
            assert name in DISTRIBUTIONS
