"""Every docstring example in the numerical modules must actually run.

Four of them had stopped reproducing before this file existed. Two claimed a
tail index estimate of 0.5 on samples too small for their own sampling
variability, so the value shown had been transcribed from what the estimator
*should* give rather than from what it did give -- 0.4 in both cases.

That is a worse failure than it looks. A docstring example is the first thing a
reader runs, and one that quietly returns something else teaches them to
distrust the rest. Nothing was checking, because the main suite does not collect
doctests.

The legacy modules are excluded rather than fixed: their examples are
structurally broken, and what happens to those modules is still open (#312).

Five of the core modules turn out to carry no runnable examples at all --
``heavy_tails``, ``extra_distributions``, ``discrete``, ``plotting`` and
``_special``. They are still checked, so an example added later is covered from
the day it lands, but the "must have examples" assertion below names only the
modules that currently have any. Asserting it for all nine would fail today for
a reason that is a documentation gap rather than a regression.
"""

from __future__ import annotations

import doctest
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

from heavytails import (
    _special,
    actuarial,
    discrete,
    extra_distributions,
    heavy_tails,
    plotting,
    risk,
    tail_index,
    threshold,
)

MODULES: list[ModuleType] = [
    _special,
    actuarial,
    discrete,
    extra_distributions,
    heavy_tails,
    plotting,
    risk,
    tail_index,
    threshold,
]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples_reproduce(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} docstring examples in "
        f"{module.__name__} did not reproduce; see the captured output above."
    )


def test_the_viz_examples_reproduce() -> None:
    """Separate because it needs the optional plotting extra."""
    pytest.importorskip("matplotlib", reason="matplotlib is the optional 'plot' extra")
    from heavytails import viz  # noqa: PLC0415

    results = doctest.testmod(viz, verbose=False)
    assert results.failed == 0


DOCUMENTED = [actuarial, risk, tail_index, threshold]


@pytest.mark.parametrize("module", DOCUMENTED, ids=lambda m: m.__name__)
def test_the_examples_have_not_been_deleted(module: ModuleType) -> None:
    """A module with no examples passes the check above vacuously.

    Deleting a failing example is an easier repair than fixing it, and would
    leave the guard green while removing the thing it guards.
    """
    assert doctest.testmod(module, verbose=False).attempted > 0
