# Code Review

This page describes what reviewers look for in a `heavytails` pull request, so
that authors can anticipate it and reviewers apply it consistently.

The automated checks — lint, formatting, types, tests, security, documentation —
run in CI. Review exists for the things a machine cannot judge: whether the
mathematics is right, whether the design fits, and whether the next reader will
understand it.

## Before requesting review

Run everything CI runs:

```bash
make check
```

That covers linting, formatting, type checking, the test suite and the security
linter. A pull request that fails these has not yet started review.

Then confirm:

- [ ] The change is described in `CHANGELOG.md` under `## [Unreleased]`.
- [ ] New behaviour has tests; fixed bugs have a regression test.
- [ ] Public API changes are reflected in the documentation.
- [ ] The branch follows the flow in [Contributing](contributing.md).
- [ ] Commits are scoped and their messages explain *why*.

## What reviewers check

### Correctness of the mathematics

This is the part that matters most, and the part automation cannot help with.

- Does the formula match the cited source? Reviewers will look it up, so cite
  it — a reference in the docstring turns a code review into a check rather than
  a derivation.
- Are the parameter constraints exactly the ones the family requires, neither
  looser nor tighter?
- Is the behaviour at the boundaries of the support correct, including the
  endpoints themselves?
- Are the degenerate cases handled — $\xi \to 0$ in the GEV, $\nu \to \infty$ in
  the Student-t, and similar removable singularities?
- Does the implementation agree with a known closed form at the points where one
  exists?

### Numerical behaviour

Pure-Python floating point in the far tail is where this library earns or loses
its reputation.

- Is there catastrophic cancellation? `1 - cdf(x)` in the far tail is the
  canonical example: once `cdf(x)` rounds to `1.0` the answer is exactly zero.
- Are logarithms and exponentials arranged to avoid overflow and underflow? Prefer
  `log1p`, `expm1` and working in log space where the arguments get large.
- Does an iterative solver have a guaranteed-termination fallback, or can it spin?
- Are tolerances justified rather than tuned until the test passed?

### Tests

- Do the tests fail without the change? A test that passes either way tests
  nothing.
- Are the properties in [Validation Studies](../theory/validation.md) covered:
  non-negativity, monotonicity, quantile inversion, normalisation?
- Are the boundaries tested, not just the comfortable middle of the range?
- Is randomness seeded, so a failure can be reproduced?
- Are timing assertions absent? Shared CI runners make them flaky; performance
  belongs in the [benchmark suite](benchmarks.md).

### Design

- Does it belong in this library? The scope is heavy-tailed distributions and
  their estimation.
- Does it introduce a runtime dependency? That needs an explicit discussion, not
  a review comment — see [Architecture](architecture.md).
- Is existing machinery reused — the RNG wrapper, the numeric PPF solver, the
  shared validation helpers — rather than reimplemented?
- Is the public surface as small as it can be? Everything exported is something
  that must keep working.

### Documentation

- Do the docstrings follow NumPy style, with parameters, returns and raised
  exceptions?
- Do the examples actually run? Reviewers paste them in.
- Are new distributions added to the README table and the
  [distributions guide](../guide/distributions.md)?
- Does new mathematics carry a citation?

### Compatibility

- Does it work on Python 3.10 through 3.13? Watch for syntax and standard-library
  APIs newer than 3.10.
- Is it platform-independent? `time.time()` resolution, path separators and
  default encodings differ across the CI matrix.
- Does it change existing behaviour? If so, is that a documented breaking change
  with a version bump to match?

## How to give a review

**Separate the blocking from the optional.** State plainly which comments must be
addressed before merge and which are suggestions. "Consider" and "must" should not
look alike.

**Explain the why.** "This loses precision for x > 1e15 because cdf(x) rounds to
1.0" teaches; "use sf here" does not.

**Ask when unsure.** Reviewing statistical code means meeting unfamiliar
mathematics. A question is more useful than a guess, and the answer often belongs
in the documentation.

**Acknowledge good work.** A clean derivation, a well-chosen test case, or a
tricky numerical fix is worth saying so.

**Review promptly.** A small review that arrives today is worth more than a
thorough one next week.

## How to receive a review

**Assume good faith.** Comments are about the change, not the author.

**Reply to every comment.** Fix it, or explain why not. Silent dismissal leaves
the reviewer unsure whether it was seen.

**Push fixes as new commits** during review, so the reviewer can see what changed.
Squash at merge.

**Push back when you are right.** You know the mathematics you just wrote better
than the reviewer does. Explain it — and if the explanation is needed, it probably
belongs in a comment or the docs too.

## Merging

A pull request merges when:

- Every CI check passes.
- At least one maintainer has approved it.
- Every blocking comment is resolved.
- `CHANGELOG.md` records the change.

The branch flow is enforced automatically: feature branches target `develop`, and
only `develop` targets `main`. See [Contributing](contributing.md).

## See also

- [Contributing](contributing.md) — workflow, branches and commit conventions
- [Testing](testing.md) — the test suite and its markers
- [Architecture](architecture.md) — design constraints a review will apply
- [Releasing](releasing.md) — what happens after merge
