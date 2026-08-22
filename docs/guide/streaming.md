# Streaming Estimation

The Hill estimator needs the top $k$ order statistics, which sounds like it
rules out a streaming version — order statistics are a property of the whole
sample. But only the top $k+1$ values ever enter the calculation, and *which*
values those are can be maintained as the data arrives.

A min-heap of size $k+1$ does it in $O(k)$ memory and $O(\log k)$ time per
observation, whatever the length of the stream.

## The estimate is exact, not approximate

```python
from heavytails import Pareto
from heavytails.streaming import StreamingTailIndex
from heavytails.tail_index import hill_estimator

data = Pareto(alpha=2.0, xm=1.0).rvs(200_000, seed=1)

stream = StreamingTailIndex(k=1000)
stream.extend(data)

stream.hill() == hill_estimator(data, k=1000)   # True
```

Not "close to". **The same float.** Both depend on the sample only through its
top $k+1$ values, and the streaming version keeps exactly those, in the same
order, and combines them with the same arithmetic. The test suite asserts
equality rather than closeness, because a tolerance would pass on an
implementation that quietly dropped or duplicated an order statistic.

Nothing is traded away except the ability to ask for a different $k$ later.

```python
stream.moment()      # (gamma, alpha), also bit-for-bit the batch result
stream.threshold     # the (k+1)-th order statistic
stream.n_seen        # 200000
```

## Memory does not grow with the stream

| Observations | Retained |
| --- | --- |
| 10,000 | 1,001 |
| 100,000 | 1,001 |
| 1,000,000 | 1,001 |

Values below the current threshold are discarded as they arrive and never
looked at again. They may be anything at all — including values the estimator
could not take a logarithm of, since they never reach it.

## Whole stream, or a window

These are different questions and the module keeps them separate.

| | Memory | Answers |
| --- | --- | --- |
| `StreamingTailIndex` | $O(k)$ | What is the tail index of everything seen? |
| `WindowedTailIndex` | $O(\text{window})$ | What is it *now*? |

For monitoring, the second is usually the one wanted. A portfolio whose tail
index has moved from 3 to 1.5 is not well described by an average of the two:

```python
from heavytails.streaming import WindowedTailIndex

monitor = WindowedTailIndex(window=5000, k=400)
monitor.extend(Pareto(alpha=3.0, xm=1.0).rvs(5000, seed=1))
monitor.hill()          # 0.330, against a true gamma of 0.333

monitor.extend(Pareto(alpha=1.5, xm=1.0).rvs(5000, seed=2))
monitor.hill()          # 0.667 — the old regime has left the window
```

The whole-stream estimator over the same data reports 0.650, describing neither
regime.

!!! warning "The window costs $O(\text{window})$ memory, not $O(k)$"
    This is inherent, not an implementation gap. When the largest value in a
    window expires, the new largest can be any of the survivors, so nothing
    smaller than the window itself determines the answer. Anything claiming to
    track windowed order statistics in $O(k)$ is either approximating or wrong.

A window also handles contamination that is *temporary*, which fixed trimming
cannot: one bad observation corrupts every whole-stream estimate from the moment
it arrives, whereas in a window it corrupts `window` of them and is then gone.

## Before there is enough data

Both refuse rather than returning something:

```python
stream = StreamingTailIndex(k=100)
stream.extend(data[:10])
stream.ready            # False
stream.hill()           # ValueError: need at least 101 observations for k=100, seen 10
```

An estimate from ten observations at `k = 100` would be a number with no
meaning attached, and the caller has `ready` to test if they would rather
branch than catch.

## Choosing between this and the batch estimators

Use `heavytails.tail_index` when the sample fits in memory and you want to sweep
`k`, look at a Hill plot, or apply the robust estimators — `StreamingTailIndex`
fixes `k` when it is constructed, and the trimmed and bias-reduced estimators
are not available over a stream.

Use `heavytails.streaming` when the data does not fit, arrives continuously, or
is being watched for change.
