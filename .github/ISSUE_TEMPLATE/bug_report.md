---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: 'diogoribeiro7'
---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

1. Import distribution: `from heavytails import ...`
2. Set parameters: `dist = Distribution(param1=..., param2=...)`
3. Call method: `result = dist.method(argument)`
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Code Example

```python
from heavytails import Pareto

# Minimal reproducible example
dist = Pareto(alpha=2.0, xm=1.0)
result = dist.pdf(5.0)  # This causes the issue
print(result)
```

## Error Message

```
Paste the complete error message and traceback here
```

## Environment

**System Information:**
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13.1]
- Python version: [e.g., 3.11.2]
- HeavyTails version: [e.g., 0.1.0]

**Installation method:**
- [ ] pip install heavytails
- [ ] poetry add heavytails
- [ ] Installed from source
- [ ] Development installation

**Dependencies (if relevant):**
```bash
# Output of: pip list | grep -E "(numpy|scipy|matplotlib|pandas)"
```

## Mathematical Context

*Fill this section if the bug is related to mathematical computations*

**Distribution:** [e.g., Pareto, Student-t, GPD]
**Parameter values:** [e.g., alpha=1.5, xm=1.0]
**Input values:** [e.g., x=10.0, u=0.95]
**Expected mathematical result:** [e.g., based on formula/reference]

## Severity Assessment

- [ ] Critical (library unusable, security issue)
- [ ] High (major functionality broken)
- [ ] Medium (some functionality broken)
- [ ] Low (minor issue, workaround exists)

## Additional Context

**Possible Cause:**
*If you have insights into what might be causing the issue*

**Workaround:**
*If you found a temporary workaround*

**Related Issues:**
*Link to any related issues*

**Screenshots/Plots:**
*If applicable, add visual evidence*

## Validation

**Have you tested this with:**
- [ ] Different parameter values
- [ ] Different Python versions
- [ ] Different operating systems
- [ ] Reference implementation (R, SciPy, etc.)

**Reference comparison:**
*If you compared against other implementations, include results*

## Reproducibility

- [ ] Bug occurs consistently
- [ ] Bug occurs intermittently
- [ ] Bug depends on specific conditions

**Random seed (if applicable):** [e.g., 42]

---

## For Maintainers

**Priority:** [ ] P0 | [ ] P1 | [ ] P2 | [ ] P3

**Component:** [ ] Core | [ ] CLI | [ ] Documentation | [ ] Tests | [ ] CI/CD

**Regression:** [ ] Yes | [ ] No | [ ] Unknown
*Since version:* [if regression]
