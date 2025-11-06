# Documentation Overview

This document provides an overview of the HeavyTails documentation structure and what has been created.

--------------------------------------------------------------------------------

## Documentation Structure

The HeavyTails documentation is organized into the following sections:

### 1\. Getting Started

**Location:** `docs/getting-started/`

**Purpose:** Help new users get up and running quickly

**Files Created:**

- `installation.md` - Installation instructions for various use cases
- `quickstart.md` - 10-minute tutorial covering essential operations
- `concepts.md` - Fundamental concepts of heavy-tailed distributions

**Target Audience:** Beginners, students, first-time users

--------------------------------------------------------------------------------

### 2\. User Guide

**Location:** `docs/guide/`

**Purpose:** Comprehensive documentation for using the library

**Files Created:**

- `distributions.md` - Complete overview of all distributions with usage examples
- `tail-estimation.md` - Guide to tail index estimation methods (Hill, Pickands, Moment)
- `pdf-cdf.md` - Working with probability functions (TODO)
- `sampling.md` - Random sampling and reproducibility (TODO)
- `fitting.md` - Parameter estimation from data (TODO)
- `diagnostics.md` - Goodness-of-fit and diagnostic tools (TODO)

**Target Audience:** Regular users, practitioners, data scientists

--------------------------------------------------------------------------------

### 3\. Examples

**Location:** `docs/examples/`

**Purpose:** Real-world application examples

**Planned Files:**

- `basic_usage.ipynb` - Basic usage patterns
- `finance.ipynb` - Financial applications (VaR, ES, risk management)
- `risk.ipynb` - Risk management applications
- `extreme_values.ipynb` - Extreme value analysis
- `comparisons.ipynb` - Comparing distributions

**Target Audience:** Applied researchers, quantitative analysts

--------------------------------------------------------------------------------

### 4\. Mathematical Background

**Location:** `docs/theory/`

**Purpose:** Rigorous mathematical foundations

**Files Created:**

- `heavy-tails.md` - Complete mathematical theory of heavy tails
- `evt.md` - Extreme value theory (TODO)
- `tail-estimation.md` - Theory of tail index estimators (TODO)
- `special-functions.md` - Special functions implementation (TODO)

**Target Audience:** Academic researchers, PhD students, mathematicians

--------------------------------------------------------------------------------

### 5\. API Reference

**Location:** `docs/reference/`

**Purpose:** Auto-generated API documentation

**Configuration:**

- `gen_ref_pages.py` - Script to generate API docs from docstrings

**Auto-Generated Files:**

- `heavy_tails.md` - Core distributions API
- `extra_distributions.md` - Extra distributions API
- `discrete.md` - Discrete distributions API
- `tail_index.md` - Tail estimators API
- `plotting.md` - Plotting utilities API
- `utilities.md` - Utility functions API

**Target Audience:** Developers, advanced users

--------------------------------------------------------------------------------

### 6\. Development

**Location:** `docs/development/`

**Purpose:** Contributing guidelines and development information

**Files Created:**

- `contributing.md` - Comprehensive contributing guide
- `testing.md` - Testing guidelines (TODO)
- `benchmarks.md` - Performance benchmarks (TODO)
- `changelog.md` - Release notes (TODO)

**Target Audience:** Contributors, maintainers

--------------------------------------------------------------------------------

### 7\. About

**Location:** `docs/about/`

**Purpose:** Project information and metadata

**Files Created:**

- `license.md` - MIT License text and explanation
- `citation.md` - Citation guidelines for academic use
- `authors.md` - Author and contributor information

**Target Audience:** All users, especially academic researchers

--------------------------------------------------------------------------------

## Documentation Features

### Mathematical Notation

All documentation uses **KaTeX** for rendering mathematical notation:

- **Inline math:** `$P(X > x)$`
- **Display math:** `$$P(X > x) = x^{-\alpha}$$`
- **Custom macros** defined in `docs/javascripts/katex.js`

**Examples:**

- $\alpha$ - Tail index
- $\mathbb{E}[X]$ - Expected value
- $\bar{F}(x) = P(X > x)$ - Survival function

### Code Examples

All code examples are:

- **Runnable** - Can be copy-pasted and executed
- **Self-contained** - Include all necessary imports
- **Reproducible** - Use `seed` parameters for random sampling
- **Commented** - Explain key concepts

### Admonitions

Documentation uses Material for MkDocs admonitions:

- `!!! note` - Additional information
- `!!! tip` - Helpful suggestions
- `!!! warning` - Important caveats
- `!!! example` - Code examples

### Cross-References

Extensive cross-referencing between sections:

- Concepts → Theory → Examples
- User Guide → API Reference
- Getting Started → User Guide

--------------------------------------------------------------------------------

## Building the Documentation

### Local Development

```bash
# Install dependencies
poetry install --with docs

# Serve documentation locally
poetry run mkdocs serve

# Access at http://127.0.0.1:8000
```

### Build Static Site

```bash
# Build documentation
poetry run mkdocs build

# Output in site/ directory
```

### Deploy to GitHub Pages

```bash
# Deploy to gh-pages branch
poetry run mkdocs gh-deploy
```

--------------------------------------------------------------------------------

## Documentation Standards

### Writing Style

- **Academic but accessible** - Rigorous but understandable
- **Example-driven** - Show, don't just tell
- **Mathematically precise** - Use proper notation
- **Practical focus** - Emphasize real-world applications

### Code Style

- **Type hints** for all examples
- **Docstrings** in NumPy style
- **Consistent formatting** with ruff
- **Reproducible** with seeds

### Mathematical Style

- **LaTeX notation** for all formulas
- **Define symbols** before using
- **Reference theorems** properly
- **Include proofs** when instructive

--------------------------------------------------------------------------------

## Maintenance Tasks

### Regular Updates

- [ ] Keep installation instructions current
- [ ] Update version numbers
- [ ] Add new examples as requested
- [ ] Fix typos and clarifications
- [ ] Update API docs when code changes

### Quality Checks

- [ ] Test all code examples
- [ ] Verify mathematical formulas
- [ ] Check cross-references
- [ ] Validate external links
- [ ] Review for accessibility

--------------------------------------------------------------------------------

## Documentation TODO List

### High Priority

1. ✅ Homepage and Getting Started
2. ✅ User Guide (distributions, tail estimation)
3. ✅ Theory (heavy-tails overview)
4. ✅ Development (contributing guide)
5. ✅ About (license, citation, authors)
6. ⏳ User Guide (pdf-cdf, sampling, fitting, diagnostics)
7. ⏳ Examples (Jupyter notebooks)
8. ⏳ Theory (EVT, detailed tail estimation theory)

### Medium Priority

- [ ] Tutorial videos (if applicable)
- [ ] FAQ section
- [ ] Troubleshooting guide
- [ ] Performance tips
- [ ] Advanced topics

### Low Priority

- [ ] Internationalization
- [ ] PDF export
- [ ] Printable documentation
- [ ] Interactive examples (Binder)

--------------------------------------------------------------------------------

## Contributing to Documentation

Documentation contributions are highly valued! You can help by:

1. **Fixing typos** - Even small fixes matter
2. **Adding examples** - Real-world use cases
3. **Clarifying explanations** - Make concepts clearer
4. **Writing tutorials** - Step-by-step guides
5. **Translating** - Make docs accessible globally

See [Contributing Guide](development/contributing.md) for details.

--------------------------------------------------------------------------------

## Documentation Analytics

Once deployed, track:

- **Page views** - Which sections are most popular
- **Search queries** - What users are looking for
- **Feedback** - Use Material's built-in feedback
- **GitHub issues** - Documentation improvement requests

--------------------------------------------------------------------------------

## Questions?

For documentation-related questions:

- **Typos/errors:** [GitHub Issues](https://github.com/diogoribeiro7/heavytails/issues)
- **Suggestions:** [GitHub Discussions](https://github.com/diogoribeiro7/heavytails/discussions)
- **Contact:** [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)

--------------------------------------------------------------------------------

**Last Updated:** 2025-01-06 **Documentation Version:** 0.1.0
