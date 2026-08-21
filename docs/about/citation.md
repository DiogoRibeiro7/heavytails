# Citation

If you use HeavyTails in your research, teaching, or professional work, please cite it appropriately.

--------------------------------------------------------------------------------

## BibTeX Entry

Use this entry until the first Zenodo archive DOI has been minted:

```bibtex
@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo},
  title = {heavytails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/diogoribeiro7/heavytails},
  version = {0.1.0}
}
```

After the first GitHub release is archived by Zenodo, cite the version DOI for
the exact release used:

```bibtex
@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo},
  title = {heavytails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  year = {2025},
  publisher = {Zenodo},
  url = {https://github.com/diogoribeiro7/heavytails},
  version = {0.1.0},
  doi = {ZENODO_VERSION_DOI}
}
```

!!! note "Zenodo DOI"
    Do not cite a placeholder DOI. The Zenodo DOI will be added here after the
    first archived GitHub release.

--------------------------------------------------------------------------------

## APA Style

Ribeiro, D. F. (2025). _HeavyTails: A Pure-Python library for heavy-tailed probability distributions_ (Version 0.1.0) [Computer software]. <https://github.com/diogoribeiro7/heavytails>

--------------------------------------------------------------------------------

## IEEE Style

D. F. Ribeiro, "HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions," 2025\. [Online]. Available: <https://github.com/diogoribeiro7/heavytails>

--------------------------------------------------------------------------------

## MLA Style

Ribeiro, Diogo F. _HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions_. Version 0.1.0, GitHub, 2025, <https://github.com/diogoribeiro7/heavytails>.

--------------------------------------------------------------------------------

## Chicago Style

Ribeiro, Diogo F. 2025\. "HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions." Computer software. Version 0.1.0\. <https://github.com/diogoribeiro7/heavytails>.

--------------------------------------------------------------------------------

## Text Citation Examples

### In-text (author-year)

- "We used the HeavyTails library (Ribeiro, 2025) to estimate tail indices..."
- "Tail index estimation was performed using the Hill estimator implemented in HeavyTails (Ribeiro, 2025)."

### In-text (numbered)

- "We used the HeavyTails library [1] to estimate tail indices..."

--------------------------------------------------------------------------------

## Citing Specific Distributions or Methods

If you use a specific distribution or estimator, also cite the original paper:

### Example: Pareto Distribution

```bibtex
@article{pareto1896cours,
  author = {Pareto, Vilfredo},
  title = {Cours d'économie politique},
  journal = {Lausanne: F. Rouge},
  year = {1896}
}

@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo F.},
  title = {HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  year = {2025},
  url = {https://github.com/diogoribeiro7/heavytails}
}
```

**Text:** "We modeled wealth distribution using the Pareto distribution (Pareto, 1896) implemented in HeavyTails (Ribeiro, 2025)."

### Example: Hill Estimator

```bibtex
@article{hill1975simple,
  author = {Hill, Bruce M.},
  title = {A Simple General Approach to Inference About the Tail of a Distribution},
  journal = {The Annals of Statistics},
  volume = {3},
  number = {5},
  pages = {1163--1174},
  year = {1975}
}

@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo F.},
  title = {HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  year = {2025},
  url = {https://github.com/diogoribeiro7/heavytails}
}
```

**Text:** "Tail indices were estimated using the Hill estimator (Hill, 1975) as implemented in HeavyTails (Ribeiro, 2025)."

### Example: Generalized Pareto Distribution

```bibtex
@article{pickands1975statistical,
  author = {Pickands, James},
  title = {Statistical Inference Using Extreme Order Statistics},
  journal = {The Annals of Statistics},
  volume = {3},
  number = {1},
  pages = {119--131},
  year = {1975}
}

@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo F.},
  title = {HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  year = {2025},
  url = {https://github.com/diogoribeiro7/heavytails}
}
```

--------------------------------------------------------------------------------

## Academic Papers Using HeavyTails

If you publish a paper using HeavyTails, please let us know! We'll list it here:

- _(Your paper could be listed here!)_

--------------------------------------------------------------------------------

## Acknowledgments in Papers

If citing is not appropriate (e.g., in acknowledgments), you can use:

> "This research was supported by the HeavyTails Python library (<https://github.com/diogoribeiro7/heavytails>)."

Or:

> "Computational analyses were performed using HeavyTails, a pure-Python library for heavy-tailed distributions."

--------------------------------------------------------------------------------

## Zenodo Metadata

The repository includes `.zenodo.json`, which Zenodo uses to override GitHub's
default release metadata. It records the software title, author ORCID, MIT
license, keywords, scholarly references, source repository, and documentation
URL.

Shared citation fields are kept in sync with `CITATION.cff` by the repository
validator.

Maintainers should validate it before releases:

```bash
python scripts/validate_zenodo_metadata.py
```

The full release workflow is documented in
[Releasing](../development/releasing.md).

--------------------------------------------------------------------------------

## Software Metadata

For software repositories and metadata:

```yaml
name: HeavyTails
description: Pure-Python library for heavy-tailed probability distributions
author: Diogo F. Ribeiro
version: 0.1.0
license: MIT
repository: https://github.com/diogoribeiro7/heavytails
keywords:
  - heavy-tailed distributions
  - extreme value theory
  - Pareto distribution
  - tail index estimation
  - risk management
  - pure Python
```

--------------------------------------------------------------------------------

## CITATION.cff File

HeavyTails includes a `CITATION.cff` file for automatic citation generation:

```yaml
cff-version: 1.2.0
message: "If you use this library, please cite as below."
title: "heavytails: A Pure-Python Library for Heavy-Tailed Probability Distributions"
authors:
  - family-names: Ribeiro
    given-names: Diogo
    orcid: "https://orcid.org/0009-0001-2022-7072"
version: 0.1.0
date-released: 2025-10-25
url: https://diogoribeiro7.github.io/heavytails
license: MIT
repository-code: https://github.com/DiogoRibeiro7/heavytails
keywords:
  - heavy tails
  - heavy-tailed distributions
  - extreme value theory
  - tail index estimation
```

--------------------------------------------------------------------------------

## Contact

For citation-related questions or to report your publication:

- **Email:** [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- **ORCID:** [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
- **GitHub:** [@diogoribeiro7](https://github.com/diogoribeiro7)

--------------------------------------------------------------------------------

## Contributing to HeavyTails

If you've benefited from HeavyTails, consider:

- ⭐ **Starring** the repository on GitHub
- 📝 **Citing** in your publications
- 🐛 **Reporting** bugs or suggesting features
- 💻 **Contributing** code or documentation
- 📢 **Sharing** with colleagues and students

See [Contributing Guide](../development/contributing.md) for details.

--------------------------------------------------------------------------------

**Thank you for using and citing HeavyTails!** 🎓📊
