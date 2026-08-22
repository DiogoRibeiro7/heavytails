# Citation

If you use heavytails in your research, teaching, or professional work, please
cite it.

--------------------------------------------------------------------------------

## Which DOI to use

The software is archived on Zenodo, which mints two kinds of DOI. They are not
interchangeable and the difference matters for reproducibility.

| DOI | Resolves to | Cite it when |
| --- | --- | --- |
| [10.5281/zenodo.22045594](https://doi.org/10.5281/zenodo.22045594) | **All versions** — always the most recent release | You mean "this software", and the exact version is not part of the claim |
| [10.5281/zenodo.22050721](https://doi.org/10.5281/zenodo.22050721) | **Version 0.3.0 only** | A result depends on the version you ran, which for numerical work it usually does |

The first is the *concept DOI*. It is the one to use by default, and the one on
the badge in the README.

!!! tip "Reproducibility wants the version DOI"
    If a reviewer should be able to reproduce a number, cite the version DOI
    and state the version in the text. A concept DOI resolves to whatever is
    newest at the time of reading, which may not be what you ran. Every release
    has its own version DOI, listed under "Versions" on the Zenodo record.

--------------------------------------------------------------------------------

## BibTeX

Citing all versions:

```bibtex
@software{ribeiro_heavytails,
  author    = {Ribeiro, Diogo},
  title     = {heavytails: A Pure-Python Library for Heavy-Tailed
               Probability Distributions},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22045594},
  url       = {https://doi.org/10.5281/zenodo.22045594}
}
```

Citing release 0.3.0 exactly:

```bibtex
@software{ribeiro_heavytails_0_3_0,
  author    = {Ribeiro, Diogo},
  title     = {heavytails: A Pure-Python Library for Heavy-Tailed
               Probability Distributions},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22050721},
  url       = {https://doi.org/10.5281/zenodo.22050721}
}
```

--------------------------------------------------------------------------------

## Other styles

### APA

> Ribeiro, D. (2026). *heavytails: A Pure-Python library for heavy-tailed
> probability distributions* (Version 0.3.0) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.22050721>

### IEEE

> D. Ribeiro, "heavytails: A Pure-Python Library for Heavy-Tailed Probability
> Distributions," version 0.3.0, Zenodo, 2026. doi: 10.5281/zenodo.22050721.

### MLA

> Ribeiro, Diogo. *heavytails: A Pure-Python Library for Heavy-Tailed
> Probability Distributions*. Version 0.3.0, Zenodo, 2026,
> doi:10.5281/zenodo.22050721.

### Chicago

> Ribeiro, Diogo. 2026. "heavytails: A Pure-Python Library for Heavy-Tailed
> Probability Distributions." Version 0.3.0. Zenodo.
> <https://doi.org/10.5281/zenodo.22050721>.

--------------------------------------------------------------------------------

## In-text

- "Tail indices were estimated with the Hill estimator (Hill, 1975) as
  implemented in heavytails (Ribeiro, 2026)."
- "Aggregate losses were computed by Panjer recursion (Panjer, 1981) using
  heavytails 0.3.0 (Ribeiro, 2026)."

--------------------------------------------------------------------------------

## Cite the method as well as the software

The library is an implementation, not the source of the methods it implements.
Where a result rests on a particular estimator, cite the paper it comes from
alongside the software. Every one of these is listed in `CITATION.cff` and
`.zenodo.json`, so the entry can be copied rather than hunted down.

| What you used | Cite |
| --- | --- |
| `hill_estimator` | Hill (1975), [10.1214/aos/1176343247](https://doi.org/10.1214/aos/1176343247) |
| `pickands_estimator` | Pickands (1975), [10.1214/aos/1176343003](https://doi.org/10.1214/aos/1176343003) |
| `moment_estimator` | Dekkers, Einmahl & de Haan (1989), [10.1214/aos/1176347397](https://doi.org/10.1214/aos/1176347397) |
| `smoothed_hill_estimator` | Resnick & Stărică (1997), [10.2307/1427870](https://doi.org/10.2307/1427870) |
| `generalized_hill_estimator` | Beirlant, Vynckier & Teugels (1996), *Bernoulli* 2(4), 293–318 |
| `bias_reduced_hill_estimator` | Caeiro, Gomes & Pestana (2005), *REVSTAT* 3(2), 113–136 |
| `harmonic_moment_estimator`, `t_hill_estimator` | Beran, Schell & Stehlík (2014), [10.1007/s10463-013-0412-2](https://doi.org/10.1007/s10463-013-0412-2) |
| `trimmed_hill_estimator`, `adaptive_trimmed_hill_estimator` | Bhattacharya, Kallitsis & Stoev (2019), [arXiv:1705.03088](https://arxiv.org/abs/1705.03088) |
| `gpd_mle_estimator`, `fit_generalized_pareto` | Grimshaw (1993), [10.1080/00401706.1993.10485040](https://doi.org/10.1080/00401706.1993.10485040) |
| Peaks-over-threshold generally | Balkema & de Haan (1974), [10.1214/aop/1176996548](https://doi.org/10.1214/aop/1176996548) |
| `panjer_recursion` | Panjer (1981), [10.1017/S0515036100006796](https://doi.org/10.1017/S0515036100006796) |
| Anderson–Darling *p*-values | Marsaglia & Marsaglia (2004), [10.18637/jss.v009.i02](https://doi.org/10.18637/jss.v009.i02) |

!!! note "The adaptive trimming rule is not from the paper"
    `adaptive_trimmed_hill_estimator` uses the *trimmed* estimator of
    Bhattacharya, Kallitsis and Stoev, but the rule for choosing how much to
    trim is a sequential exact test on the log-spacings rather than their
    adaptive procedure. Cite their paper for the estimator; the selection rule
    is part of this software.

--------------------------------------------------------------------------------

## Acknowledgement instead of citation

Where a citation is not appropriate:

> "Computational analyses were performed using heavytails, a pure-Python
> library for heavy-tailed distributions (<https://doi.org/10.5281/zenodo.22045594>)."

--------------------------------------------------------------------------------

## How the metadata is kept honest

`CITATION.cff` holds the shared citation fields, and `.zenodo.json` overrides
what Zenodo would otherwise infer from the GitHub release. The two are checked
against each other, including that every reference in one appears in the other:

```bash
python scripts/validate_zenodo_metadata.py
```

`tests/test_zenodo_metadata.py` runs the same checks in CI, so the two files
cannot drift apart unnoticed. The release workflow is in
[Releasing](../development/releasing.md).

--------------------------------------------------------------------------------

## Contact

- **Email:** [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- **ORCID:** [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
- **GitHub:** [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)

If you publish work that uses heavytails, an email is welcome — knowing what
the library is used for is what decides which parts of it get attention.
