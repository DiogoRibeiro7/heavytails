# 📅 Monthly Maintenance Reminder

This issue tracks recurring housekeeping for this repository. A bot will add a monthly comment with fresh stats (commits, PRs, issues, stars, and CI status). Use the checklist below to guide maintenance tasks.

---

## ✅ Core Maintenance

- [ ] Review **open pull requests** and close or merge stale ones
- [ ] Triage **open issues** (confirm repro, label, assign, close if resolved)
- [ ] Update **README** (quickstart, examples, badges, links)
- [ ] Verify **license** year and author info
- [ ] Refresh **CHANGELOG.md** with notable changes since last month
- [ ] Check **.gitignore** / **.gitattributes** coverage
- [ ] Verify **Dependabot** / update strategy is in place

---

## 🧪 Quality & CI

- [ ] Ensure CI passes on supported Python versions
- [ ] Run local tests: `pytest -q` (or your test runner)
- [ ] Coverage target met (e.g. ≥ 90%): `pytest --cov`
- [ ] Static analysis (linters/type checks):
  - [ ] Ruff (style): `ruff check .`
  - [ ] MyPy (types): `mypy src/`
  - [ ] Bandit (security): `bandit -r src/`
  - [ ] pip-audit: `pip-audit` (or `uv pip audit`)
- [ ] Pre-commit hooks updated & passing: `pre-commit run --all-files`
- [ ] Verify latest CI workflow runs and cache settings

---

## 📦 Packaging (if published)

- [ ] Build artifacts clean: `python -m build`
- [ ] Metadata valid: `twine check dist/*`
- [ ] Install test: `pip install dist/*.whl` (or `uv pip install`)
- [ ] Version bump policy followed (semver / semantic-release)
- [ ] Wheels include correct files (no large/unneeded assets)

---

## 📚 Docs & Examples

- [ ] Docs build cleanly (Sphinx/MkDocs): `make docs` or `mkdocs build`
- [ ] Examples & notebooks run end-to-end
- [ ] API reference reflects current code
- [ ] Tutorial / “Getting Started” tested on a clean env

---

## 🔐 Security & Compliance

- [ ] Secrets **not** in repo (scan)
- [ ] Third-party licenses reviewed
- [ ] Minimal required GitHub permissions in workflows
- [ ] Pin critical dependencies or use constraints file

---

## 🚀 Release Readiness

- [ ] Milestones/labels reflect upcoming work
- [ ] Roadmap updated
- [ ] Tag & release notes drafted (features, fixes, breaking changes)
- [ ] Benchmarks (if any) updated and reproducible

---

## 🧠 Notes

- A monthly bot comment will summarize recent activity and CI status.
- Checklists are suggestive. Adapt or extend to match this project.
- Consider setting a repo variable `MAINTAINER` to auto-assign this issue.

_Thanks for keeping the project healthy._
