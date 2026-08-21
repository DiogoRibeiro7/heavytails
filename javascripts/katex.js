// KaTeX auto-render configuration for mathematical notation
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$",  right: "$$",  display: true },
      { left: "$",   right: "$",   display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    throwOnError: false,
    strict: false,
    trust: true,
    macros: {
      "\\R": "\\mathbb{R}",
      "\\N": "\\mathbb{N}",
      "\\E": "\\mathbb{E}",
      "\\P": "\\mathbb{P}",
      "\\Var": "\\text{Var}",
      "\\Cov": "\\text{Cov}",
      "\\ind": "\\mathbf{1}",
      "\\iid": "\\overset{\\text{iid}}{\\sim}",
      "\\as": "\\overset{\\text{a.s.}}{\\to}",
      "\\prob": "\\overset{\\text{p}}{\\to}",
      "\\dist": "\\overset{\\text{d}}{\\to}"
    }
  })
})
