import math

from heavytails import Pareto


def test_pareto_pdf_cdf_consistency():
    p = Pareto(alpha=2, xm=1)
    for x in [1, 2, 5]:
        pdf = p.pdf(x)
        cdf = p.cdf(x)
        assert 0 <= cdf <= 1
        assert pdf >= 0
        # numerical differentiation sanity
        eps = 1e-6
        dF = (p.cdf(x+eps) - p.cdf(x-eps)) / (2*eps)
        assert math.isclose(pdf, dF, rel_tol=1e-3)
