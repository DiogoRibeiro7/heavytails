from heavytails import LogNormal


def test_log_normal_ppf_inverse():
    ln = LogNormal(mu=0.0, sigma=1.0)
    for u in [0.1, 0.5, 0.9]:
        x = ln.ppf(u)
        assert abs(ln.cdf(x) - u) < 1e-6
