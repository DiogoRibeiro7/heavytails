from heavytails.extra_distributions import GeneralizedPareto


def test_gpd_inverse():
    gpd = GeneralizedPareto(xi=0.2, sigma=1.0, mu=0.0)
    for u in [0.1, 0.5, 0.9]:
        x = gpd.ppf(u)
        assert abs(gpd.cdf(x) - u) < 1e-6
