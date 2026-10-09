\name{dpareto_ht}
\alias{dpareto_ht}
\alias{ppareto_ht}
\alias{qpareto_ht}
\alias{rpareto_ht}
\title{Pareto Type I Distribution}
\usage{
dpareto_ht(x, shape, scale = 1, log = FALSE)
ppareto_ht(q, shape, scale = 1, lower.tail = TRUE, log.p = FALSE)
qpareto_ht(p, shape, scale = 1, lower.tail = TRUE, log.p = FALSE)
rpareto_ht(n, shape, scale = 1)
}
\arguments{
\item{x, q}{Numeric vector of evaluation points.}
\item{p}{Numeric vector of probabilities.}
\item{n}{Non-negative integer sample size.}
\item{shape}{Positive finite Pareto exponent.}
\item{scale}{Positive finite lower support bound.}
\item{log}{Return log density.}
\item{lower.tail}{Return lower rather than upper probabilities.}
\item{log.p}{Return or accept log probabilities.}
}
\value{Numeric vector.}
\description{Pareto type I density, CDF, quantile and random generation.}
\details{For x greater than or equal to the scale, survival is
\eqn{(scale/x)^{shape}}. Outside support the density and CDF are zero.}
\examples{
dpareto_ht(2, shape = 2)
qpareto_ht(0.99, shape = 2)
}
