\name{pareto_var}
\alias{pareto_var}
\alias{pareto_es}
\title{Pareto Value at Risk and Expected Shortfall}
\usage{
pareto_var(level, shape, scale = 1)
pareto_es(level, shape, scale = 1)
}
\arguments{
\item{level}{Probability strictly between zero and one.}
\item{shape}{Positive finite Pareto exponent.}
\item{scale}{Positive finite lower support bound.}
}
\value{Upper quantile or conditional tail mean. Expected shortfall is
infinite if shape is at most one.}
\description{Analytic tail-risk measures for Pareto type I losses.}
\details{For shape greater than one, expected shortfall is value at risk
multiplied by shape divided by (shape minus one).}
\examples{
pareto_var(0.99, 2)
pareto_es(0.99, 2)
}
