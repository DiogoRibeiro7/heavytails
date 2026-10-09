\name{hill_index}
\alias{hill_index}
\title{Hill Extreme-Value Index Estimator}
\usage{hill_index(x, k)}
\arguments{
\item{x}{Positive finite numeric observations.}
\item{k}{Number of upper order statistics.}
}
\value{Numeric scalar estimating gamma, the reciprocal of the Pareto exponent.}
\description{Hill estimator of the positive extreme-value index.}
\details{Uses the average of log ratios of the top k values to the
(k+1)th descending order statistic. Results depend on the threshold choice.}
\references{Hill (1975). \doi{10.1214/aos/1176343247}}
\examples{hill_index(c(1, 2, 4, 8, 16), 2)}
