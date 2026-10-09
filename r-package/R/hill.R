#' Hill Extreme-Value Index Estimator
#'
#' Estimates the extreme-value index gamma (not Pareto exponent alpha).
#' @param x Positive finite numeric observations.
#' @param k Integer number of upper order statistics, 1 <= k < length(x).
#' @return A non-negative numeric scalar estimating gamma = 1 / alpha.
#' @details Uses the average log excess over the (k+1)th descending
#' order statistic. Assumes an approximately regularly varying upper tail.
#' @references Hill (1975), doi:10.1214/aos/1176343247.
#' @examples
#' hill_index(c(1, 2, 4, 8, 16), 2)
#' @export
hill_index <- function(x, k) {
  .assert_vector(x)
  if (length(x) < 2L || any(!is.finite(x)) || any(x <= 0))
    stop("x must contain at least two finite positive observations", call. = FALSE)
  if (!is.numeric(k) || length(k) != 1L || is.na(k) ||
      !is.finite(k) || k < 1 || k >= length(x) || k != floor(k))
    stop("k must be an integer in [1, length(x)-1]", call. = FALSE)
  sorted <- sort(x, decreasing = TRUE)
  mean(log(sorted[seq_len(k)]) - log(sorted[k + 1L]))
}
