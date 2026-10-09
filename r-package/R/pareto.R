#' Pareto Type I Distribution
#'
#' Density, distribution, quantile, and random sampling, following R conventions.
#' @param x,q Numeric vectors of evaluation points.
#' @param p Numeric vector of probabilities.
#' @param n Non-negative integer sample size.
#' @param shape Positive Pareto exponent alpha.
#' @param scale Positive lower support bound.
#' @param log Return log density.
#' @param lower.tail Return lower CDF rather than upper survival.
#' @param log.p Return or accept log probabilities.
#' @return Numeric vector of results, respecting missing values.
#' @details For x >= scale, the survival function is (scale/x)^shape.
#' @examples
#' dpareto_ht(2, shape = 2)
#' qpareto_ht(0.99, shape = 2)
#' @export
dpareto_ht <- function(x, shape, scale = 1, log = FALSE) {
  .assert_vector(x); .assert_pos(shape, "shape"); .assert_pos(scale, "scale")
  .assert_bool(log)
  ans <- rep(if (log) -Inf else 0, length(x))
  ans[is.na(x)] <- NA_real_
  inside <- !is.na(x) & x >= scale
  ld <- base::log(shape) - base::log(scale) -
    (shape + 1) * (base::log(x[inside]) - base::log(scale))
  ans[inside] <- if (log) ld else exp(ld)
  ans
}
#' @rdname dpareto_ht
#' @export
ppareto_ht <- function(q, shape, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  .assert_vector(q); .assert_pos(shape, "shape"); .assert_pos(scale, "scale")
  .assert_bool(lower.tail); .assert_bool(log.p)
  ans <- rep(NA_real_, length(q))
  below <- !is.na(q) & q < scale
  inside <- !is.na(q) & q >= scale
  ls <- shape * (base::log(scale) - base::log(q[inside]))
  if (lower.tail) {
    ans[below] <- if (log.p) -Inf else 0
    pr <- -expm1(ls)
    ans[inside] <- if (log.p) base::log(pr) else pr
  } else {
    ans[below] <- if (log.p) 0 else 1
    ans[inside] <- if (log.p) ls else exp(ls)
  }
  ans
}
#' @rdname dpareto_ht
#' @export
qpareto_ht <- function(p, shape, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  .assert_vector(p); .assert_pos(shape, "shape"); .assert_pos(scale, "scale")
  .assert_bool(lower.tail); .assert_bool(log.p)
  invalid <- if (log.p) !is.na(p) & p > 0 else !is.na(p) & (p < 0 | p > 1)
  if (any(invalid)) stop("invalid probability", call. = FALSE)
  ls <- if (lower.tail) {
    if (log.p) log1p(-exp(p)) else log1p(-p)
  } else {
    if (log.p) p else base::log(p)
  }
  exp(base::log(scale) - ls / shape)
}
#' @rdname dpareto_ht
#' @export
rpareto_ht <- function(n, shape, scale = 1) {
  if (!is.numeric(n) || length(n) != 1L || is.na(n) ||
      !is.finite(n) || n < 0 || n != floor(n) || n > .Machine$integer.max)
    stop("n must be a non-negative integer", call. = FALSE)
  .assert_pos(shape, "shape"); .assert_pos(scale, "scale")
  qpareto_ht(stats::runif(n), shape = shape, scale = scale)
}
