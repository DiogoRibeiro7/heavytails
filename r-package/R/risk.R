#' Pareto Value at Risk and Expected Shortfall
#'
#' @param level Probability strictly between zero and one.
#' @param shape Positive finite Pareto exponent.
#' @param scale Positive finite Pareto scale.
#' @return The upper quantile or conditional tail mean; the latter
#' is infinite for shape <= 1.
#' @examples
#' pareto_var(0.99, 2)
#' pareto_es(0.99, 2)
#' @export
pareto_var <- function(level, shape, scale = 1) {
  .assert_level(level)
  qpareto_ht(level, shape, scale)
}
#' @rdname pareto_var
#' @export
pareto_es <- function(level, shape, scale = 1) {
  .assert_level(level); .assert_pos(shape, "shape"); .assert_pos(scale, "scale")
  if (shape <= 1) return(Inf)
  pareto_var(level, shape, scale) * shape / (shape - 1)
}
