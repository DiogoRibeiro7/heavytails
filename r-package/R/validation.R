# Internal argument checks.
.assert_pos <- function(x, name) {
  if (!is.numeric(x) || is.object(x) || length(x) != 1L ||
      is.na(x) || !is.finite(x) || x <= 0)
    stop(sprintf("%s must be a single finite positive number", name), call. = FALSE)
}
.assert_vector <- function(x) {
  if (!is.numeric(x) || is.object(x)) stop("input must be numeric", call. = FALSE)
}
.assert_bool <- function(x) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop("flag must be TRUE or FALSE", call. = FALSE)
}
.assert_level <- function(level) {
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
      !is.finite(level) || level <= 0 || level >= 1)
    stop("level must be strictly between zero and one", call. = FALSE)
}
