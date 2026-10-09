test_that("Hill estimates gamma and is invariant to scale", {
  x <- c(1, 2, 4, 8, 16)
  expect_equal(hill_index(x, 2), mean(log(c(16, 8) / 4)))
  expect_equal(hill_index(5 * x, 2), hill_index(x, 2))
  expect_error(hill_index(c(1, 0, 2), 1), "positive")
  expect_error(hill_index(x, 5), "k")
})
test_that("Pareto risk obeys the mean existence boundary", {
  expect_equal(pareto_var(0.99, 2), 10)
  expect_equal(pareto_es(0.99, 2), 20)
  expect_identical(pareto_es(0.99, 1), Inf)
  expect_identical(pareto_es(0.99, 0.5), Inf)
  expect_error(pareto_var(1, 2), "level")
})
