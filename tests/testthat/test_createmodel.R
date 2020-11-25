context("create_model")

test_that("outputs are correct in the Dijkstra algorithm.", {
  inp = c(20,24,24,24,16,16,16,16,16,15)
  rec_drop = rep(0.1,10)
  l_drop = c(0.1,0.1)
  testthat::expect_equal(typeof(create_model(inp,rec_drop,232,c(0.1,0.1),l_drop,'mean_squared_error','adam','mean_absolute_error')),"closure")
})