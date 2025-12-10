
library(isengrin)

constructL_1d_dense(5, filter = c(-1, 1))
constructL_1d_dense(7, filter = c(1, -2, 1))


x <- 1:100
y <- rnorm(length(x), mean = sqrt(x), sd = 1)


fit2 <- lspen_dense(x, y, L.method = "d2", lambda = 1000)
fit1 <- lspen_dense(x, y, L.method = "d2", lambda = 100, niter = 10)

plot(x, y)
lines(x, fit2$fitted)
lines(x, fit1$fitted, col = 3)

## choose lambda by df

lfit <- loess(y ~ x)
lines(x, lfit$fitted, col = 2)
lfit$enp
