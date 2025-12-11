
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
plot(x, y)
lines(x, lfit$fitted, col = 1) # loess
lfit$enp

## Use this as target df for L2

fit2 <- lspen_dense(x, y, L.method = "d2", df = lfit$enp)
fit1 <- lspen_dense(x, y, L.method = "d2", df = lfit$enp, niter = 50)

lines(x, fit2$fitted, col = 2) # L2
lines(x, fit1$fitted, col = 3) # L1

plot(diff(fit1$fitted, differences = 2))



fit1 <- lspen_dense(x, y, L.method = "d2", df = 23, niter = 50)
plot(diff(fit1$fitted, differences = 2))






