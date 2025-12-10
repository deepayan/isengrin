
library(isengrin)

x <- 1:100
y <- rnorm(length(x), mean = sqrt(x), sd = 1)

plot(y ~ x)

fit <- lspen_dense(x, y, L.method = "d2", lambda = 1000)

lines(x, fit$fitted)



