

## Ad hoc algorithm to fit a piecewise linear y ~ x regression, and
## use heuristics to find some knots that may be reasonable for a y ~
## bs(x) model.


df2knots <- function(df, x, y, n = 1, niter = 20)
{
    if (missing(n) && anyDuplicated(x))
        stop("'x' values must be distinct (with weights in 'n')")

    lssfit1 <- lspen_dense(x, y, n = n, L.method = "d2", df = df, niter = niter)

    d2s <- diff(lssfit1$fitted, differences = 2)
    d2s[abs(d2s) < 0.0001] <- 0

    srle <- list2DF(rle(sign(d2s))) |>
        within({
            ends <- cumsum(lengths)
            starts <- 1 + c(0, head(ends, -1))
        }) |>
        subset(values != 0)

    eknots <- c(ux[1] - 1,
                0.5 * (ux[srle$starts - 1] + ux[srle$ends - 1]),
                ux[length(ux)] + 1)

    eknots <- 0.5 * (ux[srle$starts - 1] + ux[srle$ends - 1])

    list(knots = eknots,
         lambda = lambda,
         enp = enp)
}

