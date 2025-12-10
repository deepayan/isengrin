


irls_wts <- function(x, eps = 0.0001) {
    w <- sqrt(as.vector(abs(x)))
    w[w < eps] <- eps
    1 / w
}


lambda2df <- function(lambda, eta, L, n, B = sqrt(n) * L)
{
    if (missing(eta))
        eta <- eigen(tcrossprod(B), symmetric = TRUE, only.values = TRUE)$values
    sum(1 / (1 + lambda * eta))
}


## Proof-of-concept reference implementation using dense matrix operations

constructL_1d_dense <- function(p, filter = c(1, -2, 1))
{
    flen <- length(filter)
    stopifnot("'p' must be larger than 'length(filter)'" = p > flen)
    col1 <- c(filter, rep(0, p - flen))
    L <- toeplitz(col1, # 1st column
                  rep(c(1, 0), c(1, p-1))) # 1st row
    L <- L[, seq_len(p - flen + 1)]
    L
}



lspen_dense <- function(x, y, n = 1, L, L.method = "d1", df = NULL,
                        lambda = df2lambda(df), niter = 0)
{
    p <- length(y)
    stopifnot("'n' must be scalar or have same length as y" = length(n) %in% c(1, p))
    if (length(n) == 1) n <- rep(n, p)
    N <- diag(x = n) # DENSE but could be SPARSE

    if (missing(df)) df <- sqrt(sum(n)) # default df = sqrt(nobs)

    if (missing(L)) {
        stopifnot("Unsupported 'L.method'" = L.method %in% c("d1", "d2"))
        filter <- switch(L.method, d1 = c(-1, 1), d2 = c(1, -2, 1))
        L <- constructL_1d_dense(p, filter)
    }

    eta <- eigen(tcrossprod(sqrt(n) * L), symmetric = TRUE, only.values = TRUE)$values

    
    mu_hat <- solve(N + lambda * tcrossprod(L), N %*% y) # or n * y

    if (niter) {
        for (i in seq_len(niter)) {
            sqrtW <- diag(irls_wts(t(mu_hat) %*% L))
            mu_hat <- solve(N + lambda * tcrossprod(L %*% sqrtW), N %*% y)
        }
        ## FIXME: update
        eta1 <- eigen(tcrossprod(sqrt(n) * L), symmetric = TRUE, only.values = TRUE)$values
        enp <- lambda2df(lambda, eta1)
    }
    else 
        enp <- lambda2df(lambda, eta)

    print(df)

    structure(list(fitted = mu_hat,
                   enp = enp,
                   niter = niter),
              class = c("lpen"))
}



