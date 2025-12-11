


irls_wts <- function(x, eps = 0.0001) {
    w <- sqrt(as.vector(abs(x)))
    w[w < eps] <- eps
    1 / w
}


lambda2df <- function(lambda, eta, L, n, B = (1/sqrt(n)) * L)
{
    if (missing(eta))
        eta <- eigen(tcrossprod(B), symmetric = TRUE, only.values = TRUE)$values
    sum(1 / (1 + lambda * eta))
}


df2lambda <- function(df, eta, L, n, B = (1/sqrt(n)) * L,
                      rank = sum(zapsmall(eta) > 0))
{
    if (missing(eta))
        eta <- eigen(tcrossprod(B), symmetric = TRUE, only.values = TRUE)$values
    M <- length(eta) / eta[[rank]]
    stopifnot("'M' is too small; check 'eta' and 'rank'" = M > 0.5)
    f <- function(x) { sum(1 / (1 + x * eta)) - df }
    u <- uniroot(f, c(0.5, M))
    u$root
}






## Proof-of-concept reference implementation using dense matrix operations

constructL_1d_dense <- function(p, filter)
{
    flen <- length(filter)
    str(filter)
    stopifnot("'p' must be larger than 'length(filter)'" = p > flen)
    row1 <- c(filter, rep(0, p - flen))
    col1 <- c(filter[[1]], rep(0, p - flen))
    stats::toeplitz(col1, row1)
}


constructL_1d_sparse <- function(p, filter)
{
    constructL_1d_dense(p, filter) |> Matrix::sparseMatrix()
}



lspen_dense <- function(x, y, n = 1, L, L.method = "d1",
                        df = NULL, lambda = NULL,
                        niter = 0)
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

    likely_rank <- min(dim(L))
    BtB <- tcrossprod((1/sqrt(n)) * t(L))
    eta <- eigen(BtB, symmetric = TRUE, only.values = TRUE)$values
    if (missing(lambda)) lambda <- df2lambda(df, eta = eta, rank = likely_rank)

    mu_hat <- solve(N + lambda * crossprod(L), n * y) # or N %*% y

    if (niter) {
        for (i in seq_len(niter)) {
            sqrtW <- irls_wts(L %*% mu_hat)
            mu_hat <- solve(N + lambda * crossprod(sqrtW * L), n * y)
        }
        BtB <- tcrossprod((1/sqrt(n)) * t(sqrtW * L))
        eta1 <- eigen(BtB, symmetric = TRUE, only.values = TRUE)$values
        enp <- lambda2df(lambda, eta1)
    }
    else 
        enp <- lambda2df(lambda, eta)

    print(enp)

    structure(list(fitted = mu_hat,
                   enp = enp,
                   niter = niter),
              class = c("lpen"))
}



