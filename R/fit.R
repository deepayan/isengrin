


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
    ## str(filter)
    stopifnot("'p' must be larger than 'length(filter)'" = p > flen)
    row1 <- c(filter, rep(0, p - flen))
    col1 <- c(filter[[1]], rep(0, p - flen))
    stats::toeplitz(col1, row1)
}


constructL_1d_sparse <- function(p, filter)
{
    constructL_1d_dense(p, filter) |> Matrix::Matrix()
}



lspen_dense <- function(x, y, n = 1, L, L.method = "d1",
                        df = NULL, lambda = NULL,
                        niter = 0, verbose = TRUE)
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
        sqrtW <- irls_wts(L %*% mu_hat) # for final estimate
        BtB <- tcrossprod((1/sqrt(n)) * t(sqrtW * L))
        eta1 <- eigen(BtB, symmetric = TRUE, only.values = TRUE)$values
        enp <- lambda2df(lambda, eta1)
    }
    else 
        enp <- lambda2df(lambda, eta)

    if (verbose) message("Equiv d.f. (trace(H)) = ", format(enp),
                         " with λ = ", format(lambda))

    structure(list(x = x, y = y, n = n,
                   fitted = mu_hat,
                   lambda = lambda,
                   enp = enp,
                   niter = niter),
              class = c("lpen"))
}


## general method, delegating sparse / dense to Matrix. TODO: calculate enp = tr(H)

lspen <-
    function(x, y, n = 1, L, A = NULL, L.method = "d1",
             df = NULL, lambda = NULL,
             niter = 0, verbose = TRUE)
{
    p <- length(y)
    stopifnot("'n' must be scalar or have same length as y" = length(n) %in% c(1, p))
    if (length(n) == 1) n <- rep(n, p)
    N <- Matrix::Diagonal(x = n)
    sqrt_n <- sqrt(n)
    if (!is.null(A)) {
        A <- Matrix::Matrix(A)
        stopifnot("Mismatch in dim(A)" = all.equal(dim(A), c(p, length(x))))
        if (is.null(lambda)) stop("lambda must be specified (for now) if A ≠ I")
    }
    stopifnot("'n' must be scalar or have same length as y" = length(n) %in% c(1, p))

    if (missing(L)) {
        stopifnot("Unsupported 'L.method'" = L.method %in% c("d1", "d2"))
        filter <- switch(L.method, d1 = c(-1, 1), d2 = c(1, -2, 1))
        L <- constructL_1d_sparse(length(x), filter)
    }

    if (missing(lambda)) {
        likely_rank <- min(dim(L))
        BtB <- tcrossprod((1/sqrt(n)) * t(L))
        eta <- eigen(BtB, symmetric = TRUE, only.values = TRUE)$values

        if (missing(df)) df <- sqrt(sum(n)) # default df = sqrt(nobs)
        lambda <- df2lambda(df, eta = eta, rank = likely_rank)
    }

    mu_hat <-
        if (is.null(A))
            solve(N + lambda * crossprod(L), n * y) # or N %*% y
        else 
            solve(crossprod(sqrt_n * A) + lambda * crossprod(L), crossprod(A, n * y))

    if (niter) {
        for (i in seq_len(niter)) {
            sqrtW <- irls_wts(L %*% mu_hat)
            mu_hat <-
                if (is.null(A))
                    solve(N + lambda * crossprod(sqrtW * L), n * y)
                else
                    solve(crossprod(sqrt_n * A) + lambda * crossprod(sqrtW * L),
                          crossprod(A, n * y))
        }
        sqrtW <- irls_wts(L %*% mu_hat) # for final estimate
        if (is.null(A)) {
            BtB <- tcrossprod((1/sqrt_n) * t(sqrtW * L))
            eta1 <- eigen(BtB, symmetric = TRUE, only.values = TRUE)$values
            enp <- lambda2df(lambda, eta1)
        }
        else { # this works even when A = I [TODO: compare]
            C <- crossprod(sqrt_n * A)
            enp <- sum(diag(solve(C + lambda * crossprod(sqrtW * L), C)))
        }
    }
    else 
        if (is.null(A)) {
            enp <- lambda2df(lambda, eta)
        } else
        {
            C <- crossprod(sqrt_n * A)
            enp <- sum(diag(solve(C + lambda * crossprod(L), C)))
        }

    if (verbose) message("Equiv d.f. (trace(H)) = ", format(enp),
                         " with λ = ", format(lambda))

    structure(list(x = x, y = y, n = n,
                   fitted = as.vector(mu_hat),
                   lambda = lambda,
                   enp = enp,
                   niter = niter),
              class = c("lpen"))

}


