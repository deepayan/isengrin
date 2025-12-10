
# Introduction

This package attempts to implement a form of smoothing for timeseries
/ spatial data. The approach is analogous to (a discrete version of)
smoothing splines, with obvious similarities to ridge / lasso
regression (and possibly Gaussian Process priors?). However, instead
of a continuous covariate space, we view the problem as a discrete
problem indexed by a graph to generalize it beyond one covariate. The
reason for doing this is to enable experimentation in various kinds of
inverse problems, primarly in the domains of image and signal
processing, as well as changepoint detection.

The idea is simplest to illustrate for a one dimensional discrete time
series signal $Y_t$, where $t \in \{ 1, 2, \dotsc, n \}$. We want to
estimate an underlying mean function $\mu_t$ such that 

$$
Y_t \sim N(\mu_t, \sigma^2)
$$

We impose no structure on $\{ \mu_t \}$ except to assume that it is
"smooth" in some sense. For continuous time, we could penalize high
second derivative, which would give us smoothing splines. The discrete
analog would be to penalize large magnitudes of the second order
differences, which are essentially of the form

$$
(\mu_{t+2} - \mu_{t+1}) - (\mu_{t+1} - \mu_{t}) = \mu_{t+2} - 2 \mu_{t+1} + \mu_{t}
$$

If we wanted to instead encourage piecewise constant fits (e.g., if we
wanted to do changepoint detection), we could penalize large
magnitudes of $\mu_{t+1} - \mu_{t}$. Note that in both cases the terms
that are penalized are linear combinations of the $\mu_i$-s.

In both the above cases, we can reparameterize the problem to become a
Ridge / LASSO problem. In the second case, this is obvious --- just
use $\beta_{t+1} = \mu_{t+1} - \mu_{t}$ and penalize them, leaving
$\mu_1$ as it is, unpenalized. For the second case, we would
reparameterize using quadratic splines, and penalize the $(x -
\mu_{t})^2_{+}$ terms.

# Formulation

We reinterpret this problem as a discrete problem on a graph as
follows: timepoints $t = 1, 2, \dotsc, n$ are the nodes $V$ of a graph
$G = (V, E)$. We can imagine edges being present between all
neighbours, but this is not strictly required for the specification of
the problem. Penalities are accumulated as linear combinations, so
that we want to minimize

$$
\sum\limits_{i=1}^n (Y_i - \mu_i)^2 + \lambda \sum\limits_{k=1}^K (\ell_{k}^\top \mu)^2
$$

or 

$$
\sum\limits_{i=1}^n (Y_i - \mu_i)^2 + \lambda \sum\limits_{k=1}^K \lvert \ell_{k}^\top \boldsymbol{\mu} \rvert
$$

depending on whether we want a $L_2$ or $L_1$ penalty. Focusing on the
$L_2$ version for now, we can rewrite this as minimizing

$$
\lVert \boldsymbol{y} - \boldsymbol{\mu} \rVert^2  + \lambda \boldsymbol{\mu}^\top L L^\top \boldsymbol{\mu}
$$

where 

$$
L = \begin{bmatrix} \ell_1 & \ell_2 & \cdots & \ell_K \end{bmatrix}
$$

Differentiating and solving we get

$$
\hat{\boldsymbol{\mu}} = (I + L L^\top)^{-1} \boldsymbol{y}.
$$

This is going to be a very sparse problem in general, so we want to
use suitable tools, possibly something like

```r
library(Matrix)
A <- I + lambda * tcrossprod(L)
C <- Cholesky(A, perm = TRUE)
solve(C, b)
```

Using Woodbury's identity, we can identify this as being equivalent to
the Ridge inverse $(L^\top L + \lambda^{-1} I)^{-1}$, which may be a
smaller or larger problem depending on the dimensions of
$L$. Generally, the number of penalties $K$ will be larger than the
length of $\boldsymbol{\mu}$, so the $K \times K$ matrix $L^\top L$
will be larger.



# $L_1$ solution using IRLS

Using the standard IRLS approach, we can write the $L_1$ solution
$\boldsymbol{\mu}$ as minimizing

$$
\lVert \boldsymbol{y} - \boldsymbol{\mu} \rVert^2  + \lambda \boldsymbol{\mu}^\top L W_{\boldsymbol{\mu}} L^\top \boldsymbol{\mu}
$$

where $W_{\boldsymbol{\mu}}$ is a diagonal weight matrix with diagonal
entries $w_k(\boldsymbol{\mu}) = 1 / \lvert \ell_k^\top
\boldsymbol{\mu} \rvert$. We can hope to get an approximate solution
by reiteratively updating $W_{\boldsymbol{\mu}}$ as a function of
current $\hat{\boldsymbol{\mu}}$ and then use it to estimate the next
$\hat{\boldsymbol{\mu}}$.


# Graph structure

All this does not require a graph structure. However, it is useful to
consider a graph structure when reasoning about spatial data. Suppose
our "nodes" are located on $\mathbb{R}^2$, or more likely on a lattice
structure like $\mathbb{Z}^2$ (such as images). Which linear
combinations should be penalized? Possibly we would want to penalize
the changes between neighbours (or second order differences between
2-neighbours). To make this formal, it would help to impose a
neighbourhood structure on the nodes. For spatial data on
$\mathbb{Z}^2$, standard four or eight neighbour structures may be
used. For point process data on $\mathbb{R}^2$, a Delauney
triangulation may give the edge structure. If the units are
geographical units (e.g., US counties), neighbours may be other units
with a common border.


# Weights using covariates

We have so far assumed equispaced data, but this will not be true in
regression contexts. In such cases, we may have a covariates $x =
(x_1, \dots, x_p)$ associated with each $t$. If $x_t$ represents
geometric location, for example, then slope would be calculated as 

$$
\frac{\mu_{t+1} - \mu_{t}}{x_{t+1} - x_{t}}
$$

and so on. In other words, the linear combinations to be penalized now
become functions of the covariates as well. In general, the covariates
should define some kind of distance (or edge weight) between $t$-s
that determines how similar they are in covariate space, with the
understanding that corresponding $\mu_t$-s should be close (there
difference should be penalized).

TODO: check whether the second order difference stays a linear
combination, and think about how best to interpret covariates.

TODO: In general, covariates may define edges --- makes the graph
potentially dense, which will make life difficult. However, this may
make something like patch similarity for images feasible, say via PCA
basis.

# Interpolation

Typically "smoothing" involves interpolation. Here we can only work on
a discrete graph, but that graph may have more nodes than
observations, as long as each observation is on a node. In other
words, we have $\mu_t$ for all nodes $t$, but $Y_t$ for only some of
them. A natural example of this is image inpainting or upscaling.  The
sum of squares term in the objective function, $\sum (Y_i - \mu_i)^2$,
only involves this smaller set of nodes.

This is easily incorporated in the model by adding an initial
selection operation on the larger $\boldsymbol{\mu}$ to get the subset
that corresponds to the observed data. In other words, we minimise

$$
\lVert \boldsymbol{y} - A \boldsymbol{\mu} \rVert^2  + \lambda \boldsymbol{\mu}^\top L L^\top \boldsymbol{\mu}
$$

where each row of $A$ has exactly one 1 (with rest 0).

Note that having $A \ne I$ means that the problem may not be full rank
--- the rank would be determined by the structure of the penalty.


# Multiple observations

In regression contexts there may be multiple observations at the same
$t$, which have a common $\mu_t$. In that case, we may simply replace
$Y_t$ by $\bar{Y}_t$, the average of the observations. However, we
then need to add precision weights that take into account the number
of observations averaged. Specifically, ignoring terms free of
$\boldsymbol{\mu}$, we want to minimize

$$
\sum n_i (y_i - \mu_i)^2 + \lambda \boldsymbol{\mu}^\top L L^\top \boldsymbol{\mu} = 
\lVert N^{\frac12} ( \boldsymbol{y} - \boldsymbol{\mu} ) \rVert^2  + \lambda \boldsymbol{\mu}^\top L L^\top \boldsymbol{\mu}
$$

where $N$ is diagonal with entries $n_i$.

# General formulation

Combining everything together, and differentiating
w.r.t. $\boldsymbol{\mu}$, the final equations we need to solve are

$$
\left( A^\top N A + \lambda L L^\top \right) \boldsymbol{\mu} = A^\top N \boldsymbol{y}
$$

For implementation, defaults would be $A = I, N = I$. Everything else
needs to be specified, with $L$ determined by the structure of the
problem.


# Degrees of freedom

The choice of smoothing parameter $\lambda$ is always tricky, and will
probably depend on context. It is useful to interpret it in terms of a
"degrees of freedom" or "equivalent number of parameters". By analogy
with a linear model, we can define it as the trace of the "hat matrix"
when that makes sense.

It is unclear what we mean by the hat matrix when $A \neq I$, but
otherwise a natural interpretation is

$$
\hat{\boldsymbol{y}} = \hat{\boldsymbol{\mu}} = \left( N + \lambda L L^\top \right)^{-1} N \boldsymbol{y},
$$

so that the hat matrix is 

$$
H = \left( N + \lambda L L^\top \right)^{-1} N
$$

We need to compute $\text{trace}(H)$. To do this, it is first useful
to note that the precision weights $N$ do not complicte matters much,
by observing that

$$
\text{trace}\left\{ \left( N + \lambda L L^\top \right)^{-1} N \right\} = 
	\text{trace}\left\{ \left( I + \lambda B B^\top \right)^{-1} \right\},
$$

where $B = N^{-\frac12} L$ (this follows by splitting $N = N^{\frac12}
N^{\frac12}$ and moving one of them to the left inside the trace,
which is allowed because $\text{trace}(AB) = \text{trace}(BA)$).

Next, we note that $\text{trace}(A^{-1}) = \sum \gamma_i^{-1}$, where
$\gamma_i$-s are the eigenvalues of $A$. Further, note that the
eigenvalues of $I + \lambda B B^\top$ are of the form

$$
\gamma_i = 1 + \lambda \eta_i,
$$

where $\eta_i$ are the eigenvalues of $B B^{\top}$. Thus, once we
(numerically) compute the $\eta_i$-s, the corresponding "degrees of
freedom" $\nu$ can be easily calculated as

$$
\nu(\lambda) = \sum_{i=1}^n \frac1{1 + \lambda \eta_i}
$$

One may want to do the inverse calculation, i.e., choose $\lambda$
based on a user-specified $\nu$. There is no closed form solution, so
the solution has to be obtained numerically. It is clear that $\nu(0)
= n$ (no penalty, so solution is pointwise mean) and $\nu(M) < 1$ for
$M = n / \min \{ \eta_i \}$, and presumably the target value of $\nu$
will lie within these bounds. $\nu(\lambda)$ is clearly a decreasing
function, so any iterative method should be able to find the solution
easily (I should use `uniroot()` but will probably use _regula falsi_
just because I have always wanted to use it somewhere).


We can also blindly apply this is the $L_1$ case, inserting
$W_{\hat{\boldsymbol{\mu}}}$ where appropriate, to get a post-fitting
estimate of $\nu$, though choosing $\lambda$ based on a target $\nu$
will not be feasible using this approach.


