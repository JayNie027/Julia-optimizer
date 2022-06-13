# GMM-SMM

In this repo, I'm teaching myself how to excute the GMM and SMM in Julia. Start with simple linear model,

$$ Y_{i} = \beta_{0} + \beta_{1}X_{i} + \varepsilon_{i}$$

OLS estimator minimize $(\hat{Y_{i}} - Y_{i})^2$, we can exucte in `Julia`. First, generate data based on the true model

```Julia
using Random, Statistics

N = 1000
Random.seed!(1234)

X = randn(N)
eps = randn(N)

Y = 2.0 .+ 3.0 .* X .+ eps

```
Based on the DGP, true $\beta_{0} = 1$ and $\beta_{1} = 3$. \
First, I use the `GLM` package to estimate model

```Julia
using GLM
using DataFrames

df = DataFrame(Y = Y, X = X)
# model = fit(LinearModel, @formula(y~  x), data)
model1 = lm(@formula( Y ~ X ), df)

```
Y ~ 1 + X

Coefficients:
────────────────────────────────────────────────────────────────────────
               Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  1.98658   0.0303284   65.50    <1e-99    1.92707    2.0461
X            3.0203    0.0289095  104.47    <1e-99    2.96357    3.07703
