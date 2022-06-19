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
## linear regression

Based on the DGP, true $\beta_{0} = 2,\beta_{1} = 3$.\
First, I use the `GLM` package to estimate model

```Julia
using GLM
using DataFrames

df = DataFrame(Y = Y, X = X)
# model = fit(LinearModel, @formula(y~  x), data)
model1 = lm(@formula( Y ~ X ), df)
coef(model1) # acces to coefficients
stderror(model1) # access to standard errors
predict(model1) # geenrate model prediction 
# it's nice that all paramters stored as vectors 

Coefficients:
────────────────────────────────────────────────────────────────────────
               Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  1.98658   0.0303284   65.50    <1e-99    1.92707    2.0461
X            3.0203    0.0289095  104.47    <1e-99    2.96357    3.07703
────────────────────────────────────────────────────────────────────────
```

## Optimizer, GMM

Instead of using linear model from `GLM`. We can use `Optim` optimizer. There are lots of options for using optimizer, here's link: https://julianlsolvers.github.io/Optim.jl/stable/. Obejctive function is 

  $$\text{Min} \sum_{i}^{N}(Y_{i} - \hat{Y}_{i})^{2} $$
  
 ```Julia
 using Optim

function sqerror(betas)
    err = 0.0
    for i in 1:length(X)
        pred_i = betas[1] + betas[2] * X[i]
        err += (Y[i] - pred_i)^2
    end
    return err
end
res = optimize(sqerror, [0.0, 0.0])
Optim.minimizer(res)
```
or 

```Julia
function sqerror(beta)
    predict = beta[1] .+ beta[2] .* X
    err = Y .- predict
    sqrerr = sum(err.^2)

    return sqrerr
end
res1 = optimize(sqerror, [0.0, 0.0])
Optim.minimizer(res1)
```
Think about GMM. Linear regression is a special case of GMM because we have two parameters $\beta_{0}, \beta_{1}$, and two moment conditions ,
$$E[\varepsilon] = 0$$
$$E[X\varepsilon] = 0$$
We can stack these two conditions and 
$$ \text{Min} \,\,\,\,\,\,  W'W $$ 
Notice, because two parameters and two moment conditions so paramters are excat identified. In a more general case, have more moment conditions than parameters then the weighting matrix is need.

```Julia
function moment(beta)
    predict = beta[1] .+ beta[2] .* X
    err = Y .- predict
    mome1 = sum(err)/N  # first moment condition
    mome2 = sum(X.*err)/N # second moment condition
    moment = [mome1, mome2]
    object =  moment' * moment
    return object
end
res = optimize(moment, [ 0.0, 0.0])
Optim.minimizer(res)
```
