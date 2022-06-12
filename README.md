# GMM-SMM

In this repo, I'm teaching myself how to excute the GMM and SMM in Julia. Start with simple linear model,

$$ Y_{i} = \beta_{0} + \beta_{1}X_{i} + \varepsilon_{i}$$

OLS estimator minimize $(\hat{Y_{i}} - Y_{i})^2$, we can exucte in `Julia`. First, generate data based on the true model

```Julia
using Random

N = 1000 
Random.seed!(1234)

X = rand(N)
eps = rand(N)

Y = 1.0 + 3.0 .* X .+ eps

```
