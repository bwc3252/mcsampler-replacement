from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import argparse

import mcsampler_new as mcsampler

parser = argparse.ArgumentParser()
parser.add_argument("--ndim", type=int, default=2, help="Number of dimensions to integrate over")
parser.add_argument("--width", type=float, default=10.0, help="Width of domain of integration (same for each dimension)")
parser.add_argument("--correlated", default=False, action="store_true", help="Use a highly-correlated distribution for integrand")
parser.add_argument("--fname", type=str, default="cdf.png", help="File name for CDF plot")
args = parser.parse_args()

### generate list of named parameters
params = [str(i) for i in range(args.ndim)]

llim = -1 * (args.width / 2)
rlim = args.width / 2

### generate a covariance matrix
### make it narrower in the last dimension
cov = np.identity(args.ndim)
cov[args.ndim - 1][args.ndim - 1] = 0.05

### add some covariances if it should be correlated
if args.correlated:
    cov[0][args.ndim - 1] = -0.1
    cov[args.ndim - 1][0] = -0.1

mu = np.zeros(args.ndim)

### define integrand
def f(*x):
    arr = np.empty((len(x[0]), args.ndim))
    for i in range(args.ndim):
        arr[:,i] = x[i]
    return multivariate_normal.pdf(arr, mu, cov)

### initialize sampler
sampler = mcsampler.MCSampler() 

### add parameters
for p in params:
    sampler.add_parameter(p, left_limit=llim, right_limit=rlim)

### integrate
integral, var, eff_samp, _ = sampler.integrate(f, *params, min_iter=30, max_iter=30)

print("integral value:", integral)

### get our posterior samples as a single array
arr = np.empty((len(sampler._rvs["0"]), args.ndim))
for i in range(args.ndim):
    arr[:,i] = sampler._rvs[str(i)].flatten()

colors = ["black", "red", "blue", "green", "orange"]

for i in range(args.ndim):
    s = np.sqrt(cov[i][i])
    ### get sorted samples (for the current dimension)
    x = arr[:,i][np.argsort(arr[:,i])]
    ### plot true cdf
    plt.plot(x, truncnorm.cdf(x, llim, rlim, mu[i], s), label="True CDF", color=colors[i])
    lnL = sampler._rvs["integrand"]
    L = np.exp(lnL)
    p = sampler._rvs["joint_prior"]
    ps = sampler._rvs["joint_s_prior"]
    ### compute weights of samples
    weights = (L * p / ps)[np.argsort(arr[:,i])]
    y = np.cumsum(weights)
    y /= y[-1] # normalize
    ### plot recovered cdf
    plt.plot(x, y, "--", label="Recovered CDF", color=colors[i], linewidth=2)

plt.legend()

print("Saving CDF figure as " + args.fname + "...")

plt.savefig(args.fname)
