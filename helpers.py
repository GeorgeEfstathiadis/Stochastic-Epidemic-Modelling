import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if len(a.shape) == 2:
        m = np.mean(a, axis=1)
        se = scipy.stats.sem(a, axis=1)
    else:
        m = np.mean(a)
        se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def gelman_rubin_test(chains):
    M = len(chains)
    N, no_params = chains[0].shape

    # Calculate the mean of the parameters
    # for each chain
    mean_params = np.zeros((M, no_params))
    # also compute the intra-chain variance
    var_params = np.zeros((M, no_params))
    for m, chain in enumerate(chains):
        for param in range(no_params):
            mean_param = np.mean(chain[:, param])
            var_param = 1.0 / (N - 1) * np.sum((chain[:, param] - mean_param) ** 2)
            mean_params[m, param] = mean_param
            var_params[m, param] = var_param

    # Calculate the averaged mean and variance
    # across chains
    theta_hat = np.mean(mean_params, axis=0)
    W = np.mean(var_params, axis=0)

    # compute how indiv. means scatter around the joint mean
    B = N / (M - 1) * np.sum((mean_params - theta_hat) ** 2, axis=0)

    # compute the estimated variance of the joint mean
    V = (N - 1) / N * W + (M + 1) / (M * N) * B

    # assert it is close to 1
    return np.sqrt(V / W)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
 