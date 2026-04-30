"""Parameter settings for the selected epidemic simulation.
"""

import model
import numpy as np
import scipy.stats


def transmission_rate(t):
    if t < 50:
        return 0.2
    elif 50 <= t < 100:
        return 0.2 + (t - 50) / 50 * -0.06
    elif 100 <= t < 150:
        return 0.14
    elif 150 <= t < 200:
        return 0.14 + (t - 150) / 50 * 0.06
    elif 200 <= t < 250:
        return 0.2
    elif 250 <= t < 300:
        return 0.2 + (t - 250) / 50 * -0.06
    else:
        return 0.14


def delays():
    # Duration of infectious
    f_mean = 7
    f_var = 2 ** 2
    theta = f_var / f_mean
    k = f_mean / theta
    f_dist = scipy.stats.gamma(k, scale=theta)
    f_infectious = f_dist.pdf(np.arange(25))

    # Duration of recovered
    f_mean = 300
    f_var = 100 ** 2
    theta = f_var / f_mean
    k = f_mean / theta
    f_dist = scipy.stats.gamma(k, scale=theta)
    f_recovered = f_dist.pdf(np.arange(2000))

    # Compute approx. probability of being infectious as a function of time since infection
    num_samples = 10000
    delay_inf = [0] * 25
    for _ in range(num_samples):
        num_days = model.sample_discrete(f_infectious)
        for i in range(num_days):
            delay_inf[i] += 1 
    delay_inf = np.asarray(delay_inf) / num_samples

    # Compute approx. probability of being either infectious or recovered as a function of time since infection
    num_samples = 100000
    delay_recov = [0] * 10000
    for _ in range(num_samples):
        num_days = model.sample_discrete(f_recovered) + model.sample_discrete(f_infectious)
        for i in range(num_days):
            delay_recov[i] += 1 
    delay_recov = np.asarray(delay_recov) / num_samples
    delay_recov = delay_recov[:365]

    return f_infectious, f_recovered, delay_inf, delay_recov