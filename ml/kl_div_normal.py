"""
KL Divergence Between Two Normal Distributions
Task: Implement KL Divergence Between Two Normal Distributions
Your task is to compute the Kullback-Leibler (KL) divergence between two normal distributions. KL divergence measures how one probability distribution differs from a second, reference probability distribution.

Write a function kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) that calculates the KL divergence between two normal distributions, where 
P \sim N\left(\mu_P, \sigma_P^2\right) \text { and } Q \sim N\left(\mu_Q, \sigma_Q^2\right) .

The function should return the KL divergence as a floating-point number.
"""

import numpy as np


def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    return (
        np.log(sigma_q / sigma_p)
        + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
        - 0.5
    )


if __name__ == "__main__":
    mu_p = 0.0
    sigma_p = 1.0
    mu_q = 1.0
    sigma_q = 1.0

    print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))