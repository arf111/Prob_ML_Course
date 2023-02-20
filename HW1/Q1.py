import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt

def pdf_t_distribution(x, dof):
    """pdf of t-distribution with given degrees of freedom"""
    numerator = np.exp(gammaln((dof+1)/2) - gammaln(dof/2))

    denominator = np.sqrt(np.pi*dof) * (1 + x**2/dof)**((dof+1)/2)

    return numerator / denominator

def standard_gaussian_pdf(x):
    """pdf of standard gaussian"""
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

if __name__ == '__main__':
    dof = [0.1, 1, 10, 100, 1000000]

    # mean = 0, variance = 1
    x = np.linspace(-5, 5, 1000)

    for d in dof:
        plt.plot(x, pdf_t_distribution(x, d), label=f'dof = {d}')

    # If dof increases, the t-distribution converges to the standard gaussian distribution as shown in the plot.

    plt.plot(x, standard_gaussian_pdf(x), label='standard gaussian')
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('HW1/Q1.png')
    
