import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

def beta_function(a, b):
    """beta function"""
    return np.exp(gammaln(a) + gammaln(b) - gammaln(a+b))

def beta_density(x, a, b):
    """pdf of beta distribution"""
    return x**(a-1) * (1-x)**(b-1) / beta_function(a, b)

if __name__ == '__main__':
    a = [1, 5, 10]

    # mean = 0, variance = 1
    x = np.linspace(0, 1, 1000)

    # subplot 1
    # If a and b are equal, the beta distribution is a symmetric distribution.
    # Increasing a and b will increase the peak of the distribution.
    # For a = b = 1, the beta distribution is a uniform distribution.

    for i in range(len(a)):
        plt.plot(x, beta_density(x, a[i], a[i]), label=f'a = {a[i]}, b = {a[i]}')

    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('HW1/Q2_1.png')

    a = [1, 5, 10]
    b = [2, 6, 11]

    # mean = 0, variance = 1
    x = np.linspace(0, 1, 1000)

    # subplot 2
    plt.clf()

    # If a and b are not equal, the beta distribution is an asymmetric distribution.
    # For a = 1 and b = 2, the beta distribution is a left-skewed distribution.

    for i in range(len(a)):
        plt.plot(x, beta_density(x, a[i], b[i]), label=f'a = {a[i]}, b = {b[i]}')

    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('HW1/Q2_2.png')