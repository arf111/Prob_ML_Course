import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import minimize

def mle_for_gaussian_distribution(samples):
    """maximum likelihood estimation for gaussian distribution with given samples using LBFGS"""
    def negative_log_likelihood(params):
        """negative log likelihood"""
        mu, sigma = params
        return -np.sum(np.log(1 / np.sqrt(2 * np.pi * sigma**2)) - (samples - mu)**2 / (2 * sigma**2))
    
    # initial guess
    mu, sigma = [0, 1]
    
    # minimize negative log likelihood
    result = minimize(negative_log_likelihood, [mu, sigma], method='L-BFGS-B')
    
    return result.x

def mle_for_student_t_distribution(samples):
    """maximum likelihood estimation for student t-distribution with given samples using LBFGS"""
    def negative_log_likelihood(params):
        """negative log likelihood"""
        dof = params[0]
        return -np.sum(gammaln((dof+1)/2) - gammaln(dof/2) - np.log(np.sqrt(np.pi*dof)) - (dof+1)/2 * np.log(1 + (samples**2)/dof))
    
    # initial guess
    dof = 1
    
    # minimize negative log likelihood
    result = minimize(negative_log_likelihood, [dof], method='L-BFGS-B')
    
    return result.x

def gaussian_pdf(x, mu, sigma):
    """pdf of gaussian distribution with given mean and variance"""
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def student_t_pdf(x, dof):
    """pdf of student t-distribution with given degrees of freedom"""
    return np.exp(gammaln((dof+1)/2) - gammaln(dof/2) - np.log(np.sqrt(np.pi*dof)) - (dof+1)/2 * np.log(1 + (x**2)/dof))

if __name__ == '__main__':
    # generate samples, mean = 0, variance = 2
    samples = np.random.normal(0, np.sqrt(2), 30)
    
    # mle
    mu, sigma = mle_for_gaussian_distribution(samples)
    dof = mle_for_student_t_distribution(samples)
    
    # mean = 0, variance = 1
    x = np.linspace(-5, 5, 1000)
    
    # plot samples
    plt.hist(samples, bins=20, density=True, label='samples')
    
    # plot gaussian distribution
    plt.plot(x, gaussian_pdf(x, mu, sigma), label='gaussian distribution')
    
    # plot student t-distribution
    plt.plot(x, student_t_pdf(x, dof), label='student t-distribution')
    
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('HW1/Q3.png')

    plt.clf()

    # add [8, 9, 10] to samples
    samples_2 = np.append(samples, [8, 9, 10])

    # mle
    mu, sigma = mle_for_gaussian_distribution(samples_2)
    dof = mle_for_student_t_distribution(samples_2)

    # mean = 0, variance = 1
    x = np.linspace(-5, 11, 1000)

    # adding noise to samples will make the student t-distribution more similar to the gaussian distribution.
    # this is because the student t-distribution is more robust to outliers than the gaussian distribution.
    # More similar in this case means that the student t-distribution will have a higher probability density at the same point. 
    # This is because the student t-distribution has a heavier tail than the gaussian distribution.

    # plot samples
    plt.hist(samples_2, bins=20, density=True, label='samples')

    # plot gaussian distribution
    plt.plot(x, gaussian_pdf(x, mu, sigma), label='gaussian distribution')

    # plot student t-distribution
    plt.plot(x, student_t_pdf(x, dof), label='student t-distribution')

    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('HW1/Q3_2.png')