import numpy as np


def noisy_resids_from_one_resid(residual, noise_lvl, nr_of_generated_residuals=40, random_seed=0):
    '''Generate array of noisy residuals based on one residual value'''
    
    residuals = np.array([residual]*nr_of_generated_residuals)
    
    np.random.seed(random_seed)
    noise = np.random.normal(loc=0, scale=noise_lvl, size=nr_of_generated_residuals)
    residuals += noise
    
    return residuals