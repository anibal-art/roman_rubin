import numpy as np
import pandas as pd
import pygtc
import matplotlib.pyplot as plt
params_str = ["t0","u0", "te","rho","s","q","alpha","piEN", "piEE" ]

path = "/share/storage3/rubin/microlensing/romanrubin/PB/set_fit1/"
import os
files_fit = os.listdir(path)
rand_index = np.random.randint(0,len(files_fit))
data_fit = np.load(path+files_fit[rand_index], allow_pickle=True).item()
mean_for_sampling = data_fit["best_model"][0:9]
cov_for_sampling = data_fit['covariance_matrix'][0:9,0:9]

n_samples_plot = 30000
samples = np.random.multivariate_normal(mean_for_sampling, cov_for_sampling, n_samples_plot)

piE = np.sqrt(mean_for_sampling[7]**2+mean_for_sampling[8]**2)

piEN_dist = np.array(samples)[:,7]
piEE_dist = np.array(samples)[:,8]
piE_dist = np.sqrt(piEN_dist**2+piEE_dist**2)



errors = np.sqrt(np.diag(data_fit['covariance_matrix'][0:9,0:9]))
piEE_err = errors[7]
piEN_err = errors[8]
cov_piEE_piEN = data_fit['covariance_matrix'][8:9,8:9]

piEE = mean_for_sampling[7]
piEN = mean_for_sampling[8]

piE_err = (1/piE)*np.sqrt((piEN_err*piEN)**2+(piEE_err*piEE)**2+2*piEE*piEN*cov_piEE_piEN)
print("The value estimated is ", piE)
print("The uncertainty using MC is ", np.std(piE_dist))
print("The uncertainty using formulae for error propagation is ", piE_err)
