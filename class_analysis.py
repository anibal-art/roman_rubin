import os, sys
import re
from typing import List
from astropy.time import Time
import h5py
from astropy.table import QTable
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import warnings
from astropy import constants as const
from astropy import units as u

class Analysis_Event:
    description = "This is a simple class example."
    
    def __init__(self, model, path_model, path_fit_rr, path_fit_roman, 
                 path_dataslice, trilegal_params):
        self.model = model
        self.path_model = path_model
        self.path_fit_rr   = path_fit_rr
        self.path_fit_roman   = path_fit_roman
        self.path_dataslice = path_dataslice
        self.trilegal_params = trilegal_params
        
    def labels_params(self):
        if self.model == "USBL":
            labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
        elif self.model == "FSPL": 
            labels_params: list[str] = ['t0','u0','te','rho','piEN','piEE']
        elif self.model == "PSPL":
            labels_params: list[str] = ['t0','u0','te','piEN','piEE']
        return labels_params
    
    def error_pyLIMA(self, data):
        if any(np.diag(data['covariance_matrix'])<0):
            fit_error = np.zeros(len(self.labels_params()))
        else:
            fit_error= np.sqrt(np.diag(data['covariance_matrix']))[0:len(self.labels_params())]
        return fit_error
    
    def data_fit(self):
        data_rr = np.load(self.path_fit_rr, allow_pickle=True).item()
        data_roman = np.load(self.path_fit_roman, allow_pickle=True).item()
        return data_rr, data_roman
    
    def fit_true(self):

        fit_rr = {}
        fit_roman = {}
        
        data_rr, data_roman = self.data_fit()
        
        fit_error_rr = self.error_pyLIMA(data_rr)
        fit_error_roman = self.error_pyLIMA(data_roman)

        for i,key in enumerate(self.labels_params()):
            fit_rr[key] = data_rr["best_model"][i]
            fit_rr[key+"_err"] = fit_error_rr[i]
            fit_roman[key] = data_roman["best_model"][i]
            fit_roman[key+"_err"] = fit_error_roman[i]
        true = data_rr["true_params"][self.labels_params()].to_dict()
        return true, fit_rr, fit_roman
    

    def read_data(self):
        # Open the HDF5 file and load data using specified names
        with h5py.File(self.path_model, 'r') as file:
            # Load array with string with info of dataset using its name
            info_dataset = file['Data'][:]
            info_dataset = [file['Data'][:][0].decode('UTF-8'), file['Data'][:][1].decode('UTF-8'),
                            [file['Data'][:][2].decode('UTF-8'), [0, 0]]]
            # Dictionary using its name
            pyLIMA_parameters = {key: file['pyLIMA_parameters'].attrs[key] for key in file['pyLIMA_parameters'].attrs}
            # Load table using its name
            bands = {}
            for band in ("W149", "u", "g", "r", "i", "z", "y"):
                loaded_table = QTable()
                for col in file[band]:
                    loaded_table[col] = file[band][col][:]
                bands[band] = loaded_table
            return info_dataset, pyLIMA_parameters, bands
    
    
    def chichi(self):
        '''
        name_file(str):This function receives as input the name of the file
        example: /home/user/model/set_sim1/Event_RR_42_trf.npy.
        '''

        info_dataset, model_params, curves = self.read_data()
        data_rr, data_roman = self.data_fit()

        
        chi_rr = data_rr["chi2"]
        chi_roman = data_roman["chi2"]

        dof_rr = sum([len(curves[key]) for key in curves]) - len(
            [len(curves[key]) for key in curves if not len(curves[key]) == 0]) * 2 - len(self.labels_params())
        dof_roman = len(curves['W149']) - 2 - len(self.labels_params())

        return chi_rr, chi_roman, dof_rr, dof_roman


    def group_consecutive_numbers(self,numbers):
        '''
        Defino comienzo y finalizacion de temporadas de observacion de
        Rubin en el campo de Roman
        '''
        numbers.sort()
        groups = []
        current_group = [numbers[0]]
    
        for i in range(1, len(numbers)):
            if numbers[i] - current_group[-1] < 60:
                current_group.append(numbers[i])
            else:
                groups.append(current_group)
                current_group = [numbers[i]]
    
        if current_group:
            groups.append(current_group)
    
        return groups


    def intervals_overlap(self,interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        return (start1 <= end2 and end1 >= start2) or (start2 <= end1 and end2 >= start1)

 
    def montecarlo_propagation_piE(best_model, covariance_matrix, indx_piE):
        
        samples = np.random.multivariate_normal(best_model,
                                                covariance_matrix,
                                                30000)
        
        piEN_dist = np.array(samples)[:,indx_piE[0]]
        piEE_dist = np.array(samples)[:,indx_piE[1]]
        piE_dist = np.sqrt(piEN_dist**2+piEE_dist**2)        
        return np.std(piE_dist)

    def MC_tE_rho_piE(best_model, covariance_matrix, indx_tE_rho, indx_piE):
        """
        Parameters
        ----------
        best_model : numpy array
            Array containing best model from fit.
        covariance_matrix : numpy matrix
            Covariance matrix.
        indx_piE : TYPE
            indices for tE and rho in best_model_array or covariance_matrix.
        Returns
        -------
        thetaE_true
        thetaE_MC_tE
        thetaE_MC_rho
    
        """
    
        samples = np.random.multivariate_normal(best_model, covariance_matrix, 30000)
        if len(indx_tE_rho)==2:
            MC_tE = np.array(samples)[:,indx_tE_rho[0]]
            MC_rho = np.array(samples)[:,indx_tE_rho[1]]
            piEN_dist = np.array(samples)[:, indx_piE[0]]
            piEE_dist = np.array(samples)[:, indx_piE[1]]
            MC_piE = np.sqrt(piEN_dist ** 2 + piEE_dist ** 2)
    
        else:
            MC_tE = np.array(samples)[:,indx_tE_rho[0]]
            piEN_dist = np.array(samples)[:, indx_piE[0]]
            piEE_dist = np.array(samples)[:, indx_piE[1]]
            MC_piE = np.sqrt(piEN_dist ** 2 + piEE_dist ** 2)
            MC_rho = None
        return MC_tE, MC_rho, MC_piE
    
    
    def samples(self):
        data_rr, data_roman = self.data_fit()

        samples_rr = np.random.multivariate_normal(data_rr["best_model"],
                                                   data_rr["covariance_matrix"],
                                                   30000)

        samples_roman = np.random.multivariate_normal(data_roman["best_model"],
                                                   data_roman["covariance_matrix"],
                                                   30000)
        return samples_rr, samples_roman
        
    def MC_propagation_piE(self):
        """
        """
        
        samples_rr, samples_roman = self.samples()


        piEN_dist_rr = samples_rr[:, self.labels_params().index('piEN')]
        piEE_dist_rr = samples_rr[:, self.labels_params().index('piEE')]
        piEN_dist_roman = samples_roman[:, self.labels_params().index('piEN')]
        piEE_dist_roman = samples_roman[:, self.labels_params().index('piEE')]
        
        MC_piE_rr = np.sqrt(piEN_dist_rr ** 2 + piEE_dist_rr ** 2)
        MC_piE_roman = np.sqrt(piEE_dist_roman ** 2 + piEN_dist_roman ** 2)

        return np.std(MC_piE_rr), np.std(MC_piE_roman)

    def mass_MC(self):
        
        samples_rr, samples_roman = self.samples()
        te_dist_rr = samples_rr[:, self.labels_params().index('te')]
        te_dist_roman = samples_roman[:, self.labels_params().index('te')]

        if "rho" in self.labels_params():
            rho_dist_roman = samples_roman[:, self.labels_params().index('rho')]
            rho_dist_rr = samples_rr[:, self.labels_params().index('rho')]        
            rad_to_mas = 206264806.24709633
            theta_s = np.arctan(self.trilegal_params["radius"]/self.trilegal_params["D_S"])*rad_to_mas            
            thE_rho_rr = theta_s/rho_dist_rr 
            thE_rho_roman = theta_s/rho_dist_roman
            
                
        thE_te_rr = self.trilegal_params["mu_rel"]*te_dist_rr 
        thE_te_roman = self.trilegal_params["mu_rel"]*te_dist_roman
        
        k=(4*const.G/const.c).value
        
        err_mass_rr = np.std(thE_te_rr/(k*self.piE()[0]))
        err_mass_roman = np.std(thE_te_roman/(k*self.piE()[0]))
        
        return err_mass_rr, err_mass_roman

        

    def piE_propagation(self, piEE, piEN, err_piEE, err_piEN, cov_piEE_piEN):

        piE = np.sqrt(piEE ** 2 + piEN ** 2)
       
        err_piE = (1 / piE) * np.sqrt((err_piEN * piEN) ** 2 + (
                    err_piEE * piEE) ** 2 +2*piEE * piEN*cov_piEE_piEN)
        
        return piE, err_piE
    
    
    def piE(self):
        
        data_rr, data_roman = self.data_fit()
        
        piEN_rr = data_rr["best_model"][self.labels_params().index("piEN")]
        piEE_rr = data_rr["best_model"][self.labels_params().index("piEE")]
        err_piEN_rr = np.sqrt(np.diag(data_rr["covariance_matrix"]))[self.labels_params().index('piEN')]
        err_piEE_rr = np.sqrt(np.diag(data_rr["covariance_matrix"]))[self.labels_params().index('piEE')]
        cov_piEE_piEN_rr = data_rr["covariance_matrix"][self.labels_params().index('piEE'),
                                                        self.labels_params().index('piEN')] 
        
        
        piEN_roman = data_rr["best_model"][self.labels_params().index("piEN")]
        piEE_roman = data_rr["best_model"][self.labels_params().index("piEE")]
        err_piEN_roman = np.sqrt(np.diag(data_rr["covariance_matrix"]))[self.labels_params().index('piEN')]
        err_piEE_roman = np.sqrt(np.diag(data_rr["covariance_matrix"]))[self.labels_params().index('piEE')]
        cov_piEE_piEN_roman = data_rr["covariance_matrix"][self.labels_params().index('piEE'),
                                                        self.labels_params().index('piEN')]        

        piE_rr, err_piE_rr = self.piE_propagation(piEE_rr, piEN_rr, 
                                                  err_piEE_rr, err_piEN_rr, 
                                                  cov_piEE_piEN_rr)
        piE_roman, err_piE_roman = self.piE_propagation(piEE_roman, 
                                                        piEN_roman, 
                                                        err_piEE_roman, 
                                                        err_piEN_roman,
                                                        cov_piEE_piEN_roman)
        
        return piE_rr, err_piE_rr, piE_roman, err_piE_roman
    
    def categories_function(self):
        
        nominal_seasons = [
            {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
            {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
            {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
            {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
            {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
            {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
        ]
    
        dataSlice = np.load(self.path_dataslice, allow_pickle=True)
        consecutive_numbers = dataSlice['observationStartMJD']
        result = self.group_consecutive_numbers(consecutive_numbers)
        rubin_seasons = []
        roman_seasons = []
        for group in result:
            rubin_seasons.append((min(group) + 2400000.5, max(group) + 2400000.5))
        for season in nominal_seasons:
            roman_seasons.append((Time(season['start'], format='isot').jd, Time(season['end'], format='isot').jd))

        t0 = self.fit_true()[0]["t0"]
        tE = self.fit_true()[0]["te"]
        interval1 = (t0 - tE, t0 + tE)
        overlap_rubin = False

        for j in range(len(rubin_seasons)):
            # interval2 = rubin_seasons[j]
            if self.intervals_overlap(interval1, rubin_seasons[j]):
                overlap_rubin = True
                break
        overlap_roman = False
        for k in range(len(roman_seasons)):
            # interval2 = roman_seasons[k]
            if self.intervals_overlap(interval1, roman_seasons[k]):
                overlap_roman = True
                break

        if (overlap_rubin == True) and (overlap_roman == True):
            category='A'
        if (overlap_rubin == True) and (not overlap_roman == True):
            category='B'
        if (not overlap_rubin == True) and (not overlap_roman == True):
            category='C'
        if (not overlap_rubin == True) and (overlap_roman == True):
            category='D'
        return category

