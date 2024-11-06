#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:32:05 2024

@author: anibal
"""

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
    
    def __init__(self, model, path_model, path_fit_rr, path_fit_roman):
        self.model = model
        self.path_model = path_model
        self.path_fit_rr   = path_fit_rr
        self.path_fit_roman   = path_fit_roman
        
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


    def group_consecutive_numbers(numbers):
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
    
    def intervals_overlap(interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        return (start1 <= end2 and end1 >= start2) or (start2 <= end1 and end2 >= start1)
    
    
    # def chichi_to_fits_files(self,path,fit_rr, fit_roman):
        
    #     id_to_chi2_rr = {}
    #     id_to_chi2_roman = {}
    #     id_to_dof_rr = {}
    #     id_to_dof_roman = {}
    #     sets = [int(re.search(r'\d+', f).group()) for f in os.listdir(path) if 'set_sim' in f]
        
    #     for i in tqdm(sets):
    #         common_elements_list = self.event_fits(path+f"set_fit{i}/")
    #         if not len(common_elements_list)==0:
    #             for j in range(len(common_elements_list)):
    #                 name_file = f"Event_{common_elements_list[j]}.h5"
    #                 chi2rr, chi2roman,dof_rr,dof_roman = self.chichi(path+f"set_sim{i}/"+name_file)
    #                 # print(chi2rr, chi2roman)
    #                 id_to_chi2_rr[int(common_elements_list[j]+i*5000)] =chi2rr
    #                 id_to_chi2_roman[int(common_elements_list[j]+i*5000)] =chi2roman
    #                 id_to_dof_rr[int(common_elements_list[j]+i*5000)] = dof_rr
    #                 id_to_dof_roman[int(common_elements_list[j]+i*5000)] = dof_roman
    #     fit_rr['chi2'] = fit_rr['Source'].map(id_to_chi2_rr)
    #     fit_roman['chi2'] = fit_roman['Source'].map(id_to_chi2_roman)
    #     fit_roman['dof']= fit_roman['Source'].map(id_to_dof_roman)
    #     fit_rr['dof']= fit_rr['Source'].map(id_to_dof_rr)
    #     return fit_rr, fit_roman
    
    
    def montecarlo_propagation_piE(best_model, covariance_matrix, indx_piE):
        
        samples = np.random.multivariate_normal(best_model, covariance_matrix, 30000)
        piEN_dist = np.array(samples)[:,indx_piE[0]]
        piEE_dist = np.array(samples)[:,indx_piE[1]]
        piE_dist = np.sqrt(piEN_dist**2+piEE_dist**2)        
        return np.std(piE_dist)
        
    # def piE_cov_terms(path, fit_rr, fit_roman, labels_params):
    #     cov_piEE_piEN = {}
    #     cov_piEE_piEN_rom = {}
        
    #     piE_MC_rr = {}
    #     piE_MC_roman = {}
        
    #     if len(labels_params)==len(['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']):
    #         indx_piE = [7,8]
    #     elif len(labels_params)==len(['t0','u0','te','rho','piEN','piEE']):
    #         indx_piE = [4,5]
    #     elif len(labels_params)==len(['t0','u0','te','piEN','piEE']):
    #         indx_piE = [3,4]
            
            
    #     for i in tqdm(range(len(fit_rr))):
    #         nsource = fit_rr["Source"].iloc[i]
    #         nset = int(nsource / 5000)
    #         nevent = nsource - nset * 5000
    #         data = np.load(path + f"set_fit{nset}/Event_RR_{nevent}_TRF.npy", allow_pickle=True)
    #         data_rom = np.load(path + f"set_fit{nset}/Event_Roman_{nevent}_TRF.npy", allow_pickle=True)
            
    #         best_model = data.item()['best_model']
    #         covariance_matrix = data.item()['covariance_matrix']
    #         cov_piEE_piEN[nsource] = covariance_matrix[indx_piE[0], indx_piE[1]]
    
    #         best_model_rom = data_rom.item()['best_model']
    #         covariance_matrix_rom = data_rom.item()['covariance_matrix']
    #         cov_piEE_piEN_rom[nsource] = covariance_matrix_rom[indx_piE[0], indx_piE[1]]
    
    #         piE_MC_rr[nsource] = montecarlo_propagation_piE(best_model, covariance_matrix, indx_piE)
    #         piE_MC_roman[nsource] = montecarlo_propagation_piE(best_model_rom, covariance_matrix_rom, indx_piE)
            
    #     fit_rr["cov_piEE_piEN"] = fit_rr['Source'].map(cov_piEE_piEN)
    #     fit_roman["cov_piEE_piEN"] = fit_rr['Source'].map(cov_piEE_piEN_rom)
    
    #     fit_rr['piE'] = np.sqrt(fit_rr['piEN'] ** 2 + fit_rr['piEE'] ** 2)
    #     fit_rr['piE_err'] = (1 / fit_rr['piE']) * np.sqrt((fit_rr['piEN_err'] * fit_rr['piEN']) ** 2 + (
    #                 fit_rr['piEE_err'] * fit_rr['piEE']) ** 2 +2*fit_rr['piEE']*fit_rr['piEN']*fit_rr['cov_piEE_piEN'])
    #     fit_roman['piE'] = np.sqrt(fit_roman['piEN'] ** 2 + fit_roman['piEE'] ** 2)
    #     fit_roman['piE_err'] = (1 / fit_roman['piE']) * np.sqrt((fit_roman['piEN_err'] * fit_roman['piEN']) ** 2 + (
    #                 fit_roman['piEE_err'] * fit_roman[
    #             'piEE']) ** 2 +2*fit_roman['piEE']*fit_roman['piEN']*fit_roman['cov_piEE_piEN'])
    #     #true['piE'] = np.sqrt(true['piEN'] ** 2 + true['piEE'] ** 2)
        
    #     fit_rr['piE_err_MC'] = fit_rr['Source'].map(piE_MC_rr)
    #     fit_roman['piE_err_MC'] = fit_roman['Source'].map(piE_MC_roman)
        
    #     return fit_rr, fit_roman
    
    
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
    

    
    
    def MC_propagation(self):
        """
        Parameters
        ----------
        path : str
            path to the directory where the fit and sims are saved.
        fit_rr : dataframe
            Pandas dataframe with the results of the fitting process 
        fit_roman : dataframe
            Pandas dataframe with the results of the fitting process
        Returns
        -------
        fit_rr : dataframe
            Add a column with the mass_true.
            Add a column with the mass obtained from MonteCarlo propagation.
            Add a column with the mass obtained from propagation formulae.
        fit_roman : dataframe
            Add a column with the mass_true.
            Add a column with the mass obtained from MonteCarlo propagation.
            Add a column with the mass obtained from propagation formulae.
        """
 
        cov_piEE_piEN = {}
        cov_piEE_piEN_rom = {}
    
        piE_MC_rr = {}
        piE_MC_roman = {}
    
        self.labels_params().index('piEE')
        self.labels_params().index('piEN')
        self.labels_params().index('te')
        try:
            self.labels_params().index('rho')
        except:
            pass

        data_rr, data_roman = self.data_fit()
        
        samples_rr = np.random.multivariate_normal(data_rr["best_model"],
                                                   data_rr["covariance_matrix"],
                                                   30000)
        
        samples_roman = np.random.multivariate_normal(data_roman["best_model"],
                                                   data_roman["covariance_matrix"],
                                                   30000)
        
        piEN_dist_rr = samples_rr[:, self.labels_params().index('piEN')]
        
        piEE_dist_rr = samples_rr[:, self.labels_params().index('piEE')]
        
        piEN_dist_roman = samples_roman[:, self.labels_params().index('piEN')]
        
        piEE_dist_roman = samples_roman[:, self.labels_params().index('piEE')]
        
        MC_piE_rr = np.sqrt(piEN_dist_rr ** 2 + piEE_dist_rr ** 2)
        
        MC_piE_roman = np.sqrt(piEE_dist_roman ** 2 + piEN_dist_roman ** 2)


        #opero solo con los df
        # fit_rr['piE'] = np.sqrt(fit_rr['piEN'] ** 2 + fit_rr['piEE'] ** 2)
        # fit_rr['piE_err'] = (1 / fit_rr['piE']) * np.sqrt((fit_rr['piEN_err'] * fit_rr['piEN']) ** 2 + (
        #             fit_rr['piEE_err'] * fit_rr['piEE']) ** 2 +2*fit_rr['piEE']*fit_rr['piEN']*fit_rr['cov_piEE_piEN'])
        
        # fit_roman['piE'] = np.sqrt(fit_roman['piEN'] ** 2 + fit_roman['piEE'] ** 2)
        # fit_roman['piE_err'] = (1 / fit_roman['piE']) * np.sqrt((fit_roman['piEN_err'] * fit_roman['piEN']) ** 2 + (
        #             fit_roman['piEE_err'] * fit_roman[
        #         'piEE']) ** 2 +2*fit_roman['piEE']*fit_roman['piEN']*fit_roman['cov_piEE_piEN'])
        
    
        return np.std(MC_piE_rr), np.std(MC_piE_roman)
    
    
    def categories_function(true,path_dataslice):
        nominal_seasons = [
            {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
            {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
            {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
            {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
            {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
            {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
        ]
    
        dataSlice = np.load(path_dataslice, allow_pickle=True)
        # dataSlice['observationStartMJD']
        consecutive_numbers = dataSlice['observationStartMJD']
        result = group_consecutive_numbers(consecutive_numbers)
        rubin_seasons = []
        roman_seasons = []
        for group in result:
            rubin_seasons.append((min(group) + 2400000.5, max(group) + 2400000.5))
        for season in nominal_seasons:
            roman_seasons.append((Time(season['start'], format='isot').jd, Time(season['end'], format='isot').jd))
    
        categories = {}
        for i in range(len(true)):
            Source = true['Source'].iloc[i]
            t0 = true['t0'].iloc[i]
            tE = true['te'].iloc[i]
            interval1 = (t0 - tE, t0 + tE)
            overlap_rubin = False
            for j in range(len(rubin_seasons)):
                interval2 = rubin_seasons[j]
                if intervals_overlap(interval1, rubin_seasons[j]):
                    overlap_rubin = True
                    break
            overlap_roman = False
            for k in range(len(roman_seasons)):
                interval2 = roman_seasons[k]
                if intervals_overlap(interval1, roman_seasons[k]):
                    overlap_roman = True
                    break
    
            if (overlap_rubin == True) and (overlap_roman == True):
                categories[Source]='A'
            if (overlap_rubin == True) and (not overlap_roman == True):
                categories[Source]='B'
            if (not overlap_rubin == True) and (not overlap_roman == True):
                categories[Source]='C'
            if (not overlap_rubin == True) and (overlap_roman == True):
                categories[Source]='D'
    
        true['categories'] = true['Source'].map(categories)
        return true


path_fit_rr = "/home/anibal/roman_rubin/test_sim_fit/Event_RR_18_TRF.npy"
path_fit_roman = "/home/anibal/roman_rubin/test_sim_fit/Event_Roman_18_TRF.npy"
path_model = "/home/anibal/roman_rubin/test_sim_fit/Event_18.h5"

Event = Analysis_Event("USBL", path_model, path_fit_rr, path_fit_roman)
true, fit_rr, fit_roman = Event.fit_true()

print(fit_rr)
print(Event.read_data())
print(Event.chichi())
print(Event.MC_propagation())