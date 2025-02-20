import h5py
import math
import os, re
import numpy as np
from astropy.time import Time
from astropy.table import QTable
from astropy import constants as const
from astropy import units as u

def labels_params(model):
    if model == "USBL":
        labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
    elif model == "FSPL": 
        labels_params: list[str] = ['t0','u0','te','rho','piEN','piEE']
    elif model == "PSPL":
        labels_params: list[str] = ['t0','u0','te','piEN','piEE']
    return labels_params

def event_fits(path_fits):
    '''
    return events in common with roman and rubin
    we have events that fits only one of two for unknown reasons
    '''

    files_fits = os.listdir(path_fits)

    files_roman = [f for f in files_fits if 'Roman' in f]
    files_rr = [f for f in files_fits if not 'Roman' in f]

    n_rom = []  # list with the event number
    for j in files_roman:
        number = int(re.findall(r'\d+', j)[0])
        n_rom.append(number)

    n_rr = []  # # list with the event number
    for j in files_rr:
        number = int(re.findall(r'\d+', j)[0])
        n_rr.append(number)

    # Convert lists to sets
    set1 = set(n_rom)
    set2 = set(n_rr)
    # Find the common elements using intersection
    common_elements = set1.intersection(set2)
    # Convert the result back to a list (if needed)
    common_elements_list = list(common_elements)
    return common_elements_list


class Analysis_Event:
    description = "This is a class to analize one event."
    
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
    
    def data_fit_rr(self):
        data_rr = np.load(self.path_fit_rr, allow_pickle=True).item()
        return data_rr
    
    def data_fit_roman(self):
        data_roman = np.load(self.path_fit_roman, allow_pickle=True).item()
        return data_roman
    
    def data_fit(self):
        data_rr = self.data_fit_rr()
        data_roman = self.data_fit_roman()
        return data_rr, data_roman
    
    def fit_values_rr(self):
        fit_rr = {}
        data_rr = self.data_fit_rr()    
        fit_error_rr = self.error_pyLIMA(data_rr)

        for i,key in enumerate(self.labels_params()):
            fit_rr[key] = data_rr["best_model"][i]
            fit_rr[key+"_err"] = fit_error_rr[i]
        
        return fit_rr
    
    def fit_values_roman(self):
        fit_roman = {}
        data_roman = self.data_fit_roman()
        fit_error_roman = self.error_pyLIMA(data_roman)

        for i,key in enumerate(self.labels_params()):
            fit_roman[key] = data_roman["best_model"][i]
            fit_roman[key+"_err"] = fit_error_roman[i]
        
        return fit_roman
        

    def fit_values(self):
        fit_rr = self.fit_values_rr()
        fit_roman = self.fit_values_roman()
        return fit_rr, fit_roman
        
    
    def true_values(self):
        data_rr, data_roman = self.data_fit()
        true = data_rr["true_params"][self.labels_params()].to_dict()
        return true
        
    def fit_true(self):
        fit_rr, fit_roman =self.fit_values()
        true = self.true_values()
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

    def has_consecutive_numbers(self, lst):
        """
        check if there at least 3 consecutive numbers in a list lst
        """
        sorted_lst = sorted(lst)
        for i in range(len(sorted_lst) - 2):
            if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
                return True
        return False


    
    def deviation_from_constant(self):
        '''
         There at least four points in the range
         $[t_0-tE, t_0+t_E]$ with the magnification deviating from the
         constant flux by more than 3$\sigma$
        '''
        
        ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
              'i': 27.85, 'z': 27.46, 'y': 26.68}
    
        info_dataset, pyLIMA_parameters, pyLIMA_telescopes = self.read_data()
        
        t0 = pyLIMA_parameters['t0']
        tE = pyLIMA_parameters['tE']
        satis_crit = {}
        for telo in pyLIMA_telescopes:
            if not len(pyLIMA_telescopes[telo]['mag']) == 0:
                mag_baseline = ZP[telo] - 2.5 * np.log10(pyLIMA_parameters['ftotal_' + f'{telo}'])
                x = pyLIMA_telescopes[telo]['time'].value
                y = pyLIMA_telescopes[telo]['mag'].value
                z = pyLIMA_telescopes[telo]['err_mag'].value
                mask = (t0 - tE < x) & (x < t0 + tE)
                consec = []
                if len(x[mask]) >= 3:
                    combined_lists = list(zip(x[mask], y[mask], z[mask]))
                    sorted_lists = sorted(combined_lists, key=lambda item: item[0])
                    sorted_x, sorted_y, sorted_z = zip(*sorted_lists)
                    for j in range(len(sorted_y)):
                        if sorted_y[j] + 3 * sorted_z[j] < mag_baseline:
                            consec.append(j)
                    result = self.has_consecutive_numbers(consec)
                    if result:
                        satis_crit[telo] = True
                    else:
                        satis_crit[telo] = False
                else:
                    satis_crit[telo] = False
            else:
                satis_crit[telo] = False
        return any(satis_crit.values())


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
    
    def mass(self, thetaE, piEN, piEE):
        aconv = (180 * 60 * 60 * 1000) / math.pi
        k= 4 * const.G / (const.c ** 2)
        mest = ((thetaE/aconv**2)*u.kpc/(k*np.sqrt(piEN**2+piEE**2))).decompose().to('M_sun')
        return mest
    
    def mass_true(self):
        yr2day = 365.25
        thetaE = self.trilegal_params["te"]*self.trilegal_params["mu_rel"]/yr2day
        piEN = self.true_values()["piEN"]
        piEE =self.true_values()["piEE"]
        return self.mass(thetaE, piEN, piEE)
    
    
    def fit_mass_rr1(self):
        '''
        Returns the mass as theta_E known
        -------
        mass_rr : float
            DESCRIPTION.

        '''
        
        yr2day = 365.25
        fit_rr = self.fit_values_rr()
        thetaE = self.trilegal_params["te"]*self.trilegal_params["mu_rel"]/yr2day
        mass_rr = self.mass(thetaE, fit_rr["piEN"], fit_rr["piEE"])
        return mass_rr
    
    def fit_mass_rr2(self):
        '''
        Returns the mass with theta_E estimated with known mu_rel
        -------
        mass_rr : float
            DESCRIPTION.

        '''
        yr2day = 365.25
        fit_rr = self.fit_values_rr()
        thetaE = fit_rr['te']*self.trilegal_params["mu_rel"]/yr2day
        mass_rr = self.mass(thetaE, fit_rr["piEN"], fit_rr["piEE"])
        return mass_rr

    def fit_mass_rr3(self):
        '''
        Returns the mass with theta_E estimated with theta_star known
        -------
        mass_rr : float
            DESCRIPTION.

        '''
        fit_rr = self.fit_values_rr()
        theta_s = np.arctan((self.trilegal_params["radius"]*u.R_sun/(self.trilegal_params["D_S"]*u.pc))).to('mas').value      
        thetaE = theta_s/fit_rr['rho']
        mass_rr = self.mass(thetaE, fit_rr["piEN"], fit_rr["piEE"])
        return mass_rr

    
    def fit_mass_roman1(self):
        yr2day = 365.25
        fit_roman = self.fit_values_roman()
        thetaE = self.trilegal_params["te"]*self.trilegal_params["mu_rel"]/yr2day   
        mass_roman = self.mass(thetaE, fit_roman["piEN"], fit_roman["piEE"])
        return mass_roman
    
    def fit_mass_roman2(self):
        yr2day = 365.25
        fit_roman = self.fit_values_roman()
        thetaE = fit_roman['te']*self.trilegal_params["mu_rel"]/yr2day   
        mass_roman = self.mass(thetaE, fit_roman["piEN"], fit_roman["piEE"])
        return mass_roman
    
    def fit_mass_roman3(self):
        fit_roman = self.fit_values_roman()
        theta_s = np.arctan((self.trilegal_params["radius"]*u.R_sun/(self.trilegal_params["D_S"]*u.pc))).to('mas').value
        thetaE = theta_s/fit_roman['rho']   
        mass_roman = self.mass(thetaE, fit_roman["piEN"], fit_roman["piEE"])
        return mass_roman
    

    def mass_MC(self):
        
        samples_rr, samples_roman = self.samples()
        
        piEE_dist_rr = samples_rr[:, self.labels_params().index('piEE')]
        piEN_dist_rr = samples_rr[:, self.labels_params().index('piEN')]
        te_dist_rr = samples_rr[:, self.labels_params().index('te')]

        piEE_dist_roman = samples_roman[:, self.labels_params().index('piEE')]
        piEN_dist_roman = samples_roman[:, self.labels_params().index('piEN')]
        te_dist_roman = samples_roman[:, self.labels_params().index('te')]


        if "rho" in self.labels_params():
            rho_dist_roman = samples_roman[:, self.labels_params().index('rho')]
            rho_dist_rr = samples_rr[:, self.labels_params().index('rho')]        
            theta_s = np.arctan((self.trilegal_params["radius"]*u.R_sun/(self.trilegal_params["D_S"]*u.pc))).to('mas').value      
            thE_rho_rr = theta_s/rho_dist_rr 
            thE_rho_roman = theta_s/rho_dist_roman
            
            err_mass_rr2 = self.mass( thE_rho_rr, piEN_dist_rr, piEE_dist_rr)
            err_mass_roman2 = self.mass(thE_rho_roman, piEN_dist_roman, piEE_dist_roman)
            
        else:
            err_mass_rr2 = [0,0]
            err_mass_roman2 = [0,0]
            
        yr2day = 365.25        
        
        thE = self.trilegal_params["mu_rel"]* self.trilegal_params["te"]/yr2day 
        thE_te_rr = self.trilegal_params["mu_rel"]*te_dist_rr/yr2day 
        thE_te_roman = self.trilegal_params["mu_rel"]*te_dist_roman/yr2day 
        
        err_mass_rr1 = self.mass( thE_te_rr, piEN_dist_rr, piEE_dist_rr)
        err_mass_roman1 = self.mass(thE_te_roman, piEN_dist_roman, piEE_dist_roman)
        

        err_mass_rr3 = self.mass( thE, piEN_dist_rr, piEE_dist_rr)
        err_mass_roman3 = self.mass(thE, piEN_dist_roman, piEE_dist_roman)
        
        return {'sigma_m_thetaS_rr':np.std(err_mass_rr1), 
                'sigma_m_thetaS_roman':np.std(err_mass_roman1),
                'sigma_m_mu_rr':np.std(err_mass_rr2),
                'sigma_m_mu_roman':np.std(err_mass_roman2),
                'sigma_m_thetaE_rr':np.std(err_mass_rr3),
                'sigma_m_thetaE_roman':np.std(err_mass_roman3)
                }
    
    def formula_mass_uncertainty(self, piEE_fit, piEN_fit, sigma_piee, sigma_pien,cov_piEE_piEN,thetaE):
        aconv = (180 * 60 * 60 * 1000) / math.pi
        c = const.c
        G = const.G
        k = 4 * G / (c ** 2)
        yr2day = 365.25  
        piE_fit  = np.sqrt(piEE_fit**2 +piEN_fit**2)*(1 / u.kpc)
        # Calculate the fitted mass
        M_fit = ((thetaE / aconv**2) / (k * piE_fit)).decompose().to('M_sun')
        # Derivatives for error propagation
        dm_dpiee = ((thetaE * (-piEE_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dpien = ((thetaE * (-piEN_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dthetaE = ((1  / aconv**2) / (k * piE_fit)).decompose()
        # Quadratic terms for the error
        cuad_terms =  (dm_dpiee * sigma_piee)**2 + (dm_dpien * sigma_pien)**2 + (dm_dthetaE * 0.1 * thetaE)**2
        # Covariance terms for the error
        cov_terms = 2*dm_dpiee * dm_dpien * cov_piEE_piEN * (1 / u.kpc)**2
        sigma_m = cuad_terms + cov_terms
        return np.sqrt(sigma_m).to('M_sun').value
        
    
    def formula_mass_uncertainty_rr(self):
        '''
        Expression of change of variables in covariance at first order
        '''
        aconv = (180 * 60 * 60 * 1000) / math.pi
        c = const.c
        G = const.G
        k = 4 * G / (c ** 2)
        yr2day = 365.25
        piEE_fit = self.fit_values_rr()['piEE'] * (1 / u.kpc)
        piEN_fit = self.fit_values_rr()['piEN'] * (1 / u.kpc)
        piE_fit  = np.sqrt(piEE_fit**2 +piEN_fit**2)

        sigma_piee = self.fit_values_rr()['piEE_err'] * (1 / u.kpc)
        sigma_pien = self.fit_values_rr()['piEN_err'] * (1 / u.kpc)
        
        thetaE = self.trilegal_params["mu_rel"]* self.trilegal_params["te"]/yr2day 
        # Covariance terms for the error
        data_rr, data_roman = self.data_fit()
        cov_piEE_piEN = data_rr["covariance_matrix"][self.labels_params().index('piEN'),self.labels_params().index('piEE')]

        # # Derivatives for error propagation
        dm_dpiee = ((thetaE * (-piEE_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dpien = ((thetaE * (-piEN_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dthetaE = ((1  / aconv**2) / (k * piE_fit)).decompose()
        
        cuad_terms =  (dm_dpiee * sigma_piee)**2 + (dm_dpien * sigma_pien)**2 + (dm_dthetaE * 0.1 * thetaE)**2
        cov_terms = 2*dm_dpiee * dm_dpien * cov_piEE_piEN * (1 / u.kpc)**2
        sigma_m = cuad_terms + cov_terms
        
    
        return np.sqrt(sigma_m).to('M_sun').value

    
    def formula_mass_uncertainty_roman(self):
        '''
        Expression of change of variables in covariance at first order
        '''
        aconv = (180 * 60 * 60 * 1000) / math.pi
        c = const.c
        G = const.G
        k = 4 * G / (c ** 2)
        yr2day = 365.25  
        
        piEE_fit = self.fit_values_roman()['piEE'] * (1 / u.kpc)
        piEN_fit = self.fit_values_roman()['piEN'] * (1 / u.kpc)
        piE_fit  = np.sqrt(piEE_fit**2 +piEN_fit**2)

        sigma_piee = self.fit_values_roman()['piEE_err'] * (1 / u.kpc)
        sigma_pien = self.fit_values_roman()['piEN_err'] * (1 / u.kpc)
        
        thetaE = self.trilegal_params["mu_rel"]* self.trilegal_params["te"]/yr2day 
        # Calculate the fitted mass
        M_fit = ((thetaE / aconv**2) / (k * piE_fit)).decompose().to('M_sun')
        # Derivatives for error propagation
        dm_dpiee = ((thetaE * (-piEE_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dpien = ((thetaE * (-piEN_fit) / aconv**2) / (k * piE_fit**3)).decompose()
        dm_dthetaE = ((1  / aconv**2) / (k * piE_fit)).decompose()
        # Quadratic terms for the error
        cuad_terms =  (dm_dpiee * sigma_piee)**2 + (dm_dpien * sigma_pien)**2 + (dm_dthetaE * 0.1 * thetaE)**2
        # Covariance terms for the error
        data_rr, data_roman = self.data_fit()
        cov_piEE_piEN = data_rr["covariance_matrix"][self.labels_params().index('piEN'),self.labels_params().index('piEE')]
        cov_terms = 2*dm_dpiee * dm_dpien * cov_piEE_piEN * (1 / u.kpc)**2
        sigma_m = cuad_terms + cov_terms
        return np.sqrt(sigma_m).to('M_sun').value            


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
            roman_seasons.append((Time(season['start'], format='isot').jd, 
                                  Time(season['end'], format='isot').jd))

        t0 = self.fit_true()[0]["t0"]
        tE = self.fit_true()[0]["te"]
        interval1 = (t0 - tE, t0 + tE)

        overlap_rubin = False

        for j in range(len(rubin_seasons)):
            if self.intervals_overlap(interval1, rubin_seasons[j]):
                overlap_rubin = True
                break
        overlap_roman = False
        for k in range(len(roman_seasons)):
            if self.intervals_overlap(interval1, roman_seasons[k]):
                overlap_roman = True
                break

        if (overlap_rubin == True) and (overlap_roman == True):
            category='A'
        if (overlap_rubin == True) and (not overlap_roman == True):
            category='B'
        if (not overlap_rubin == True) and (not overlap_roman == True):
            category='D'
        if (not overlap_rubin == True) and (overlap_roman == True):
            category='C'
        return category
