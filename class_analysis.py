import os
import re
from typing import List, Tuple, Dict
from astropy.table import QTable
from astropy import constants as const
from astropy import units as u
import numpy as np
import pandas as pd
import h5py
from tqdm.auto import tqdm
import warnings


class EventAnalysis:
    def __init__(self, path: str, labels_params: List[str]):
        self.path = path
        self.labels_params = labels_params

    def event_fits(self, path_fits: str) -> List[int]:
        """
        Returns events that are common between Roman and Rubin.
        """
        files_fits = os.listdir(path_fits)
        files_roman = [f for f in files_fits if 'Roman' in f]
        files_rr = [f for f in files_fits if 'RR' in f]

        n_rom = [int(re.findall(r'\d+', f)[0]) for f in files_roman]
        n_rr = [int(re.findall(r'\d+', f)[0]) for f in files_rr]

        common_elements = set(n_rom).intersection(set(n_rr))
        return list(common_elements)

    def new_rows(self, camino: str, st: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes data from Rubin and returns data frames with true and fitted values.
        """
        data_rr = np.load(camino, allow_pickle=True).item()
        fit_values = dict(zip(self.labels_params, data_rr['best_model'][0:len(self.labels_params)]))

        if any(np.diag(data_rr['covariance_matrix']) < 0):
            fit_error = np.zeros(len(self.labels_params))
        else:
            fit_error = np.sqrt(np.diag(data_rr['covariance_matrix']))[0:len(self.labels_params)]

        for j, key in enumerate(self.labels_params):
            fit_values[f"{key}_err"] = fit_error[j]

        fit_values['Source'] = data_rr['true_params'].name + st * 5000
        true_values = data_rr['true_params'][self.labels_params].to_dict()
        true_values['Source'] = data_rr['true_params'].name + st * 5000

        return pd.DataFrame([true_values]), pd.DataFrame([fit_values])

    def fit_true(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Combines true and fitted values across Rubin and Roman datasets.
        """
        cols_true = ['Source'] + self.labels_params
        cols_fit = cols_true + [f"{t}_err" for t in self.labels_params]
        true, fit_rr, fit_roman = pd.DataFrame(columns=cols_true), pd.DataFrame(columns=cols_fit), pd.DataFrame(
            columns=cols_fit)
        fit_completed = []

        for st in tqdm(range(1, 5)):
            PATH = self.path + f'set_fit{st}/'
            nevent = self.event_fits(PATH)
            list_files_rr = [f'Event_RR_{int(f)}_TRF.npy' for f in nevent]
            list_files_roman = [f'Event_Roman_{int(f)}_TRF.npy' for f in nevent]

            for i in range(len(nevent)):
                path_rr, path_roman = PATH + list_files_rr[i], PATH + list_files_roman[i]
                try:
                    new_row_true, new_row_rr = self.new_rows(path_rr, st)
                    new_row_true2, new_row_roman = self.new_rows(path_roman, st)
                    true = pd.concat([true, new_row_true], ignore_index=True)
                    fit_rr = pd.concat([fit_rr, new_row_rr], ignore_index=True)
                    fit_roman = pd.concat([fit_roman, new_row_roman], ignore_index=True)
                    fit_completed.append(1)
                except Exception as e:
                    print(f"Error with event {nevent[i]}: {e}")
                    fit_completed.append(0)
        return true, fit_rr, fit_roman

    def read_data(self, path_model: str) -> Tuple[List, Dict, Dict[str, QTable]]:
        """
        Reads model data from an HDF5 file.
        """
        with h5py.File(path_model, 'r') as file:
            info_dataset = [file['Data'][i].decode('UTF-8') for i in range(3)]
            pyLIMA_parameters = {key: file['pyLIMA_parameters'].attrs[key] for key in file['pyLIMA_parameters'].attrs}
            bands = {band: QTable({col: file[band][col][:] for col in file[band]}) for band in
                     ("W149", "u", "g", "r", "i", "z", "y")}
            return info_dataset, pyLIMA_parameters, bands

    def chichi(self, name_file: str) -> Tuple[float, float, int, int]:
        """
        Calculates chi-squared values and degrees of freedom for given file.
        """
        nset = int(name_file[name_file.index('Event_') - 2:name_file.index('Event_') - 1])
        nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
        directory_simset = name_file[:name_file.index('set')] + f'set_sim{nset}/'
        model_file = f"{directory_simset}Event_{nevent}.h5"

        info_dataset, model_params, curves = self.read_data(model_file)
        name_file_rr, name_file_roman = f"{name_file[:name_file.index('set')]}set_fit{nset}/Event_RR_{nevent}_TRF.npy", \
            f"{name_file[:name_file.index('set')]}set_fit{nset}/Event_Roman_{nevent}_TRF.npy"
        data_rr, data_roman = np.load(name_file_rr, allow_pickle=True), np.load(name_file_roman, allow_pickle=True)

        try:
            chi_rr, chi_roman = data_rr.item()["chi2"], data_roman.item()["chi2"]
            dof_rr = sum(len(curves[key]) for key in curves) - len(
                [len(curves[key]) for key in curves if len(curves[key]) != 0]) * 2 - 9
            dof_roman = len(curves['W149']) - 2 - 9
            return chi_rr, chi_roman, dof_rr, dof_roman
        except:
            return 0, 0, 0, 0

    def group_consecutive_numbers(self, numbers: List[int]) -> List[List[int]]:
        """
        Groups consecutive numbers, with a gap threshold of 60.
        """
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

    def intervals_overlap(self, interval1: Tuple[int, int], interval2: Tuple[int, int]) -> bool:
        """
        Checks if two intervals overlap.
        """
        start1, end1 = interval1
        start2, end2 = interval2
        return (start1 <= end2 and end1 >= start2) or (start2 <= end1 and end2 >= start1)

    def chichi_to_fits_files(self, fit_rr: pd.DataFrame, fit_roman: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Maps chi-squared values and degrees of freedom to fit files for Rubin and Roman data.
        """
        id_to_chi2_rr, id_to_chi2_roman = {}, {}
        id_to_dof_rr, id_to_dof_roman = {}, {}
        ndir = len([f for f in os.listdir(self.path) if 'set_' in f])

        for i in tqdm(range(1, int(ndir + 1))):
            common_elements_list = self.event_fits(f"{self.path}set_fit{i}/")
            if common_elements_list:
                for j in common_elements_list:
                    name_file = f"Event_{j}.h5"
                    chi2rr, chi2roman, dof_rr, dof_roman = self.chichi(f"{self.path}set_sim{i}/{name_file}")
                    id_to_chi2_rr[j + i * 5000] = chi2rr
                    id_to_chi2_roman[j + i * 5000] = chi2roman
                    id_to_dof_rr[j + i * 5000] = dof_rr
                    id_to_dof_roman[j + i * 5000] = dof_roman

        fit_rr['chi2'] = fit_rr['Source'].map(id_to_chi2_rr)
        fit_roman['chi2'] = fit_roman['Source'].map(id_to_chi2_roman)
        fit_roman['dof'] = fit_roman['Source'].map(id_to_dof_roman)
        fit_rr['dof'] = fit_rr['Source'].map(id_to_dof_rr)
        return fit_rr, fit_roman

    def montecarlo_propagation_piE(self, best_model: List[float], covariance_matrix: np.ndarray,
                                   indx_piE: List[int]) -> np.ndarray:
        """
        Monte Carlo propagation of piE parameters.
        """
        num_samples = 10000
        samples = np.random.multivariate_normal(best_model, covariance_matrix, num_samples)
        piE_samples = samples[:, indx_piE]
        return np.linalg.norm(piE_samples, axis=1)

    def MC_tE_rho_piE(self, best_model: List[float], covariance_matrix: np.ndarray, indx_tE_rho: List[int],
                      indx_piE: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo propagation of tE, rho, and piE parameters.
        """
        num_samples = 10000
        samples = np.random.multivariate_normal(best_model, covariance_matrix, num_samples)
        tE_samples = samples[:, indx_tE_rho[0]]
        rho_samples = samples[:, indx_tE_rho[1]]
        piE_samples = self.montecarlo_propagation_piE(best_model, covariance_matrix, indx_piE)
        return tE_samples, rho_samples, piE_samples

    def piE_cov_terms(self, fit_rr: pd.DataFrame, fit_roman: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Monte Carlo propagated mass and other derived parameters.
        """
        masses_rubin, masses_roman = [], []

        for index, row in fit_rr.iterrows():
            covariance_matrix = np.load(f"{self.path}covariances/Event_{int(row['Source']) % 5000}_covariance.npy")
            masses_rubin.append(
                np.mean(self.MC_tE_rho_piE(row[self.labels_params], covariance_matrix, indx_tE_rho, indx_piE)))

        for index, row in fit_roman.iterrows():
            covariance_matrix = np.load(f"{self.path}covariances/Event_{int(row['Source']) % 5000}_covariance.npy")
            masses_roman.append(
                np.mean(self.MC_tE_rho_piE(row[self.labels_params], covariance_matrix, indx_tE_rho, indx_piE)))

        return np.array(masses_rubin), np.array(masses_roman)


