import numpy as np
import pandas as pd
import os, sys
from pyLIMA import event
from scipy.signal import find_peaks
from tqdm.auto import tqdm
from pyLIMA import telescopes
from pyLIMA.models import USBL_model
from astropy.time import Time
import matplotlib.pyplot as plt


home = '/home/anibal/'
sys.path.append(home + '/roman_rubin/fit_codes')
# this codes are in the /fit_codes directory
# https://github.com/anibal-art/roman_rubin/tree/main/fit_codes
from fit_results import chi_dof, best_model, event_fits, sigmas
from filter_curves import read_curves
from analysis_metrics import m1,m2,m3, fit_true, metrics_from_df, sigma_ratio, bias_ratio, fit_values
from plot_models import plot_LCmodel
from plot_lightcurves import model
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# from plot_models import plot_only_model
path_ephemerides = '/home/anibal/files_db/james_webb.txt'
path_dataslice = '/home/anibal/roman_rubin/simulation/dataSlice_custom_roman.npy'
sys.path.append(home + '/results_roman_rubin/codigo_modelos_unificado/')
from sim_event import sim_event
sim_event


DATA = pd.read_csv('/home/anibal/results_roman_rubin/TRILEGAL_sets/PB_split_1.csv')
many_peak_index=[]
one_peak_index = []
for i in tqdm(range(len(DATA))):
    data = DATA.iloc[i]
    params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'], 'rho': data['rho'], 's': data['s'],
              'q': data['q'], 'alpha': data['alpha'], 'piEN': data['piEN'], 'piEE': data['piEE']}
    # t0 = params['t0']
    tE = params['tE']
    # params
    # print(tE)
    simulated_event = event.Event()
    simulated_event.name = 'Simulated'
    simulated_event.ra = 170
    simulated_event.dec = -70
    t0guess = 2457777.777
    u0guess=0.01
    rhoguess= 0.01
    time_sim = np.linspace(t0guess-10*tE,t0guess+10*tE,240)
    lightcurve_sim = np.c_[time_sim,[19.] * len(time_sim),[0.01] * len(time_sim)]
    telescope = telescopes.Telescope(name = 'Simulation',
                                     camera_filter = 'G',
                                     light_curve = lightcurve_sim.astype(float),
                                     light_curve_names = ['time','mag','err_mag'],
                                     light_curve_units = ['JD','mag','mag'])

    simulated_event.telescopes.append(telescope)

    # def plot_curve_usbl(u0=u0guess, tE=40, q=1e-3, alpha=3.14/180):
    u0=params['u0']
    # tE=
    q=params['q']
    alpha=params['alpha']

    t0 = t0guess
    rho=params['rho']
    s=params['s']

    usbl = USBL_model.USBLmodel(simulated_event, parallax=['None', 0])
    usbl.define_model_parameters()
    event_parameters = [t0, u0, tE, rho, s, q, alpha]

    pyLIMA_parameters2 = usbl.compute_pyLIMA_parameters(event_parameters)
    model = usbl.compute_the_microlensing_model(telescope, pyLIMA_parameters2)
    magnification = usbl.model_magnification(telescope, pyLIMA_parameters2)
    peaks, _ = find_peaks(2.5*np.log10(magnification), height=0)
    latex_strings = [r'$u_0$', r'$t_E$', r'$\rho$','s','q','$\\alpha$']
    units = [' [JD]', ' ', ' [day]', ' ', ' ', ' ',' ', ' ']
    many_peak_index.append([i,time_sim[peaks],magnification[peaks]])

nominal_seasons = [
    {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
    {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
    {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
    {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
    {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
    {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
]


def find_max_overlap_with_points(intervals, points):
    max_overlap = 0
    best_shift = 0

    for interval in intervals:
        for shift in [interval[0] - point for point in points]:
            overlap = sum(1 for point in points if interval[0] - shift <= point <= interval[1] - shift)
            if overlap > max_overlap:
                max_overlap = overlap
                best_shift = shift
    Best_Shift = best_shift + 5
    shifted_points = [point + Best_Shift for point in points]
    return Best_Shift


# Example usage:

# plt.close('all')
new_many_peak_index = []
intervals = [(Time(nominal_seasons[j]['start'], format='isot').jd, Time(nominal_seasons[j]['end'], format='isot').jd)
             for j in range(6)]
t0guess = 2457777.777

for i in tqdm(range(len(DATA))):
    tpeaks = many_peak_index[i][1]
    result = find_max_overlap_with_points(intervals, tpeaks)
    t0 = t0guess + result

    new_many_peak_index.append([i, tpeaks + result, 2.5 * np.log10(magnification)[peaks]])
    DATA.iloc[i]['t0'] = t0
    # plt.plot(tpeaks,np.ones(len(tpeaks)),marker='o',color='r')
    # plt.axvline(t0)
# # # print([abs(points[f]-points[f+1]) for f in range(len(points)-1)])
# # # plt.close('all')
# i = 3
DATA.to_csv('/home/anibal/results_roman_rubin/TRILEGAL_sets/PB_split_1_critpeak.csv', index=False)
tpeaks = new_many_peak_index[i][1]


# %matplotlib widget

# for i in tqdm(range(0,5000)):
#     data = DATA.iloc[i]
#     # data['t0']=
# # new_creation = sim_event(i, path_to_save, data, opsim, path_ephemerides, path_phot_series, path_dataslice)
#     print(many_peak_index[i][2])
#     new_creation = sim_event(i, path_to_save, data, path_ephemerides, path_dataslice,'USBL')


import pandas as pd
from time import time

def not_parallel(n1, n2, path_ephemerides, path_dataslice):
    path_to_save = home + '/results_roman_rubin/' + 'PB_peaks_crit/'

    for i in range(n1, n2):
        tinit = time()
        print('***********STARTING EVENT SIMULATION ', i,'***********')
        # magstar_params = data.iloc[i].to_dict()
        data = DATA.iloc[i]
        sim_event(i, path_to_save, data, path_ephemerides, path_dataslice,'USBL')
        print('*********** ENDING EVENT SIMULATION ', i, time()-tinit, '***********')

