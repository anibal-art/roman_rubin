import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import numpy as np
import pandas as pd
from scipy.signal import find_peaks,peak_widths
import os, sys, re, copy, math
from tqdm.auto import tqdm

home = '/home/anibal/'
sys.path.append(home + '/roman_rubin/fit_codes')
# this codes are in the /fit_codes directory
# https://github.com/anibal-art/roman_rubin/tree/main/fit_codes
# from fit_results import chi_dof, best_model, event_fits, sigmas
# from filter_curves import read_curves
# from analysis_metrics import m1,m2,m3, fit_true, metrics, sigma_ratio, bias_ratio, fit_values
# from plot_models import plot_LCmodel
# from plot_lightcurves import model

sys.path.append(home + '/che/archive/photutils')
from bandpass import Bandpass
from signaltonoise import calc_mag_error_m5
from photometric_parameters import PhotometricParameters

#astropy
import astropy.units as u
from astropy import constants as const
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

#pyLIMA
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.toolbox import time_series
from pyLIMA.simulations import simulator
from pyLIMA.models import PSBL_model
from pyLIMA.models import USBL_model
from pyLIMA.models import FSPLarge_model
from pyLIMA.models import PSPL_model
from pyLIMA.fits import TRF_fit
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.outputs import file_outputs
import multiprocessing as mul

import h5py


def tel_roman_rubin(path_ephemerides, path_dataslice):
    '''
    :param opsim:
    :return:
    '''
    gc = SkyCoord(l=0.5 * u.degree, b=-1.25 * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    LSST_BandPass = {}
    lsst_filterlist = 'ugrizy'
    for f in lsst_filterlist:
        LSST_BandPass[f] = Bandpass()
        LSST_BandPass[f].read_throughput('/home/anibal/che/archive/troughputs/' + f'total_{f}.dat')
    dataSlice = np.load(path_dataslice, allow_pickle=True)
    rubin_ts = {}
    for fil in lsst_filterlist:
        m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)]
        mjd = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == fil)] + 2400000.5
        int_array = np.column_stack((mjd, m5, m5)).astype(float)
        rubin_ts[fil] = int_array

    tlsst = 60413.26382860778 + 2400000.5
    tstart_Roman = 2461508.763828608  # tlsst + 3*365 #Roman is expected to be launch in may 2027

    my_own_creation = event.Event(ra=Ra, dec=Dec)
    my_own_creation.name = 'An event observed by Roman'
    nominal_seasons = [
        {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
        {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
        {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
        {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
        {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
        {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
    ]
    Roman_tot = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107,
                                               time_end=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107 + 72,
                                               sampling=0.25,
                                               location='Space', camera_filter='W149', uniform_sampling=True,
                                               astrometry=False)
    lightcurve_fluxes = []
    for season in nominal_seasons:
        tstart = Time(season['start'], format='isot').jd
        tend = Time(season['end'], format='isot').jd
        Roman = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart,
                                               time_end=tend,
                                               sampling=0.25,
                                               location='Space',
                                               camera_filter='W149',
                                               uniform_sampling=True,
                                               astrometry=False)
        lightcurve_fluxes.append(Roman.lightcurve_flux)
    # Combine all the lightcurve_flux tables into one array
    combined_array = np.concatenate([lc.as_array() for lc in lightcurve_fluxes])
    # Convert the combined array back into a QTable
    new_table = QTable(combined_array, names=['time', 'flux', 'err_flux'], units=['JD', 'W/m^2', 'W/m^2'])

    Roman_tot.lightcurve_flux = new_table
    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:, 0] = ephemerides[:, 0]
    ephemerides[:, 3] *= 60 * 300000 / 150000000
    deltaT = tlsst - ephemerides[:, 0][0]
    ephemerides[:, 0] = ephemerides[:, 0] + deltaT
    Roman_tot.location = 'Space'
    Roman_tot.spacecraft_name = 'WFIRST_W149'
    Roman_tot.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    my_own_creation.telescopes.append(Roman_tot)

    for band in lsst_filterlist:
        lsst_telescope = telescopes.Telescope(name=band, camera_filter=band, location='Earth',
                                              light_curve=rubin_ts[band],
                                              light_curve_names=['time', 'mag', 'err_mag'],
                                              light_curve_units=['JD', 'mag', 'mag'])
        my_own_creation.telescopes.append(lsst_telescope)

    return my_own_creation, dataSlice, LSST_BandPass


def deviation_from_constant(pyLIMA_parameters, pyLIMA_telescopes):
    '''
     There at least four points in the range
     $[t_0-tE, t_0+t_E]$ with the magnification deviating from the
     constant flux by more than 3$\sigma$
    '''
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    satis_crit = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve_magnitude['mag']) == 0:
            mag_baseline = ZP[telo.name] - 2.5 * np.log10(pyLIMA_parameters['ftotal_' + f'{telo.name}'])
            x = telo.lightcurve_magnitude['time'].value
            y = telo.lightcurve_magnitude['mag'].value
            z = telo.lightcurve_magnitude['err_mag'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            consec = []
            if len(x[mask]) >= 3:
                combined_lists = list(zip(x[mask], y[mask], z[mask]))
                sorted_lists = sorted(combined_lists, key=lambda item: item[0])
                sorted_x, sorted_y, sorted_z = zip(*sorted_lists)
                for j in range(len(sorted_y)):
                    if sorted_y[j] + 3 * sorted_z[j] < mag_baseline:
                        consec.append(j)
                result = has_consecutive_numbers(consec)
                if result:
                    satis_crit[telo.name] = True
                else:
                    satis_crit[telo.name] = False
            else:
                satis_crit[telo.name] = False
        else:
            satis_crit[telo.name] = False
    return any(satis_crit.values())


def filter5points(pyLIMA_parameters, pyLIMA_telescopes):
    '''
    Check that at least one light curve
    have at least 5 pts in the t0+-tE
    '''
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    crit5pts = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve_magnitude['mag']) == 0:
            x = telo.lightcurve_magnitude['time'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            if len(x[mask]) >= 5:
                crit5pts[telo.name] = True
            else:
                crit5pts[telo.name] = False
    return any(crit5pts.values())


def mag(zp, Flux):
    '''
    Transform the flux to magnitude
    inputs
    zp: zero point
    Flux: vector that contains the lightcurve flux
    '''
    return zp - 2.5 * np.log10(abs(Flux))


def filter_band(mjd, mag, magerr, m5, fil):
    '''
    *Save the points of the lightcurve greater and smaller than
      1sigma fainter and brighter that the saturation and 5sigma_depth
    * check that the lightcurve have more than 10 points
    * check if the lightcurve have at least 1 point at 5 sigma from the 5sigma_depth
    '''
    mag_sat = {'W149': 14.8, 'u': 14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    MJD = []
    MAG = []
    MAGERR = []
    M5 = []
    five_sigmas = False
    for i in range(len(mjd)):
        if (mag[i] - magerr[i] > mag_sat[fil]) and (mag[i] + magerr[i] < m5[i]):
            MJD.append(mjd[i])
            MAGERR.append(magerr[i])
            MAG.append(mag[i])
            M5.append(m5[i])
        elif mag[i] + 5 * magerr[i] < m5[i]:
            five_sigmas = True

    if not len(MAG) > 10:
        MJD, MAG, MAGERR, M5 = [], [], [], []
    return MJD, MAG, MAGERR, M5


def has_consecutive_numbers(lst):
    """
    check if there at least 3 consecutive numbers in a list lst
    """
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 2):
        if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
            return True
    return False


def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams


def fit_rubin_roman(Source, event_params, path_save, path_ephemerides, model, algo, Origin, wfirst_lc, lsst_u, lsst_g,
                    lsst_r, lsst_i, lsst_z,
                    lsst_y):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
    tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)

    if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
        e.name = 'Event_Roman_' + str(int(Source))
    else:
        e.name = 'Event_RR_' + str(int(Source))
    tel_list = []

    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name='Roman', camera_filter='W149',
                                light_curve=wfirst_lc,
                                light_curve_names=['time', 'mag', 'err_mag'],
                                light_curve_units=['JD', 'mag', 'mag'],
                                location='Space')

    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:, 0] = ephemerides[:, 0]
    ephemerides[:, 3] *= 60 * 300000 / 150000000
    deltaT = tlsst - ephemerides[:, 0][0]
    ephemerides[:, 0] = ephemerides[:, 0] + deltaT
    tel1.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    e.telescopes.append(tel1)
    tel_list.append('Roman')

    lsst_lc_list = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
    lsst_bands = "ugrizy"
    for j in range(len(lsst_lc_list)):
        if not len(lsst_lc_list[j]) == 0:
            tel = telescopes.Telescope(name=lsst_bands[j], camera_filter=lsst_bands[j],
                                       light_curve=lsst_lc_list[j],
                                       light_curve_names=['time', 'mag', 'err_mag'],
                                       light_curve_units=['JD', 'mag', 'mag'],
                                       location='Earth')
            e.telescopes.append(tel)
            tel_list.append(lsst_bands[j])
    e.check_event()
    # Give the model initial guess values somewhere near their actual values so that the fit doesn't take all day
    t0 = float(event_params['t0'])
    u0 = float(event_params['u0'])
    tE = float(event_params['te'])
    rho = float(event_params['rho'])
    piEN = float(event_params['piEN'])
    piEE = float(event_params['piEE'])
    s = float(event_params['s'])
    q = float(event_params['q'])
    alpha = float(event_params['alpha'])

    rango = 0.5
    if model == 'FSPL':
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e, parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, piEN, piEE]
    elif model == 'USBL':
        pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t0])
        # pyLIMAmodel = USBL_model.USBLmodel(e, origin=Origin,
        #                                    blend_flux_parameter='ftotal',
        #                                    parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, s, q, alpha, piEN, piEE]
    elif model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e, parallax=['Full', t0])
        param_guess = [t0, u0, tE, piEN, piEE]

    if algo == 'TRF':
        fit_2 = TRF_fit.TRFfit(pyLIMAmodel)
        pool = None
    elif algo == 'MCMC':
        fit_2 = MCMC_fit.MCMCfit(pyLIMAmodel, MCMC_links=5000)
        pool = mul.Pool(processes=32)
    elif algo == 'DE':
        fit_2 = DE_fit.DEfit(pyLIMAmodel, telescopes_fluxes_method='polyfit', DE_population_size=20,
                             max_iteration=10000,
                             display_progress=True)

    fit_2.model_parameters_guess = param_guess

    if model == 'USBL':
        fit_2.fit_parameters['separation'][1] = [s - np.abs(s) * rango, s + np.abs(s) * rango]
        fit_2.fit_parameters['mass_ratio'][1] = [q - rango * q, q + rango * q]
        fit_2.fit_parameters['alpha'][1] = [0, np.pi]

    if (model == 'USBL') or (model == 'FSPL'):
        fit_2.fit_parameters['rho'][1] = [0, rho + rango * abs(rho)]

    # fit_2.fit_parameters['t0'][1] = [t0 - 10, t0 + 10]  # t0 limits
    # fit_2.fit_parameters['u0'][1] = [u0 - abs(u0) * rango, u0 + abs(u0) * rango]  # u0 limits
    fit_2.fit_parameters['tE'][1] = [tE - tE * rango, tE + tE * rango]  # tE limits in days
    fit_2.fit_parameters['piEE'][1] = [piEE - rango * abs(piEE),
                                       piEE + rango * abs(piEE)]  # parallax vector parameter boundaries
    fit_2.fit_parameters['piEN'][1] = [piEN - rango * abs(piEN),
                                       piEN + rango * abs(piEN)]  # parallax vector parameter boundaries

    if algo == "MCMC":
        fit_2.fit(computational_pool=pool)
    else:
        fit_2.fit()

    true_values = np.array(event_params)
    fit_2.fit_results['true_params'] = event_params
    np.save(path_save + e.name + '_' + algo + '.npy', fit_2.fit_results)
    return fit_2, e, pyLIMAmodel


def save(iloc, path_TRILEGAL_set, path_to_save, my_own_model, pyLIMA_parameters):
    print('saving...')
    # Save to an HDF5 file with specified names
    with h5py.File(path_to_save + 'Event_' + str(iloc) + '.h5', 'w') as file:
        # Save array with a specified name
        file.create_dataset('Data', data=np.array([iloc, path_TRILEGAL_set, my_own_model.origin[0]], dtype='S'))
        # Save dictionary with a specified name
        dict_group = file.create_group('pyLIMA_parameters')
        for key, value in pyLIMA_parameters.items():
            dict_group.attrs[key] = value
        # Save table with a specified name
        for telo in my_own_model.event.telescopes:
            table = telo.lightcurve_magnitude
            table_group = file.create_group(telo.name)
            for col in table.colnames:
                table_group.create_dataset(col, data=table[col])


def read_data(path_model):
    # Open the HDF5 file and load data using specified names
    with h5py.File(path_model, 'r') as file:
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


def sim_event(i, data, path_ephemerides, path_dataslice, model):
    '''
    i (int): index of the TRILEGAL data set
    data (dictionary): parameters including magnitude of the stars
    path_ephemerides (str): path to the ephemeris of Gaia
    path_dataslice(str): path to the dataslice obtained from OpSims
    model(str): model desired
    '''
    magstar = {'W149': data["W149"], 'u': data["u"], 'g': data["g"], 'r': data["r"],
               'i': data["i"], 'z': data["z"], 'y': data["Y"]}
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    my_own_creation, dataSlice, LSST_BandPass = tel_roman_rubin(path_ephemerides,
                                                                path_dataslice)
    photParams = set_photometric_parameters(15, 2)
    new_creation = copy.deepcopy(my_own_creation)
    np.random.seed(i)
    t0 = data['t0']
    tE = data['te']
    if model == 'USBL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'], 'rho': data['rho'],
                  's': data['s'], 'q': data['q'], 'alpha': data['alpha'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
        my_own_model = USBL_model.USBLmodel(new_creation, origin=["third_caustic", [0, 0]],
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0])
        print(my_own_model.origin)
        # my_own_model = USBL_model.USBLmodel(new_creation,origin=[choice, [0, 0]], parallax=['Full', t0])
    elif model == 'FSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'],
                  'rho': data['rho'], 'piEN': data['piEN'],
                  'piEE': data['piEE']}
        my_own_model = FSPLarge_model.FSPLargemodel(new_creation, parallax=['Full', t0])
    elif model == 'PSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        my_own_model = PSPL_model.PSPLmodel(new_creation, parallax=['Full', t0])

    my_own_parameters = []
    for key in params:
        my_own_parameters.append(params[key])

    my_own_flux_parameters = []
    fs, G, F = {}, {}, {}
    np.random.seed(i)
    for band in magstar:
        flux_baseline = 10 ** ((ZP[band] - magstar[band]) / 2.5)
        g = np.random.uniform(0, 1)
        f_source = flux_baseline / (1 + g)
        fs[band] = f_source
        G[band] = g
        F[band] = f_source + g * f_source  # flux_baseline
        f_total = f_source * (1 + g)
        if my_own_model.blend_flux_parameter == "ftotal":
            my_own_flux_parameters.append(f_source)
            my_own_flux_parameters.append(f_total)
        else:
            my_own_flux_parameters.append(f_source)
            my_own_flux_parameters.append(f_source * g)  # *f_source #esto antes era g

    my_own_parameters += my_own_flux_parameters
    print(my_own_parameters)
    pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)
    simulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters)

    for k in range(0, len(new_creation.telescopes)):
        model_flux = my_own_model.compute_the_microlensing_model(new_creation.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
        new_creation.telescopes[k].lightcurve_flux['flux'] = model_flux

    Roman_band = False
    Rubin_band = False

    all_residuals = {}
    for telo in new_creation.telescopes:
        if telo.name == 'W149':
            x = telo.lightcurve_magnitude['time'].value
            y = telo.lightcurve_magnitude['mag'].value
            z = telo.lightcurve_magnitude['err_mag'].value
            m5 = np.ones(len(x)) * 27.6

            residuals = mag(27.4, telo.lightcurve_flux['flux']) - y

            X, Y, Z, sigma_5 = filter_band(x, y - 27.4 + ZP[telo.name], z, m5, telo.name)

            pairs = list(zip(x, residuals))
            selected_pairs = [pair for pair in pairs if pair[0] in X]
            selected_residuals = [pair[1] for pair in selected_pairs]
            RESIDUALS = list(zip(X, selected_residuals))
            all_residuals[telo.name] = RESIDUALS
            telo.lightcurve_magnitude = QTable([X, Y, Z],
                                               names=['time', 'mag', 'err_mag'], units=['JD', 'mag', 'mag'])
            if not len(telo.lightcurve_magnitude['mag']) == 0:
                Roman_band = True

        else:
            X = telo.lightcurve_flux['time'].value
            # print(len(X))
            ym = mag(ZP[telo.name], telo.lightcurve_flux['flux'].value)
            z, y, x, M5 = [], [], [], []
            residuals = []
            for k in range(len(ym)):
                m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == telo.name)][k]
                magerr = calc_mag_error_m5(ym[k], LSST_BandPass[telo.name], m5, photParams)[0]
                mag_var = np.random.normal(ym[k], magerr)
                z.append(magerr)
                y.append(mag_var)
                x.append(X[k])
                M5.append(m5)
                residuals.append(ym[k] - mag_var)

            X, Y, Z, sigma_5 = filter_band(x, y, z, M5, telo.name)

            pairs = list(zip(x, residuals))
            selected_pairs = [pair for pair in pairs if pair[0] in X]
            selected_residuals = [pair[1] for pair in selected_pairs]
            RESIDUALS = list(zip(X, selected_residuals))
            all_residuals[telo.name] = RESIDUALS
            telo.lightcurve_magnitude = QTable([X, Y, Z],
                                               names=['time', 'mag', 'err_mag'],
                                               units=['JD', 'mag', 'mag'])

            if not len(telo.lightcurve_magnitude['mag']) == 0:
                Rubin_band = True

    # This first if holds for an event with at least one Roman and Rubin band
    if Rubin_band and Roman_band:
        # This second if holds for a "detectable" event to fit
        if filter5points(pyLIMA_parameters, new_creation.telescopes) and deviation_from_constant(pyLIMA_parameters,
                                                                                                 new_creation.telescopes):
            print("A good event to fit")
            return my_own_model, pyLIMA_parameters, all_residuals, True
        else:
            print(
                "Not a good event to fit.\nFail 5 points in t0+-tE\nNot have 3 consecutives points that deviate from constant flux in t0+-tE")
            return my_own_model, pyLIMA_parameters, all_residuals, False
    else:
        print("Not a good event to fit since no Rubin band")
        return my_own_model, pyLIMA_parameters, all_residuals, False


def modelo(params, model, fit):
    ts = []
    teles = model.event.telescopes
    for tel in teles:
        ts += list(tel.lightcurve_magnitude['time'].value)
        # print(tel.name,len(tel.lightcurve_magnitude['time'].value))
    ts_event = np.array(ts)
    # print(type(ts_event),len(ts_event))
    Origin = model.origin
    path_ephemerides = '/home/anibal/files_db/james_webb.txt'
    gc = SkyCoord(l=0.5 * u.degree, b=-1.25 * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    simulated_event = event.Event(ra=Ra, dec=Dec)
    simulated_event.name = 'Simulated'

    # if type(params)==dict:
    tE = params['tE']
    if model.model_type() == 'USBL':
        u0 = params['u_center']
        t0 = params['t_center']
    else:
        t0 = params['t0']
        u0 = params['u0']

    tlsst = 60413.26382860778 + 2400000.5

    time_sim = concatenated_array = np.concatenate((np.linspace(tlsst, tlsst + 10 * 365, 5000), ts))
    # print('*********************time_sim**********************',time_sim)

    lightcurve_sim = np.c_[time_sim, [19.] * len(time_sim), [0.01] * len(time_sim)]
    telescope = telescopes.Telescope(name='W149', camera_filter='W149', location='Space',
                                     light_curve=lightcurve_sim.astype(float),
                                     light_curve_names=['time', 'mag', 'err_mag'],
                                     light_curve_units=['JD', 'mag', 'mag'])

    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:, 0] = ephemerides[:, 0]
    ephemerides[:, 3] *= 60 * 300000 / 150000000
    deltaT = tlsst - ephemerides[:, 0][0]
    ephemerides[:, 0] = ephemerides[:, 0] + deltaT
    # telescope.location = 'Space'
    telescope.spacecraft_name = 'W149'
    telescope.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    simulated_event.telescopes.append(telescope)
    flux_parameters = []
    flux_parameters.append(params['fsource_' + 'W149'])
    flux_parameters.append(params['ftotal_' + 'W149'])
    # for b in 'ugrizy':
    b = 'r'
    telescope2 = telescopes.Telescope(name=b,
                                      camera_filter=b, location='Earth',
                                      light_curve=lightcurve_sim.astype(float),
                                      light_curve_names=['time', 'mag', 'err_mag'],
                                      light_curve_units=['JD', 'mag', 'mag'])
    # telescope2.location = 'Earth'
    simulated_event.telescopes.append(telescope2)
    flux_parameters.append(params['fsource_' + 'W149'])
    flux_parameters.append(params['ftotal_' + 'W149'])

    piEE = params['piEE']
    piEN = params['piEN']

    new_creation = copy.deepcopy(simulated_event)
    if model.model_type() == 'USBL':
        q = params['mass_ratio']
        alpha = params['alpha']
        rho = params['rho']
        s = params['separation']

        if fit:
            usbl = USBL_model.USBLmodel(new_creation,
                                        # origin=['third_caustic', [0, 0]],
                                        blend_flux_parameter='ftotal',
                                        parallax=['Full', t0])
        else:
            usbl = USBL_model.USBLmodel(new_creation, origin=Origin,
                                        blend_flux_parameter='ftotal',
                                        parallax=['Full', t0])

        event_parameters = [t0, u0, tE, rho, s, q, alpha, piEN, piEE] + flux_parameters

    elif model.model_type() == 'FSPLarge':
        rho = params['rho']

        usbl = FSPLarge_model.FSPLargemodel(new_creation,
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0])
        event_parameters = [t0, u0, tE, rho, piEN, piEE] + flux_parameters
    else:

        usbl = PSPL_model.PSPLmodel(new_creation,
                                    blend_flux_parameter='ftotal',
                                    parallax=['Full', t0])
        event_parameters = [t0, u0, tE, piEN, piEE] + flux_parameters

    # print(event_parameters)
    pyLIMA_parameters2 = usbl.compute_pyLIMA_parameters(event_parameters)

    # print(pyLIMA_parameters2)
    simulator.simulate_lightcurve_flux(usbl, pyLIMA_parameters2, add_noise=False)
    return new_creation, usbl


def data_aligned(params, model, residuals):
    Origin = model.origin
    path_ephemerides = '/home/anibal/files_db/james_webb.txt'
    gc = SkyCoord(l=0.5 * u.degree, b=-1.25 * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    simulated_event = event.Event(ra=Ra, dec=Dec)
    simulated_event.name = 'Simulated'

    tE = params['tE']
    if 't_center' in params.keys():
        t0 = params['t_center']
        u0 = params['u_center']
    else:
        t0 = params['t0']
        u0 = params['u0']

    tlsst = 60413.26382860778 + 2400000.5

    time_sim = [pair[0] for pair in all_residuals['W149']]
    lightcurve_sim = np.c_[time_sim, [19.] * len(time_sim), [0.01] * len(time_sim)]

    telescope = telescopes.Telescope(name='W149',
                                     camera_filter='W149',
                                     light_curve=lightcurve_sim,
                                     light_curve_names=['time', 'mag', 'err_mag'],
                                     light_curve_units=['JD', 'mag', 'mag'])

    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:, 0] = ephemerides[:, 0]
    ephemerides[:, 3] *= 60 * 300000 / 150000000
    deltaT = tlsst - ephemerides[:, 0][0]
    ephemerides[:, 0] = ephemerides[:, 0] + deltaT
    # telescope.location = 'Space'
    telescope.spacecraft_name = 'W149'
    telescope.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    simulated_event.telescopes.append(telescope)
    flux_parameters = []
    flux_parameters.append(params['fsource_' + 'W149'])
    flux_parameters.append(params['ftotal_' + 'W149'])
    for b in 'ugrizy':
        time_sim = [pair[0] for pair in all_residuals[b]]
        lightcurve_sim = np.c_[time_sim, [19.] * len(time_sim), [0.01] * len(time_sim)]

        telescope2 = telescopes.Telescope(name=b,
                                          camera_filter=b, location='Earth',
                                          light_curve=lightcurve_sim.astype(float),
                                          light_curve_names=['time', 'mag', 'err_mag'],
                                          light_curve_units=['JD', 'mag', 'mag'])
        # telescope2.location = 'Earth'
        simulated_event.telescopes.append(telescope2)
        flux_parameters.append(params['fsource_W149'])
        flux_parameters.append(params['ftotal_W149'])

    new_creation = copy.deepcopy(simulated_event)
    piEE = params['piEE']
    piEN = params['piEN']

    if model.model_type() == 'FSPLarge':
        usbl = FSPLarge_model.FSPLargemodel(new_creation,
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0])
        rho = params['rho']
        event_parameters = [t0, u0, tE, rho, piEN, piEE] + flux_parameters
    elif model.model_type() == 'USBL':
        usbl = USBL_model.USBLmodel(new_creation, origin=Origin,
                                    blend_flux_parameter='ftotal',
                                    parallax=['Full', t0])
        rho = params['rho']
        q = params['mass_ratio']
        alpha = params['alpha']
        s = params['separation']
        event_parameters = [t0, u0, tE, rho, s, q, alpha, piEN, piEE] + flux_parameters
    else:
        usbl = PSPL_model.PSPLmodel(new_creation,
                                    blend_flux_parameter='ftotal',
                                    parallax=['Full', t0])

        event_parameters = [t0, u0, tE, piEN, piEE] + flux_parameters

    # event_parameters = [t0, u0, tE, rho, s, q, alpha,piEN,piEE]+flux_parameters
    # print(event_parameters)
    pyLIMA_parameters2 = usbl.compute_pyLIMA_parameters(event_parameters)

    print(pyLIMA_parameters2)
    simulator.simulate_lightcurve_flux(usbl, pyLIMA_parameters2, add_noise=False)
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    for telo in usbl.event.telescopes:
        residuals = [pair[1] for pair in all_residuals[telo.name]]
        X = telo.lightcurve_magnitude['time'].value
        Y = mag(ZP['W149'], telo.lightcurve_flux['flux'].value) - np.array(residuals)
        Z = telo.lightcurve_magnitude['err_mag'].value
        telo.lightcurve_magnitude = QTable([X, Y, Z],
                                           names=['time', 'mag', 'err_mag'],
                                           units=['JD', 'mag', 'mag'])

    return new_creation, usbl


def plot_lightcurves(my_own_model, pyLIMA_parameters, all_residuals):
    simulated_event2, usbl = data_aligned(pyLIMA_parameters, my_own_model.origin, all_residuals)  # modelo
    simulated_event, model = modelo(pyLIMA_parameters, my_own_model.origin, False)  # modelo
    j = 1
    index_bands = {'W149': 0, 'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6}
    Markers = {'W149': '>', 'u': 'd', 'g': '<', 'r': 'v', 'i': '^', 'z': 'o', 'y': '*'}
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    # if decision:
    for telo in my_own_model.event.telescopes:
        if not len(telo.lightcurve_magnitude['mag'].value) == 0:
            plt.errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                         simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude['mag'].value,
                         telo.lightcurve_magnitude['err_mag'].value,
                         marker=Markers[telo.name], ls=" ", alpha=0.6, color=colorbands[telo.name])
            j = j + 1
            plt.plot(simulated_event.telescopes[0].lightcurve_flux['time'].value,
                     mag(ZP['W149'], simulated_event.telescopes[0].lightcurve_flux['flux'].value), marker="", ls="-",
                     lw=2, color='blue')
            plt.plot(simulated_event.telescopes[1].lightcurve_flux['time'].value,
                     mag(ZP['W149'], simulated_event.telescopes[1].lightcurve_flux['flux'].value), marker="", ls="-",
                     lw=2, color='orange')
    plt.gca().invert_yaxis()
    plt.xlabel('Time JD [day]')
    plt.ylabel('Magnitude')

    plt.title(r'$t_E$ = ' + str(pyLIMA_parameters['tE']))
    plt.legend(loc='best')
    # plt.show()


def plot_fits(axes, ylim, xlim, my_own_model, pyLIMA_parameters, all_residuals, data_fit_rr, data_fit_roman):
    # print(my_own_model.model_type)
    simulated_event2, usbl = data_aligned(pyLIMA_parameters, my_own_model, all_residuals)  # modelo

    pyLIMA_parameters_rr = {}
    pyLIMA_parameters_roman = {}
    zp_pyLIMA_to_Rubin = (10 ** ((27.4 - ZP['W149']) / 2.5))
    if my_own_model.model_type() == 'USBL':
        for i, p in enumerate(
                ['t_center', 'u_center', 'tE', 'rho', 'separation', 'mass_ratio', 'alpha', 'piEN', 'piEE']):
            pyLIMA_parameters_rr[p] = data_fit_rr['best_model'][i]
            pyLIMA_parameters_roman[p] = data_fit_roman['best_model'][i]
        pyLIMA_parameters_rr['fsource_W149'] = data_fit_rr['best_model'][9] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_W149'] = data_fit_rr['best_model'][10] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['fsource_r'] = data_fit_rr['best_model'][9] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_r'] = data_fit_rr['best_model'][
                                               10] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']

        pyLIMA_parameters_roman['fsource_W149'] = data_fit_roman['best_model'][9] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_roman['ftotal_W149'] = data_fit_roman['best_model'][
                                                     10] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']


    elif my_own_model.model_type() == 'FSPLarge':
        for i, p in enumerate(['t0', 'u0', 'tE', 'rho', 'piEN', 'piEE']):
            pyLIMA_parameters_rr[p] = data_fit_rr['best_model'][i]
            pyLIMA_parameters_roman[p] = data_fit_roman['best_model'][i]
        pyLIMA_parameters_rr['fsource_W149'] = data_fit_rr['best_model'][6] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_W149'] = data_fit_rr['best_model'][7] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['fsource_r'] = data_fit_rr['best_model'][6] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_r'] = data_fit_rr['best_model'][
                                               7] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']

        pyLIMA_parameters_roman['fsource_W149'] = data_fit_roman['best_model'][6] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_roman['ftotal_W149'] = data_fit_roman['best_model'][
                                                     7] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']

    elif my_own_model.model_type() == 'PSPL':
        for i, p in enumerate(['t0', 'u0', 'tE', 'piEN', 'piEE']):
            pyLIMA_parameters_rr[p] = data_fit_rr['best_model'][i]
            pyLIMA_parameters_roman[p] = data_fit_roman['best_model'][i]

        pyLIMA_parameters_rr['fsource_W149'] = data_fit_rr['best_model'][5] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_W149'] = data_fit_rr['best_model'][6] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['fsource_r'] = data_fit_rr['best_model'][5] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_rr['ftotal_r'] = data_fit_rr['best_model'][
                                               6] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']

        pyLIMA_parameters_roman['fsource_W149'] = data_fit_roman['best_model'][5] / zp_pyLIMA_to_Rubin
        pyLIMA_parameters_roman['ftotal_W149'] = data_fit_roman['best_model'][
                                                     6] / zp_pyLIMA_to_Rubin  # +pyLIMA_parameters['ftotal_r']-pyLIMA_parameters['fsource_r']

    # pyLIMA_parameters_roman[q] = pyLIMA_parameters[q]
    simulated_event_rr, model1 = modelo(pyLIMA_parameters_rr, my_own_model, True)  # modelo rubin
    simulated_event_r, model2 = modelo(pyLIMA_parameters_roman, my_own_model, True)  # modelo roman
    simulated_event_true, model = modelo(pyLIMA_parameters, my_own_model, False)  # modelo roman

    index_bands = {'W149': 0, 'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6}
    Markers = {'W149': '>', 'u': 'd', 'g': '<', 'r': 'v', 'i': '^', 'z': 'o', 'y': '*'}
    labels = {'W149': 'F146', 'u': 'u', 'g': 'g', 'r': 'r', 'i': 'i', 'z': 'z', 'y': 'y'}
    # if decision:
    for telo in my_own_model.event.telescopes:
        if not len(telo.lightcurve_magnitude['mag'].value) == 0:
            axes[0].errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                             simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude['mag'].value,
                             telo.lightcurve_magnitude['err_mag'].value,
                             marker=Markers[telo.name], ls=" ", alpha=0.6, color=colorbands[telo.name],
                             label=labels[telo.name], zorder=1)
            if telo.name == 'W149':
                indices_rr_w149 = np.where(np.in1d(simulated_event_rr.telescopes[0].lightcurve_flux['time'].value,
                                                   simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux[
                                                       'time'].value))[0]
                indices_r_w149 = np.where(np.in1d(simulated_event_r.telescopes[0].lightcurve_flux['time'].value,
                                                  simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux[
                                                      'time'].value))[0]
                y_r_rr = mag(ZP['W149'], simulated_event_rr.telescopes[0].lightcurve_flux['flux'].value)[
                    indices_rr_w149]
                y_r_r = mag(ZP['W149'], simulated_event_r.telescopes[0].lightcurve_flux['flux'].value)[indices_r_w149]
                axes[1].errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                                 simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude[
                                     'mag'].value - y_r_rr, telo.lightcurve_magnitude['err_mag'].value,
                                 marker=Markers[telo.name], ls=" ", lw=2, color=colorbands[telo.name])
                axes[2].errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                                 simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude[
                                     'mag'].value - y_r_r, telo.lightcurve_magnitude['err_mag'].value,
                                 marker=Markers[telo.name], ls=" ", lw=2, color=colorbands[telo.name])
            else:
                indices_rr_lsst = np.where(np.in1d(simulated_event_rr.telescopes[0].lightcurve_flux['time'].value,
                                                   simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux[
                                                       'time'].value))[0]
                y_rr = mag(ZP['W149'], simulated_event_rr.telescopes[1].lightcurve_flux['flux'].value)[indices_rr_lsst]
                axes[1].errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                                 simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude[
                                     'mag'].value - y_rr, telo.lightcurve_magnitude['err_mag'].value,
                                 marker=Markers[telo.name], ls="", lw=2, color=colorbands[telo.name])

    axes[0].plot(simulated_event_rr.telescopes[0].lightcurve_flux['time'].value,
                 mag(ZP['W149'], simulated_event_rr.telescopes[0].lightcurve_flux['flux'].value), marker=" ", ls="--",
                 lw=2, color='royalblue', label='Fit Roman+Rubin (Space)', zorder=4)
    axes[0].plot(simulated_event_rr.telescopes[1].lightcurve_flux['time'].value,
                 mag(ZP['W149'], simulated_event_rr.telescopes[1].lightcurve_flux['flux'].value), marker=" ", ls="-",
                 lw=2, color='royalblue', label='Fit Roman+Rubin (Earth)', zorder=4)
    axes[0].plot(simulated_event_r.telescopes[0].lightcurve_flux['time'].value,
                 mag(ZP['W149'], simulated_event_r.telescopes[0].lightcurve_flux['flux'].value), marker=" ", ls="--",
                 lw=2, color='orange', label='Fit Roman (Space)', zorder=3)
    axes[0].plot(simulated_event_true.telescopes[0].lightcurve_flux['time'].value,
                 mag(ZP['W149'], simulated_event_true.telescopes[0].lightcurve_flux['flux'].value), marker=" ", ls="--",
                 lw=2, color='crimson', label='True model (Space)', zorder=2)
    axes[0].plot(simulated_event_true.telescopes[1].lightcurve_flux['time'].value,
                 mag(ZP['W149'], simulated_event_true.telescopes[1].lightcurve_flux['flux'].value), marker=" ", ls="-",
                 lw=2, color='crimson', label='True model (Earth)', zorder=2)
    axes[0].invert_yaxis()
    axes[2].set_xlabel('Time JD [day]', fontsize=16)
    axes[0].set_ylabel('Magnitude', fontsize=16)
    axes[0].legend(shadow=True, fontsize='large',
                   bbox_to_anchor=(0, 1.02, 1, 0.2),
                   loc="lower left",
                   mode="expand", borderaxespad=0, ncol=3)
    # -------------RESIDUALS---------------------------------------------------------
    # for telo in my_own_model.event.telescopes:
    #     if not len(telo.lightcurve_magnitude['mag'].value)==0:
    axes[1].set_ylabel(r'$\Delta m (RR)$', fontsize=16)
    axes[2].set_ylabel(r'$\Delta m (R)$', fontsize=16)
    # axes[0].set_xticks([])
    # axes[1].set_xticks([])
    tE = pyLIMA_parameters['tE']
    if my_own_model.model_type() == 'USBL':
        t0 = pyLIMA_parameters['t_center']
    else:
        t0 = pyLIMA_parameters['t0']

    # axes[2].set_xticks([t0 -  2*tE,t0,t0 +  2*tE],
    # [r'$t_0-2t_E$','$t_0$',r'$t_0+2t_E$'],fontsize=12)
    # -------------------------Annotate---------------------------------------
    # m1_te_rr = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)
    # m1_rho_rr = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)
    # m1_piE_rr = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)
    # m1_te_rom = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)
    # m1_rho_rom = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)
    # m1_piE_rom = round(abs(pyLIMA_parameters['tE']-pyLIMA_parameters_rr['tE'])/pyLIMA_parameters['tE'],3)

    #     string_roman = 'Roman:\n'
    #     for p in ('tE','rho'):
    #         m1_p_rr = round(abs(pyLIMA_parameters[p]-pyLIMA_parameters_rr[p])/pyLIMA_parameters[p],3)
    #         string_roman+= r'$\frac{|t_E^{true}-t_E^{fit}|}{t_E^{true}}$='+f'{m1_p_rr}'+'     '

    #     axes[2].annotate(string_roman,
    #             xy=(0.5, -1.5), xycoords='axes fraction',
    #             ha='center', va='center', fontsize=15)

    # axes[2].annotate(f'Fraction of events Roman\nwith {3}<0.1 = {3}',
    #         xy=(0.5, -2), xycoords='axes fraction',
    #         ha='center', va='center', fontsize=15)
    # --------------------------------------------------------------------------------
    axins = inset_axes(axes[0], width="40%", height="40%", loc='upper left')
    mins = []
    maxs = []
    for telo in my_own_model.event.telescopes:
        if not len(telo.lightcurve_magnitude['mag'].value) == 0:
            axins.errorbar(simulated_event2.telescopes[index_bands[telo.name]].lightcurve_flux['time'].value,
                           simulated_event2.telescopes[index_bands[telo.name]].lightcurve_magnitude['mag'].value,
                           telo.lightcurve_magnitude['err_mag'].value,
                           marker=Markers[telo.name], ls=" ", alpha=0.6, color=colorbands[telo.name], zorder=1)
    axins.plot(simulated_event_rr.telescopes[0].lightcurve_flux['time'].value,
               mag(ZP['W149'], simulated_event_rr.telescopes[0].lightcurve_flux['flux'].value), marker="", ls="--",
               lw=2, color='royalblue', label='Fit Roman+Rubin (Space)', zorder=4)
    axins.plot(simulated_event_rr.telescopes[1].lightcurve_flux['time'].value,
               mag(ZP['W149'], simulated_event_rr.telescopes[1].lightcurve_flux['flux'].value), marker="", ls="-", lw=2,
               color='royalblue', label='Fit Roman+Rubin (Earth)', zorder=4)
    axins.plot(simulated_event_r.telescopes[0].lightcurve_flux['time'].value,
               mag(ZP['W149'], simulated_event_r.telescopes[0].lightcurve_flux['flux'].value), marker="", ls="--", lw=2,
               color='orange', label='Fit Roman (Space)', zorder=3)
    axins.plot(simulated_event_true.telescopes[0].lightcurve_flux['time'].value,
               mag(ZP['W149'], simulated_event_true.telescopes[0].lightcurve_flux['flux'].value), marker="", ls="--",
               lw=2, color='crimson', label='True model (Space)', zorder=2)
    axins.plot(simulated_event_true.telescopes[1].lightcurve_flux['time'].value,
               mag(ZP['W149'], simulated_event_true.telescopes[1].lightcurve_flux['flux'].value), marker="", ls="-",
               lw=2, color='crimson', label='True model (Earth)', zorder=2)

    # Set limits for the inset axis
    frac_d = 10
    frac = 1 / frac_d
    # axins.legend(loc='best')

    axins.set_xlim(t0 - xlim[0] * tE,
                   t0 + xlim[1] * tE)
    axins.set_ylim(ylim[0], ylim[1])
    axins.set_yticks([])
    axins.set_xticks([])  # [t0 -  2*tE,t0,t0 +  2*tE],
    # [r'$t_0-2t_E$','$t_0$',r'$t_0+2t_E$'],fontsize=12)
    axins.invert_yaxis()
    # Mark the region in the main plot corresponding to the inset axis
    mark_inset(axes[0], axins, loc1=3, loc2=4, fc="none", ec="gray")


