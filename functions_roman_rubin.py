import numpy as np
import os, sys, re, copy, math
import pandas as pd
from pathlib import Path
# Get the directory where the script is located
script_dir = Path(__file__).parent
print(script_dir)
sys.path.append(str(script_dir)+'/photutils/')
# print(os.getcwd()+'/photutils/')
from bandpass import Bandpass
from signaltonoise import calc_mag_error_m5
from photometric_parameters import PhotometricParameters

#astropy
import astropy.units as u
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
    print(str(script_dir))
    gc = SkyCoord(l=0.5 * u.degree, b=-1.25 * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    LSST_BandPass = {}
    lsst_filterlist = 'ugrizy'
    for f in lsst_filterlist:
        LSST_BandPass[f] = Bandpass()
        LSST_BandPass[f].read_throughput(str(script_dir)+'/troughputs/' + f'total_{f}.dat')
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

# I add rango as input
def fit_rubin_roman(Source, event_params, path_save, path_ephemerides, model, algo, Origin,rango, wfirst_lc, lsst_u, lsst_g,
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
        name_roman = 'Roman'
    else:
        e.name = 'Event_RR_' + str(int(Source))
        name_roman = 'Roman'
    tel_list = []

    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name = name_roman, camera_filter='W149',
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
   # rho = float(event_params['rho'])
    piEN = float(event_params['piEN'])
    piEE = float(event_params['piEE'])
   # s = float(event_params['s'])
   # q = float(event_params['q'])
   # alpha = float(event_params['alpha'])

    # rango = 1
    if model == 'FSPL':
        rho = float(event_params['rho'])
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e,blend_flux_parameter='ftotal', parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, piEN, piEE]
    elif model == 'USBL':
        rho = float(event_params['rho'])
        s = float(event_params['s'])
        q = float(event_params['q'])
        alpha = float(event_params['alpha'])
        # pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t0])
        pyLIMAmodel = USBL_model.USBLmodel(e,
                                           blend_flux_parameter='ftotal',
                                           parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, s, q, alpha, piEN, piEE]
    elif model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e,blend_flux_parameter='ftotal', parallax=['Full', t0])
        param_guess = [t0, u0, tE, piEN, piEE]



    
    if algo == 'TRF':
        fit_2 = TRF_fit.TRFfit(pyLIMAmodel)
        pool = None
    elif algo == 'MCMC':
        fit_2 = MCMC_fit.MCMCfit(pyLIMAmodel, MCMC_links=7000)
        pool = mul.Pool(processes=36)
    elif algo == 'DE':
        pool = mul.Pool(processes=16)
        fit_2 = DE_fit.DEfit(pyLIMAmodel, telescopes_fluxes_method='polyfit', DE_population_size=20,
                             max_iteration=10000,
                             display_progress=True)

    fit_2.model_parameters_guess = param_guess

    if model == 'USBL':
        fit_2.fit_parameters['separation'][1] = [s - np.abs(s) * rango, s + np.abs(s) * rango]
        fit_2.fit_parameters['mass_ratio'][1] = [q - rango * q, q + rango * q]
        fit_2.fit_parameters['alpha'][1] = [alpha - rango * abs(alpha), alpha + rango * abs(alpha)]

    if (model == 'USBL') or (model == 'FSPL'):
        if (rho - rango * abs(rho))<0:
            fit_2.fit_parameters['rho'][1] = [0, rho + rango * abs(rho)]
        else:
            fit_2.fit_parameters['rho'][1] = [rho - rango * abs(rho), rho + rango * abs(rho)]

    fit_2.fit_parameters['t0'][1] = [t0 - 10, t0 + 10]  # t0 limits
    fit_2.fit_parameters['u0'][1] = [u0 - abs(u0) * rango, u0 + abs(u0) * rango]  # u0 limits

    fit_2.fit_parameters['tE'][1] = [tE - tE * rango, tE + tE * rango]  # tE limits in days
    fit_2.fit_parameters['piEE'][1] = [piEE - rango * abs(piEE),
                                       piEE + rango * abs(piEE)]  # parallax vector parameter boundaries
    fit_2.fit_parameters['piEN'][1] = [piEN - rango * abs(piEN),
                                       piEN + rango * abs(piEN)]  # parallax vector parameter boundaries

    if algo == "MCMC" or algo =='DE' :
        fit_2.fit(computational_pool=pool)
    else:
        fit_2.fit()

    true_values = np.array(event_params)
    fit_2.fit_results['true_params'] = event_params
    fit_2.fit_results['rango'] = rango
    np.save(path_save + e.name + '_' + algo +'.npy', fit_2.fit_results)
    return fit_2, e, pyLIMAmodel


def save(iloc, path_TRILEGAL_set, path_to_save, my_own_model, pyLIMA_parameters):
    print('Saving...')
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
    print('File saved:',path_to_save + 'Event_' + str(iloc) + '.h5' )


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
        my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
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
            my_own_flux_parameters.append(f_source * g)

    my_own_parameters += my_own_flux_parameters
    pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)
    simulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters)

    for k in range(1, len(new_creation.telescopes)):
        model_flux = my_own_model.compute_the_microlensing_model(new_creation.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
        new_creation.telescopes[k].lightcurve_flux['flux'] = model_flux
    Roman_band = False
    Rubin_band = False
    for telo in new_creation.telescopes:
        if telo.name == 'W149':
            x = telo.lightcurve_magnitude['time'].value
            y = telo.lightcurve_magnitude['mag'].value
            z = telo.lightcurve_magnitude['err_mag'].value
            m5 = np.ones(len(x)) * 27.6
            X, Y, Z, sigma_5 = filter_band(x, y - 27.4 + ZP[telo.name], z, m5, telo.name)
            telo.lightcurve_magnitude = QTable([X, Y, Z],
                                               names=['time', 'mag', 'err_mag'], units=['JD', 'mag', 'mag'])
            if not len(telo.lightcurve_magnitude['mag']) == 0:
                Roman_band = True
        else:
            X = telo.lightcurve_flux['time'].value
            ym = mag(ZP[telo.name], telo.lightcurve_flux['flux'].value)
            z, y, x, M5 = [], [], [], []
            for k in range(len(ym)):
                m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == telo.name)][k]
                magerr = calc_mag_error_m5(ym[k], LSST_BandPass[telo.name], m5, photParams)[0]
                z.append(magerr)
                y.append(np.random.normal(ym[k], magerr))
                x.append(X[k])
                M5.append(m5)
            X, Y, Z, sigma_5 = filter_band(x, y, z, M5, telo.name)
            telo.lightcurve_magnitude = QTable([X, Y, Z],
                                               names=['time', 'mag', 'err_mag'],
                                               units=['JD', 'mag', 'mag'])

            if not len(telo.lightcurve_magnitude['mag']) == 0:
                Rubin_band = True

    # This first if holds for an event with at least one Roman and Rubin band
    if Rubin_band and Roman_band:
        # This second if holds for a "detectable" event to fit
        if filter5points(pyLIMA_parameters, new_creation.telescopes) and deviation_from_constant(pyLIMA_parameters, new_creation.telescopes):
            print("A good event to fit")
            return my_own_model, pyLIMA_parameters, True
        else:
            print(
                "Not a good event to fit.\nFail 5 points in t0+-tE\nNot have 3 consecutives points that deviate from constant flux in t0+-tE")
            return my_own_model, pyLIMA_parameters, False
    else:
        print("Not a good event to fit since no Rubin band")
        return my_own_model, pyLIMA_parameters, False

def sim_fit(i, model, algo,path_TRILEGAL_set, path_to_save_model, path_to_save_fit, path_ephemerides, path_dataslice):
    pd_planets = pd.read_csv(path_TRILEGAL_set)
    event_params = pd_planets.iloc[int(i)]
    my_own_model, pyLIMA_parameters, decision = sim_event(i, event_params, path_ephemerides, path_dataslice, model)
    Source = i
    print(event_params)
    print(pyLIMA_parameters)
    if decision:
        print("Save the data and fit with ",algo)
        save(i, path_TRILEGAL_set, path_to_save_model, my_own_model, pyLIMA_parameters)
        lc_to_fit = {}
        for telo in my_own_model.event.telescopes:
            if not len(telo.lightcurve_magnitude['mag'])==0:
                df = telo.lightcurve_magnitude.to_pandas()
                lc_to_fit[telo.name] = df.values
            else:
                lc_to_fit[telo.name] = []
        origin = my_own_model.origin
        fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(Source, event_params, path_to_save_fit, path_ephemerides,model,algo,origin,
                                   lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
                                               lc_to_fit["i"], lc_to_fit["z"],lc_to_fit["y"])
        fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(Source,event_params, path_to_save_fit, path_ephemerides,model,algo,origin,
                                   lc_to_fit["W149"], [], [], [], [], [],[])


# def model_rubin_roman(Source,true_model,event_params, path_ephemerides,model, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z,
#                     lsst_y):
#     '''
#     Model Rubin and Roman data for plot
#     '''
    
#     tlsst = 60350.38482057137 + 2400000.5
#     RA, DEC = 267.92497054815516, -29.152232510353276
#     e = event.Event(ra=RA, dec=DEC)

#     if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
#         e.name = 'Event_Roman_' + str(int(Source))
#     else:
#         e.name = 'Event_RR_' + str(int(Source))
#     tel_list = []

#     # Add a PyLIMA telescope object to the event with the Gaia lightcurve
#     tel1 = telescopes.Telescope(name='Roman', camera_filter='W149',
#                                 light_curve=wfirst_lc,
#                                 light_curve_names=['time', 'mag', 'err_mag'],
#                                 light_curve_units=['JD', 'mag', 'mag'],
#                                 location='Space')

#     ephemerides = np.loadtxt(path_ephemerides)
#     ephemerides[:, 0] = ephemerides[:, 0]
#     ephemerides[:, 3] *= 60 * 300000 / 150000000
#     deltaT = tlsst - ephemerides[:, 0][0]
#     ephemerides[:, 0] = ephemerides[:, 0] + deltaT
#     tel1.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
#     e.telescopes.append(tel1)
#     tel_list.append('Roman')
    
#     lsst_lc_list = [lsst_u,lsst_g,lsst_r,lsst_i,lsst_z,lsst_y]
#     lsst_bands = "ugrizy"
#     for j in range(len(lsst_lc_list)):
#         if not len(lsst_lc_list[j])==0:
#             tel = telescopes.Telescope(name=lsst_bands[j], camera_filter=lsst_bands[j],
#                                 light_curve=lsst_lc_list[j],
#                                 light_curve_names=['time', 'mag', 'err_mag'],
#                                 light_curve_units=['JD', 'mag', 'mag'],
#                                 location='Earth')
#             e.telescopes.append(tel)
#             tel_list.append(lsst_bands[j])
#     e.check_event()
#     # Give the model initial guess values somewhere near their actual values so that the fit doesn't take all day
#         # Determine the initial guess values for the model
#     t0 = float(event_params['t0']) if 't0' in event_params else None
#     t_center = float(event_params['t_center']) if 't_center' in event_params else None
    
#     # Use t_center if available; otherwise, fall back to t0
#     t_guess = t_center if t_center is not None else t0

#     # t0 = float(event_params['t0'])
#         # t0 = float(event_params['t_center'])
#     # u0 = float(event_params['u_center'])
    
#     # rango = 0.5
#     if model == 'FSPL':
#         pyLIMAmodel = FSPLarge_model.FSPLargemodel(e, parallax=['Full', t_guess])
#     elif model=='USBL':
#         if true_model:
#             pyLIMAmodel = USBL_model.USBLmodel(e, origin=['third_caustic', [0, 0]],
#                                                blend_flux_parameter='ftotal',
#                                                parallax=['Full', t_guess])
#         else:
#             pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t_guess])
#     elif model=='PSPL':
#         pyLIMAmodel = PSPL_model.PSPLmodel(e, parallax=['Full', t_guess])
#     return pyLIMAmodel

def model_rubin_roman(Source, true_model, event_params, path_ephemerides, model, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
    
    tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)

    if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
        e.name = 'Event_Roman_' + str(int(Source))
        name_roman = 'Roman (Roman)'
    else:
        e.name = 'Event_RR_' + str(int(Source))
        name_roman = 'Roman (Roman+Rubin)'
    tel_list = []

    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name='W149', camera_filter='W149',
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
        if len(lsst_lc_list[j]) != 0:
            tel = telescopes.Telescope(name=lsst_bands[j], camera_filter=lsst_bands[j],
                                       light_curve=lsst_lc_list[j],
                                       light_curve_names=['time', 'mag', 'err_mag'],
                                       light_curve_units=['JD', 'mag', 'mag'],
                                       location='Earth')
            e.telescopes.append(tel)
            tel_list.append(lsst_bands[j])
    
    e.check_event()
    
    # Use t_center if available; otherwise, use t0
    t_guess = float(event_params['t_center']) if 't_center' in event_params else float(event_params.get('t0', None))

    # Check if model is specified and create the appropriate model instance
    if model == 'FSPL':
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e, parallax=['Full', t_guess])
    elif model == 'USBL':
        if true_model:
            pyLIMAmodel = USBL_model.USBLmodel(e, origin=['third_caustic', [0, 0]],
                                               blend_flux_parameter='ftotal',
                                               parallax=['Full', t_guess])
        else:
            pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t_guess])
    elif model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e, parallax=['Full', t_guess])

    return pyLIMAmodel
