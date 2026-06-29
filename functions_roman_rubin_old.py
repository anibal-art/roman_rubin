import numpy as np
import os, sys, re, copy, math
import pandas as pd
from pathlib import Path
import time
# Get the directory where the script is located
script_dir = Path(__file__).parent
# print(script_dir)
home_dir = os.path.expanduser("~")
# sys.path.append(str(script_dir)+'/photutils/')
# from bandpass import Bandpass
# from signaltonoise import calc_mag_error_m5
# from photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_mag_error_m5
from rubin_sim.phot_utils.bandpass import Bandpass
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline
#astropy
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'
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
from class_analysis import Analysis_Event
from ulens_params import microlensing_params, event_param
import multiprocessing as mul
import h5py
from fit_lc import fit_rubin_roman
from detection_criteria import filter5points, deviation_from_constant, has_consecutive_numbers, filter_band, mag
from read_save import save_sim, save_fit, read_data




def tel_roman_rubin(path_ephemerides, path_dataslice):
    '''
    :param opsim:
    :return:
    '''
    # print(str(script_dir))
    gc = SkyCoord(l=0.5 * u.degree, b=-1.25 * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    LSST_BandPass = {}
    lsst_filterlist = 'ugrizy'
    for f in lsst_filterlist:
        LSST_BandPass[f] = Bandpass()
        path_che = '/home/anibal/rubin_sim_data/throughputs/baseline/'
        #path_che = '/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data/throughputs/baseline/'
        # LSST_BandPass[f].read_throughput(str(script_dir)+'/troughputs/' + f'total_{f}.dat')
        LSST_BandPass[f].read_throughput(path_che + f'total_{f}.dat')
    baseline_file = get_baseline()
    conn = baseline_file
    outDir = 'temp'
    resultsDb = maf.db.ResultsDb()
    metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth'])
    slicer = maf.slicers.UserPointsSlicer(ra=[Ra], dec=[Dec])
    sql = ''
    metric_bundle = maf.MetricBundle(metric, slicer, sql)
    bundleDict = {'my_bundle': metric_bundle}
    bg = maf.MetricBundleGroup(bundleDict, conn, out_dir=outDir, results_db=resultsDb)
    bg.run_all()
    dataSlice = metric_bundle.metric_values[0]
    # dataSlice = np.load(path_dataslice, allow_pickle=True)
    
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
    
    off_seasons = [
        {"start": "2028-08-15T00:00:00", "end": "2028-10-27T00:00:00"},
        {"start": "2029-02-11T00:00:00", "end": "2029-04-24T00:00:00"},
        {"start": "2029-08-16T00:00:00", "end": "2029-10-27T00:00:00"}
    ]
    
    
    Roman_tot = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107,
                                               time_end=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107 + 72,
                                               sampling=121/600,
                                               location='Space', camera_filter='W149', uniform_sampling=True,
                                               astrometry=False)
    lightcurve_fluxes = []
    for season in nominal_seasons:
        tstart = Time(season['start'], format='isot').jd
        tend = Time(season['end'], format='isot').jd
        Roman = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart,
                                               time_end=tend,
                                               sampling=121/600,
                                               location='Space',
                                               camera_filter='W149',
                                               uniform_sampling=True,
                                               astrometry=False)
        lightcurve_fluxes.append(Roman.lightcurve)
    
    for season in off_seasons:
        tstart = Time(season['start'], format='isot').jd
        tend = Time(season['end'], format='isot').jd
        Roman = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart,
                                               time_end=tend,
                                               sampling=24*3,
                                               location='Space',
                                               camera_filter='W149',
                                               uniform_sampling=True,
                                               astrometry=False)
        lightcurve_fluxes.append(Roman.lightcurve)
    # Combine all the lightcurve_flux tables into one array
    combined_array = np.concatenate([lc.as_array() for lc in lightcurve_fluxes])
    # print(combined_array)
    # Convert the combined array back into a QTable
    new_table = QTable(combined_array,names=['time','mag','err_mag', 'flux', 'err_flux','inv_err_flux'], 
                       units=['JD', 'mag','mag','W/m^2', 'W/m^2','m^2/W'])
    # display(new_table)
    Roman_tot.lightcurve = new_table
    ephemerides = np.load(path_ephemerides)
    Roman_tot.location = 'Space'
    Roman_tot.spacecraft_name = 'L2'
    Roman_tot.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    my_own_creation.telescopes.append(Roman_tot)
    for band in lsst_filterlist:
        lsst_telescope = telescopes.Telescope(name=band, camera_filter=band, location='Earth',
                                              lightcurve=rubin_ts[band],
                                              lightcurve_names=['time', 'mag', 'err_mag'],
                                              lightcurve_units=['JD', 'mag', 'mag'])
        my_own_creation.telescopes.append(lsst_telescope)
        # display(lsst_telescope.lightcurve)

    return my_own_creation, dataSlice, LSST_BandPass

def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams

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
    print(magstar)
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    my_own_creation, dataSlice, LSST_BandPass = tel_roman_rubin(path_ephemerides,
                                                                path_dataslice)
    photParams = set_photometric_parameters(15, 2)
    new_creation = copy.deepcopy(my_own_creation)
    np.random.seed(i)
    t0 = data['t0']
    tE = data['tE']

    if model == 'USBL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'], 'rho': data['rho'],
                  's': data['s'], 'q': data['q'], 'alpha': data['alpha'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
        my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0])
    elif model == 'USBL_NoPiE':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'], 'rho': data['rho'],
          's': data['s'], 'q': data['q'], 'alpha': data['alpha']}
        
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
                                    blend_flux_parameter='ftotal')

    elif model == 'FSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                  'rho': data['rho'], 'piEN': data['piEN'],
                  'piEE': data['piEE']}
        my_own_model = FSPLarge_model.FSPLargemodel(new_creation, parallax=['Full', t0])
    elif model == 'PSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        my_own_model = PSPL_model.PSPLmodel(new_creation, parallax=['Full', t0])
    elif model == 'PSPL_NoPiE':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        my_own_model = PSPL_model.PSPLmodel(new_creation)

    my_own_parameters = []
    for key in params:
        my_own_parameters.append(params[key])

    my_own_flux_parameters = []
    fs, G, F = {}, {}, {}
    # np.random.seed(i)
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
    simulator.simulate_lightcurve(my_own_model, pyLIMA_parameters)

    for k in range(1, len(new_creation.telescopes)):
        model_flux = my_own_model.compute_the_microlensing_model(new_creation.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
        new_creation.telescopes[k].lightcurve['flux'] = model_flux

    Roman_band = False
    Rubin_band = False
    for telo in new_creation.telescopes:
        if telo.name == 'W149':
            # display(telo.lightcurve)
            telo.lightcurve['mag'] = (telo.lightcurve['mag'].value- 27.4 + ZP[telo.name])*u.mag
            m5 = np.ones(len(telo.lightcurve['mag'])) * 27.6
            telo.lightcurve = filter_band(telo.lightcurve, m5, telo.name)
            if not len(telo.lightcurve['mag']) == 0:
                Roman_band = True
        else:
            # display(telo.lightcurve)
            X = telo.lightcurve['time'].value
            ym = mag(ZP[telo.name], telo.lightcurve['flux'].value)
            z, y, x, M5 = [], [], [], []
            for k in range(len(ym)):
                m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == telo.name)][k]
                magerr = calc_mag_error_m5(ym[k], LSST_BandPass[telo.name], m5, photParams)[0]
                z.append(magerr)
                y.append(np.random.normal(ym[k], magerr))
                x.append(X[k])
                M5.append(m5)
            data = QTable([telo.lightcurve['err_flux'].value, np.array(z), telo.lightcurve['flux'].value,
                           telo.lightcurve['inv_err_flux'].value, np.array(M5), np.array(y), np.array(x)],
                          names=('err_flux', 'err_mag', 'flux', 'inv_err_flux', 'm5', 'mag', 'time'))
            
            telo.lightcurve = filter_band(data, m5, telo.name)
            if not len(telo.lightcurve['mag']) == 0:
                Rubin_band = True
    # This first if holds for an event with at least one Roman and Rubin band
    if Rubin_band and Roman_band:
        # This second if holds for a "detectable" event to fit
        if deviation_from_constant(pyLIMA_parameters, new_creation.telescopes):
            print("A good event to fit")
            return my_own_model, pyLIMA_parameters, True
        else:
            print(
                "Not a good event to fit.\nFail 5 points in t0+-tE\nNot have 3 consecutives points that deviate from constant flux in t0+-tE")
            return my_own_model, pyLIMA_parameters, False
    else:
        print("Not a good event to fit since no Rubin band")
        return my_own_model, pyLIMA_parameters, False


def new_data(Event, nset, nevent, cols_fit, data):
    new_row = dict.fromkeys(cols_fit)
    new_row['Source'] = [nevent]
    new_row['Set'] = [nset]
    fit_vals = Event.dict_fit_vals(data)
    for key in fit_vals:
        new_row[key] = [fit_vals[key]]
    piemc = Event.piE_MC(data)
    new_row['piE'] = [piemc['piE']]
    new_row['piE_err'] = [piemc['err_piE']]

    chichidof = Event.chichi_dof(data)
    new_row['chichi'] = [chichidof['chi2']]
    new_row['dof'] = [chichidof['dof']]

    if not Event.model == 'PSPL':
        fitmassv3 = Event.fit_mass_v3(data)
        new_row['err_mass_v3'] = [fitmassv3['err_mass']]
        new_row['mass_v3'] = [fitmassv3['mass']]
    fitmassv2 = Event.fit_mass_v2(data)
    fitmassv1 = Event.fit_mass_v1(data)
    new_row['err_mass_v2'] = [fitmassv2['err_mass']]
    new_row['mass_v2'] = [fitmassv2['mass']]
    new_row['err_mass_v1'] = [fitmassv1['err_mass']]
    new_row['mass_v1'] = [fitmassv1['mass']]
    new_row['ln_likelihood'] = [data['ln_likelihood']]

    return new_row

def extract_data_event(Event, model, nevent, system_type,nset):
    # script_dir = paths["script_dir"]
    # path_run_model = paths["path_run_model"]
    # path_model = paths["model"](path_run_model, nset, nevent)
    Event.load_data_fit()
    Event.load_data_sim()
    labels_params = Event.labels_params()
    if 'dfit' not in system_type:
        if model == 'USBL':
            cols_true = ['Source', 'Set'] + labels_params + ['Category', 'Category_p', 'mass', 'sel_crit', 'piE']
        elif model == 'FSPL':
            cols_true = ['Source', 'Set'] + labels_params + ['Category', 'mass', 'sel_crit', 'piE', 'crit_FFP_Rubin']
        else:
            cols_true = ['Source', 'Set'] + labels_params + ['Category', 'mass', 'sel_crit', 'piE', 'W149', 'u', 'g', 'r', 'i', 'z', 'y']

    if "Planets_systems" in system_type:
        st = "Planets_systems"
    elif "FFP" in system_type:
        st = "FFP"
    elif "BH" in system_type:
        st = "BH"

    # path_run_fit = paths["path_run_fit"]
    # path_fit_rr = paths["fit_rr"](path_run_fit, nset, nevent)
    # path_fit_roman = paths["fit_roman"](path_run_fit, nset, nevent)

    # Event = Analysis_Event(model, path_model, path_fit_rr, path_fit_roman)


    # df true
    if 'dfit' not in system_type:
        true = Event.true_values()
        fit_rr, fit_roman = Event.fit_values()

        new_data_true = dict.fromkeys(cols_true)
        new_data_true['Source'] = [nevent]
        new_data_true['Set'] = [nset]
        for key in Event.labels_params():
            new_data_true[key] = [true[key]]
        new_data_true['piE'] = [np.sqrt(true['piEN']**2 + true['piEE']**2)]

        if model == 'USBL':
            npts_speak = Event.count_points_secon_peak()
            for f in npts_speak:
                new_data_true['anomaly_' + f] = [npts_speak[f]]

        new_data_true['mass'] = [Event.mass_true()]

        npts = Event.count_points()
        for f in npts:
            new_data_true[f] = [npts[f]]

        npts_pk = Event.count_points_peak()
        for f in npts_pk:
            new_data_true[f + '_peak'] = [npts_pk[f]]

        npts_pk_narrow = Event.count_points_narrow_peak()
        for f in npts_pk:
            new_data_true[f + '_npeak'] = [npts_pk_narrow[f]]

        npts_pk_left = Event.count_points_left_peak()
        for f in npts_pk:
            new_data_true[f + '_lpeak'] = [npts_pk_left[f]]

        npts_pk_right = Event.count_points_right_peak()
        for f in npts_pk:
            new_data_true[f + '_rpeak'] = [npts_pk_right[f]]

    else:
        new_data_true = []

    cols_fit = ['Source', 'Set'] + labels_params + \
               [f + '_err' for f in labels_params] + \
               ['piE', 'piE_err'] + ['chichi', 'dof'] + \
               ['mass_v1', 'mass_v2', 'err_mass_v1', 'err_mass_v2']
    if not model == 'PSPL':
        cols_fit = cols_fit + ['mass_v3', 'err_mass_v3']

    new_data_rr = new_data(Event, nset, nevent, cols_fit, Event.fit_rr_data)
    new_data_roman = new_data(Event, nset, nevent, cols_fit, Event.fit_roman_data)

    return new_data_true, new_data_rr, new_data_roman

def sim_fit(i, system_type, model, algo ,path_TRILEGAL_set,path_GENULENS_set, 
            path_to_save_model, path_to_save_fit, path_ephemerides, path_dataslice):
    tstart = time.time()
    
    seed = i
    np.random.seed(seed)
    ROW_G = np.random.randint(0,10000)   #ROW = i-10000*int(i/10000)
    ROW_T = np.random.randint(0,10000)
    
    TRILEGAL_row = pd.read_csv(path_TRILEGAL_set,
                           skiprows=lambda x: x not in (0, ROW_T + 1))
    GENULENS_row = pd.read_csv(path_GENULENS_set,
                           skiprows=lambda x: x not in (0, ROW_G + 1))

    # print(GENULENS_row)
    magstar = TRILEGAL_row[["W149","u","g","r","i","z","Y"]].iloc[0]
    event_params = {**magstar.to_dict(), 
                    **event_param(i, TRILEGAL_row.iloc[0], GENULENS_row.iloc[0], system_type)}
    
    my_own_model, pyLIMA_parameters, decision = sim_event(i,event_params, path_ephemerides, path_dataslice, model)
    
    Source = seed

    if decision:
        print("Criteria satisfied. Save the simulated light-curve.")

        save_sim(i,ROW_G, ROW_T, path_TRILEGAL_set,path_GENULENS_set, path_to_save_model, 
                 my_own_model, pyLIMA_parameters, event_params,GENULENS_row,TRILEGAL_row)
        lc_to_fit = {}
        lc_to_save = {}
        for telo in my_own_model.event.telescopes:
            if not len(telo.lightcurve['mag'])==0:
                tbl  =telo.lightcurve[['time', 'mag','err_mag']]
                tbl['time']=tbl['time'].value
                df = tbl.to_pandas()
                lc_to_fit[telo.name] = df.values
                lc_to_save[telo.name] = tbl
            else:
                lc_to_fit[telo.name] = []
        origin = my_own_model.origin
        rango=1
        print("Start the fit using Roman and Rubin data:")
        fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(Source, pyLIMA_parameters, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
                                   lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
                                               lc_to_fit["i"], lc_to_fit["z"],lc_to_fit["y"])
        print("Start the fit using only the Roman data:")
        fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(Source, pyLIMA_parameters, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
                                   lc_to_fit["W149"], [], [], [], [], [],[])

        tend = time.time()
        if True:
            Event = Analysis_Event(model, path_model=path_to_save_model+f"Event_{i}.h5", 
                                   path_fit_rr=path_to_save_fit+f"Event_{algo}_{i}_RR.npy", 
                                   path_fit_roman=path_to_save_fit+f"Event_{algo}_{i}_Roman.npy",
                                   genulens_params=GENULENS_row.iloc[0], trilegal_params=TRILEGAL_row.iloc[0],
                                   computed_params=event_params, fit_rr_data=fit_rr.fit_results, fit_roman_data = fit_roman.fit_results, 
                                   origin=origin, lightcurves=lc_to_save,
                                   model_params=pyLIMA_parameters, info= None, indices= None)
            
            new_data_true, new_data_rr, new_data_roman = extract_data_event(Event, model, i, system_type, re.findall(r'\d+', path_GENULENS_set)[0])
            print(new_data_true, new_data_rr, new_data_roman)

        print("El tiempo transcurrido fue de : ", tend-tstart)
        return fit_rr, pyLIMAmodel_rr, fit_roman, pyLIMAmodel_roman




def read_fit(nsource, nset, path_run, model, algo,
             path_to_save_fit, path_ephemerides):

    path_event =path_run + f'/set_sim{nset}/Event_{nsource}.h5'

    print('path_event', path_event)
    print('os.path.getsize(path_model):' , os.path.getsize(path_event))
    if os.path.getsize(path_event) == 0:
        return 
    else:
            
        info_event, pyLIMA_parameters, curves = read_data(path_event)
    
        lc_to_fit = {}
        for telo in curves:
            if not len(curves[telo]['mag'])==0:
                df = curves[telo][['time', 'mag','err_mag']].to_pandas()
                lc_to_fit[telo] = df.values
            else:
                lc_to_fit[telo] = []
    
        origin = info_event[2]
        rango = 1
        
        print("Start the fit using Roman and Rubin data:")
        fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
                                   lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
                                               lc_to_fit["i"], lc_to_fit["z"],lc_to_fit["y"])
        print("Start the fit using only the Roman data:")
        fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
                                   lc_to_fit["W149"], [], [], [], [], [],[])
    
