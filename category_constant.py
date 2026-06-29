import sys
import os
import numpy as np
import pandas as pd
import rubin_sim
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline
from astropy.table import QTable
from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_mag_error_m5
from rubin_sim.phot_utils.bandpass import Bandpass
import matplotlib.pyplot as plt
import h5py
sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import sim_fit,sim_event,filter_band
from functions_roman_rubin import read_data, save_sim


script_dir = os.path.dirname(os.path.abspath(__file__))

def save_sim_const(iloc, path_TRILEGAL_set, path_to_save, lightcurves, event_params):
    print('Saving Simulation...')
    with h5py.File(path_to_save + 'Event_' + str(iloc) + '.h5', 'w') as file:
        file.create_dataset('Data', data=np.array([iloc, path_TRILEGAL_set, "Constant type"], dtype='S'))

        dict_group_tril = file.create_group('TRILEGAL_params')
        for key, value in event_params.items():
            dict_group_tril.attrs[key] = value

        for telo, table in lightcurves.items():
            table_group = file.create_group(telo)
            for col in table.colnames:
                table_group.create_dataset(col, data=table[col])

    print('File saved:', path_to_save + 'Event_' + str(iloc) + '.h5')


# Obtener el path al archivo .db del baseline simulado
baseline_file = get_baseline()
conn = baseline_file  # usar directamente el path como string
outDir = 'temp'
resultsDb = maf.db.ResultsDb()  # <- CORREGIDO
# Coordenadas
Ra, Dec = 270, -30
ra = [Ra]
dec = [Dec]
# Métrica
metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth'])
# Slicer
slicer = maf.slicers.UserPointsSlicer(ra=ra, dec=dec)
# SQL vacío
sql = ''
# Crear MetricBundle
metric_bundle = maf.MetricBundle(metric, slicer, sql)
bundleDict = {'my_bundle': metric_bundle}
# Ejecutar
bg = maf.MetricBundleGroup(bundleDict, conn, out_dir=outDir, results_db=resultsDb)
bg.run_all()
dataSlice = metric_bundle.metric_values[0]
LSST_BandPass = {}
lsst_filterlist = 'ugrizy'
for f in lsst_filterlist:
    LSST_BandPass[f] = Bandpass()
    LSST_BandPass[f].read_throughput('/home/anibalvarela/rubin_sim_data/throughputs/baseline/' + f'total_{f}.dat')

def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams
photParams = set_photometric_parameters(15, 2)

for i in range(10000):
    ROW = i-10000*int(i/10000)
# <<<<<<< HEAD
    print()
    path_TRILEGAL_set = os.path.join(script_dir,'/chunks_TRILEGAL_GENULENS/uniform.csv')
# =======
#     path_TRILEGAL_set = os.path.join(script_dir,'chunks_TRILEGAL_GENULENS/TRILEGAL_chunk_1.csv')
# >>>>>>> f45920f6be0c0cad4ee5d4e01020942704c5cc99
    TRILEGAL_row = pd.read_csv(path_TRILEGAL_set, skiprows=ROW+1, nrows=1)
    TRILEGAL_row.columns = pd.read_csv(path_TRILEGAL_set, nrows=0).columns
    TRILEGAL_row['y']=TRILEGAL_row['Y']
    constant_lc = {}
    filters = 'ugrizy'
    for f in filters:
        mag_model = TRILEGAL_row[f].iloc[0]
        time = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == f)]
        m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == f)]
        mag = []
        mag_err = []
        
        for day in range(len(time)):
            magerr = calc_mag_error_m5(mag_model, LSST_BandPass[f], m5[day], photParams)[0]
            mag_err.append(magerr)
            mag.append(np.random.normal(mag_model, magerr))
        
        data = QTable([np.array(mag_err),m5,np.array(mag), time],
                      names=('err_mag' , 'm5', 'mag', 'time'))
    
        constant_lc[f] = filter_band(data, m5, f)

    path_save = '/share/storage3/rubin/microlensing/romanrubin/RR2025/Baseline4_set/constant/'
    save_sim_const(i, path_TRILEGAL_set, path_save, constant_lc, TRILEGAL_row)
