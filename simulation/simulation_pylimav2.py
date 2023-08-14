# %reset -f
import os, sys
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

import rubin_sim
import rubin_sim.maf as maf
import rubin_sim.utils as rsUtils
from rubin_sim.data import get_baseline
from rubin_sim.maf.utils import m52snr

#version 2
from pyLIMA.simulations import simulator
from pyLIMA import event
from pyLIMA import telescopes

from scipy.interpolate import interp1d

from rubin_sim.phot_utils import signaltonoise
from rubin_sim.phot_utils import PhotometricParameters
from rubin_sim.phot_utils import bandpass_dict

import matplotlib.pyplot as plt
import numpy as np

def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams

photParams = set_photometric_parameters(30,1)
LSST_BandPass = bandpass_dict.BandpassDict.load_total_bandpasses_from_files()#bandpass_dict.load_total_bandpasses_from_files()
default5sigma = {'u':23.78, 'g':24.81, 'r':24.35 , 'i':23.92, 'z': 23.34 , 'y':22.45}
# for f in ()
f = 'u'
magerr, gamma = signaltonoise.calc_mag_error_m5(17., LSST_BandPass[f], default5sigma[f], photParams)

print("mag_err: {0} gamma {1}".format(magerr, gamma))


import astropy.units as u
from astropy.coordinates import SkyCoord
gc = SkyCoord(l=0.5*u.degree, b=-1.25*u.degree, frame='galactic')
gc.icrs.dec.value
Ra = gc.icrs.ra.value
Dec = gc.icrs.dec.value
print(Ra,Dec)

baseline_file = get_baseline()
name = os.path.basename(baseline_file).replace('.db','')

outDir = 'temp'
resultsDb = maf.db.ResultsDb()

print(baseline_file)
print(name)

bundleList = []
ra = [Ra]
dec = [Dec]
metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth'])
sql = ''
slicer = maf.slicers.UserPointsSlicer(ra=ra, dec=dec)
bundleList.append(maf.MetricBundle(metric, slicer, sql))

example1_bg = maf.MetricBundleGroup(
    bundleList, baseline_file, out_dir=outDir)
example1_bg.run_all()
dataSlice = bundleList[0].metric_values[0]



filters = ['u','g','r','i','z','y']

for fil in filters:
    int_array = np.empty((0,3), int)
    for i in range(len(dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)])):
        m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)][i]
        mjd = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == fil)][i]
        mags = np.random.uniform(15.,m5-0.15)
        # mags = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)]
        magerr = signaltonoise.calc_mag_error_m5(mags,  LSST_BandPass[fil],m5 , photParams)
        int_array = np.append(int_array, [[mjd,mags,magerr[0]]],axis=0)
    np.savetxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_'+fil+'.txt', int_array)

magu = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_u.txt')
magg = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_g.txt')
magr = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_r.txt')
magi = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_i.txt')
magz = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_z.txt')
magy = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/simulation/lc_timestamps_lsst/lightcurve_y.txt')

magu[:,0] = magu[:,0]+2400000.5
magg[:,0] = magg[:,0]+2400000.5
magr[:,0] = magr[:,0]+2400000.5
magi[:,0] = magi[:,0]+2400000.5
magz[:,0] = magz[:,0]+2400000.5
magy[:,0] = magy[:,0]+2400000.5


data = pd.read_csv('/home/anibal-art/ROMAN-RUBIN/output445007434654.dat', sep="\s+", decimal ='.', header = [0])

data = data.drop([len(data['u'])-1])
# data = data[data['W149']>14.8]
# data = data[data['W149']<29]
data = data[data['i']<28]
data['W149'] = data['W149']+1.2258 #transform Vega magnitudes into AB
display(data)

def mag(zp, Flux):
    return zp-2.5*np.log10(abs(Flux))

def magerr(Flux,Flux_err):
    return np.abs(2.5*Flux_err/(np.log(10)*Flux))


from pyLIMA.models import PSBL_model
from IPython.core.debugger import Pdb 
from astropy.table import Table
from pyLIMA.toolbox import time_series
import matplotlib.pyplot as plt
path='/home/anibal-art/ROMAN-RUBIN/simulation/lightcurves/binarios_pylimav2/'

tlsst = min(list(dataSlice['observationStartMJD']))+2400000.5
tstart_Roman = tlsst + 3*365 #Roman is expected to be launch in may 2027

photParams_u = set_photometric_parameters(30,1)
photParams_grizy = set_photometric_parameters(15,2)
photParams = [photParams_u,photParams_grizy]

tinicial = time.time()

for i in range(0,2000):

# i=3
    print(i)
    random.seed(i)
    np.random.seed(i)
    tE = np.random.lognormal(3.1, 1)
    t0 = np.random.uniform(tlsst+3*365-60,tstart_Roman+1740.36) # from 100 day before Roman is launched to one year later
    u0 = np.random.uniform(1e-6,1)
    q = np.random.uniform(1e-6,1)
    alpha = np.random.uniform(0,2)
    s = np.random.uniform(1e-2, 2)
    piEN = np.random.normal(0.164267,0.749409)
    piEE = np.random.normal(0.044996,0.259390)
    log_s = np.log10(s)
    log_q = np.log10(q)
    rho = np.random.uniform(0,7)
    my_own_parameters = [t0, u0, tE, log_s, log_q, alpha, piEN, piEE]
    #     my_own_parameters = [t0, u0, tE, rho, log_s, log_q, alpha, piEN, piEE]
    #     my_own_parameters ={'to': t0, 'uo': u0, 'log(tE)': tE,  'log(s)': log_s, 'log(q)': log_q, 'alpha': alpha}# + [piEN,piEE] 
    RA = 267.92497054815516 
    DEC = -29.152232510353276
    my_own_creation = event.Event(ra=RA, dec=DEC)
    my_own_creation.name = 'An event observed by Roman'


    Roman1 = simulator.simulate_a_telescope(name='W149', time_start=tstart_Roman,time_end=tstart_Roman+72,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman2 = simulator.simulate_a_telescope(name='W149',time_start =tstart_Roman+107+72,time_end=tstart_Roman+107+72*2,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman3 = simulator.simulate_a_telescope(name='W149',time_start=tstart_Roman+107+72*2+113, time_end=tstart_Roman+107+72*3+113,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman4 = simulator.simulate_a_telescope(name='W149',time_start=tstart_Roman+107+72*3+113+838.36, time_end=tstart_Roman+107+72*4+113+838.36,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman5 = simulator.simulate_a_telescope(name='W149',time_start=tstart_Roman+107+72*4+113*2+838.36,time_end=tstart_Roman+107+72*5+113*2+838.36,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman6 = simulator.simulate_a_telescope(name='W149',time_start=tstart_Roman+107+72*4+113*2+838.36+72+107, time_end=tstart_Roman+107+72*4+113*2+838.36+72+107+72,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)
    Roman_tot = simulator.simulate_a_telescope(name='W149',time_start=tstart_Roman+107+72*5+113*2+838.36+107, time_end=tstart_Roman+107+72*5+113*2+838.36+107+72,sampling=0.25, 
                                            location='Space',camera_filter='W149',uniform_sampling=True, astrometry=False)

    # Roman_tot.lightcurve_flux = np.r_[Roman1.lightcurve_flux,Roman2.lightcurve_flux,Roman3.lightcurve_flux,Roman4.lightcurve_flux,Roman5.lightcurve_flux,Roman6.lightcurve_flux]
    # display(Roman_tot.lightcurve_flux)
    new_array = np.r_[Roman1.lightcurve_flux,Roman2.lightcurve_flux,Roman3.lightcurve_flux,Roman4.lightcurve_flux,Roman5.lightcurve_flux,Roman6.lightcurve_flux]
    new_table = time_series.construct_time_series(new_array,['time','flux','err_flux'],['JD','W/m^2','W/m^2'])

    Roman_tot.lightcurve_flux = new_table

    ephemerides = np.loadtxt('/home/anibal-art/ROMAN-RUBIN/ephemeris/james_webb.txt')
    ephemerides[:,0] = ephemerides[:,0]- 2400000.5+2400000.5
    print(ephemerides[:,0])
    ephemerides[:,3] *=  60*300000/150000000
    deltaT = tlsst-ephemerides[:,0][0]
    ephemerides[:,0] = ephemerides[:,0]+deltaT

    Roman_tot.location = 'Space'
    Roman_tot.spacecraft_name = 'WFIRST_W149'
    # Roman_tot.spacecraft_positions = ephemerides
    Roman_tot.spacecraft_positions = {'astrometry':[],'photometry': ephemerides}

    my_own_creation.telescopes.append(Roman_tot)
    # Pdb().set_trace()
    #Create LSST 6 bands, 1 points per week for 10 yr
    #     print('ts_rubin', magu[:,0]+2400000.5)

    LSST_u = telescopes.Telescope(name='u', camera_filter='u', location='Earth', light_curve=magu.astype(float),
                                  light_curve_names = ['time','mag','err_mag'], light_curve_units = ['JD','mag','mag'])

    LSST_g = telescopes.Telescope(name='g', camera_filter='g', location='Earth', light_curve=magg.astype(float),
                                  light_curve_names = ['time','mag','err_mag'], light_curve_units = ['JD','mag','mag'])


    LSST_r = telescopes.Telescope(name='r', camera_filter='r', location='Earth', light_curve=magr.astype(float),
                                  light_curve_names = ['time','mag','err_mag'],light_curve_units = ['JD','mag','mag'])


    LSST_i = telescopes.Telescope(name='i', camera_filter='i', location='Earth', light_curve=magi.astype(float),
                                  light_curve_names = ['time','mag','err_mag'],light_curve_units = ['JD','mag','mag'])


    LSST_z = telescopes.Telescope(name='z', camera_filter='z', location='Earth', light_curve=magz.astype(float),
                                  light_curve_names = ['time','mag','err_mag'], light_curve_units = ['JD','mag','mag'])

    LSST_y = telescopes.Telescope(name='y', camera_filter='y', location='Earth', light_curve=magy.astype(float),
                                  light_curve_names = ['time','mag','err_mag'], light_curve_units = ['JD','mag','mag'])


    my_own_creation.telescopes.append(LSST_u)
    my_own_creation.telescopes.append(LSST_g)
    my_own_creation.telescopes.append(LSST_r)
    my_own_creation.telescopes.append(LSST_i)
    my_own_creation.telescopes.append(LSST_z)
    my_own_creation.telescopes.append(LSST_y)
    # breakpoint()

    my_own_model = PSBL_model.PSBLmodel(my_own_creation ,parallax=['Full', t0])

    pyLIMA_parameters_1 = my_own_model.compute_pyLIMA_parameters(my_own_parameters)
    #-----------what does this--------------------------------
    simulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters_1)
    # for telo in my_own_creation.telescopes:
    #     x = telo.lightcurve_magnitude['time'].value
    #     y = telo.lightcurve_magnitude['mag'].value
    #     z = telo.lightcurve_magnitude['err_mag'].value       
    #     plt.errorbar(x,y,z,marker='o',linestyle=' ',capsize=2)
    #     plt.gca().invert_yaxis()
    # plt.show()
    #-----------end--------------------------------
    my_own_flux_parameters = []
    magstar = [data['W149'].values[i],data['u'].values[i],data['g'].values[i],
               data['r'].values[i],data['i'].values[i],data['z'].values[i],data['Y'].values[i]]

    #     magstar = [22.4888, 27.94, 25.112, 23.791, 23.052, 22.687, 22.509]
    cero_p = [27.615, 27.03, 28.38, 28.16, 27.85,  27.46, 26.68]
    mag_sat = [14.8, 14.7, 15.7, 15.8, 15.8, 15.3, 13.9]

    fs = []
    for m in range(len(magstar)):
        ZP = cero_p[m] 
        mag_baseline = magstar[m]
        flux_baseline = 10**((ZP-mag_baseline)/2.5)
        g = np.random.uniform(0,1)
        f_source = flux_baseline/(1+g)
    #         flux_baseline=f_source*(1+g)
        fs.append(f_source)
        my_own_flux_parameters.append(f_source)
        my_own_flux_parameters.append(g)

    my_own_parameters += my_own_flux_parameters
    pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)


    simulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters)
    filterset = ['w146','u','g','r','i','z','y']
    msource = [mag(cero_p[j],fs[j]) for j in range(len(fs))]

    for k in range(1,len(my_own_creation.telescopes)):
        model_flux = my_own_model.compute_the_microlensing_model(my_own_creation.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
    #     print(model_flux)
        my_own_creation.telescopes[k].lightcurve_flux['flux'] = model_flux
    #     print(my_own_creation.telescopes[k].lightcurve_flux['flux'].value)
    telescope_bands = ['F146', 'u', 'g', 'r','i','z','y']
    with open(path+'Event_'+str(i)+'.txt', 'w') as w:
        w.write('TRILEGAL simulation: output445007434654 \n')
        w.write('Source:'+ str(i)+'\n')
        w.write('w:'+str(magstar[0])+'\n')
        w.write('u:'+str(magstar[1])+'\n')
        w.write('g:'+str(magstar[2])+'\n')
        w.write('r:'+str(magstar[3])+'\n')
        w.write('i:'+str(magstar[4])+'\n')
        w.write('z:'+str(magstar[5])+'\n')
        w.write('y:'+str(magstar[6])+'\n')
        w.write('Microlensing event parameters:\n')
        w.write('t0:'+str(t0)+'\n')
        w.write('u0:'+str(u0)+'\n')
        w.write('tE:'+str(tE)+'\n')
        w.write('rho:'+str(rho)+'\n')
        w.write('s:'+str(s)+'\n')
        w.write('q:'+str(q)+'\n')
        w.write('alpha:'+str(alpha)+'\n')
        w.write('piEN:'+str(piEN)+'\n')
        w.write('piEE:'+str(piEE)+'\n')
        w.write('\n')
        w.write('band,mjd,mag,magerr,m5\n')

        j=0
        for telo in my_own_creation.telescopes:
            if j==0:
    #             print(telo)
                x = telo.lightcurve_magnitude['time'].value
                y = telo.lightcurve_magnitude['mag'].value
                z = telo.lightcurve_magnitude['err_mag'].value       
                m5 = np.ones(len(x))*27.6
                w149 = np.c_[['w']*len(x),x,y-27.4+cero_p[0],z,m5]
                np.savetxt(w,w149, delimiter=', ',fmt="%s")
                plt.errorbar(x,y,z,marker='o',linestyle=' ',capsize=2,label=telescope_bands[j])

            if j==1:
                X = telo.lightcurve_flux['time'].value
    #             print(telo.lightcurve_flux)
                ym = mag(cero_p[j],telo.lightcurve_flux['flux'].value)
                z = []
                y = []
                x = []
                M5 = []
                for k in range(len(ym)): 
                    m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == filters[j-1])][k]
                    magerr = signaltonoise.calc_mag_error_m5(ym[k],  LSST_BandPass[filters[j-1]], m5 , photParams[0])[0]
                    z.append(magerr)
                    y.append(np.random.normal(ym[k],magerr))
                    x.append(X[k])
                    M5.append(        m5)
                u = np.c_[['u']*len(x),x,y,z,M5]
                np.savetxt(w,u, delimiter=', ',fmt="%s")
                plt.errorbar(x,y,z,marker='o',linestyle=' ',capsize=2,alpha=0.6,label=telescope_bands[j])
            if j>=2:
                filtersets = ['w146','u','g','r','i','z','y']
                X =  telo.lightcurve_flux['time'].value
                ym = mag(cero_p[j],telo.lightcurve_flux['flux'].value)
                z = []
                y = []
                x = []
                M5 = []
                for k in range(len(ym)):
                    m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == filters[j-1])][k]
                    magerr = signaltonoise.calc_mag_error_m5(ym[k],  LSST_BandPass[filters[j-1]], m5 , photParams[1])[0]
                    z.append(magerr)
                    y.append(np.random.normal(ym[k],magerr))
                    x.append(X[k])
                    M5.append(m5)
                array = np.c_[[filterset[j]]*len(x),x,y,z,M5]
                np.savetxt(w,array, delimiter=', ',fmt="%s")
                plt.errorbar(x,y,z,marker='o',linestyle=' ',capsize=2,alpha=0.6,label=telescope_bands[j])
            j += 1
        plt.legend(loc='best')
        plt.axvspan(t0-2*tE,t0+2*tE,alpha=0.2,color='crimson')
        plt.gca().invert_yaxis()
        plt.show()
    print((time.time()-tinicial)/60,' minutes')

