import os
 
from astropy import units as u
from astropy.time import Time
from matplotlib import pyplot as plt
import numpy as np
 
import synphot as syn
import stsynphot as stsyn
import stpsf

# Set up the Roman WFI object and retrieve the throughput of a filter
roman = stpsf.WFI()
wfi_f129 = roman._get_synphot_bandpass('F129')
print(wfi_f129(1.29 * u.micron))


band = stsyn.band('roman, wfi, f129')
print('WFI F129:')
print(f'\tBandwidth: {band.photbw():.5f}')
print(f'\tPivot wavelength: {band.pivot():.5f}')
print(f'\tFWHM: {band.fwhm():.5f}')
print(f'\tThroughput at 1.29 um: {band(1.29 * u.micron):.4f}')

# Set up the Roman WFI object and make a list of the optical element names
roman = stpsf.WFI()
roman_filter = ["F146"]
  
# Set up wavelengths from 0.4 to 2.5 microns in increments of 0.01 microns.
waves = np.arange(0.4, 2.5, 0.01) * u.micron
  
# Set up figure
fig, ax = plt.subplots(dpi=200)
  
# For each optical element, plot the throughput in a different color
# and shade the area below the curve.
colors = plt.cm.rainbow(np.linspace(0.2, 1, len(roman_filter)))
 
for i, f in enumerate(roman_filter):
    band = roman._get_synphot_bandpass(f)
    clean = np.where(band(waves) > 0)
 
    ax.plot(waves[clean]*1000, band(waves[clean]),
            color=colors[i],label=roman_filter[i])
     
    ax.fill_between(waves[clean].value*1000,
                    band(waves[clean]).value,
                    alpha=0.2,
                    color=colors[i])
  
# Set plot axis labels, ranges, and add grid lines
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel('Throughput')
 
ax.set_ylim(0, 1.05)
# ax.set_xlim(0.4, 2.5)
 
ax.grid(':', alpha=0.3)


#/////////////////////////////////
from rubin_sim.phot_utils.bandpass import Bandpass
filterdir ='/home/anibal/rubin_sim_data/throughputs/baseline/'
lsst = {}
filterlist = ('u', 'g', 'r', 'i', 'z', 'y')
filtercolors = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
for f in filterlist:
    lsst[f] = Bandpass()
    lsst[f].read_throughput(os.path.join(filterdir, 'total_'+f+'.dat'))
atmos = Bandpass()
atmos.read_throughput(os.path.join(filterdir, 'atmos_std.dat'))

# plt.figure(figsize=(8,7))
for f in filterlist:
    ax.plot(lsst[f].wavelen, lsst[f].sb, color=filtercolors[f], lw=2, label='LSST %s' % (f))
# plt.plot(atmos.wavelen, atmos.sb, 'k:', label='X=1.2 std atmosphere')
ax.set_xlabel('Wavelenght [nm]')
ax.set_ylabel('Throughput')
# plt.title('LSST Throughput Curves')
# plt.xlim(300, 1100)
# plt.legend(loc=(0.87, 0.5),ncols=4, fancybox=True, fontsize='smaller')
ax.legend(shadow=True, fontsize='large',
                      bbox_to_anchor=(0, 1.02, 1, 0.2),
                      loc="lower left",
                      mode="expand", borderaxespad=0, ncol=4)
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()