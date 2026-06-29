import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

def mag(zp, Flux):
    '''
    Transform the flux to magnitude
    inputs
    zp: zero point
    Flux: vector that contains the lightcurve flux
    '''
    return zp - 2.5 * np.log10(abs(Flux))



def debug_nsigma_global(pyLIMA_parameters, pyLIMA_telescopes,
                        nsigma=3.0, nmin=6, window="all"):

    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}

    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']

    rows = []  # (t, band, mag, err, mag_base, sig)

    for telo in pyLIMA_telescopes:
        if len(telo.lightcurve['mag']) == 0:
            continue

        band = telo.name
        key = f'ftotal_{band}'
        if key not in pyLIMA_parameters:
            continue

        x = np.asarray(telo.lightcurve['time'].value)
        y = np.asarray(telo.lightcurve['mag'].value)
        z = np.asarray(telo.lightcurve['err_mag'].value)

        # --- NUEVO BLOQUE: saneamiento mínimo ---
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
        x, y, z = x[good], y[good], z[good]
        if x.size == 0:
            continue
        # ----------------------------------------

        if window == "t0pm_tE":
            mask = (t0 - tE < x) & (x < t0 + tE)
        elif window == "t0pm_2tE":
            mask = (t0 - 2*tE < x) & (x < t0 + 2*tE)
        else:
            mask = np.ones_like(x, dtype=bool)

        if np.count_nonzero(mask) == 0:
            continue

        mag_base = ZP[band] - 2.5*np.log10(pyLIMA_parameters[key])

        xm, ym, zm = x[mask], y[mask], z[mask]
        sig = (mag_base - ym) / zm  # >0 si está más brillante que baseline

        for ti, mi, ei, si in zip(xm, ym, zm, sig):
            rows.append((ti, band, mi, ei, mag_base, si))

    if len(rows) == 0:
        return {"passes": False, "max_run": 0, "run_rows": []}

    # ordenar por tiempo global
    rows = sorted(rows, key=lambda r: r[0])
    cond = np.array([r[5] >= nsigma for r in rows], dtype=bool)

    # racha máxima y guardar inicio/fin
    max_run = 0
    best_i = None
    run = 0
    start = 0

    for i, v in enumerate(cond):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > max_run:
                max_run = run
                best_i = (start, i)
        else:
            run = 0

    run_rows = []
    if best_i is not None:
        i0, i1 = best_i
        run_rows = rows[i0:i1+1]

    return {
        "passes": (max_run >= nmin),
        "max_run": int(max_run),
        "run_rows": run_rows  # lista con tuplas (t, band, mag, err, mag_base, sig)
    }



def deviation_from_constant(pyLIMA_parameters, pyLIMA_telescopes,
                          nsigma=3.0, nmin=6, window="all"):

    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}

    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']

    all_t = []
    all_cond = []

    for telo in pyLIMA_telescopes:
        if len(telo.lightcurve['mag']) == 0:
            continue

        band = telo.name
        key = f'ftotal_{band}'
        if key not in pyLIMA_parameters:
            continue

        mag_baseline = ZP[band] - 2.5*np.log10(pyLIMA_parameters[key])

        x = telo.lightcurve['time'].value
        y = telo.lightcurve['mag'].value
        z = telo.lightcurve['err_mag'].value

        if window == "t0pm_tE":
            mask = (t0 - tE < x) & (x < t0 + tE)
        elif window == "t0pm_2tE":
            mask = (t0 - 2*tE < x) & (x < t0 + 2*tE)
        else:
            mask = np.ones_like(x, dtype=bool)

        cond = y[mask] < (mag_baseline - nsigma*z[mask])

        all_t.append(x[mask])
        all_cond.append(cond)

    if len(all_t) == 0:
        return False, 0

    t = np.concatenate(all_t)
    cond = np.concatenate(all_cond)

    idx = np.argsort(t)
    cond_sorted = cond[idx]

    # racha máxima de True consecutivos
    max_run = run = 0
    for v in cond_sorted:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    return (max_run >= nmin), max_run



def filter5points(pyLIMA_parameters, pyLIMA_telescopes):
    '''
    Check that at least one light curve
    have at least 5 pts in the t0+-tE
    '''
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    crit5pts = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve['mag']) == 0:
            x = telo.lightcurve['time'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            if len(x[mask]) >= 5: #cambiar a 10, 15
                crit5pts[telo.name] = True
            else:
                crit5pts[telo.name] = False
    return any(crit5pts.values())

def FilterNpoints(n, pyLIMA_parameters, pyLIMA_telescopes):
    '''
    Check that at least one light curve
    have at least N pts in the t0+-tE
    '''
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    critNpts = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve['mag']) == 0:
            x = telo.lightcurve['time'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            if len(x[mask]) >= n: #cambiar a 10, 15
                critNpts[telo.name] = True
            else:
                critNpts[telo.name] = False
    return any(critNpts.values())

def filter_band(lightcurve, m5, fil):
    '''
    * Save the points of the lightcurve greater and smaller than
      1sigma fainter and brighter that the saturation and 5sigma_depth
    * check that the lightcurve have more than 10 points
    * check if the lightcurve have at least 1 point at 5 sigma from the 5sigma_depth
    '''
    # print(len(lightcurve))
    mag_sat = {'W149': 14.8, 'u': 14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    lightcurve['m5'] =  m5

    b1 = lightcurve['mag'].value - lightcurve['err_mag'].value > mag_sat[fil]
    b2 = lightcurve['mag'].value + lightcurve['err_mag'].value < lightcurve['m5']
    lc_fil1 = lightcurve[b1&b2]
    # display(lc_fil1)
    return lc_fil1

def has_consecutive_numbers(lst):
    """
    check if there at least 3 consecutive numbers in a list lst
    """
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 2):
        if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
            return True
    return False
