#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:39:16 2024

@author: anibal
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy import units as u
# from astropy import constants as c
from astropy.constants import L_sun, sigma_sb

data = pd.read_csv("/home/anibal/roman_rubin/TRILEGAL_KOSHIMOTO/output445007434654.dat", sep="\s+", decimal ='.', header = [0])
data = data.drop([len(data['u'])-1])
# data = data[data['W149']>14.8]
# data = data[data['W149']<29]
data = data[data['i']<28]
data['W149'] = data['W149']+1.2258 #transform Vega magnitudes into AB

Radii_star = []
for i in tqdm(range(len(data))):
    logL = data['logL'][i]  # log10 of the luminosity in Lsun from TRILEGAL
    logTe = data['logTe'][i]  # log10 of effective temperature in K from TRILEGAL
    L_star = 10**(logL)
    Teff = (10**(logTe))*u.K
    top = L_star*L_sun
    sigma = sigma_sb
    bot = 4*np.pi*sigma*Teff**4
    Rstar = np.sqrt(top/bot)
    Radii_star.append(Rstar.to('R_sun').value)

data['radius']=Radii_star
df_koshimoto_large = pd.read_csv('/home/anibal/TRILEGAL_KOSHIMOTO/tmp_new.dat', delim_whitespace=True, comment='#', header=None)
header = ["wtj", "M_L", "D_L", "D_S", "tE", "thetaE", "piE", "piEN", "piEE", "mu_rel", "muSl", "muSb", "i_L", "iS",
          "iL", "fREM"]
df_koshimoto_large.columns = header
data["D_S"]=df_koshimoto_large["D_S"]
data["D_L"]=df_koshimoto_large["D_L"]
data["mu_rel"]=df_koshimoto_large["mu_rel"]

data = data[['u','g','r','i','z','Y','W149','radius','D_S','D_L','mu_rel']]
data = data[data['D_S']<8000]
data.to_csv('/home/anibal/roman_rubin/TRILEGAL_KOSHIMOTO/df_trilegal_radios.csv', index=False)