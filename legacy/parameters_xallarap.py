import numpy as np
from astropy import units as u
from astropy import constants as C
def thetaE(Ds, Dl, Ml):
    """
    Ds (float): distance to the source in kpc
    Dl (float): distance to the lens in kpc
    Ml (float): mass of the lens in solar masses
    """
    dl = Dl*u.kpc
    ds = Ds*u.kpc
    M = Ml*u.M_sun
    k = 4*C.G/C.c**2
    pi_rel = 1/dl-1/ds
    arg = k*pi_rel*M
    thetaE = np.sqrt(arg)
    return thetaE.decompose()*u.rad
    
def generate_xiE(a_s, Ds, Dl, Ml, theta):
    """
    a_s (float): semi-major axis of the binary source system in AU
    Ds (float): distance to the source in kpc
    Dl (float): distance to the lens in kpc
    Ml (float): mass of the lens in solar masses
    theta (float): angle in radians
    """
    ds = Ds*u.kpc
    bot = thetaE(Ds, Dl, Ml)
    top = ((a_s*u.AU)/ds).decompose()*u.rad
    xiE = top/bot
    print(xiE)
    return xiE*np.cos(theta), xiE*np.sin(theta)  

def xi_mass_ratio(m1, m2): 
    """
    m1 (float): mass of primary source that is being magnified
    m2 (float): mass of companion
    """
    return ((m2*u.M_sun)/(m1*u.M_sun)).decompose()


def angular_velocity(P):
    """
    P (float): period in days
    return angular velocity in radians/day
    """
    return 2*np.pi/P

def tE(Ds, Dl, Ml, mu_rel):
    """
    mu_rel (float): relative proper motion in mas/year
    return tE in day
    """
    murel = mu_rel*u.mas/u.year
    return ((thetaE(Ds, Dl, Ml).to("mas"))/murel).to("day")

#example of use

a_s = 1
Ds = 8
Dl = 4
Ml = 0.1
mu_rel = 5
theta = np.pi/2
P = 0.4
M1_source = 2
M2_source = 1
t0 = 50
u0 = 0.05
q_flux = 0.1
xiEE, xiEN = generate_xiE(a_s, Ds, Dl, Ml, theta)
xi_q = xi_mass_ratio(2, 1)
omega = angular_velocity(P)
tE = tE(Ds, Dl, Ml, mu_rel).value
xi_inclination = 0
xi_phase = 0
ftotal=1000