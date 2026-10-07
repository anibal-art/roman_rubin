import astropy.units as u
from astropy import constants as const
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.constants import c, L_sun, sigma_sb, M_jup, M_earth, G
import numpy as np
import pandas as pd

# Constants
c = const.c
G = const.G
k = 4 * G / (c ** 2)
tstart_Roman = 2461508.763828608
t0 = tstart_Roman + 20





def get_sampled_or_default(name, param_samplers, rng, context, default_sampler):
    """
    Si name está en param_samplers, usa ese sampler.
    Si no, usa default_sampler().
    """
    if param_samplers is not None and name in param_samplers:
        return sample_from_spec(
            name,
            param_samplers[name],
            rng,
            context=context,
        )

    return default_sampler()


def sample_from_spec(name, spec, rng, context=None):
    """
    Samplea un parámetro desde una especificación flexible.

    spec puede ser:
    - número: valor fijo
    - tuple/list de largo 2: uniforme entre esos límites
    - dict con type:
        {"type": "uniform", "low": a, "high": b}
        {"type": "loguniform", "low": a, "high": b}
        {"type": "normal", "loc": mu, "scale": sigma}
        {"type": "choice", "values": [...]}
        {"type": "rho_uniform", "low": a, "high": b}
        {"type": "fixed", "value": x}
    - callable: función que recibe rng y context
    """

    if context is None:
        context = {}

    # Valor fijo numérico
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return spec

    # Uniforme simple: (low, high)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.uniform(spec[0], spec[1])

    # Callable custom
    if callable(spec):
        try:
            return spec(rng=rng, context=context)
        except TypeError:
            try:
                return spec(rng, context)
            except TypeError:
                return spec()

    # Diccionario
    if isinstance(spec, dict):
        kind = spec.get("type", "uniform")

        if kind == "fixed":
            return spec["value"]

        if kind == "uniform":
            return rng.uniform(spec["low"], spec["high"])

        if kind == "loguniform":
            low = spec["low"]
            high = spec["high"]
            return 10 ** rng.uniform(np.log10(low), np.log10(high))

        if kind == "normal":
            return rng.normal(spec["loc"], spec["scale"])

        if kind == "choice":
            return rng.choice(spec["values"])

        if kind == "rho_uniform":
            rho = context["rho"]
            return rho * rng.uniform(spec["low"], spec["high"])

        if kind == "abs_rho_uniform":
            rho = context["rho"]
            return abs(rho) * rng.uniform(spec["low"], spec["high"])

        raise ValueError(f"Sampler type no reconocido para {name}: {kind}")

    raise TypeError(f"Sampler inválido para {name}: {spec}")
    
    

# ============================================================
# Default mass ranges for controlled characterization studies
# ============================================================

# Planetary-mass range:
#   0.01 M_earth <= M <= 13 M_jup
#
# mass_planet is represented internally in M_jup.
PLANET_MASS_MIN_MJUP = 0.01 / 317.828
PLANET_MASS_MAX_MJUP = 13.0

# Stellar-mass range:
#   1 M_sun <= M <= 100 M_sun
STELLAR_MASS_MIN_MSUN = 1.0
STELLAR_MASS_MAX_MSUN = 100.0

# ------------------------------------------------------------
# Controlled parameter-space scan for planetary binary lenses
# ------------------------------------------------------------
#
# These are exploration priors, not an astrophysical occurrence
# distribution.
#
# The host mass, q, and s are sampled independently in log-space.
PLANET_HOST_MASS_MIN_MSUN = 0.08
PLANET_HOST_MASS_MAX_MSUN = 10.0

# Unit conversion used for planetary binary lenses.
M_JUP_TO_M_SUN = u.M_jup.to(u.M_sun)

# Physical planetary-mass limits expressed in solar masses.
PLANET_MASS_MIN_MSUN = (
    PLANET_MASS_MIN_MJUP
    * M_JUP_TO_M_SUN
)

PLANET_MASS_MAX_MSUN = (
    PLANET_MASS_MAX_MJUP
    * M_JUP_TO_M_SUN
)

# Global q range compatible with at least one combination of
# the adopted host- and planet-mass ranges.
#
# q_min:
#     minimum planet / maximum host
#
# q_max:
#     maximum planet / minimum host
# Mass-ratio range for the controlled binary-lens scan.
#
# q is a primary parameter and is not restricted by an
# independently imposed companion-mass interval.
PLANET_Q_MIN = 1.0e-8
PLANET_Q_MAX = 1.0e-1

PLANET_S_MIN = 0.1
PLANET_S_MAX = 10.0


def log_uniform(rng, low, high):
    """
    Sample uniformly in log10(parameter).

    This is appropriate for the controlled mass scans used here,
    because the simulated lens masses span several orders of
    magnitude.

    The resulting sampling distribution is

        p(M) proportional to 1 / M

    between low and high.
    """
    low = float(low)
    high = float(high)

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or low <= 0
        or high <= low
    ):
        raise ValueError(
            "log_uniform requires 0 < low < high; "
            f"got low={low}, high={high}"
        )

    return 10.0 ** rng.uniform(
        np.log10(low),
        np.log10(high),
    )


def event_param(
    random_seed,
    data_TRILEGAL,
    data_Genulens,
    system_type,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
    param_samplers=None,
):
    """
    Genera los parámetros de microlente.

    param_samplers permite reemplazar distribuciones default.

    Ejemplos
    --------
    param_samplers = {
        "u0": {"type": "rho_uniform", "low": -1, "high": 1},
        "t0": {"type": "uniform", "low": 2461508.76, "high": 2461508.76 + 72},
    }
    """

    rng = np.random.RandomState(random_seed)

    DL = data_Genulens["D_L"]
    DS = data_Genulens["D_S"]

    # GENULENS provides the relative proper-motion vector.
    # Use its physical direction to orient the microlensing
    # parallax vector instead of drawing a random angle.
    mu_rel_N = float(data_Genulens["mu_rel_N"])
    mu_rel_E = float(data_Genulens["mu_rel_E"])

    mu_rel_vector_norm = float(
        np.hypot(mu_rel_N, mu_rel_E)
    )

    if (
        not np.isfinite(mu_rel_vector_norm)
        or mu_rel_vector_norm <= 0
    ):
        raise ValueError(
            "Invalid GENULENS relative proper-motion vector: "
            f"mu_rel_N={mu_rel_N}, mu_rel_E={mu_rel_E}"
        )

    # Keep GENULENS vector components internally self-consistent.
    # The scalar column is retained only as a consistency check.
    mu_rel_catalog = float(data_Genulens["mu_rel"])

    if (
        np.isfinite(mu_rel_catalog)
        and mu_rel_catalog > 0
    ):
        rel_diff = abs(
            mu_rel_vector_norm - mu_rel_catalog
        ) / mu_rel_catalog

        if rel_diff > 1e-6:
            raise ValueError(
                "GENULENS mu_rel is inconsistent with "
                "hypot(mu_rel_N, mu_rel_E): "
                f"mu_rel={mu_rel_catalog}, "
                f"norm={mu_rel_vector_norm}, "
                f"relative_difference={rel_diff}"
            )

    mu_rel = mu_rel_vector_norm

    logL = data_TRILEGAL["logL"]
    logTe = data_TRILEGAL["logTe"]

    context = {
        "random_seed": random_seed,
        "data_TRILEGAL": data_TRILEGAL,
        "data_Genulens": data_Genulens,
        "system_type": system_type,
        "DL": DL,
        "DS": DS,
        "mu_rel": mu_rel,
        "mu_rel_N": mu_rel_N,
        "mu_rel_E": mu_rel_E,
        "mu_rel_catalog": mu_rel_catalog,
        "logL": logL,
        "logTe": logTe,
    }

    orbital_period = get_sampled_or_default(
        "orbital_period",
        param_samplers,
        rng,
        context,
        default_sampler=lambda: 0,
    )

    if system_type == "Planets_systems":

        # For planetary binary lenses, s is the primary
        # projected-separation parameter.
        #
        # We therefore do NOT generate an independent orbital
        # semi-major axis.  NaN is passed internally because
        # microlensing_params retains the historical argument,
        # but ulens.s() is never called for Planets_systems.
        if (
            param_samplers is not None
            and "semi_major_axis" in param_samplers
        ):
            raise ValueError(
                "For Planets_systems, semi_major_axis is no "
                "longer an independent parameter. Specify s."
            )

        semi_major_axis = np.nan

    else:

        semi_major_axis = get_sampled_or_default(
            "semi_major_axis",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: rng.uniform(
                0.1,
                28,
            ),
        )

    if system_type == "Planets_systems":

        # ----------------------------------------------------
        # Controlled binary-lens parameter scan
        #
        # Primary independent variables:
        #
        #     M_star
        #     q
        #
        # with both sampled uniformly in log10.
        #
        # The companion mass is derived exactly from
        #
        #     M_planet = q * M_star.
        #
        # No independent hard cut on M_planet is imposed here.
        # ----------------------------------------------------

        if (
            param_samplers is not None
            and "mass_planet" in param_samplers
        ):
            raise ValueError(
                "For Planets_systems, mass_planet is a "
                "derived quantity. Specify q and/or "
                "star_mass instead."
            )

        q_target = get_sampled_or_default(
            "q",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: log_uniform(
                rng,
                PLANET_Q_MIN,
                PLANET_Q_MAX,
            ),
        )

        q_target = float(q_target)

        if (
            not np.isfinite(q_target)
            or q_target <= 0
        ):
            raise ValueError(
                "Invalid planetary-binary mass ratio: "
                f"q={q_target}"
            )

        star_mass = get_sampled_or_default(
            "star_mass",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: log_uniform(
                rng,
                PLANET_HOST_MASS_MIN_MSUN,
                PLANET_HOST_MASS_MAX_MSUN,
            ),
        )

        star_mass = float(star_mass)

        if (
            not np.isfinite(star_mass)
            or star_mass <= 0
        ):
            raise ValueError(
                "Invalid planetary-binary host mass: "
                f"Mstar={star_mass} Msun"
            )

        # q = M_planet / M_star.
        #
        # star_mass is stored in Msun while mass_planet is
        # stored internally in Mjup.
        mass_planet = (
            q_target
            * star_mass
            / M_JUP_TO_M_SUN
        )

        context.update(
            {
                "q_target": q_target,
                "planet_host_mass_msun":
                    star_mass,
            }
        )

    elif system_type == "Binary_stars":

        star_mass = get_sampled_or_default(
            "star_mass",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: rng.uniform(1, 50),
        )

        mass_planet = get_sampled_or_default(
            "mass_planet",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: rng.uniform(1, 50) * u.M_sun.to("M_jup"),
        )

    elif system_type == "BH":

        star_mass = get_sampled_or_default(
            "star_mass",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: log_uniform(
                rng,
                STELLAR_MASS_MIN_MSUN,
                STELLAR_MASS_MAX_MSUN,
            ),
        )

        mass_planet = get_sampled_or_default(
            "mass_planet",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: 0,
        )

    elif system_type == "FFP":

        star_mass = get_sampled_or_default(
            "star_mass",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: 0,
        )

        mass_planet = get_sampled_or_default(
            "mass_planet",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: log_uniform(
                rng,
                PLANET_MASS_MIN_MJUP,
                PLANET_MASS_MAX_MJUP,
            ),
        )

    elif system_type == "custom":

        if custom_system is None:
            custom_system = {}

        star_mass = get_sampled_or_default(
            "star_mass",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: custom_system.get("star_mass"),
        )

        mass_planet = get_sampled_or_default(
            "mass_planet",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: custom_system.get("planet_mass"),
        )

    else:
        raise ValueError(f"Unknown system_type: {system_type}")

    context.update(
        {
            "orbital_period": orbital_period,
            "semi_major_axis": semi_major_axis,
            "star_mass": star_mass,
            "mass_planet": mass_planet,
        }
    )

    ulens = microlensing_params(
        system_type,
        orbital_period,
        semi_major_axis,
        DL,
        star_mass,
        mass_planet,
        DS,
        mu_rel,
        logTe,
        logL,
    )

    rho = ulens.rho()
    tE = ulens.tE()
    piE = ulens.piE()

    context.update(
        {
            "ulens": ulens,
            "rho": rho.value,
            "tE": tE.value,
            "piE": piE.value,
            "thetaE": ulens.theta_E().value,
            "thetas": ulens.thetas().value,
            "source_radius": float(ulens.source_radius().value),
        }
    )

    t0 = get_sampled_or_default(
        "t0",
        param_samplers,
        rng,
        context,
        default_sampler=lambda: rng.uniform(*t0_range),
    )

    if system_type == "Planets_systems":
        default_u0_sampler = lambda: rho.value * rng.uniform(-3, 3)
    else:
        default_u0_sampler = lambda: rng.uniform(-2, 2)

    u0 = get_sampled_or_default(
        "u0",
        param_samplers,
        rng,
        context,
        default_sampler=default_u0_sampler,
    )

    alpha = get_sampled_or_default(
        "alpha",
        param_samplers,
        rng,
        context,
        default_sampler=lambda: rng.uniform(0, 2 * np.pi),
    )

    # --------------------------------------------------------
    # Physical microlensing-parallax direction
    # --------------------------------------------------------
    #
    # pi_E is parallel to the lens-source relative proper-motion
    # vector. Its magnitude is still computed from the imposed
    # lens mass together with GENULENS D_L and D_S.
    #
    # GENULENS components are North/East:
    #
    #   piEN = piE * mu_rel_N / |mu_rel|
    #   piEE = piE * mu_rel_E / |mu_rel|
    #
    # This replaces the previous random parallax angle.
    # --------------------------------------------------------

    muhat_N = mu_rel_N / mu_rel
    muhat_E = mu_rel_E / mu_rel

    piEN = piE * muhat_N
    piEE = piE * muhat_E

    piE_angle = float(
        np.arctan2(mu_rel_N, mu_rel_E)
    )

    context.update(
        {
            "t0": t0,
            "u0": u0,
            "alpha": alpha,
            "piE_angle": piE_angle,
            "mu_rel_N": mu_rel_N,
            "mu_rel_E": mu_rel_E,
            "muhat_N": muhat_N,
            "muhat_E": muhat_E,
            "piEN_default": piEN.value,
            "piEE_default": piEE.value,
        }
    )

    piEN_value = get_sampled_or_default(
        "piEN",
        param_samplers,
        rng,
        context,
        default_sampler=lambda: piEN.value,
    )

    piEE_value = get_sampled_or_default(
        "piEE",
        param_samplers,
        rng,
        context,
        default_sampler=lambda: piEE.value,
    )

    params_ulens = {
        "t0": t0,
        "u0": u0,
        "tE": tE.value,
        "piEN": piEN_value,
        "piEE": piEE_value,
        "radius": float(ulens.source_radius().value),
        "mass_star": star_mass,
        "mass_planet": mass_planet,
        "thetaE": ulens.theta_E().value,
        "thetas": ulens.thetas().value,
    }

    if system_type in ["FFP", "BH", "Binary_stars", "Planets_systems"]:
        params_ulens["rho"] = get_sampled_or_default(
            "rho",
            param_samplers,
            rng,
            context,
            default_sampler=lambda: rho.value,
        )

    if system_type in ["Binary_stars", "Planets_systems"]:

        q_physical = float(
            ulens.mass_ratio().value
        )

        if system_type == "Planets_systems":

            # ----------------------------------------------
            # q consistency
            # ----------------------------------------------

            if not np.isclose(
                q_physical,
                q_target,
                rtol=1e-12,
                atol=0.0,
            ):
                raise RuntimeError(
                    "Internal q inconsistency: "
                    f"sampled q={q_target}, "
                    f"Mplanet/Mstar={q_physical}"
                )

            params_ulens["q"] = q_physical

            # ----------------------------------------------
            # s is the primary projected-separation
            # parameter.
            # ----------------------------------------------

            s_final = get_sampled_or_default(
                "s",
                param_samplers,
                rng,
                context,
                default_sampler=lambda: log_uniform(
                    rng,
                    PLANET_S_MIN,
                    PLANET_S_MAX,
                ),
            )

            s_final = float(s_final)

            if (
                not np.isfinite(s_final)
                or s_final <= 0
            ):
                raise ValueError(
                    "Invalid planetary-binary "
                    f"separation s={s_final}"
                )

            params_ulens["s"] = s_final

            # ----------------------------------------------
            # Derived projected physical separation
            #
            # 1 mas at 1 kpc = 1 AU
            #
            # DL is stored in pc:
            #
            # a_perp[AU]
            #   = s * thetaE[mas] * DL[kpc].
            # ----------------------------------------------

            params_ulens["a_perp_au"] = (
                s_final
                * float(
                    ulens.theta_E().value
                )
                * float(DL)
                / 1000.0
            )

            context.update(
                {
                    "q_default":
                        q_physical,
                    "s_default":
                        s_final,
                    "a_perp_au":
                        params_ulens[
                            "a_perp_au"
                        ],
                }
            )

        else:

            # ----------------------------------------------
            # Preserve Binary_stars behavior.
            # ----------------------------------------------

            s_physical = ulens.s()

            context.update(
                {
                    "s_default":
                        s_physical.value,
                    "q_default":
                        q_physical,
                }
            )

            params_ulens["s"] = (
                get_sampled_or_default(
                    "s",
                    param_samplers,
                    rng,
                    context,
                    default_sampler=lambda:
                        s_physical.value,
                )
            )

            params_ulens["q"] = (
                get_sampled_or_default(
                    "q",
                    param_samplers,
                    rng,
                    context,
                    default_sampler=lambda:
                        q_physical,
                )
            )

        params_ulens["alpha"] = alpha

    return params_ulens

  

class microlensing_params:
    
    def __init__(self, name, orbital_period, semi_major_axis, DL, star_mass, mass_planet, DS, mu_rel, logTe, logL):
        self.name = name
        self.orbital_period = (orbital_period * u.day).to(u.year)
        self.semi_major_axis = semi_major_axis * u.au
        self.DL = DL * u.pc
        self.mass_star = star_mass * u.M_sun
        self.mass_planet = mass_planet * u.M_jup
        # self.method = method
        # self.source_radius = source_radius * u.R_sun
        self.DS = DS * u.pc
        self.mu_rel = mu_rel * (u.mas / u.year)
        self.logTe = logTe
        self.logL = logL

    def mass_ratio(self):
        return (self.mass_planet / self.mass_star).decompose()

    def m_lens(self):
        if np.isnan(self.mass_planet):
            return (self.mass_star).decompose().to(u.M_sun)
        else:
            return (self.mass_star + self.mass_planet).decompose().to(u.M_sun)

    def pi_rel(self):
        if self.DL<self.DS:
            # print(u.au, self.DL, u.au / self.DL)
            return ((1 / self.DL) - (1  / self.DS)) * u.rad

        else:
            raise Exception("Invalid distance combination DL>DS")

    def theta_E(self):
        # Calculate theta_E in radians
        theta_E_rad = np.sqrt(k * self.pi_rel() * self.m_lens())
        
        # Convert radians to milliarcseconds (mas)
        theta_E_mas = theta_E_rad.to(u.mas, equivalencies=u.dimensionless_angles())
        
        return theta_E_mas

    
    def tE(self):
        return (self.theta_E() / self.mu_rel).to(u.day)

    def piE(self):
        return (u.au*self.pi_rel() / self.theta_E()).decompose()

    def source_radius(self):
        logL = self.logL
        logTe = self.logTe
        L_star = 10**(logL)
        Teff = (10**(logTe))*u.K
        top = L_star*L_sun
        sigma = sigma_sb
        bot = 4*np.pi*sigma*Teff**4
        Radius = np.sqrt(top/bot).to('R_sun')
        # print('Radius: ',type(Radius), Radius)
        return Radius
    
    def thetas(self):
        if self.DL<self.DS:
            # print('source_radisu:',self.source_radius(),'  DS:', self.DS)
            # Calculate the angular size of the source in radians
            theta_S_rad = (self.source_radius() / self.DS).decompose()
            
            # Convert radians to milliarcseconds (mas)
            theta_S_mas = theta_S_rad.to(u.mas, equivalencies=u.dimensionless_angles())
            # print('thetaS', theta_S_mas)
            return theta_S_mas
        else:
            raise Exception("Invalid distance combination DL>DS")


    def rho(self):
        return (self.thetas() / self.theta_E()).decompose()

    def s(self):
        if self.DL<self.DS:
            # Calculate the angular separation in radians
            s_rad = (self.semi_major_axis / self.DL).decompose()
            
            # Convert radians to milliarcseconds (mas)
            s_mas = s_rad.to(u.mas, equivalencies = u.dimensionless_angles())
            
            # Divide by the Einstein radius to get the normalized separation
            return s_mas / self.theta_E()
        else:
            raise Exception("Invalid distance combination DL>DS")

            
    def u0(self, criterion = "caustic_proximity"):
        random_factor = np.random.uniform(0,3)
        if criterion == "caustic_proximity":
            return random_factor*self.rho() 
        if criterion == "resonant_region":
            return 1/self.s() - self.s()
        # np.sqrt(1 - self.s() ** 2)

    def piE_comp(self):
        phi =  np.random.uniform(0, np.pi) # np.pi/4
        piEE = self.piE() * np.cos(phi)
        piEN = self.piE() * np.sin(phi)
        return piEE, piEN

    def orbital_motion(self, sz=2, a_s=1):
        r_s = sz / self.s()
        n = 2 * np.pi / self.orbital_period
        denominator = a_s * np.sqrt((-1 + 2 * a_s) * (1 + r_s**2))
        velocity_magnitude = n * denominator
    
        def sample_velocities(magnitude):
            # Extract the value of magnitude (without units)
            magnitude_value = magnitude.value
            
            # Generate random velocities
            gamma = np.random.normal(size=3)
            gamma *= magnitude_value / np.linalg.norm(gamma)
            return gamma
    
        # Sample velocities
        gamma1, gamma2, gamma3 = sample_velocities(velocity_magnitude)
        
        # Assign velocities to components
        v_para = gamma1
        v_perp = gamma2
        v_radial = gamma3
        
        return r_s, a_s, v_para, v_perp, v_radial
    
    
def build_mu_rel_pairs(
    df,
    N,
    offset=0.1,
    min_D=1.0,
    random_state=None,
    mu_rel_mode="random_angle",
    lens_D_range_kpc=None,
    source_D_range_kpc=None,
):
    """
    Construye pares fuente-lente a partir de un catálogo de estrellas.

    La fuente es la estrella más lejana.
    La lente es una estrella más cercana.

    Parameters
    ----------
    df : pandas.DataFrame
        Catálogo de AstroDataLab/TRILEGAL.
        Debe contener:
            mu0, pmracosd, pmdec

    N : int
        Número máximo de pares a devolver.

    offset : float
        Separación mínima fuente-lente en pc, porque D_S se calcula en pc.
        Condición:
            D_L < D_S - offset

    min_D : float
        Distancia mínima en pc para la lente.

    lens_D_range_kpc : tuple or None
        Rango permitido para la lente en kpc.
        Ejemplo:
            lens_D_range_kpc=(1.0, 2.0)

    source_D_range_kpc : tuple or None
        Rango permitido para la fuente en kpc.
        Ejemplo:
            source_D_range_kpc=(2.0, 8.0)

    mu_rel_mode : str
        "random_angle":
            usa los módulos de movimiento propio y sortea un ángulo relativo.

        "vector":
            usa directamente las componentes pmracosd y pmdec.

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas de la fuente, más:
            D_S, D_L, mu_rel, theta_rad, mu_source, mu_lens,
            D_S_kpc, D_L_kpc
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(random_state)

    df = df.copy()

    # Distancia en pc a partir del módulo de distancia.
    df["D_S"] = 10 ** ((df["mu0"] + 5) / 5)

    # Columnas diagnósticas en kpc.
    df["D_S_kpc"] = df["D_S"] / 1000.0

    df["mu_s"] = np.sqrt(
        df["pmracosd"]**2 + df["pmdec"]**2
    )

    df_sorted = df.sort_values(
        by="D_S"
    ).reset_index(drop=True)

    # Convertimos rangos en kpc a pc para filtrar.
    if lens_D_range_kpc is not None:
        lens_D_min_pc = 1000.0 * float(lens_D_range_kpc[0])
        lens_D_max_pc = 1000.0 * float(lens_D_range_kpc[1])
    else:
        lens_D_min_pc = None
        lens_D_max_pc = None

    if source_D_range_kpc is not None:
        source_D_min_pc = 1000.0 * float(source_D_range_kpc[0])
        source_D_max_pc = 1000.0 * float(source_D_range_kpc[1])
    else:
        source_D_min_pc = None
        source_D_max_pc = None

    kept = []
    n = len(df_sorted)

    for i in range(n - 1, -1, -1):

        if len(kept) >= N:
            break

        source_row = df_sorted.iloc[i]
        D_s = float(source_row["D_S"])

        if D_s <= min_D + offset:
            continue

        # Filtro opcional sobre distancia de la fuente.
        if source_D_min_pc is not None and D_s < source_D_min_pc:
            continue

        if source_D_max_pc is not None and D_s > source_D_max_pc:
            continue

        closer_block = df_sorted.iloc[:i]

        if closer_block.empty:
            continue

        candidates = closer_block[
            (closer_block["D_S"] > min_D)
            & (closer_block["D_S"] < D_s - offset)
        ]

        # Filtro principal: distancia de la lente.
        if lens_D_min_pc is not None:
            candidates = candidates[
                candidates["D_S"] >= lens_D_min_pc
            ]

        if lens_D_max_pc is not None:
            candidates = candidates[
                candidates["D_S"] <= lens_D_max_pc
            ]

        if candidates.empty:
            continue

        lens_row = candidates.sample(
            n=1,
            random_state=int(rng.integers(1e9)),
        ).iloc[0]

        mu_source = float(source_row["mu_s"])
        mu_lens = float(lens_row["mu_s"])

        if mu_rel_mode == "random_angle":

            theta = rng.uniform(
                0.0,
                2 * np.pi,
            )

            mu_rel = np.sqrt(
                mu_source**2
                + mu_lens**2
                - 2 * mu_source * mu_lens * np.cos(theta)
            )

        elif mu_rel_mode == "vector":

            dpmra = float(lens_row["pmracosd"] - source_row["pmracosd"])
            dpmdec = float(lens_row["pmdec"] - source_row["pmdec"])

            mu_rel = np.sqrt(
                dpmra**2 + dpmdec**2
            )

            theta = np.arctan2(
                dpmdec,
                dpmra,
            )

        else:
            raise ValueError(
                "mu_rel_mode debe ser 'random_angle' o 'vector'."
            )

        src_dict = source_row.to_dict()

        src_dict.update(
            {
                "D_S": float(D_s),
                "D_L": float(lens_row["D_S"]),
                "D_S_kpc": float(D_s / 1000.0),
                "D_L_kpc": float(lens_row["D_S"] / 1000.0),
                "mu_rel": float(mu_rel),
                "theta_rad": float(theta),
                "mu_source": float(mu_source),
                "mu_lens": float(mu_lens),
                "lens_ra": float(lens_row["ra"]) if "ra" in lens_row else np.nan,
                "lens_dec": float(lens_row["dec"]) if "dec" in lens_row else np.nan,
            }
        )

        # Si existe gc, guardamos también el componente galáctico de la lente.
        if "gc" in lens_row.index:
            src_dict["lens_gc"] = lens_row["gc"]

        if "galb" in lens_row.index:
            src_dict["lens_galb"] = lens_row["galb"]

        if "gall" in lens_row.index:
            src_dict["lens_gall"] = lens_row["gall"]

        kept.append(src_dict)

    return pd.DataFrame(kept).reset_index(drop=True)


def _standardize_astrodatalab_columns(df):
    """
    Normaliza nombres de columnas de AstroDataLab/TRILEGAL al formato
    que espera el simulador.

    Acepta columnas originales:
        umag, gmag, rmag, imag, zmag, ymag, logl, logte

    y las renombra a:
        u, g, r, i, z, Y, logL, logTe

    También elimina duplicados, por ejemplo si vienen z y zmag.
    """

    df = df.copy()

    # Seguridad: si vienen columnas duplicadas exactas desde la query.
    df = df.loc[:, ~df.columns.duplicated()]

    col_map = {
        "umag": "u",
        "gmag": "g",
        "rmag": "r",
        "imag": "i",
        "zmag": "z",
        "ymag": "Y",
        "logl": "logL",
        "logte": "logTe",
    }

    # Si existe columna destino y también la original,
    # conservamos la original renombrada.
    # Ejemplo: si existen z y zmag, eliminamos z y usamos zmag -> z.
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col in df.columns:
            df = df.drop(columns=[new_col])

    df = df.rename(columns=col_map)

    # Seguridad final.
    df = df.loc[:, ~df.columns.duplicated()]

    return df
def _build_astrodatalab_query(
    ra_center,
    dec_center,
    radius,
    N,
    Ds_max,
    limit_extra=1000,
    table="lsst_sim.simdr2",
    select_cols=None,
    extra_where=None,
    use_radial_query=True,
    mu0_min=None,
    mu0_max=None,
    limit_override=None,
):
    """
    Construye la query estándar, permitiendo:
    - filtros extra;
    - chunks en mu0;
    - columnas custom.
    """
    import numpy as np

    if mu0_max is None:
        mu0_max = 5 * np.log10(float(Ds_max)) - 5

    default_select_cols = [
        "ra",
        "dec",
        "mu0",
        "pmracosd",
        "pmdec",
        "umag",
        "gmag",
        "rmag",
        "imag",
        "zmag",
        "ymag",
        "logl",
        "logte",
    ]

    if select_cols is None:
        select_cols = default_select_cols
    else:
        select_cols = list(select_cols)

        for col in default_select_cols:
            if col not in select_cols:
                select_cols.append(col)

    columns_sql = ",\n            ".join(select_cols)

    where_terms = []

    if use_radial_query:
        where_terms.append(
            f"q3c_radial_query(ra, dec, {ra_center}, {dec_center}, {radius})"
        )

    if mu0_min is not None:
        where_terms.append(
            f"mu0 >= ({mu0_min})"
        )

    if mu0_max is not None:
        where_terms.append(
            f"mu0 < ({mu0_max})"
        )

    if extra_where is not None:
        where_terms.append(
            f"({extra_where})"
        )

    where_sql = "\n          AND ".join(where_terms)

    if limit_override is None:
        limit_value = N + limit_extra
    else:
        limit_value = limit_override

    query = f"""
        SELECT
            {columns_sql}
        FROM {table}
        WHERE {where_sql}
        LIMIT {limit_value}
    """

    return query
def _query_astrodatalab_dataframe(
    query,
    query_format="csv",
    timeout=None,
    max_retries=2,
    sleep_seconds=5,
):
    """
    Ejecuta una query en AstroDataLab y devuelve un DataFrame.
    Reintenta si falla por timeout u otro error temporal.
    """
    import time
    from dl import queryClient as qc
    from dl.helpers.utils import convert

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            query_kwargs = {
                "sql": query,
                "fmt": query_format,
            }

            if timeout is not None:
                query_kwargs["timeout"] = timeout

            res = qc.query(**query_kwargs)
            df = convert(res, "pandas")

            return df

        except Exception as e:
            last_error = e
            print(f"    Query failed on attempt {attempt + 1}/{max_retries + 1}")
            print(f"    Error: {e}")

            if attempt < max_retries:
                time.sleep(sleep_seconds)

    raise last_error

def download_astrodatalab_pair_catalog(
    ra_center=None,
    dec_center=None,
    radius=None,
    N=1000,
    Ds_max=12000,
    offset=0.1,
    min_D=1.0,
    random_state=None,
    limit_extra=1000,
    table="lsst_sim.simdr2",
    mu_rel_mode="random_angle",
    w149_from="Y",
    select_cols=None,
    extra_where=None,
    custom_query=None,
    query_format="csv",
    timeout=None,
    extra_keep_cols=None,
    use_radial_query=True,
    chunk_query=True,
    n_mu0_chunks=10,
    limit_per_chunk=200,
    max_retries=2,
    sleep_seconds=5,
    lens_D_range_kpc=None,
    source_D_range_kpc=None,
):
    """
    Descarga estrellas desde AstroDataLab, construye pares fuente-lente
    y devuelve un DataFrame listo para usar con event_param() y sim_fit().

    Modos de uso
    ------------
    1. Query estándar:
        download_astrodatalab_pair_catalog(...)

    2. Query estándar con filtros extra:
        extra_where="gc IN (1, 2)"

    3. Query completamente custom:
        custom_query=\"\"\"
            SELECT ...
            FROM ...
            WHERE ...
            LIMIT ...
        \"\"\"

    Columnas mínimas requeridas, antes o después de renombrar
    ---------------------------------------------------------
    Para construir pares:
        ra, dec, mu0, pmracosd, pmdec

    Para simular:
        u/g/r/i/z/Y, logL, logTe
        o las originales umag/gmag/rmag/imag/zmag/ymag, logl/logte

    Returns
    -------
    pair_catalog : pandas.DataFrame
        Columnas principales:
        D_S, D_L, mu_rel, logL, logTe,
        ra, dec, u, g, r, i, z, Y, W149.
    """

    import numpy as np
    import pandas as pd
    from dl import queryClient as qc
    from dl.helpers.utils import convert

    if extra_keep_cols is None:
        extra_keep_cols = []

    if custom_query is None:
        if use_radial_query:
            if ra_center is None or dec_center is None or radius is None:
                raise ValueError(
                    "Si custom_query=None y use_radial_query=True, "
                    "tenés que pasar ra_center, dec_center y radius."
                )

        query = _build_astrodatalab_query(
            ra_center=ra_center,
            dec_center=dec_center,
            radius=radius,
            N=N,
            Ds_max=Ds_max,
            limit_extra=limit_extra,
            table=table,
            select_cols=select_cols,
            extra_where=extra_where,
            use_radial_query=use_radial_query,
        )

    else:
        query = custom_query

    print("Requesting data from AstroDataLab ...")

    if custom_query is not None:
        print("Requesting data from AstroDataLab using custom_query ...")
    
        df_raw = _query_astrodatalab_dataframe(
            custom_query,
            query_format=query_format,
            timeout=timeout,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
    
    else:
        if chunk_query:
            df_raw = _download_astrodatalab_chunks_mu0(
                ra_center=ra_center,
                dec_center=dec_center,
                radius=radius,
                N=N,
                Ds_max=Ds_max,
                min_D=min_D,
                limit_extra=limit_extra,
                table=table,
                select_cols=select_cols,
                extra_where=extra_where,
                use_radial_query=use_radial_query,
                query_format=query_format,
                timeout=timeout,
                n_mu0_chunks=n_mu0_chunks,
                limit_per_chunk=limit_per_chunk,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
    
        else:
            query = _build_astrodatalab_query(
                ra_center=ra_center,
                dec_center=dec_center,
                radius=radius,
                N=N,
                Ds_max=Ds_max,
                limit_extra=limit_extra,
                table=table,
                select_cols=select_cols,
                extra_where=extra_where,
                use_radial_query=use_radial_query,
            )
    
            print("Requesting data from AstroDataLab ...")
    
            df_raw = _query_astrodatalab_dataframe(
                query,
                query_format=query_format,
                timeout=timeout,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )

    print(f"Downloaded {len(df_raw)} stars from AstroDataLab.")

    if len(df_raw) == 0:
        raise RuntimeError(
            "La query no devolvió estrellas."
        )

    # Normalizamos nombres antes de construir pares.
    df_raw = _standardize_astrodatalab_columns(df_raw)

    required_raw_cols = [
        "ra",
        "dec",
        "mu0",
        "pmracosd",
        "pmdec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "Y",
        "logL",
        "logTe",
    ]

    missing_raw = [
        col for col in required_raw_cols
        if col not in df_raw.columns
    ]

    if missing_raw:
        raise KeyError(
            "Faltan columnas necesarias en el resultado de AstroDataLab "
            f"después de normalizar nombres: {missing_raw}\n\n"
            "Asegurate de que tu query incluya columnas equivalentes a:\n"
            "ra, dec, mu0, pmracosd, pmdec, "
            "umag/gmag/rmag/imag/zmag/ymag, logl, logte."
        )

    df_raw = df_raw.dropna(
        subset=required_raw_cols,
    ).reset_index(drop=True)

    if len(df_raw) < 2:
        raise RuntimeError(
            "Muy pocas estrellas después de limpiar NaNs. "
            "Probá aumentar radius, Ds_max, limit_extra o relajar filtros."
        )

    df_pairs = build_mu_rel_pairs(
        df_raw,
        N=N,
        offset=offset,
        min_D=min_D,
        random_state=random_state,
        mu_rel_mode=mu_rel_mode,
        lens_D_range_kpc=lens_D_range_kpc,
        source_D_range_kpc=source_D_range_kpc,
    )

    print(f"Built {len(df_pairs)} source-lens pairs.")

    if len(df_pairs) == 0:
        raise RuntimeError(
            "No se pudieron construir pares fuente-lente. "
            "Probá aumentar radius, Ds_max, limit_extra o reducir offset."
        )

    # Por seguridad, normalizamos también después del pairing.
    df_pairs = _standardize_astrodatalab_columns(df_pairs)

    required_pair_cols = [
        "D_S",
        "D_L",
        "mu_rel",
        "logL",
        "logTe",
        "ra",
        "dec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "Y",
    ]

    missing = [
        col for col in required_pair_cols
        if col not in df_pairs.columns
    ]

    if missing:
        raise KeyError(
            "Faltan columnas necesarias en el catálogo pareado: "
            f"{missing}"
        )

    if "W149" not in df_pairs.columns:
        if w149_from not in df_pairs.columns:
            raise KeyError(
                f"No existe W149 y tampoco existe la columna {w149_from} "
                "para usar como aproximación."
            )

        df_pairs["W149"] = df_pairs[w149_from]

    keep_cols = [
        "D_S",
        "D_L",
        "D_S_kpc",
        "D_L_kpc",
        "mu_rel",
        "logL",
        "logTe",
        "ra",
        "dec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "Y",
        "W149",
        "theta_rad",
        "mu_source",
        "mu_lens",
        "lens_ra",
        "lens_dec",
        "gc",
        "galb",
        "gall",
        "lens_gc",
        "lens_galb",
        "lens_gall",
    ]

    # Guardar columnas extra pedidas por el usuario si existen.
    for col in extra_keep_cols:
        if col in df_pairs.columns and col not in keep_cols:
            keep_cols.append(col)

    keep_cols = [
        col for col in keep_cols
        if col in df_pairs.columns
    ]

    pair_catalog = df_pairs[keep_cols].copy()

    # Seguridad final antes de devolver/guardar.
    pair_catalog = pair_catalog.loc[:, ~pair_catalog.columns.duplicated()]

    return pair_catalog



def _download_astrodatalab_chunks_mu0(
    ra_center,
    dec_center,
    radius,
    N,
    Ds_max,
    min_D=1.0,
    limit_extra=1000,
    table="lsst_sim.simdr2",
    select_cols=None,
    extra_where=None,
    use_radial_query=True,
    query_format="csv",
    timeout=None,
    n_mu0_chunks=10,
    limit_per_chunk=200,
    max_retries=2,
    sleep_seconds=5,
):
    """
    Descarga el catálogo en chunks de mu0 para evitar timeouts.
    """
    import numpy as np
    import pandas as pd

    mu0_min_global = 5 * np.log10(float(min_D)) - 5
    mu0_max_global = 5 * np.log10(float(Ds_max)) - 5

    edges = np.linspace(
        mu0_min_global,
        mu0_max_global,
        n_mu0_chunks + 1,
    )

    dfs = []

    print("Requesting data from AstroDataLab in mu0 chunks...")
    print(f"mu0 range: {mu0_min_global:.3f} to {mu0_max_global:.3f}")
    print(f"n_mu0_chunks={n_mu0_chunks}, limit_per_chunk={limit_per_chunk}")

    for k in range(n_mu0_chunks):
        mu0_lo = edges[k]
        mu0_hi = edges[k + 1]

        print(
            f"  chunk {k + 1}/{n_mu0_chunks}: "
            f"{mu0_lo:.3f} <= mu0 < {mu0_hi:.3f}"
        )

        query = _build_astrodatalab_query(
            ra_center=ra_center,
            dec_center=dec_center,
            radius=radius,
            N=N,
            Ds_max=Ds_max,
            limit_extra=limit_extra,
            table=table,
            select_cols=select_cols,
            extra_where=extra_where,
            use_radial_query=use_radial_query,
            mu0_min=mu0_lo,
            mu0_max=mu0_hi,
            limit_override=limit_per_chunk,
        )

        try:
            df_k = _query_astrodatalab_dataframe(
                query,
                query_format=query_format,
                timeout=timeout,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )

            print(f"    downloaded {len(df_k)} rows")

            if len(df_k) > 0:
                dfs.append(df_k)

        except Exception as e:
            print(f"    WARNING: chunk failed permanently.")
            print(f"    {e}")

    if len(dfs) == 0:
        raise RuntimeError(
            "No se pudo descargar ningún chunk desde AstroDataLab. "
            "Probá reducir radius, reducir limit_per_chunk, "
            "aumentar n_mu0_chunks o relajar extra_where."
        )

    df_raw = pd.concat(
        dfs,
        ignore_index=True,
    )

    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

    return df_raw

def event_param_from_pair_row(
    random_seed,
    pair_row,
    system_type,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
    param_samplers=None,
):
    """
    Genera parámetros de microlente usando una sola fila pareada
    fuente-lente.

    La misma fila contiene:
    - parámetros de fuente: logL, logTe, magnitudes;
    - parámetros de geometría/lente: D_S, D_L, mu_rel.

    Internamente llama a event_param() pasando pair_row como
    data_TRILEGAL y como data_Genulens.
    """

    return event_param(
        random_seed,
        pair_row,
        pair_row,
        system_type,
        t0_range=t0_range,
        custom_system=custom_system,
        param_samplers=param_samplers,
    )