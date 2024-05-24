import types
import numbers
import numpy as np
import scipy as sp
from sdeint import itoSRI2

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def tipping_equation(c, d, tau=None):
    tau = np.ones(d.shape[0]) if tau is None else np.asarray(tau)
    return lambda y, t: (-y**3 + y + c + d@(y + 1))/tau

def jac_tipping_equation(d):
    return lambda y: np.diag(-3*y**2 + 1) + d

def find_slope(ysteady):
    return -3*ysteady**2 + 1

def find_steady_state(c, d, tau=None):
    
    # number of tipping elements
    N, _ = d.shape
    
    # do not vary the t parameter, as it has no effect
    dy     = lambda y: tipping_equation(c, d, tau=tau)(y, None)
    jac_dy = jac_tipping_equation(d) 
    
    # untipped state
    untipped_state = sp.optimize.least_squares(dy, np.repeat(-1, N), jac=jac_dy)
    
    # tipped state
    tipped_state = sp.optimize.least_squares(dy, np.repeat(1, N), jac=jac_dy)
    
    return untipped_state.x, tipped_state.x


def solve(c, d, noise_level=0.05, y0=0, tspan=np.arange(0, 100, 0.1), tau=None, generator=None):
    """
        Solves the PyCascades equations for n tipping elements using a stochastic RK4 integrator.

        Parameters
        ----------
        a: numeric
            Coefficient of nonlinearity in the self-mechanics (default: -1)
        b: numeric
            Autocorrelation of the systems self-mechanics (default: 1)
        c: array-like (n,)
            External forcing parameters
        d: array-like (n,n)
            Coupling matrix for interactions
        noise-level: numeric
            Standard deviation of the noise for the stochastic integrator
        y0: numeric (n,) or "tipped" or "untipped"
            Initial conditions for the integration. For the string values the respective tipped or 
            untipped state is found in which the solution is steady.
        tspan: array-like (T,)
            Time points over which the solution should be integrated.            

        Returns
        -------
        y: array-like (T, n)
            Integrated solution of the coupled tipping equations.
    """
    
    # number of coupled elements
    N = len(d) if hasattr(d, "shape") else d.shape[0]

    if c is None:
        c = 2*((1/3)**3)**0.5
    
    if isinstance(y0, numbers.Number):
        y0 = np.repeat(y0, N)
    
    if isinstance(y0, str):
        untipped, tipped = find_steady_state(c, d, tau)
        if y0 == "tipped":
            y0 = tipped
        if y0 == "untipped":
            y0 = untipped
        
    if isinstance(y0, np.ndarray) and not y0.shape == (N,):
        raise Exception(f"Initial state vector {y0} can not be interpreted")

    noise_fn = None
    if isinstance(noise_level, numbers.Number):
        noise_fn = lambda y, t: noise_level*np.identity(N)
    if callable(noise_level):
        noise_fn = noise_level
    
    return itoSRI2(tipping_equation(c, d, tau), noise_fn, y0, tspan, generator=generator)