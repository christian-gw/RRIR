import numpy as np
from scipy.optimize import curve_fit
from sml.Signal import Signal


def fit(imp_dec: Signal, xx):
    """ fit linear function linear() to slice of im_dec
        Parameters
        ----------
        imp_dec : Signal
            Signal containing one impulse response
        xx : float
            Decay in dB over which to aprox. keyvalues

        Returns
        -------
        opt : np.array([float: float])
            m: float
                Slope of the aproximated line (should be negative)
            y0: float
                Start value of fitted line (-5 dB)
        """

    def linear(t, m, y0):
        """ Linear function for regression fit: y = m*t+y0"""
        return m*t+y0

    imp_dec.level_time()

    # Arraymask to make sure to get peaks after max
    afterPeakArray = np.ones(imp_dec.n_tot, dtype=bool)
    afterPeakArray[:np.argmax(imp_dec.y)] = False

    # Understanding: shold be in samples, comment said [s]
    # Last pos where a sample is close (+-.01) at L(peak)-5dB
    el5 = np.where(np.isclose(imp_dec.L_t,
                              max(imp_dec.L_t)-5,
                              atol=.01) &
                   afterPeakArray)[0][-1]
    # Last pos where a sample is close (+-.01) at L(peak)-5dB-xxdB
    elxx = np.where(np.isclose(imp_dec.y,
                               max(imp_dec)-5-xx,
                               atol=.01) &
                    afterPeakArray)[0][-1]

    # Slice the arrays in which the fit should take place
    t_act = imp_dec.axis_arrays['t'][el5:elxx]
    y_act = imp_dec.y[el5:elxx]

    # Get the optimisation
    # TODO: adding the R^2 value in return
    # to tell user if chosen length yields sensible results
    opt, _ = curve_fit(linear, t_act, y_act)

    return opt


def txx(imp_dec: Signal, xx=20):
    """Calculate decay time.

    Calculate the decay time by 60 dB form interpolation
    between -5 dB and -5-xx dB.
    It uses a interpolation with scipy.optimize.curve_fit on a linear function.
    This is implemneted in the fit function

    Parameters
    ----------
    imp_dec : Signal
        Signal containing one impulse only
    xx : float
        Decay for interpolation

    Returns
    -------
    T60 : float
        Time it would take the signal to decay by 60 dB.
        (If it had a sufficiently high level relatively to the noisefloor)
    """

    lin_par = fit(imp_dec, xx)
    return 60/np.abs(lin_par[0])


def cxx(imp_dec: Signal, xx: float = 80):
    """Calculates the ratio between early and late sound energy.

    Calculates the logarithmic ratio between sound energy comming before
    and after xx ms of a RIR according to DIN-ISO-3382-1:2009

    Calculation (e.g.):
    C_80 = 10*lg(\\int_0^80(p(t)^2) / \\int_80_\\inv(p(t)^2))

    Parameters
    ----------
    imp_dec : Signal
        Impulse response with only one impulse.
        Beware: the Impulse must be cut, but should not be cut to short.
    xx : float
        Time in ms to draw the line between early and late energy.
        Typical values are:
            50 -> Deutlichkeitsmaß D50
            80 -> Klarheitsmaß C80"""
    start = np.argmax(imp_dec.y)
    mid = int(xx*1e-3/imp_dec.dt) + start
    end = len(imp_dec.y)-1

    # Summing instead of integration is ok, bc its about the relation
    cxx = 10*np.log10(np.sum(np.pow(imp_dec.y[start, mid], 2)) /
                      np.sum(np.pow(imp_dec.y[mid, end], 2)))
    return cxx
