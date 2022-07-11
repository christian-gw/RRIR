from datetime import datetime
from sml.Signal import appl_win, create_band_lst, Signal
import matplotlib.pyplot as plt
from scipy.fft import ifft  # , fft,  fftfreq  # , fftshift
# from scipy.signal.windows import tukey, blackmanharris  # , hann
import scipy.signal as sg
# import librosa as lr
import numpy as np
# import os


def dbg_info(txt, verbose=False):
    """Print debug info if global VERBOSE = True"""
    if verbose:
        print(str(datetime.now().time()) + ' - ' + txt)


class TransferFunction:
    """Transfer Fkt Class
        Initialisation depends on used kwargs:
        - 'incoming_sig' and 'reflected_sig' tf of those two signal obj
        - 'x' and 'hf' to create a tf obj from a external tf
        - 'signal', 'in_win' and 're_win' (both [start, duration]) tf
          from two windowed sections of signal
        Methods:
        - 'convolute_f':
            Returns a convolution of a signal and the tf with length of signal
        - '__get_band': Returns RMS value within a specified band
        - 'get_octave_band': Returns tf in octave bands
        - 'plot_ht', 'plot_hf' and 'plot_oct':
            Plot time or frequency representation of tf
        Attributes:
        - bool: exitation,
        - double:      dt,   T, frange
        - integer:  n_tot,
        - array-axis:   t,  xf,
        - array-signal: hf, ht"""

    def __init__(self, **kwargs):
        if 'incoming_sig' in kwargs and 'reflected_sig' in kwargs:
            incoming_sig = kwargs.get('incoming_sig',  None)
            reflected_sig = kwargs.get('reflected_sig', None)

            self.dt = incoming_sig.dt
            self.T = incoming_sig.T
            self.n_tot = incoming_sig.n_tot
            self.xf = incoming_sig.axis_arrays['xf']

            self.hf = np.copy(reflected_sig.y_f**2 / incoming_sig.y_f**2)
            self.frange = incoming_sig.frange

        elif 'xf' in kwargs and 'hf' in kwargs:
            self.xf = kwargs.get('xf', None)
            self.hf = kwargs.get('hf', None)

            self.n_tot = len(self.hf)

            self.dt = None
            self.T = None

        elif 'signal' in kwargs and 'in_win' in kwargs and 're_win' in kwargs:
            sig = kwargs.get('signal', None)
            in_win = kwargs.get('in_win', None)
            re_win = kwargs.get('re_win', None)

            self.dt = sig.dt
            self.T = sig.T
            self.n_tot = sig.n_tot
            self.xf = sig.axis_arrays['xf']

            incoming_sig, _ = appl_win(sig, in_win[0], in_win[1], form='norm')
            reflected_sig, _ = appl_win(sig, re_win[0], re_win[1], form='norm')

            self.hf = np.copy(reflected_sig.y_f**2 / incoming_sig.y_f**2)
            self.frange = incoming_sig.frange

        else:
            print('No valid keywords provided. - See docstring.')

    def plot_hf(self):
        """Plots spectrum of tf before and after .get_octave_band"""
        if self.n_tot > 128:
            y = 2.0/self.n_tot * np.abs(self.hf[0:self.n_tot//2])
            frange = self.frange
            style = None
        elif self.n_tot <= 128:
            y = self.hf
            frange = [self.xf[0], self.xf[-1]]
            style = 'x'
            print(str(self.xf))
            print(str(self.hf))
        else:
            print('Invalid transfer fkt.')

        fig, ax = plt.subplots(1, figsize=(10, 3))
        ax.plot(self.xf, y, marker=style, linewidth=.25)
        ax.set_xlabel('f [Hz]')
        ax.set_xlim(*frange)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig, ax

    # Fkt for convolution of transferfct with signal
    def convolute_f(self, sig):
        """Convolutes two signals in freq domain (multiplication),
        after resample.)"""

        h_samp = sg.resample(self.hf, len(sig.y_f))
        conv_f = h_samp*sig.y_f
        return Signal(y=ifft(conv_f), dt=self.dt)

    # Fkt to generate octave based signal
    def __get_band(self, f0, f1):
        i0 = np.where(self.xf > f0-self.xf[0])[0][0]
        i1 = np.where(self.xf > f1-self.xf[0])[0][0]

        l_sum = 0
        for el in self.hf[i0:i1]:
            l_sum += np.power(el, 2)
        return np.sqrt(l_sum / (i1-i0))

    def get_octave_band(self, fact=1.):
        """Return new octave based tf."""
        x = []
        y = []
        oct_band_lst = create_band_lst(fact)

        for f in oct_band_lst:
            x.append(f[2])
            y.append(abs(self.__get_band(*f[0:2])))
        return TransferFunction(xf=np.array(x), hf=np.array(y))
