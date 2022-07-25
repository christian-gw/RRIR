from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import ifft  # , fft,  fftfreq  # , fftshift
# from scipy.signal.windows import tukey, blackmanharris  # , hann
import scipy.signal as sg
# import librosa as lr
import numpy as np
# import os

if __name__ == "__main__":
    from Signal import appl_win, create_band_lst, Signal
else:
    from sml.Signal import appl_win, create_band_lst, Signal


def dbg_info(txt, verbose=False):
    """Print debug info if global VERBOSE = True

    Parameters
    ----------
    txt: str
        Text to print in debug
    verbose : Bool
        Print takes place when True
    Returns
    -------
    None"""

    if verbose:
        print(str(datetime.now().time()) + ' - ' + txt)


class TransferFunction:
    """The TransferFunction Class holds and processes a transfer function.

    Its intended for sound signals derived from sweep measurements.
    It can be used to analyse RIRs, Room and Wall TF, directivity analysis ...
    Its initialisation depends on the used kwargs

    Parameters
    ----------
    **kwargs: dict
        incoming_sig: Signal
            Transferfunction betw. incoming_sig and reflected_sig as measured.
            Also needs reflected_sig
        reflected_sig: Signal
            Transferfunction betw. incoming_sig and reflected_sig as measured.
            Also needs incomming_sig
        x: np.array(float)
            x-axis of TF
            Also needs hf
        'hf: np.array(float)
            Externally calculated tf
            Also needs x
        signal: Signal
            Signal with incomming and outcomming component
            Also needs in_win and re_win
            Uses Adrienne Windowing see Norm # TODO: Norm
         in_win: touple(float, float)
            Time window for incomming
            Also needs signal and re_win
            start: float
                Startvalue in [s]
            duration: float
                Duration in [s]
         re_win: touple(float, float)
            Time window for incomming
            Also needs signal and in_win
            start: float
                Startvalue in [s]
            duration: float
                Duration in [s]

          from two windowed sections of signal (Adrienne windowing - Norm)

        Methods
        -------
        convolute_f() -> Signal
            Returns a convolution of a signal and the tf with length of signal
        __get_band() -> float
            Returns RMS value within a specified band
        get_octave_band() -> TransferFunction
            Returns tf in octave bands
        plot_ht() -> fig, ax
            Plot Time signal
        plot_hf() -> fig, ax
            Plot Spectrum
        plot_oct() -> fig, ax:
            Plot time or frequency representation of tf

        Attributes
        ----------
        exitation: Bool
            Specifies wether the signal is a exitation
        dt: Float
            Timebase
        frange: tuple(float)
            Total time
        n_tot: Int
            Number of samples
        ht: np.array(float)
            Time signal
        hf: np.array(float)
            Frequency signal
        axis_array: dict
            t: np.array
                Time axis
            xf: np.array
                Frequency axis"""

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
            if type(self.hf) == list:
                self.hf = np.array(kwargs.get('hf', None)).mean(axis=0)

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
        """Plots spectrum of tf before and after .get_octave_band

        Returns
        -------
        fig: plt.Figure
        ax: plt.Axis"""

        # For Octaves etc.
        if self.n_tot > 128:
            y = 2.0/self.n_tot * np.abs(self.hf[0:self.n_tot//2])
            frange = self.frange
            style = None
        # For Spectra
        elif self.n_tot <= 128:
            y = self.hf
            frange = [self.xf[0], self.xf[-1]]
            style = 'x'
            # print(str(self.xf))
            # print(str(self.hf))
        else:
            print('Invalid transfer fkt.')

        fig, ax = plt.subplots(1, figsize=(10, 3))
        ax.set_xlabel('f [Hz]')
        ax.set_xlim(*frange)
        ax.set_ylim(.1, 1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(self.xf, y, marker=style, linewidth=1)
        fig.tight_layout()
        return fig, ax

    # Fkt for convolution of transferfct with signal
    def convolute_f(self, sig):
        """Convolutes tf with sig.

        Convolution takes place by multiplication in frequency domain.
        Sampling rate is taken from sig
        Formula: convolution = F(self)*F(sig)

        Parameters
        ----------
        sig: Signal
            Signal which should be convolved with TransferFunction

        Returns
        -------
        Signal: Signal
            Convolved Signal"""

        h_samp = sg.resample(self.hf, len(sig.y_f))
        conv_f = h_samp*sig.y_f
        return Signal(y=ifft(conv_f), dt=self.dt)

    # Fkt to generate octave based signal
    def __get_band(self, f0, f1):
        """Extracts band rms by cutting out of spectrum

        Parameters
        ----------
        f0: Float
            Startfrequency [Hz]
        f1: float
            Endfrequency [Hz]

        Returns
        -------
        rms: float
            Calculated rms"""

        i0 = np.where(self.xf > f0-self.xf[0])[0][0]
        i1 = np.where(self.xf > f1-self.xf[0])[0][0]

        l_sum = 0
        for el in self.hf[i0:i1]:
            l_sum += np.power(el, 2)
        return np.sqrt(l_sum / (i1-i0))

    def get_octave_band(self, fact=1.):
        """Return new octave based tf.

        Parameters
        ----------
        fact: float
            Factor to divide up a octave
            fact = 1. means full octave
            fact = 1/3 means 1/3rd octave

        Returns
        -------
        TransferFunction: Transferfunction
            TF in bands"""

        x = []
        y = []
        oct_band_lst = create_band_lst(fact)

        for f in oct_band_lst:
            x.append(f[2])
            y.append(abs(self.__get_band(*f[0:2])))
        return TransferFunction(xf=np.array(x), hf=np.array(y))
