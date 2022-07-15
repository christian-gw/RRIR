from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq  # , fftshift, ifft
from scipy.signal.windows import tukey, blackmanharris  # , hann
import scipy.signal as sg
import librosa as lr
import numpy as np
import os
from scipy.io import wavfile


def dbg_info(txt, verbose=False):
    """Print debug info if global VERBOSE = True"""
    if verbose:
        print(str(datetime.now().time()) + ' - ' + txt)


# Signal class
class Signal:
    """Signal Class
        Initialisation depends on used kwargs:
        - 'path' and 'name' of wav file to load
            path : string - path to the file
            name : string - name of the file
        - 'sweep_par' and 'dt' to create sweep
            sweep_par : tuple of form (start_freq, sweep_dur, end_freq)
            dt : float - time base of the signal (1/fs)
        - 'signal_lst_imp' to average several time aligned impulses
            signal_lst_imp : array - Array of form [Signal, Signal, ...]
        - 'y' and 'dt' for naive input of y signal
            y : array - np.array with Soundpressurelevels (e.g. from wav)
            dt : float - timebase of the signal (1/fs)

        Methods:
        - 'impulse_response':
            Calc and return impulse response from measured sweep and exitation
        - 'filter_y': filter signal and return filtered signal
        - 'resample': Resample signal and return resampled (no antialiasing)
        - 'cut_signal':
            Cut signal between specified times and return new signal
        - 'plot_y_t' 'plot_y_f:' Plot time or frequency signal
        - 'plot_spec_transform':
            Plot spectrogram of transformation like in method impulse_response
        -
        Attributes:
        - bool: exitation,
        - double:      dt,   T,
        - integer:  n_tot,
        - array-axis:   t,  xf,
        - array-signal: y,"""

    sig_nr = 0
    entity = []

    def __init__(self, **kwargs):
        self.y = None
        self.L_t = []
        self.y_f = None

        self.frange = [60., 96.0e3/2]
        self.dt = None
        self.T = None
        self.n_tot = None
        self.par_sweep = None

        self.axis_arrays = {'t': None,
                            'xf': None}

        # Init answer signals
        if 'path' in kwargs and 'name' in kwargs:
            path = kwargs.get('path', None)
            name = kwargs.get('name', None)

            self.dt, self.T, self.n_tot, t, xf, self.y = self.__load_data(
                os.path.join(path, name))
            self.axis_arrays['t'] = t
            self.axis_arrays['xf'] = xf

        # Init exitation signals
        elif 'par_sweep' in kwargs and 'dt' in kwargs:
            #
            self.par_sweep = kwargs.get('par_sweep', None)
            self.dt = kwargs.get('dt', None)
            self.T, self.n_tot, t, xf, self.y = self.__gen_sweep(
                self.par_sweep, self.dt)
            self.axis_arrays['t'] = t
            self.axis_arrays['xf'] = xf

        # Generate avg-signal from lst
        elif 'signal_lst_imp' in kwargs:
            lst = kwargs.get('signal_lst_imp', None)

            self.dt = lst[0].dt
            self.T = lst[0].T
            self.n_tot = lst[0].n_tot
            self.axis_arrays['t'] = lst[0].axis_arrays['t']
            self.axis_arrays['xf'] = lst[0].axis_arrays['xf']

            avg_lst_imp = np.copy(lst[0].y)

            for el in lst[1:]:
                avg_lst_imp += el.y

            self.y = avg_lst_imp/len(lst)

        # Generate signal object from one y
        elif 'y' in kwargs and 'dt' in kwargs:
            self.y = kwargs.get('y', None)
            self.dt = kwargs.get('dt')

            self.n_tot = len(self.y)
            self.T = self.dt * self.n_tot
            self.axis_arrays['t'] = np.linspace(0, self.T, self.n_tot)
            self.axis_arrays['xf'] =\
                fftfreq(self.n_tot, self.dt)[:self.n_tot//2]

        else:
            print('No valid keywords provided. See docstring of Signal class.')

        self.__fft_all()

        Signal.sig_nr += 1
        Signal.entity.append(self)

    def __load_data(self, path):
        """Load Data from file:
            .wav --> Load only answer and calculate metainformation,
            the exitation must be set mannually.
            dt, T, n_tot, t, xf, y =load_data(path)"""

        raw = lr.load(path, sr=None)  # (fs,[Data])

        # Get values
        y = np.array(raw[0])

        dt = 1/raw[1]
        n_tot = len(y)
        T = dt*n_tot

        # print(dt, n_tot, T)
        # Create time and frequency axis
        t = np.linspace(0, T, n_tot)
        xf = fftfreq(n_tot, dt)[:n_tot//2]

        return dt, T, n_tot, t, xf, y

    def __gen_sweep(self, _par_sweep, dt):
        """Generate sweep signal from parmeters:
              par_sweep = [f_start, time, f_end]
              time base"""

        T = _par_sweep[1]
        n_tot = int(T/dt)
        xf = fftfreq(n_tot, dt)[:n_tot//2]
        t = np.linspace(0, _par_sweep[1], int(_par_sweep[1]/dt))
        y = sg.chirp(t,                     # time axis
                     *_par_sweep,             # Sweeppar
                     method='logarithmic',   # Logarithmic Sweep
                     phi=90)                 # start with negative slope
        return T, n_tot, t, xf, y

    def __inv_u(self, x):
        """ x = Signal to invert --> Exitation E-Sweep u
        params = [low-frequency, high-frequency]
        Get inverse of e-sweep by reverse x and get amplitude right, so that
          its rising with 3 dB/Oct
        according to:
        Farina 2007 p4 and
        Meng; Sen; Wang; Hayes - Impulse response measurement with sine sweeps
          and amplitude modulation shemes Signal Processing and Communication
          systems 2008 - ICSPCS"""

        om1 = x.par_sweep[0]*2*np.pi            # f0 --> omega0
        om2 = x.par_sweep[2]*2*np.pi            # f1 --> omega1
        T = x.par_sweep[1]                      # Sweep duration

        # Creation Frequency modulation (exp rising magnitude)
        # For modulation, K and L see paper from Docstring
        L = T/np.log(om1/om2)
#                       K             / L
        KpL = (om1*T/np.log(om1/om2)) / L

        time = np.linspace(0, T, len(x.y))

        # modulation = [(1/(KpL * np.exp(t/L))) for t in time]

        modulator = lambda t: 1/(KpL * np.exp(t/L))
        modulation = modulator(time)

        # vec_mod = np.vectorize(modulator)
        # modulation = vec_mod(time)

        x_mod = x.y*modulation

        # reverse the modulated signal
        i = x_mod[::-1]

        return i

    def __fft_all(self):
        """Calculate a fft on all present signals"""
        if self.y is not None:
            self.y_f = fft(self.y)

    # Create new Signal
    def impulse_response(self, exitation):

        """Create new signal from measured signal containing impulse response
        created by deconvolution of 'exitation' signal"""
        i_u = self.__inv_u(exitation)
        imp = sg.convolve(self.y, i_u,    # convolution of signal with inverse
                          mode='same')
        # same means conv calculates a full (zero padding) and cuts out middle
        return Signal(y=imp, dt=self.dt)

    def filter_y(self, frange=(65.0, 22000.0)):
        """Filter Signal of range 'frange' of type list."""
        sos = sg.butter(10, frange, btype='bp',
                        analog=False, output='sos', fs=1/self.dt)

        filt = sg.sosfilt(sos, self.y)
        # self.y = filt
        return Signal(y=filt, dt=self.dt)

    def resample(self, Fs_new):
        """Resample the singal to the new sampling rate"""
        factor = Fs_new * self.dt
        n_new = int(self.n_tot * factor)

        y, t = sg.resample(self.y, n_new, self.axis_arrays['t'])

        return Signal(y=y, dt=1/Fs_new)

    def cut_signal(self, t_start, t_end, force_n=None):
        """
        Cuts Signal to a new signal between times

        Parameters
        ----------
        t_start : start time in [s]
        t_end : endtime in [s]
        force_n: Force a certain sample number

        Returns
        -------
        cutted signal
        """

        n_start = int(t_start/self.dt)
        if force_n is None:
            n_dur = int((t_end-t_start)/self.dt)
        else:
            n_dur = int(force_n)

        snap = self.y[n_start: n_start + n_dur]

        return Signal(y=snap, dt=self.dt)

    # Plotting
    def plot_y_t(self, headline=''):
        """Plot time signal."""
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(self.axis_arrays['t'], self.y, linewidth=1)
        ax.set_xlabel('Time [s]')
        fig.suptitle(headline)
        # plt.show()
        # If not in, no show when exec, if in, no work on axis return
        return fig, ax

    def plot_y_f(self, headline=''):
        """Plot the magnitude spectrum of the signal."""
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(self.axis_arrays['xf'], 2.0/self.n_tot *
                np.abs(self.y_f[0:self.n_tot//2]), linewidth=.25)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(50, 22e3)
        ax.set_xscale('log')
        fig.suptitle(headline)
        plt.show()
        return fig, ax

    def plot_spec(self, n_win=2048):
        """ Create figure with spectrogram plot to visualise the sweep."""

        # Preset and debug
        f_range = [50, 1/(3*self.dt)]

        # Spectrogram
        fc, tc, Spec = sg.spectrogram(self.y.real,           # signal
                                      fs=1/self.dt,          # Samplingrate
            # tuckey is rising cos, uniform, falling cos window
                                      window=('tukey', .25),
            # alpha factor specifies the proportion wich is cos
                                      nperseg=n_win,         # Blocksize
            # Overlap Samples overlap (default nperseg//8)
                                      noverlap=1*n_win//8)
        # Alpha of tuckey.25 and nperseg//8 means, that the
        # Overlap is in the rising/faling cos range of the win

        # Generate figure
        fig, ax = plt.subplots(1, figsize=(10, 6))

        # , vmin = -280, vmax = -80) # Use this to set color bar limits
        ax.pcolormesh(tc, fc, to_db(Spec), shading='auto')
        ax.set_xlabel('t [s]')
        ax.set_ylim(f_range)
        ax.set_ylabel('f [Hz]')
        ax.set_yscale('log')

        fig.tight_layout()
        plt.show()
        return fig, ax

    def level_time(self, T=.035):
        """ Calculates the sound level with a smoothing window of length T
            Returns Level array according to DIN EN 61672-1"""

        # Square signal
        x = self.y
        x2 = np.power(x, 2)
        dt = self.dt

        # Design and apply Filter
        # Consider reviewing DIN p.8
        # "Filter with real pole at -1/T"

        filt = sg.butter(1,
                         1/T,
                         output='sos',
                         fs=1/dt)
        x_filt = sg.sosfilt(filt, x2)

        # Calculate Level array
        L = [10*np.log10(el/((2e-5)**2)*T) for el in x_filt]
        self.L_t = L

    def write_wav(self, name, F_samp=48e3):
        """
        Writes y value of signal to wave file after sampling to Samplingrate.
        Parameters
        ----------
        F_samp : Samplingrate
        name : name of file
        """
        y = Signal(y=np.copy(np.trim_zeros(self.y)), dt=self.dt)
        y.resample(F_samp)
        np.trim_zeros(y.y)
        # lr.output.write_wav(name, np.float32(y.y), int(F_samp))
        wavfile.write(name, int(F_samp), np.float32(y.y))

    def correct_refl_component(self, direct, t_start, t_dur):
        """
        Subtracts section of a cor_sig (direct Signal) Signal
        after searching for the pos with greatest overlap.

        Parameters
        ----------
        cor_sig : Signal containing correction
          (e.g. direct sound and or ground reflection)
        t_start : start of the direct sound
        t_dur : duration of the direct sound

        Changes:
        -------
        int
            Signal to corrected Signal
        """
        direct, _ = appl_win(direct, t_start, t_dur)
        corell = sg.correlate(direct.y, self.y)

        corell = np.roll(corell, int(len(corell)/2))
        pos = np.argmax(corell)
        fak = max(self.y)/max(direct.y)

        direct_sync = Signal(y=fak * np.roll(self.y, -int(len(corell)/2)-pos),
                             dt=direct.dt)
        direct_sync.plot_y_t()
        return direct_sync, Signal(y=self.y - direct_sync.y,
                                   dt=direct_sync.dt)


def rotate_sig_lst(sig_lst,
                   cor_range_pre=0,
                   start_cor_reference=0,
                   start_cor_reflection=0,
                   fix_shift=0):
    """
    Rotate a signal list so all impulses are sync and at start of file.
    Number of rotation samples is determined by correlation between
    the most highest and second highest peak +-MARGIN samples by default.
    Its possible to set the correlation range (for all signals)
    and the start of it for reference (first) and all other signals.
    Its also possible to shift all signals by a specified time,
    in order to make sure the signal is not cut by the file borders.

    Parameters
    ----------
    sig_lst : List of signal objects to rotate
    cor_range_pre : float, optional - time [s] over wich to correlate
        The default is 0. If not set otherwise
        If set otherwise it changes size of correlation window
    start_cor_reference : float, optional - time [s] to start corr on ref-sig
        The default is 0. That means Corr range is determined by max search
        If set otherwise given start of cor range is used.
    start_cor_reflection : float, optional - time to start corr on shift sig
        The default is 0. That means Corr range is determidned by max search
        If set otherwise given start is used
    fix_shift: float, optional - time to shift all signals to the right
        The default is 0. No shift is applied.
        Use if you want to avoid a impulse beeing split in parts by file end.

    Returns
    -------
    None. But changes sig_lst by rotating elements.

    """

    MARGIN = 200  # cor range enhance - see next comment
    if cor_range_pre == 0:
        # If no cor_range is specified: find it:
        # Cor between
        #   first peak1 - 200 samp and (that means highest peak)
        #   (first peak2 after first peak1 + 400) + 600

        start_cor_1 = np.argmax(sig_lst[0].y) - MARGIN
        end_cor_1 = start_cor_1 + \
            np.argmax(sig_lst[0].y[start_cor_1 + 2 * MARGIN:]) \
            + 3 * MARGIN
        # DEBUG
        print('start_cor_1 = ' + str(start_cor_1))
        print('end_cor_1 = ' + str(end_cor_1))
    else:
        # start_cor_1 = 0
        # end_cor_1 = sig_lst[0].n_tot-1
        start_cor_1 = int(start_cor_reference/sig_lst[0].dt)
        end_cor_1 = start_cor_1 \
            + int(cor_range_pre/sig_lst[0].dt)

    i = 1                                  # dont work the first signal
    for el in sig_lst[1:]:
        # find max position of cor sig relative to first sig
        if cor_range_pre == 0:
            # if range is not preset find it for every sig
            start_cor_2 = np.argmax(el.y) - MARGIN
            end_cor_2 = start_cor_2\
                + np.argmax(el.y[start_cor_2 + 2*MARGIN:]) \
                + 3 * MARGIN
        else:
            # else use preset range
            start_cor_2 = int(start_cor_reflection/el.dt)
            end_cor_2 = start_cor_2 + (end_cor_1 - start_cor_1)

        # Get shifter by corelation in determined range
        shifter = (start_cor_2-start_cor_1) + end_cor_2-start_cor_2 - \
            np.argmax(sg.correlate(sig_lst[0].y[start_cor_1:end_cor_1],
                                   el.y[start_cor_2:end_cor_2],
                                   mode='full'))

        # shift every signal by 'shifter' values
        # np.roll(sig_lst[i] and [0]) makes sure, that all imp resp. start
        # at the same time and no reflection tail is split by fileborders

        fix_shift_int = int(fix_shift/sig_lst[0].dt)
        sig_lst[i].y = np.roll(el.y, -shifter-start_cor_1+fix_shift_int)
        i += 1
    sig_lst[0].y = np.roll(sig_lst[0].y, - start_cor_1+fix_shift_int)

    # return sig_lst


def to_db(x):
    """Calculates the value of xa as an array or an scalar:\n
    Formula: L = 20*log10(p/p0)"""

    p0 = 2e-5                     # Reference
    xa = np.asarray(x)            # Cast to array so its polymorph

    return 20*np.log10(xa/p0)     # Return according to furmla


# TODO: Why is this no method of Sigal class?
def appl_win(sig, t_start, t_len, form='norm'):
    """Define and apply a Window of form
        sym = tuckey with \alpha = .03 or
        Norm = norm window)
    on pos t_start with length t_len"""

    if form == 'sym':
        sta = int(t_start/sig.dt)
        N = int(t_len/sig.dt)
        steep = .03
        win = tukey(N, steep)
        n_lead = int(.5*steep)

        y = np.zeros(sig.n_tot)
        w = np.zeros(sig.n_tot)

        y[sta-n_lead:sta+n_lead+N] = win
        w[sta-n_lead:sta+n_lead+N] = win

        y *= sig.y

    elif form == 'norm':
        sta = int((t_start)/sig.dt)
        n_lead = int(.5e-3/sig.dt)
        n_steady = int(.7*t_len/sig.dt)
        n_tail = int(.3*t_len/sig.dt)

        win = np.empty(0)

        win = np.concatenate((win, blackmanharris(2*n_lead)[:n_lead]))
        win = np.concatenate((win, np.ones(n_steady)))
        win = np.concatenate((win, blackmanharris(2*n_tail)[n_tail:]))

        y = np.zeros(sig.n_tot)
        w = np.zeros(sig.n_tot)

        y[sta-n_lead:sta+n_steady+n_tail] = win
        w[sta-n_lead:sta+n_steady+n_tail] = win

        y *= sig.y

    else:
        print("Invalid window shape.")

    return Signal(y=y, dt=sig.dt), w


def create_band_lst(fact=1):
    """Returns list of bands for octave or deca spectrum generation"""

    f_centre = [round(10**3 * 2**(exp*fact)) for exp in range(round(-6/fact), round((5-1)/fact)+1)]

    fd = 2**(1/2*fact)
    f_low = [round(centre/fd, 1) for centre in f_centre]
    f_up = [round(centre*fd, 1) for centre in f_centre]

    return [list(i) for i in zip(*[f_low, f_up, f_centre])]


def pre_process_one_measurement(PATH, sig_name, F_up, u):
    """ Performs Time synchronous averaging on impulse domain of  singal list.

    Parameters
    ----------
    PATH : Path to the wave Files to work.
    sig_name : List of filenames
    F_up : Frequency to upsample to.
    u : Signal object of exitation signal.

    Returns
    -------
    avg_sig : Signal object of the time synchron averaged signals.

    """
    sig_raw_lst = []
    sig_imp_lst = []

    for el in sig_name:
        # Load signal from path to subsignallist in signallist
        sig_raw_lst.append(Signal(path=PATH, name=el))
        sig_raw_lst[-1].resample(F_up)

        sig_imp_lst.append(sig_raw_lst[-1].impulse_response(u))

    rotate_sig_lst(sig_imp_lst)

    avg_sig = Signal(signal_lst_imp=sig_imp_lst)
    return avg_sig
