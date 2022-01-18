# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:08:55 2021

@author: gmeinwieserch
Helper functions for evaluation of RIRs from Wave files
"""

from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq #, fftshift
from scipy.signal.windows import tukey, blackmanharris #, hann
import scipy.signal as sg
from scipy.io import wavfile
import numpy as np


# Signal class
class Signal:
    """Signal Class
        Initialisation depends on used kwargs:
        - 'path' and 'name' of wav file to load
        - 'sweep_par' and 'dt' to create sweep
        - 'signal_lst_imp' to average several time aligned impulses
        - 'y' and 'dt' for naive input of y signal

        Methods:
        - 'impulse_response' calculate and return impulse response from measured sweep and exitation
        - 'filter_y' filter signal and return filtered signal
        - 'resample' resample signal and return resampled (no antialiasing)
        - 'cut_signal' cut signal between specified times and return new signal
        - 'plot_y_t' 'plot_y_f' plot time or frequency signal
        - 'plot_spec_transform' plot spectrogram of transformation like in method impulse_response
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
        self.L_t =[]
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
                path + name)
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
            self.axis_arrays['xf'] = fftfreq(self.n_tot, self.dt)[:self.n_tot//2]

        else:
            print('No valid keywords provided. - See docstring.')

        self.__fft_all()

        Signal.sig_nr += 1
        Signal.entity.append(self)

    def __load_data(self, path):
        """Load Data from file:
            .wav --> Load only answer and calculate metainformation the exitation was set mannually.
            dt, T, n_tot, t, xf, y =load_data(path)"""

        raw = wavfile.read(path)  # (fs,[Data])

        # Get values
        y = np.array(raw[1])

        dt = 1/raw[0]
        n_tot = len(y)
        T = dt*n_tot

        #print(dt, n_tot, T)
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
#                       K              /  L
        KpL = (om1*T/np.log(om1/om2))  /  L

        time = np.linspace(0, T, len(x.y))

        #modulation = [(1/(KpL * np.exp(t/L))) for t in time]

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
                          mode = 'same')
        # same means conv calculates a full (zero padding) and cuts out middle
        return Signal(y = imp, dt = self.dt)


    def filter_y(self, frange = (65.0, 22000.0)):
        """Filter Signal of range 'frange' of type list."""
        sos = sg.butter(10, frange, btype='bp',
                        analog=False, output='sos', fs=1/self.dt)

        filt = sg.sosfilt(sos, self.y)
        #self.y = filt
        return Signal(y=filt, dt=self.dt)


    def resample(self, Fs_new):
        """Resample the singal to the new sampling rate"""
        factor = Fs_new * self.dt
        n_new = int(self.n_tot * factor)

        y, t = sg.resample(self.y, n_new, self.axis_arrays['t'])

        return Signal(y=y, dt=1/Fs_new)

    def cut_signal(self, t_start, t_end, force_n = None):
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

        return Signal(y = snap, dt = self.dt)

    # Plotting
    def plot_y_t(self, headline = ''):
        """Plot time signal."""
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(self.axis_arrays['t'], self.y, linewidth=.25)
        ax.set_xlabel('Time [s]')
        fig.suptitle(headline)
        #plt.show()  # If not in, no show when exec, if in, no work on axis return
        return fig, ax

    def plot_y_f(self, headline = ''):
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
        fc, tc, Spec = sg.spectrogram(self.y.real,             # signal
            fs=1/self.dt,          # Samplingrate
            # tuckey is rising cos, uniform, falling cos window
            window=('tukey', .25),
            # alpha factor specifies the proportion wich is cos
            nperseg=n_win,         # Blocksize
            noverlap=1*n_win//8)    # Overlap Samples overlap (default nperseg//8)
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
        wavfile.write(name, int(F_samp),np.float32(y.y))

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
        direct,_ = appl_win(direct, t_start, t_dur)
        corell = sg.correlate(direct.y, self.y)

        corell = np.roll(corell,int(len(corell)/2))
        pos = np.argmax(corell)
        fak = max(self.y)/max(direct.y)

        direct_sync = Signal(y = fak * np.roll(self.y, -int(len(corell)/2)-pos), dt = direct.dt)
        direct_sync.plot_y_t()
        return direct_sync, Signal(y = self.y - direct_sync.y, dt = direct_sync.dt)

class TransferFunction:
    """Transfer Fkt Class
        Initialisation depends on used kwargs:
        - 'incoming_sig' and 'reflected_sig' tf of those two signal obj
        - 'x' and 'hf' to create a tf obj from a external tf
        - 'signal', 'in_win' and 're_win' (both [start, duration]) tf
          from two windowed sections of signal
        Methods:
        - 'convolute_f' returns a convolution of a signal and the tf with length of signal
        - '__get_band' returns RMS value within a specified band
        - 'get_octave_band' returns tf in octave bands
        - 'plot_ht', 'plot_hf' and plot_oct plot time or frequency representation of tf
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
            frange = [self.xf[0],self.xf[-1]]
            style = 'x'
            print(str(self.xf))
            print(str(self.hf))
        else:
            print('Invalid transfer fkt.')

        fig, ax = plt.subplots(1, figsize=(10, 3))
        ax.plot(self.xf, y, marker = style, linewidth=.25)
        ax.set_xlabel('f [Hz]')
        ax.set_xlim(*frange)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig, ax

    # Fkt for convolution of transferfct with signal
    def convolute_f(self, sig):
        """Convolutes two signals in freq domain (multiplication) after resample.)"""

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

    def get_octave_band(self, fact = 1.):
        """Return new octave based tf."""
        x = []
        y = []
        oct_band_lst =create_band_lst(fact)

        for f in oct_band_lst:
            x.append(f[2])
            y.append(abs(self.__get_band(*f[0:2])))
        return TransferFunction(xf = np.array(x), hf = np.array(y))

class Measurement:
    """Class for handling a whole measurement with multiple Measurement points
         containing one signal each.
       If there are multiple signals (should be), averaging before creation of mp is advised.
       Attributes:
         self.m_name   Name of the Measurement (Meta data class is planned)
         self.d_mic    Distance
         self.d_probe  Distance
         self.mp_lst   List containing all measurementpoints with signals
         self.n_mp     Over all number of mps
       Methods:
         self.create_mp()     creates a measurement point
         self.del_mp()        deletes a measurement point by number
         self.plot_overview() plots an overview of all measurement points
                              relative to source and probe
         """

    def __init__(self, name, d_mic, d_probe):
        self.m_name = name
        self.d_mic = d_mic
        self.d_probe = d_probe
        self.mp_lst = []
        self.n_mp = 0
        self.average = []

    def create_mp(self, number, _signal, pos):
        """Create measurement point object with specific number and _signal
        object for mp Position pos [x,y]"""

        self.mp_lst.append(MeasurementPoint(
            number, (self.d_mic, self.d_probe), _signal, pos))
        self.n_mp += 1

    def del_mp(self, number):
        """Delete measurement point object specified by number.
           No reaction if not present."""

        for i, el in enumerate(self.mp_lst):
            if el.no == number:
                del self.mp_lst[i]
                #return 0

    def plot_overview(self):
        """Plots an overview of Source, Probe and Mics. All units in [m]"""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = []
        y = []
        n = []
        z = []
        # bet = []
        for el in self.mp_lst:
            pos = el.pos
            x.append(pos['x'])
            y.append(pos['y'])
            n.append(pos['no'])
            z.append(self.d_mic)
            # bet.append(str(el.beta_in_deg()))

        ax.scatter(x, y, z)
        ax.scatter(0, 0, self.d_probe)
        ax.set_zlim(0, self.d_probe)
        ax.set_xlabel('Plane Coord. [m]')
        ax.set_ylabel('Plane Coord. [m]')
        ax.set_zlabel('Distance Coord. [m]')

        for i, n in enumerate(n):
            ax.text(x[i], y[i], z[i], str(n))

        ax.text(0, 0, self.d_probe, 'Source')
        ax.text(0, 0, 0, 'Probe')

        return fig, ax

    def average_mp(self):
        """
        Averages all mps
        If required make sure to apply Corrections on a mp level
        Parameters
        ----------
        Requires full mp_lst

        Returns
        -------
        Averaged tf
        """
        l = len(self.mp_lst)
        sum = np.zeros(len(self.mp_lst[0].tf.hf))

        for el in self.mp_lst:
            sum += el.tf.hf
        average = sum / l
        self.average = TransferFunction(xf = self.mp_lst[0].tf.xf,
                                        hf = average)

class MeasurementPoint:
    """Class for one measuremnet point it should be included in the 'measurement'-class.
       Norm suggests 3x3 measurement grid with distances .4 m, numbered like a phone.

       Attributes:
           self.no         (init)        number of measurement point
           self.x, self.y  (init as pos) position relative to probe normal through source
           self.tf         (init)        transferfct object from mp (if multiple average before)
           self.d_mic      (init)        distance between source and microphone
           self.d_probe    (init)        distance between microphone and probe
           self.beta       (__geo)       reflection angle
           self.c_geo      (__c_geo)     correction coefficient for sound power
                                         distribution

        Public methods:
        self.calc_c_geo()  Calculates self.c_geo, calls self.__c_geo and self.__geo
        self.calc_c_dir()  Currently not implemented bc its assumed, that
                           omnisource has no directivity
    """

    def __init__(self, number, distances, transfer_function, pos):
        self.tf = transfer_function

        self.distances = {'mic': distances[0],
                           'probe': distances[1]}

        self.corrections = {'c_geo': 1.,
                            'c_dir': 1.,
                            'c_gain': 1.,
                            'applied': False}

        self.pos = {'x': pos[0],
                    'y': pos[1],
                    'beta': 2*np.pi,
                    'no': number}

    def __geo_norm(self):
        """Performs several geometrical calculations concerning the measurement position.
           works with:
               self.x, self.y, self.d_mic, self.d_probe
           returns:
               travel_to_r    traveled distance from source to reflection point
               travel_from_r  traveled distance from reflection point to mic
               alpha          angle between probe normal and source-mic line
               beta           angle between probe normal and reflected soundray
           sets:
               self.beta      angle between probe normal and reflected soundray"""

        x = self.pos['x'] # x-pos on grid
        y = self.pos['y'] # y-pos on grid

        d_mic = self.distances['mic']     # source - mic
        d_probe = self.distances['probe'] # mic - probe

        r_xy = np.sqrt(     x**2 + y**2)     # Distance in mic plane
        r_xyz = np.sqrt(d_mic**2 + r_xy**2)  # Direct distance source -mic

        # Traveled distance to and from reflection point
        r_to_ref_xy   = r_xy*(d_mic + d_probe)/(2*d_probe + d_mic)
        travel_to_r   = np.sqrt((d_mic + d_probe)**2 + r_to_ref_xy**2)
        travel_from_r = np.sqrt(d_probe**2 + (r_xy - r_to_ref_xy)**2)

        alpha = np.arctan(r_xy/d_mic)
        self.pos['beta'] = np.arctan((d_mic + d_probe)/r_to_ref_xy)

        return travel_to_r, travel_from_r, alpha, self.pos['beta'], r_xyz

    def calc_c_geo(self):
        """Calculates c_geo, uses __geo_norm()"""

        # Get geometry
        travel_to_r, travel_from_r, _, _, r_xyz = self.__geo_norm()

        # Set c_geo
        print('d_ik\t '
              + str(round(r_xyz,3))
              + ';\t d_rk\t ' + str(round(travel_to_r + travel_from_r,3))
              + ';\t c_geo\t ' + str(round(((travel_to_r + travel_from_r) / r_xyz)**2,3)))

        self.corrections['c_geo'] = ((travel_to_r + travel_from_r) / r_xyz)**2

    def apply_c(self):
        """Applys all correction values to the transfer fkt"""
        mul = 1
        for el in self.corrections:
            mul *= el

        if abs(mul-1) < .02: # If corrections in .98-1.02
            self.calc_c_geo()
            self.calc_c_dir()

        if not self.corrections['applied']:
            self.tf.hf = np.copy(self.tf.hf) \
                                 * self.corrections['c_geo']  \
                                 * self.corrections['c_dir'] \
                                 * self.corrections['c_gain']
            self.corrections['applied'] = True

    def calc_c_dir(self):#, signal_direct, signal_ref):
        """Currently not implemented - placeholder """
        _ = self.pos['x']  # bc of code evaluation reasons
        return 0

    def beta_in_deg(self):
        """Returns reflection angle beta in degree"""
        return 180/np.pi*self.pos['beta']


def rotate_sig_lst(sig_lst, cor_range_pre = 0, start_cor_reference=0, start_cor_reflection = 0):
    """
    Rotate a signal list relative to the first element.
    Number of rotation samples is determined by correlation between
    highest and following peak +-100 samp.

    Parameters
    ----------
    sig_lst : List of signal objects to rotate
    cor_range_pre : TYPE, optional - time over wich to correlate
        DESCRIPTION. The default is 0. If not set otherwise
        If set otherwise
    start_cor_reference : TYPE, optional - time to start correlation on reference
        DESCRIPTION. The default is 0. If not set otherwise
        If set otherwise
    start_cor_reflection : TYPE, optional - time to start correlation on reflection
        DESCRIPTION. The default is 0. If not set otherwise
        If set otherwise

    Returns
    -------
    None. But changes sig_lst.

    """
# TODO: Realy refactor and rethinkt this function
    MARGIN = 200
    if cor_range_pre == 0:
        # If no cor_range is specified: find it:
        # Cor between
        #   first peak1 - 200 samp and (that means highest peak)
        #   (first peak2 after first peak1 + 400) + 600

        start_cor_1 = np.argmax(sig_lst[0].y) - MARGIN
        end_cor_1 = start_cor_1 + \
            np.argmax(sig_lst[0].y[start_cor_1 + 2 * MARGIN:]) \
                + 3 * MARGIN
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
        sig_lst[i].y = np.roll(el.y, -shifter-start_cor_1)
        i += 1
    sig_lst[0].y = np.roll(sig_lst[0].y, - start_cor_1)

    #return sig_lst


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
    f_up = [round(centre*fd,1) for centre in f_centre]

    return [list(i) for i in zip(*[f_low, f_up, f_centre])]

def dbg_info(txt, verbose = False):
    """Print debug info if global VERBOSE = True"""
    if verbose:
        print(str(datetime.now().time()) + ' - ' + txt)

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
        sig_raw_lst.append(Signal(path = PATH, name = el))
        sig_raw_lst[-1].resample(F_up)

        sig_imp_lst.append(sig_raw_lst[-1].impulse_response(u))

    rotate_sig_lst(sig_imp_lst)

    avg_sig = Signal(signal_lst_imp = sig_imp_lst)
    return avg_sig


if __name__ == "__main__":
    TARGET_DIR = "C:/Users/gmeinwieserch/Desktop/FF_Reflektionen/Messung5/"
    sig_3_name = 'm1-3.wav'

    F_up = 500e3
    par_sweep = (22.0, 1., 2.2e+04)

    u = Signal(par_sweep = par_sweep, dt = 1/F_up)

    sig_345 = (Signal(path = TARGET_DIR, name = sig_3_name))

    sig_345.resample(500e3)

    sig_345_imp = (sig_345.impulse_response(u))

    if True:
        print('Generation of exitation:')
        # %timeit for each
        Signal(par_sweep = par_sweep, dt = 1/F_up)

        print('Load of wav file:')
        Signal(path = TARGET_DIR, name = sig_3_name)

        print('Resample:')
        sig_345.resample(F_up)

        print('Create Impulse:')
        sig_345.impulse_response(u)

        sig_345.plot_y_t()
