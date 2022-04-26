# %%[markdown]
# Short learning script for inversion of e-sweep
# # 2. codecell
#  - Create and plot Sweep according to Farina_2004
#  - Create and plot algebraic inverse sweep
#  - Create and plot inverse by ifft(1/fft(x))
#
#  - Make sure, that the fft is _not_ circullar
#    by forcing a much greater length of fft input
#
# # 3. codecell
#   Load exemplaric wave file
#
# # 4. codecell
#   Comparison between conv of fft and algebraic inv
#
# # 5. codecell
#   Synchronisation and average of the loaded signals
#   impulse responses.

# %%codecell # 1. Import and fun definition
from __future__ import division
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from scipy.signal.windows import tukey
import librosa as lr
from reflection_definitions import\
    Signal, rotate_sig_lst


def filter(x_b, lo, hi, fs):
    # replace by filter_y(self, frange=())
    """Use 5.O Chebishef filter on signal
    x_b Unfiltered signal
    l   Lower Frequency
    h   Higher Frequency
    fs  Samplingrate
    Returns sampled signal
    """
    cheb = sig.cheby1(5,
                      1,
                      (lo, hi),
                      btype='bandpass',
                      output='sos',
                      fs=fs)
    return sig.sosfiltfilt(cheb, x_b)


def plot_signal(x, xf, freq, title=''):
    # subs by plot_y_t
    """Plot Timesignal, Mag and Ang spectrum
    x       Timesignal (Will be plotted w.o. time axis.)
    xf      Complex complete spectrum
    freq    Frequencyaxis for spectrum
    title   Title to set on top
    Returns figure and axis object.
    """
    db = lambda x: 20 * np.log10(abs(x)/max(abs(x)))
    ang = lambda x: np.angle(x)

    fig, ax = plt.subplots(3, figsize=(10, 6))
    ax[0].plot(np.abs(x), label='Sin-Sweep')
    ax[1].plot(freq, db(abs(xf)), linewidth=.25)
    ax[2].plot(freq, ang(xf), linewidth=.25)
    [a.set_xlim(50) for a in ax[1:]]
    [a.set_xscale('log') for a in ax[1:]]
    fig.suptitle(title, size=30)
    fig.tight_layout()
    plt.draw()
    return fig, ax


def fft_calc(x, length, win=False):
    # subs by signal.y_f, signal.axis_array['x_f']
    """Calc FFT for plot_signal fkt
    x       Signal to work with
    length  length to force on FFT
    win     Bool-Use tuckey win
    Returns input, spectrum, freq_axis"""

    if win is False:
        win = np.ones(len(x))
    else:
        win = tukey(len(x), alpha=.05, sym=False)

    x = x*win

    xf = fft(x, length)
    freq = fftfreq(n=len(xf), d=1/fs)
    return x, xf, freq


def load_lr(path):
    """Load data with librosa"""
    # Not subsable bc of 24bit
    raw = lr.load(path, sr=96000)
    return raw[0], raw[1], len(raw[0])


# Sweep Parameters
f1 = 50
f2 = 5e3
fs = 96e3
T = 10

Tp = 5
n_sweep = 2
cut_start = 7.5
path = """C:/Users/gmeinwieserch/Desktop/20220420_Messung im Gang/"""
files = {1: """Gang_1.WAV""",
         2: """Gang_2.WAV""",
         3: """Gang_3.WAV""",
         4: """Gang_4.WAV"""}

# %%codecell # 2. Sweep and inv creation

t = np.arange(0, T*fs)/fs
R = np.log(f2/f1)

fft_length = 2*T*fs

# Sweep generation
x = Signal(par_sweep=(f1, T, f2), dt = 1/fs)

print('Created sweep signal...')


# %%codecell # 3. Load wav
h_avg = []
h_tot = []
for key in files:
    f_path = path + files[key]

    y_m, _, n_tot_m = load_lr(f_path)
    y_raw = Signal(y=y_m, dt=1/fs)
    print('Loaded soundfile...')


    # Deconvolve exitation by division algebraic filter by fft*fft
    h2 = y_raw.impulse_response(x)
    h_tot.append(h2)

    print('Inverted with convolution filter...')

rotate_sig_lst(h_tot, fix_shift=int(2e3))
print('Synced signals...')

for el in zip(range(1, 5), h_tot):
    plt.plot(np.real(el[1].y), label=el[0], lw=.25)
    # plt.legend()
plt.show()
# %%
