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
from scipy.io import wavfile
from scipy.signal.windows import tukey
import librosa as lr


def filter(x_b, lo, hi, fs):
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
    raw = lr.load(path, sr=96000)
    return raw[0], len(raw[0])


def load_data(path):
    """Load Data from file:
        .wav --> Load only answer and calculate metainformation the
        exitation was set mannually.
        dt, T, n_tot, t, xf, y =load_data(path)"""

    raw = wavfile.read(path)  # (fs,[Data])

    # Get values
    y = np.array(raw[1])

    dt = 1/raw[0]
    n_tot = len(y)
    T = dt*n_tot

    # print(dt, n_tot, T)
    # Create time and frequency axis
    t = np.linspace(0, T, n_tot)
    xf = fftfreq(n_tot, dt)[:n_tot//2]

    return dt, T, n_tot, t, xf, y


# %%codecell # 2. Sweep and inv creation
# Sweep Parameters
f1 = 50
f2 = 5e3
fs = 96e3
T = 10
Tp = 5
n_sweep = 2
cut_start = 7.5

t = np.arange(0, T*fs)/fs
R = np.log(f2/f1)

fft_length = 2*T*fs

# Sweep generation
x = np.sin((2*np.pi*f1*T/R)*(np.exp(t*R/T)-1))
# fig, ax = plot_signal(*fft_calc(x, len(x)),
#                       'e-sweep')
print('Created sweep signal...')

# Algebraic sweep inversion
k = np.exp(t*R/T)
f = x[::-1]/k
# f = filter(f, f1*.9, f2*1.1, fs)

# fig, ax = plot_signal(*fft_calc(f, len(f)),
#                       'Algebraic inverse')
print('Inverted signal algebraically and nummerically...')

# %%codecell # 3. Load wav
path = """C:\\Users\\gmeinwieserch\\Desktop\\20220420_Messung im Gang\\Gang_1.WAV"""

y_m, n_tot_m = load_lr(path)

# fig, ax = plot_signal(*fft_calc(y_m, n_tot_m*2), 'Measured signal')
print('Loaded exemplaric soundfile...')

# %%codecell # 4. Comparison between conv of fft and algebraic inv
# Deconvolve exitation by division by FFT(exitation)
complete_N = int(n_tot_m+fft_length)

# Deconvolve exitation by division algebraic filter by fft*fft
h2 = ifft(fft(y_m, complete_N) * fft(f, complete_N), complete_N)

# fig, ax = plot_signal(*fft_calc(h2, len(h2)),
#                       'h - measurement*fft(algebraic inv)')
# # Plot timesignal logarithmic
# ax[0].clear()
# ax[0].plot(20*np.log10(np.abs(h2)/max(np.abs(h2))), linewidth=.25)
# ax[0].set_ylim(-100, 0)
print('Inverted with nummeric filter...')

# %%codecell # 5. Sync and Average the impulse responses of algebraic inv
h = []
delta = []
span = int((Tp + T)*fs)        # Number of samples for one sweep + silence
hred = h2[int(cut_start*fs):]  # Impulse - Number of samples to cut from begin

for i in range(n_sweep):
    h.append(np.real(hred[i*span:(i+1)*span]))

    cor = sig.correlate(h[0], h[i])  # Sync corrections
    delta.append(np.argmax(cor))

    # fak = max(h[0])/max(h[i])      # Amplitude corrections
    fak = 1

    # Syncing with np.take(... 'wrap') instead of index to assure circularity
    h[i] = fak * np.take(h[i],
                         range(delta[0]-delta[i],
                               delta[0]-delta[i]+len(h[i])),
                         mode='wrap')
h_avg = np.average(h, axis=0)        # Averaging of synced signals
print('Synced and averaged the sweeps...')

# plt.plot(h[0], lw=.25)
# plt.plot(h[1], lw=.25)
# # plt.plot(h[2][0:], lw=.25)
# plt.xlim(.6875e6, .6876e6)

# fig, ax = plt.subplots()
# [ax.plot(hi, lw=.25) for hi in h]

fig, ax = plot_signal(h_avg,
                      *fft_calc(h_avg, len(h_avg))[1:],
                      'Averaged Impulse Response')

plt.show()
# wavfile.write('Testimpulse.wav', int(48e3), np.float32(np.abs(h2)))
# %%
