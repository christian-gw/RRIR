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
import matplotlib.pyplot as plt
from reflection_definitions import\
    Signal  # , rotate_sig_lst


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
x = Signal(par_sweep=(f1, T, f2), dt=1/fs)

print('Created sweep signal...')


# %%codecell # 3. Load wav
h_avg = []
h_tot = []
for key in files:
    f_path = path + files[key]

    # y_m, _, n_tot_m = load_lr(f_path)
    # y_raw = Signal(y=y_m, dt=1/fs)
    y_raw = Signal(path=path,
                   name=files[key])
    print('Loaded soundfile...')

    # Deconvolve exitation by division algebraic filter by fft*fft
    h2 = y_raw.impulse_response(x)
    h_tot.append(h2)

    print('Inverted with convolution filter...')

# %% codecell
# rotate_sig_lst(h_tot, fix_shift=.1)
# print('Synced signals...')

for el in zip(range(1, 5), h_tot):
    plt.plot(np.real(el[1].y), label=el[0], lw=.25)
    plt.legend()
plt.show()

# %%
