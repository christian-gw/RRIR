# %%[markdown]
# Short learning script for inversion of e-sweep
# # 2. codecell
#  - Create and plot Sweep according to Farina_2004
#
# # 3. codecell
#   Load exemplaric wave file
#   Invert signal by deconvolution
#
# # 4. codecell

# %%codecell # 1. Import and fun definition
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sml.Signal as sg
# Signal  # , rotate_sig_lst()

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
# Sweep generation
x = sg.Signal(par_sweep=(f1, T, f2), dt=1/fs)

print('Created sweep signal...')


# %%codecell # 3. Load wav
h_avg = []
h_tot = []
for key in files:
    f_path = path + files[key]

    y_raw = sg.Signal(path=path,
                      name=files[key])
    print('Loaded soundfile...')

    # Deconvolve exitation deconvolution package
    h2 = y_raw.impulse_response(x)
    h_tot.append(h2)

    print('Inverted with convolution ...')

# %% codecell
# Rotation not apropriate, bc ambisonics need shift
# sg.rotate_sig_lst(h_tot, fix_shift=.1)
# print('Synced signals...')

for el in zip(range(1, 5), h_tot):
    plt.plot(np.real(el[1].y), label=el[0], lw=.25)
    plt.legend()
    # el[1].write_wav(name='imp' + str(el[0]) + '.wav',
    #                 F_samp=int(96e3))
plt.show()
# %%
