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
import sml.Ambi as ambi
# Signal  # , rotate_sig_lst()

# Sweep Parameters
f1 = 50
f2 = 5e3
fs = 48e3
T = 10

Tp = 5
n_sweep = 3
cut_start = 2
path = """C:/Users/gmeinwieserch/Desktop/20220629_Messung Schallmessraum/\
Reflektion Boden/"""
files = {1: """ZOOM0005_Tr1.WAV""",
         2: """ZOOM0005_Tr2.WAV""",
         3: """ZOOM0005_Tr3.WAV""",
         4: """ZOOM0005_Tr4.WAV"""}

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
# %% codecell
# Create Ambisonics signal

# Microphone settings
am = ambi.AmbiMic(1.47, .5)
aSig = ambi.ambiSig(h_tot, am)
print(aSig.b_format)
# %%
