# %%[markdown]
# Script for processing of reflection measurement analysis
# Measurement was performed in nonreverbant room with ambisonics mic
# 
# 1. Imports and Settings
# 2. Create exitation sweep
# 3. Load measurement
#    Perform transformation to impulse (Farina 2004)
#    Plot and/or safe
# 4. Calculate averaged Transferfunktion using Adrienne windowing
# 5. Create Ambisonics Signal

# %% Import and Settings
# 1. Import Settings
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import sml.Signal as sg
import sml.Transfer_function as tf
import sml.Ambi as ambi
# Signal  # , rotate_sig_lst()

get_ipython().run_line_magic('matplotlib', 'qt')
# get_ipython().run_line_magic('matplotlib', 'inline')

# Sweep Parameters
f1 = 50
f2 = 5e3
fs = 48e3
T = 10

Tp = 5
n_sweep = 3
cut_start = 2
# path = """C:/Users/gmeinwieserch/Desktop/20220629_Messung Schallmessraum/\
# Reflektion Boden/"""
# files = {1: """ZOOM0005_Tr1.WAV""",
#          2: """ZOOM0005_Tr2.WAV""",
#          3: """ZOOM0005_Tr3.WAV""",
#          4: """ZOOM0005_Tr4.WAV"""}

path = """C:/Users/gmeinwieserch/Desktop/20220629_Messung Schallmessraum/\
Reflektion Element/"""
files = {1: """ZOOM0006_Tr1.WAV""",
         2: """ZOOM0006_Tr2.WAV""",
         3: """ZOOM0006_Tr3.WAV""",
         4: """ZOOM0006_Tr4.WAV"""}

# %% Load and Impulse
# 2. Sweep and inv creation
x = sg.Signal(par_sweep=(f1, T, f2), dt=1/fs)
print('Created sweep signal...')

# # 3.1 Load wav
h_tot = []
for key in files:
    f_path = path + files[key]

    y_raw = sg.Signal(path=path,
                      name=files[key])
    print('Loaded soundfile...')
#   # 3.2 Transform to impulse
    # Deconvolve exitation deconvolution package
    h2 = y_raw.impulse_response(x)
    h_tot.append(h2)

    print('Inverted with convolution ...')

# %% Plotting and/or Saving
# 3.3 Plot and/or save Impulses

for el in zip(range(1, 5), h_tot):
    plt.plot(el[1].axis_arrays['t'],
             np.real(el[1].y),
             label=el[0], lw=.25)
    plt.legend()
    # If Imp should be Saved:
    # el[1].write_wav(name='imp' + str(el[0]) + '.wav',
    #                 F_samp=int(96e3))
plt.show()

# %% Calculate Ambisonics
# 4. Create Ambisonics signal

# Microphone settings
am = ambi.AmbiMic(1.47, .5)

# Generate Signal
aSig = ambi.ambiSig(h_tot, am)
# print(aSig.b_format)

# Plot and Format
fig_ax = [s.plot_y_t() for s in aSig.b_format]
for el in fig_ax:
    el[1].set_xlim(8.405, 8.425)
    el[1].set_ylim(-1.2e9, 2.5e9)

# %% Calculate Transferfunctions of A-Format
# 5. Calculate Transferfunction using Adrienne Windowing
tf_tot = []
for el in h_tot:
    tf_avg = []
    for i in range(3):
        # Calculate TF of single impulses within signal 7.6565 - 8.4085
        tf_single = tf.TransferFunction(signal=el.filter_y((50, 5e3)),
                                        in_win=[8.4085+i*(T+Tp), 2.5e-3],
                                        re_win=[8.4110+i*(T+Tp), 2.5e-3])
        tf_avg.append(tf_single.get_octave_band(1/3))
        print('Calculated TF %d of 3.' % (i+1))
        
    # Average octave spectra of one signal
    tf_tot = tf.TransferFunction(xf=tf_avg[0].xf,
                                 hf=[el.hf for el in tf_avg])
    print('Averaged TF within a signal.')

    # Plot and Format
    fig, ax = tf_tot.plot_hf()
    ax.set_xlim(50, 5e3)
    ax.set_ylim(.05, 10)


# %%
