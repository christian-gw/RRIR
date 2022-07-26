# %%[markdown]
# Script for processing of reflection measurement analysis
# Measurement was performed in nonreverbant room with ambisonics mic
#
# 1. Imports and Settings
# 2. Create exitation sweep
# 3. Load measurement
#    Perform transformation to impulse (Farina 2004)
#    Plot and/or safe
# 4. Create Ambisonics Signal
# 5. Calculate averaged Transferfunktion using Adrienne windowing
# 6. Export Aimed and perpendicular directive signals
# 7. Plot spherical max and delay


# %% Import and Settings
# 1. Import Settings
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
# from IPython import get_ipython
import sml.Signal as sg
import sml.Transfer_function as tf
import sml.Ambi as ambi
# Signal  # , rotate_sig_lst()

# get_ipython().run_line_magic('matplotlib', 'qt')
# get_ipython().run_line_magic('matplotlib', 'inline')

PLOT = True

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
    y_raw = y_raw.cut_signal(t_start=0, t_end=15)
    print('Loaded soundfile...')
#   # 3.2 Transform to impulse
    # Deconvolve exitation deconvolution package
    h2 = y_raw.impulse_response(x)
    h_tot.append(h2)

    print('Inverted with convolution ...')

# %% Plotting and/or Saving
# 3.3 Plot and/or save Impulses
if PLOT:
    fig, ax = plt.subplots()
    for el in zip(range(1, 5), h_tot):
        ax.plot(el[1].axis_arrays['t'],
                np.real(el[1].y),
                label=el[0], lw=.25)
        ax.legend()
        ax.set_xlim(8.405, 8.425)

        # el[1].write_wav(name='imp' + str(el[0]) + '.wav',
        #                 F_samp=int(96e3))
    print('Plotted raw impulses')

# %% Calculate Ambisonics
# 4. Create Ambisonics signal

# Microphone settings
am = ambi.AmbiMic(1.47, .5)

# Generate Signal
aSig = ambi.ambiSig(h_tot, am)
# print(aSig.b_format)

# Plot and Format
if PLOT:
    fig, ax = plt.subplots()
    for el in zip(['w', 'x', 'y', 'z'], aSig.b_format):
        ax.plot(el[1].axis_arrays['t'],
                np.real(el[1].y),
                label=el[0], lw=.25)
        ax.legend()
        ax.set_xlim(8.405, 8.425)
        ax.set_ylim(-1.2e9, 2.5e9)
    print('Plotted B-Format impulses')

# %% Calculate Transferfunctions of A-Format
# 5. Calculate Transferfunction using Adrienne Windowing
if False:
    tf_tot = []
    output_label = ['w', 'x', 'y', 'z']
    for el in zip(output_label, h_tot):
        tf_avg = []
        print('Signal ' + el[0] + ':')
        for i in range(3):
            # Calculate TF of single impulses within signal 7.6565 - 8.4085
            # Time ranges for impulses:
            # Boden: 7.6565 s; Element 8.4085 s; Duration 0.0025 s
            tf_single = tf.TransferFunction(signal=el[1].filter_y((50, 5e3)),
                                            in_win=[8.4085+i*(T+Tp), 2.5e-3],
                                            re_win=[8.4110+i*(T+Tp), 2.5e-3])
            tf_avg.append(tf_single.get_octave_band(1/3))
            print('Calculated TF %d of 3.' % (i+1))

        # Average octave spectra of one signal
        tf_tot = tf.TransferFunction(xf=tf_avg[0].xf,
                                     hf=[el.hf for el in tf_avg])
        print('Averaged TF within a signal.')

        if PLOT:
            # Plot and Format
            fig, ax = tf_tot.plot_hf()
            ax.set_xlim(50, 5e3)
            ax.set_ylim(.05, 10)
        #    fig.title('TF of signal ' + output_label[i])
            fig.tight_layout()
            print('Plotted B-Format Transfer Function')
# %% Get Directive mono
# 6. Export Aimed and perpendicular directive signals
if True:
    # DEBUG ONLY ############
    aSig = ambi.ambiSig(h_tot, am)
    # #######################
    # phi = 0 --> X
    # theta = 0 --> Z
    direct_sig = aSig.get_one_direction(20, 00, rad=False)
    perp1_sig = aSig.get_one_direction(110, 0, rad=False)
    perp2_sig = aSig.get_one_direction(20, 90, rad=False)
    ref_sig = aSig.get_one_direction(20, 105, rad=False)

    fig, ax = plt.subplots()
    tar_file_names = ['Direct', 'Perp1', 'Perp2', 'Reflection']

    if PLOT:
        for el in zip(range(4), [direct_sig, perp1_sig, perp2_sig, ref_sig]):
            ax.plot(el[1].axis_arrays['t'],
                    el[1].y,
                    lw=1,
                    label=tar_file_names[el[0]])
        ax.set_xlim(8.405, 8.425)
        ax.set_xlabel('t [s]')
        ax.legend()

    norm_fak = 1/(np.max(np.abs(np.concatenate((direct_sig.y,
                                                perp1_sig.y,
                                                perp2_sig.y,
                                                ref_sig.y)))))

    for el in zip(range(4), [direct_sig, perp1_sig, perp2_sig, ref_sig]):
        el[1]*norm_fak
        el[1].write_wav(tar_file_names[el[0]] + '.wav', norm=False)

    # od_sig.plot_y_t()
# %%
# 7. Plot spherical max and delay

# DEBUG ONLY ############
# aSig = ambi.ambiSig(h_tot, am)
# #######################

phi = np.linspace(0, 2*np.pi, 32)
theta = np.linspace(0, 2*np.pi, 32)
P, T = np.meshgrid(phi, theta)


def calc_R(phi, theta, sig):
    arr_max = []
    arr_argmax = []
    for t in theta:
        line_max = []
        line_argmax = []
        for p in phi:
            # print('Calling get_one_direction with:')
            print('p, t :' + str(p) + ',' + str(t))
            directive_sig = sig.get_one_direction(p, t, rad=True)
            line_max.append(np.max(directive_sig.y))
            index = np.argmax(directive_sig.y)
            line_argmax.append(directive_sig.axis_arrays['t'][index])
        arr_max.append(line_max)
        arr_argmax.append(line_argmax)
    return np.array(arr_max), np.array(arr_argmax)


R_max, R_argmax = calc_R(phi, theta, aSig)

X = (R_argmax)*np.sin(T)*np.cos(P)
Y = (R_argmax)*np.sin(T)*np.sin(P)
Z = (R_argmax)*np.cos(T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
# %%
