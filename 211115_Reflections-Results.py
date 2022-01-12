# %% markdown
# # Correct Signals for Direct Component
# ## Abstract
# - File 'Direct_.8' was measured 2-2.5 m above ground level with distance .8 m
# - This is the direct component by some margin C_dir is ommitted here
#
# ## Doing
# - Load averaged impulse
# - Sync all
# - Window 'Direct_.8' and substract it from the rest
#
# ## Files
# From: TARGET_DIR/IMP_{Measurement Position}.wav
# %% codecell
# 1. Import, Base Settings (System) and Base Settings (User)
from reflection_definitions import *

%matplotlib notebook
plt.rcParams['axes.grid'] = True    # Set grid in plots on per default
VERBOSE = False
F_up = 500e3


############################################################
########## User Interaction Starts Here       ##############
########## Please specify where the Files are ##############
############################################################

TARGET_DIR = 'C:/Users/gmeinwieserch/Desktop/04  St-Martha Kirche/211007_Martha/Wall_Refl/'
NAME = 'IMP_%s.wav'

NR = {'Wand_0_0': ['Wall - LR = 0 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_+_0': ['Wall - LR = +.4 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_-_0': ['Wall - LR = -.4 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_0_+': ['Wall - LR = 0 m, H = +.4 m, NeutrH = 1.6 m'],
      'Wand_0_-': ['Wall - LR = 0 m, H = -.4 m, NeutrH = 1.6 m'],
      'Boden_1.5': ['Floor - A = 1.5 m, H = .93 m'],
      'Boden_.2': ['Floor - A = .2 m, H = .93 m'],
      'Direct_.8': ['Direct - A  =  .8 m, H = 2 m']}

################################################################
######### Please specify the exitation pars        #############
################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]
# %% codecell
for position in NR.keys():
    # Loop will load wav signals to a list wich is the value of the NR dict
    # = Complete Reflection
    del NR[position][1:]  # Reset List after title field
    NR[position].append(Signal(path=TARGET_DIR, name=NAME %
                        (position)))  # Append signal obj

direct = NR.pop('Direct_.8')

# Distance Correction: Norm everything to a 1 m distance
# direct: .8
# Reflection: .25 * 2 + 1
# p_2/p_1 = (d_1/d_2)^2

d_fak_dir = (1/.8)**2
d_fak_ref = (1/1.5)**2

for position in NR.keys():
    NR[position][-1].y *= d_fak_ref
direct[-1].y *= d_fak_dir
# %% codecell
for position in NR.keys():
    del NR[position][2:]  # Reset List after input signal field

    # Complete Reflection - external Direct signal = isolated Reflection
    sig = np.copy(NR[position][1].resample(F_up).y)    # Complete
    cor = np.copy(direct[1].resample(F_up).y)  # External Direct
    fak = max(sig)/max(cor)
    #                         Complete - External Direct
    NR[position].append(Signal(y=sig - fak*cor, dt=1 /
                        F_up).resample(1/direct[1].dt))
# %% codecell
for position in NR.keys():
    del NR[position][3:]  # Reset List after input signal field

    # Calculate Transfer fkt
    tf = TransferFunction(
        incoming_sig=direct[1], reflected_sig=NR[position][2])
    tf_oct = tf.get_octave_band(fact=1/3)
    NR[position].append(tf)
    NR[position].append(tf_oct)
# %% codecell
def plot_t_f(ax, sig):
    ax[0].plot(sig.axis_arrays['t'],
               sig.y,
               linewidth=.25)

    ax[1].plot(sig.axis_arrays['xf'],
               np.absolute(sig.y_f[:int(sig.n_tot/2)]),
               linewidth=.25)
    ax[0].set_xlim(0, .012)
    ax[0].set_xlabel('t [s]')
    ax[0].set_ylabel('p')

    ax[1].set_xlim(50, 5e3)
    ax[1].set_xlabel('f [Hz]')
    ax[1].set_yscale('log')
    ax[1].set_ylim(5e2, 5e6)


for position in NR.keys():
    if True:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 2, figsize=(10, 9))
        plot_t_f(ax1, NR[position][1])
        ax1[0].set_title('Raw Reflected Signal')
        ax1[1].set_title('Raw Reflected Signal - Frequency')

        plot_t_f(ax2, direct[1])
        ax2[0].set_title('Direct Reflected Signal')
        ax2[1].set_title('Direct Reflected Signal - Frequency')

        plot_t_f(ax3, NR[position][2])
        ax3[0].set_title('Corrected Reflected Signal')
        ax3[1].set_title('Corrected Reflected Signal - Frequency')

        # Plot Transfer Fkt
        tf = NR[position][3]
        ax4[0].plot(tf.xf,
                    np.absolute(tf.hf[:int(tf.n_tot/2)]),
                    linewidth=.25)

        tf = NR[position][4]
        ax4[1].bar(tf.xf,
                   np.absolute(tf.hf),
                   .1*tf.xf)
        for ax in ax4:
            ax.set_xlim(50, 5e3)
            ax.set_xlabel('f [Hz]')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(.05, 5)

        ax4[0].set_title('Transfkt (not Amlitude correct) - Spectrum')
        ax4[1].set_title('Transfkt (not Amlitude correct) - Terz')

        fig.suptitle(NR[position][0])
        fig.tight_layout()
    else:
        tf = NR[position][4]
        fig, ax = plt.subplots(1, figsize=(10, 6))

        ax.bar(tf.xf,
               np.absolute(tf.hf),
               .1*tf.xf)

        ax.set_xlim(50, 5e3)
        ax.set_xlabel('f [Hz]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_ylim(.05, 1)

        ax.set_title('Transfkt (Magnitude roughly corrected) - Terz')

        fig.suptitle(NR[position][0])
        fig.tight_layout()
# %% codecell
for position in NR.keys():
    print(position + ': ')
    freq = NR[position][4].xf
    pin = NR[position][4].hf
    print('\tF [Hz]\tFaktor []')
    for i, fi in enumerate(freq):
        print('\t' + str(fi) + '\t' + str(pin[i]))
# %% codecell
NR
# %% codecell
