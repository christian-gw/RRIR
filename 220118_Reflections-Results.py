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
# From: TARGET_DIR/avg/IMP_{Measurement Position}.wav
#
# 1. Import, Base Settings (System) and Base Settings (User)
#
# 2. Load Data
# - Load all elements from the NR dict from Array pos 0 into a Signal object.
# - Save the direct sound to a direct signal object.
#
# 3. Correct Signal
# - sig = singal
# - cor = correction signal from .8 direct sound
# - Create corrected signals
#
# 4. Calculate transferfunction
# - Uses direct .8 as incomming_sig
# - uses corrected reflected signal as reflected
#
# 5. Create Measurement object and add it to MP objects
#

# %% codecell
# 1. Import, Base Settings (System) and Base Settings (User)
from reflection_definitions import Signal, TransferFunction, Measurement
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib notebook
plt.rcParams['axes.grid'] = True    # Set grid in plots on per default
VERBOSE = False
F_up = 500e3

############################################################
########## User Interaction Starts Here       ##############
########## Please specify where the Files are ##############
############################################################

AVG_DIR = 'C:/Users/gmeinwieserch/Documents/Python/Reflection/Work_Dir/20220119_1513_avg/'
NAME = 'IMP_%s.wav'

NR = {'Wand_0_0': ['Wall - LR = 0 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_+_0': ['Wall - LR = +.4 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_-_0': ['Wall - LR = -.4 m, H = 0 m, NeutrH = 1.6 m'],
      'Wand_0_+': ['Wall - LR = 0 m, H = +.4 m, NeutrH = 1.6 m'],
      'Wand_0_-': ['Wall - LR = 0 m, H = -.4 m, NeutrH = 1.6 m'],
      # 'Boden_1.5': ['Floor - A = 1.5 m, H = .93 m'],
      # 'Boden_.2': ['Floor - A = .2 m, H = .93 m'],
      'Direct_.8': ['Direct - A  =  .8 m, H = 2 m']}
POS = {'Wand_0_0':  [0,   0],
       'Wand_+_0':  [.4,   0],
       'Wand_-_0': [-.4,   0],
       'Wand_0_+':   [0,  .4],
       'Wand_0_-':   [0, -.4],
       # 'Boden_1.5': [0,   0],
       # 'Boden_.2':  [0,   0],
       'Direct_.8':  [0,   0]}

################################################################
######### Please specify the exitation pars        #############
################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]

# %% codecell
# 2. Load Data
# - Load all elements from the NR dict from Array pos 0 into a Signal object.
# - Save the direct sound to a direct signal object.

for position in NR.keys():
    # Loop will load wav signals to a list wich is the value of the NR dict
    # = Complete Reflection
    del NR[position][1:]  # Reset List after title field
    NR[position].append(Signal(path=AVG_DIR, name=NAME %
                        (position)))  # Append signal obj

direct = NR.pop('Direct_.8')

fig, ax = direct[1].plot_y_t()
ax.set_xlim(0, .025)
ax.set_title('Direct_.8')
fig.canvas.manager.set_window_title('Direct_.8')

# plt.show()

# %% codecell
# 3. Correct Signal
# - sig = singal
# - cor = correction signal from .8 direct sound
# - Create corrected signals

for position in NR.keys():
    del NR[position][2:]  # Reset List after input signal field

    # Complete Reflection - external Direct signal = isolated Reflection
    sig = np.copy(NR[position][1].resample(F_up).y)  # Complete
    cor = np.copy(direct[1].resample(F_up).y)        # External Direct
    fak = max(sig)/max(cor)

    fig, ax = NR[position][1].plot_y_t()
    ax.set_xlim(0, .025)
    ax.set_title(position + '_Direct_Reflection')
    fig.canvas.manager.set_window_title(position + '_Direct_Reflection')
    # plt.show()

    #                       Complete - External Direct
    NR[position].append(Signal(y=sig - fak*cor,
                               dt=1 / F_up).resample(1/direct[1].dt))

    fig, ax = _, ax = NR[position][2].plot_y_t()
    ax.set_xlim(0, .025)
    ax.set_title(position + '_Reflection')
    fig.canvas.manager.set_window_title(position + '_Reflection')
    # plt.show()

    # Transfer function with scaled direct as in and scaled corrected as refl
    tf = TransferFunction(
        incoming_sig=Signal(y=fak*cor,
                            dt=1/F_up).resample(1/direct[1].dt),
        reflected_sig=NR[position][2])

    # tf_oct = tf.get_octave_band(fact=1/3)
    NR[position].append(tf)
# plt.show()

# %% codecell
# 4. Calculate transferfunction
# - Uses direct .8 as incomming_sig
# - uses corrected reflected signal as reflected
#
# 5. Create Measurement object and add it to MP objects
mea_Marth = Measurement('Measurement_Martha_Kirche', 1., .25)
reference_dist = .8
for i, position in zip(range(len(NR)), NR.keys()):
    mea_Marth.create_mp(i,
                        NR[position][3].get_octave_band(fact=1),
                        POS[position])

    reflection_dist = mea_Marth.mp_lst[i].calc_c_geo(norm=False)
    print(reflection_dist)
    mea_Marth.mp_lst[i].corrections['c_geo'] = (reference_dist/reflection_dist)**2
    # print(mea_Marth.mp_lst[i].corrections['c_geo'])
    mea_Marth.mp_lst[i].apply_c()

    fig, ax = mea_Marth.mp_lst[i].tf.plot_hf()
    fig.canvas.manager.set_window_title(position + '_TF')
    ax.set_title(position + '_TF')
    ax.set_ylim(5e-2, 2e0)
mea_Marth.average_mp()
fig, ax = mea_Marth.average.plot_hf()
fig.canvas.manager.set_window_title('Averaged_TF')
ax.set_title('Averaged TF')
ax.set_ylim(5e-2, 2e0)
plt.show()

# %%
