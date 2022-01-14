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

#%matplotlib notebook
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
      #'Boden_1.5': ['Floor - A = 1.5 m, H = .93 m'],
      #'Boden_.2': ['Floor - A = .2 m, H = .93 m'],
      'Direct_.8': ['Direct - A  =  .8 m, H = 2 m']}
POS = {'Wand_0_0': [ 0,   0],
      'Wand_+_0': [ .4,   0],
      'Wand_-_0': [-.4,   0],
      'Wand_0_+': [  0,  .4],
      'Wand_0_-': [  0, -.4],
      #'Boden_1.5': [ 0,   0],
      #'Boden_.2': [  0,   0],
      'Direct_.8': [ 0,   0]}

################################################################
######### Please specify the exitation pars        #############
################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]

# %% markdown
# 2. Load Data
# - Load all elements from the NR dict from Array pos 0 into a Signal object.
# - Save the direct sound to a direct signal object.

# %% codecell
for position in NR.keys():
    # Loop will load wav signals to a list wich is the value of the NR dict
    # = Complete Reflection
    del NR[position][1:]  # Reset List after title field
    NR[position].append(Signal(path=TARGET_DIR, name=NAME %
                        (position)))  # Append signal obj

direct = NR.pop('Direct_.8')

# %% markdown
# 3. Correct Signal
# - sig = singal
# - cor = correction signal from .8 direct sound
# - Create corrected signals

# %% codecell
for position in NR.keys():
    del NR[position][2:]  # Reset List after input signal field

    # Complete Reflection - external Direct signal = isolated Reflection
    sig = np.copy(NR[position][1].resample(F_up).y)    # Complete
    cor = np.copy(direct[1].resample(F_up).y)  # External Direct
    fak = max(sig)/max(cor)
    #                         Complete - External Direct
    NR[position].append(Signal(y=sig - fak*cor,
                               dt=1 / F_up).resample(1/direct[1].dt))
    # Transfer function with scaled direct as in and scaled corrected as refl
    tf = TransferFunction(
        incoming_sig=Signal(y=fak*cor,
                            dt = 1/F_up).resample(1/direct[1].dt),
        reflected_sig=NR[position][2])

    #tf_oct = tf.get_octave_band(fact=1/3)
    NR[position].append(tf)
# %% markdown
# 4. Calculate transferfunction
# - Uses direct .8 as incomming_sig
# - uses corrected reflected signal as reflected

# %% codecell
for position in NR.keys():
    del NR[position][3:]  # Reset List after input signal field

    # Calculate Transfer fkt

    #NR[position].append(tf_oct)

# %% markdown
# 5. Create Measurement object and add it to MP objects

# %% codecell
mea_Marth = Measurement('Measurement_Martha_Kirche', 1., .25)
for i, position in zip(range(len(NR)), NR.keys()):
    mea_Marth.create_mp(i,
                        NR[position][3].get_octave_band(fact = 1/3),
                        POS[position])
    mea_Marth.mp_lst[i].apply_c()

    fig, ax = mea_Marth.mp_lst[i].tf.plot_hf()
    ax.set_title(NR[position][0])
    ax.set_ylim(1e-1, 1e0)
mea_Marth.average_mp()
_, ax = mea_Marth.average.plot_hf()
ax.set_title('Averaged TF')
ax.set_ylim(1e-1, 1e0)

# %% codecell
