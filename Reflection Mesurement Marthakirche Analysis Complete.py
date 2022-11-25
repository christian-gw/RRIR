# %%markdown
# # Cut Signals for each Measurement
# ## Abstract
# - The measurent is performed with a e-sweep
# - It needs to be transformed into Impulse
# - This is performed for everything
#
# ## Doing
# - Load Test Files
# - Create Impulse Response
# - Write Impulse Files
#
# ## Files
# From: TARGET_DIR/raw/REC{Nr}.wav --> To: ./imp/IMP_REC{Nr}.wav

# %%codecell
# 1. Import, Base Settings (System) and Base Settings (User)
# from reflection_definitions import Signal, rotate_sig_lst
from RRIR.Signal import Signal
from RRIR.Signal import rotate_sig_lst
from RRIR.Transfer_function import TransferFunction
from RRIR.Reflection import Measurement
import numpy as np

from os import mkdir, path, getcwd
import matplotlib.pyplot as plt

from datetime import datetime
now = datetime.now().strftime('%Y%m%d_%H%M')

VERBOSE = False
F_up = 500e3

############################################################
# ######## User Interaction Starts Here       ##############
# ######## Please specify where the Files are ##############
############################################################

TARGET_DIR = getcwd() + \
    "/Work_Dir_Reflection/"


NAME = 'REC0%s.wav'

NR = {'Wand_0_0':  ['27'],  # LR =   0, H =   0, NeutrH = 1.6
      'Wand_+_0':  ['28'],  # LR = +.4, H =   0, NeutrH = 1.6
      'Wand_-_0':  ['29'],  # LR = -.4, H =   0, NeutrH = 1.6
      'Wand_0_+':  ['30'],  # LR =   0, H = +.4, NeutrH = 1.6
      'Wand_0_-':  ['31'],  # LR =   0, H = -.4, NeutrH = 1.6
      'Boden_1.5': ['32'],  # A  = 1.5, H = .93
      'Boden_.2':  ['33'],  # A  =  .2, H = .93
      'Direct_.8': ['35']}  # A  =  .8, H = 2


################################################################
# ####### Please specify the exitation pars        #############
################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]
in_Sample = 48e3

################################################################

SRC_DIR = path.join(TARGET_DIR, 'raw')
IMP_DIR = path.join(TARGET_DIR, now + '_imp')
mkdir(IMP_DIR)

# Create exitation
signal = Signal(path=SRC_DIR, name=NAME % (NR['Wand_0_0'][0]))
u_sig = Signal(par_sweep=par_sweep, dt=signal.dt)

print('Create and Save Impulse Responses from Sweeps')

# Process all sigs
for position in NR.keys():
    # Load sig
    signal = Signal(path=SRC_DIR, name=NAME % (NR[position][0]))

    # Create Impulse
    impulse = signal.impulse_response(u_sig)
    del signal.y, signal.y_f, signal.axis_arrays['t']
    del signal.axis_arrays['xf'], signal

    # Save Impulse
    # impulse.plot_y_t(headline='Impulse Rail.')
    plt.show()
    impulse.write_wav(name=IMP_DIR + '/IMP_' + NAME % (NR[position][0]))
    del impulse.y, impulse.y_f, impulse.axis_arrays['t']
    del impulse.axis_arrays['xf'], impulse

# %%markdown
# # Average the Multimeasurements for Reflection
# ## Abstract
# - The Measuremnts for reflection where performed 3 at a time.
# - Its necesarry to cut them up.
# - Cutting is performed on the transformed Impulses
#
# ## Doing
# - Load Impulse Group
# - Cut it Up
# - Upsample
# - Average
# - Write to Imp File
#
# ## Files
# From: TARGET_DIR/avg/IMP_REC{Nr}.wav --> To: ./avg/{Measurement Pos}.wav

# %%codecell
# 1. Import, Base Settings (System) and Base Settings (User)

############################################################
# ######## User Interaction Starts Here       ##############
# ######## Please specify where the Files are ##############
############################################################

IMP_DIR = path.join(TARGET_DIR, now + '_imp')
NAME = 'IMP_REC0%s.wav'
AVG_DIR = path.join(TARGET_DIR, now + '_avg')
mkdir(AVG_DIR)
################################################################
print('Average and Save the impulses.')

for position in NR.keys():
    # position = 'Wand_0_0'#for position in NR.keys():
    all_imp = Signal(path=IMP_DIR, name=NAME % (NR[position][0]))

    all_cut = []
    all_cut.append(all_imp.cut_signal(3, 10))
    all_cut.append(all_imp.cut_signal(17, 24))
    all_cut.append(all_imp.cut_signal(33, 40))

    cut_up = [imp.resample(F_up) for imp in all_cut]
    rotate_sig_lst(cut_up)

    end_cut = int(3.2/cut_up[0].dt)   # min([el.n_tot for el in cut_up])
    imp_avg = Signal(
        signal_lst_imp=[sig.cut_signal(t_start=0,
                                       t_end=.1,
                                       force_n=end_cut) for sig in cut_up])

    imp_avg_down = imp_avg.resample(in_Sample)
    fig, _ = imp_avg_down.plot_y_t(headline='Averaged Impulse of one position.')
    fig.canvas.manager.set_window_title('Averaged Impulses')
    imp_avg_down.write_wav(name=AVG_DIR + '/IMP_' + position + '.wav',
                           F_samp=in_Sample)

# %%
# ####################################################################################
#
# %%[markdown]
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
# from reflection_definitions import Signal, TransferFunction, Measurement
# from RRIR.Signal import Signal
# from RRIR.Transfer_function import TransferFunction
# from RRIR.Reflection import Measurement

# import matplotlib.pyplot as plt
# import numpy as np

# # %matplotlib notebook
# plt.rcParams['axes.grid'] = True    # Set grid in plots on per default
# VERBOSE = False
# F_up = 500e3

############################################################
# ######## User Interaction Starts Here       ##############
# ######## Please specify where the Files are ##############
############################################################

# AVG_DIR = "C:/Users/gmeinwieserch/Documents/Python/Reflection/" + \
#     "Work_Dir_Reflection/20220210_0957_avg/"
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
# ####### Please specify the exitation pars        #############
################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]

# %% codecell
# 2. Load Data
# - Load all elements from the NR dict from Array pos 0 into a Signal object.
# - Save the direct sound to a direct signal object.

for position in NR.keys():
    # Loop will load wav signals to a list wich is the value of the NR dict
    # = Complete Reflection
    del NR[position][1:]  # Reset List after title field (Multirun reason)
    NR[position].append(Signal(path=AVG_DIR, name=NAME %
                        (position)))  # Append signal obj

direct = NR.pop('Direct_.8')

fig, ax = direct[1].plot_y_t()
ax.set_xlim(0, .025)
ax.set_title('Direct_.8')
fig.canvas.manager.set_window_title('Direct_.8')

# plt.show()  # Required for command line exec

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
    mea_Marth.mp_lst[i].corrections['c_geo'] = (reference_dist /
                                                reflection_dist)**2
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
