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

from os import mkdir, path
import matplotlib.pyplot as plt

from datetime import datetime
now = datetime.now().strftime('%Y%m%d_%H%M')

VERBOSE = False
F_up = 500e3

############################################################
# ######## User Interaction Starts Here       ##############
# ######## Please specify where the Files are ##############
############################################################

TARGET_DIR = "C:/Users/gmeinwieserch/Documents/Python/" + \
    "Reflection/Work_Dir_Reflection/"


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
    impulse.plot_y_t(headline='Impulse Rail.')
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
    imp_avg_down.plot_y_t(headline='Averaged Impulse of one position.')
    imp_avg_down.write_wav(name=AVG_DIR + '/IMP_' + position + '.wav',
                           F_samp=in_Sample)

# %%
