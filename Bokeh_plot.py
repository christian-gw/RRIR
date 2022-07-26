# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 08:33:46 2021

@author: gmeinwieserch
"""


# 1. Import, Base Settings (System) and Base Settings (User)
from sml.Signal import Signal, rotate_sig_lst

from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True    # Set grid in plots on per default
VERBOSE = False
F_up = 500e3


# ###########################################################
# ######### User Interaction Starts Here       ##############
# ######### Please specify where the Files are ##############
# ###########################################################

TARGET_DIR = "C:/Users/gmeinwieserch/Desktop/20220629_Messung Schallmessraum/\
Reflektion Boden/"


NAME = ["""ZOOM0005_Tr1.WAV"""]
# ###########################################################

# Load whole sig
signal_all = Signal(path=TARGET_DIR, name=NAME[0])


# 2. Cut whole signal and create time sync average of impulses

# #################################################
# #### Please specify the sweep segments ##########
# ####   where they start (t_cut)        ##########
# ####   how long they last (t_dur)      ##########
# ####   the Sweepparams (par_sweep)     ##########
# ####     par_sweep = [fstart, T, fend] ##########
# #################################################

t_dur = 12                   # duration of cut [s]
t_rep = 15
n = 3
t_cut_sig = np.linspace(0, (n-1)*t_rep, n) + 1
t_cut_dir = np.linspace(0, (n-1)*t_rep, n) + 1

par_sweep = [50, 10, 5e3]   # parameter of sweep [fstart, T, fend]

################################################################

# Cut it up
signal = [signal_all.cut_signal(t_start,
                                t_start + t_dur,
                                force_n=int(t_dur/signal_all.dt))
          for t_start in t_cut_sig]

del signal_all

# Create exitation
u_sig = Signal(par_sweep=par_sweep, dt=signal[0].dt)

# Create Impulses
impulses_sig = [sig.impulse_response(u_sig) for sig in signal]
del signal

# Upsample and sync impulses
[imp.resample(F_up) for imp in impulses_sig]
rotate_sig_lst(impulses_sig)


# Average Impulses
impulse_sig = Signal(signal_lst_imp=impulses_sig)


# %%
def plot_sig(signal):
    import numpy as np

    from bokeh.layouts import row  # ,column
    # from bokeh.models import CustomJS, Slider
    from bokeh.plotting import ColumnDataSource, figure, show

    signal.resample(20e3)

    # Start Values
    x_t = signal.axis_arrays['t']
    y_t = signal.y
    x_f = signal.axis_arrays['xf']
    y_f = 2.0/signal.n_tot * np.abs(signal.y_f[0:signal.n_tot//2])

    source_t = ColumnDataSource(data=dict(x=x_t, y=y_t))
    source_f = ColumnDataSource(data=dict(x=x_f, y=y_f))

    # Define Tools
    TOOLS = "hover," +\
            "crosshair," +\
            "pan,wheel_zoom," +\
            "zoom_in," +\
            "zoom_out," +\
            "box_zoom," +\
            "reset," +\
            "save,"

    TOOLTIPS = [("(x,y)", "($x, $y)")]

    # Create Plot-figure with tools and dimensions
    plot_t = figure(title="Time",
                    x_axis_label='t [s]',
                    width=400,
                    height=400,
                    tools=TOOLS,
                    tooltips=TOOLTIPS)

    plot_f = figure(title='Spectrum',
                    x_axis_label='f [Hz]',
                    width=400,
                    height=400,
                    y_axis_type="log",
                    tools=TOOLS,
                    tooltips=TOOLTIPS)

    # Plot initial line
    plot_t.line('x',
                'y',
                source=source_t,
                line_width=.25,
                line_alpha=1)

    plot_f.line('x',
                'y',
                source=source_f,
                line_width=.25,
                line_alpha=1)

    # Define layout of plot
    layout = row(
        plot_t,
        plot_f)

    show(layout)

    print('There is a browser tab with your plot now.')


plot_sig(impulse_sig)


# %%
