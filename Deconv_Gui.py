# %%
""" Functioins for calculating the impulse response of a room from a measured e-sweep answer
+ plotting and keyvalues"""

# Load modules and define functions
import matplotlib
import PySimpleGUI as gui
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from scipy.fft import fft  # , ifft, fftshift
# from scipy.signal.windows import hann, tukey
import scipy.signal as sg
from scipy.io import wavfile
import scipy.optimize as op
import numpy as np

# Use inline plotting in Jupyter

plt.rcParams['axes.grid'] = True    # Set grid in plots on per default


def to_dB(x):
    """Calculates the value of xa as an array or an scalar:\n
    Formula: L = 20*log10(p/p0)"""

    p0 = 2e-5                     # Reference
    xa = np.asarray(x)            # Cast to array so its polymorph

    return 20*np.log10(xa/p0)     # Return according to furmla


def inv_u(x, params):
    """ x = Signal to invert --> Exitation E-Sweep u
        params = [low-frequency, high-frequency]
        Get inverse of e-sweep by reverse x and get amplitude right, so that its rising with 3 dB/Oct
    according to:
        Farina 2007 p4 and
        Meng; Sen; Wang; Hayes - Impulse response measurement with sine sweeps and amplitude modulation shemes Signal Processing and Communication systems 2008 - ICSPCS"""

    om1 = params[0]*2*np.pi            # f0 --> omega0
    om2 = params[2]*2*np.pi            # f1 --> omega1
    T = params[1]                      # Sweep duration

    # Creation Frequency modulation (exp rising magnitude)
    # For modulation, K and L see paper from Docstring
    K = om1*T/np.log(om1/om2)
    L = T/np.log(om1/om2)
    modulation = [(1/(K/L * np.exp(t/L))) for t in np.linspace(0, T, len(x))]

    x_mod = x*modulation

    # reverse the modulated signal
    i = x_mod[::-1]

    return i


def load_data(path):
    """Load Data from file:
        .wav --> Load only answer and calculate metainformation the exitation was set mannually.
        dt, T, N_tot, t, xf, y =load_data(path)"""

    raw = wavfile.read(path)  # (fs,[Data])

    # Get values
    y = np.array(raw[1])

    dt = 1/raw[0]
    N_tot = len(y)
    T = dt*N_tot

    # Create time and frequency axis
    t = np.linspace(0, T, N_tot)
    xf = np.linspace(0, 1/(2/dt), N_tot//2)

    return dt, T, N_tot, t, xf, y


def prep_f(signal, dt, f_range=(1, 10000), downsample=True):
    """Filter the signal to the specified frequencyrange and apply downsampling if specified
        If downsampling is True, the new samplingrate = 2*f_range[1]"""

    # Consider to rework that using scipy.signal.decimate()

    # Define butterworth filter in second order sections format
    sos = sg.butter(10,            # Order
                    f_range,       # (from, to)
                    'bp',          # type
                    fs=1/dt,       # samplingrate
                    output='sos')  # result format

    # Apply filter
    filt = sg.sosfilt(sos, signal)

    # Apply downsampling
    if downsample:
        n0 = len(signal)          # Initial lenght of signal
        ratio = 2*f_range[1]*dt   # Get reduction factor
        n1 = round(ratio*n0)      # New array length
        dt = dt/ratio             # New dt

        res = sg.resample(filt, n1)  # Resampled Signal

        return dt, res
    else:                            # if no resampling
        return dt, filt


def analysis_all(path,
                 par_sweep=[6.39893794e+01, 5.13363160e+00, 2.19951833e+04]):
    """ Handles the basic transformation from e-sweep to impulse.
        Input: path = path of wavefile to transform
               par_sweep = [fbegin,T,fend]
        Output: dt of signal
                t = time array of signal
                tu = timearray of exitation
                u = array of exitation
                i_u = inverse array of exitation
                u_f = Spektrum of  exitation
                i_uf = Spectrum of inverse exitation
                y = answer signal
                y_f = Spectrum of signal
                imp = Array of impulse after transformation
                imp_f = Spectrum of impulse after transformation"""

    # Load Data
    dt, T, N_tot, t, xf, y = load_data(path)

    # Create time array of u and u
    tu = np.linspace(0, par_sweep[1], int(par_sweep[1]/dt))

    u = sg.chirp(tu,                     # time axis
                 *par_sweep,             # Sweeppar
                 method='logarithmic',   # Logarithmic Sweep
                 phi=90)                 # start with negative slope

    # Calculate Signals (see Docstring)
    i_u = inv_u(u, par_sweep)
    u_f = fft(u)
    i_uf = fft(i_u)

    y_f = fft(y)

    imp = sg.convolve(y, i_u,         # convolution of signal with inverse
                      mode='same')    # same means conv calculates a full (zero padding) and cuts out middle
    imp_f = fft(imp)

    return dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f


def analyse_decay(dt, t, imp, f_range):
    """From the time based impulse response calculate the logarithmic decay
    Filtering the impulse response is supported"""

    # Create and apply Butterworth Bandpass for frequency band selection
    sos = sg.butter(10, f_range, btype='bp',
                    analog=False, output='sos', fs=1/dt)
    imp_filt = sg.sosfilt(sos, imp)

    # Get the decay level from level_time
    imp_dec = level_time(imp_filt, .035, dt)

    return t, imp_dec


def create_deca_plot(t, imp_dec, f_range):
    """ Create a figure containing the decay curve with the predefined settings.
        Returns figure"""

    # Debug:
    vis_dec = False            # if vis_dec is False the function will not print the figure
    cut = 0                    # Possibility to cut the signal before plotting

    # Plotting
    fig3, ax21 = plt.subplots(1, 1, figsize=(10, 6))
    # Plot from cut to end
    ax21.plot(t[cut:], imp_dec[cut:])
    ax21.set_title('Abklingkurve - L(t) für %.0f - %.0f Hz' %  # Caption according to f_range
                   (f_range[0], f_range[1]))
    ax21.set_xlabel('t [s]')
    ax21.set_ylim(.5*np.mean(imp_dec), 1.1*np.max(imp_dec))

    fig3.tight_layout()

    # No plot if vis_dec
    if not vis_dec:
        plt.close(fig3)
    return fig3


def create_step_plot(dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f):
    """ Create Figure containing the step by step transformation process with predefined settings.
        Returns figure"""

    vis_step = False          # if vis_dec is False the function will not print the figure

    # Step by step analysis - visu
    fig1, ((ax1, ax3), (ax2, ax4), (ax5, ax6), (ax7, ax8)
           ) = plt.subplots(4, 2, figsize=(10, 6))

    ax1.plot(tu, u)
    ax2.plot(tu, i_u)
    ax3.plot(np.linspace(0, 1/(2*dt), int(len(u_f)/2)),
             np.log10(np.absolute(u_f[:int(len(u_f)/2)])))
    ax4.plot(np.linspace(0, 1/(2*dt), int(len(i_uf)/2)),
             np.log10(np.absolute(i_uf[:int(len(i_uf)/2)])))
    ax5.plot(t, y)
    ax6.plot(np.linspace(0, 1/(2*dt), int(len(y_f)/2)),
             np.log10(np.absolute(y_f[:int(len(y_f)/2)])))
    ax7.plot(t, imp)
    ax8.plot(np.linspace(0, 1/(2*dt), int(len(imp_f)/2)),
             np.absolute(np.absolute(imp_f[:int(len(imp_f)/2)])))

    ax1.set_title('u(t)')
    ax2.set_title('iu(t)')
    ax3.set_title('u(f)')
    ax4.set_title('iu(f)')
    ax5.set_title('y(t)')
    ax6.set_title('y(f)')
    ax7.set_title('h(t)')
    ax8.set_title('H(f)')

    for ax in [ax3, ax4, ax6, ax8]:
        ax.set_xscale('log')
        ax.set_xlim(par_sweep[::2])
        ax.set_xlabel('f [Hz]')

    for ax in [ax1, ax2, ax5, ax7]:
        ax.set_xlabel('t [s]')
    fig1.tight_layout()

    if not(vis_step):
        plt.close(fig1)
    return fig1


def create_comp_plot(dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f):
    """ Create figure with two spectrogram plots to visualise the transformation
        Returns figure"""

    # Preset and debug
    f_range = [50, 10000]
    vis_comp = False
    N_win = 512

    # Spectrogram
    fc, tc, Spec_imp = sg.spectrogram(np.absolute(imp),      # impulse signal
                                      fs=round(1/dt),        # Samplingrate
                                      # tuckey is rising cos, uniform, falling cos window
                                      window=('tukey', .25),
                                      # alpha factor specifies the proportion wich is cos
                                      nperseg=N_win,         # Blocksize
                                      noverlap=1*N_win//8)   # Overlap Samples overlap (default nperseg//8)
    # Alpha of tuckey.25 and nperseg//8 means, that the
    # Overlap is in the rising/faling cos range of the win

    fa, ta, Spec_y = sg.spectrogram(np.absolute(y),          # Answer signal
                                    fs=round(1/dt),
                                    window=('tukey', .25),
                                    nperseg=N_win,
                                    noverlap=1*N_win//8)

    # Generate figure
    fig2, (ax11, ax12) = plt.subplots(2, 1, figsize=(10, 6))

    # , vmin = -280, vmax = -80) # Use this to set color bar limits
    ax11.pcolormesh(tc, fc, to_dB(Spec_imp), shading='auto')
    ax11.set_title('Impulse')
    ax11.set_xlabel('t [s]')
    ax11.set_ylim(f_range)
    ax11.set_ylabel('f [Hz]')
    ax11.set_yscale('log')

    ax12.pcolormesh(ta, fa, to_dB(Spec_y), shading='auto')
    ax12.set_title('Sweep')
    ax12.set_xlabel('t [s]')
    ax12.set_ylim(f_range)
    ax12.set_ylabel('f [Hz]')
    ax12.set_yscale('log')

    fig2.tight_layout()

    if not(vis_comp):
        plt.close(fig2)
    return fig2


def level_time(x, T, dt):
    """ Calculates the sound level with a smoothing window of length T
        Returns Level array according to DIN EN 61672-1"""

    # Square signal
    x2 = np.power(x, 2)

    # Design and apply Filter
    # Consider reviewing DIN p.8
    # "Filter with real pole at -1/T"

    filt = sg.butter(1,
                     1/T,
                     output='sos',
                     fs=1/dt)
    x_filt = sg.sosfilt(filt, x2)

    # Calculate Level array
    L = [10*np.log10(el/((2e-5)**2)*T) for el in x_filt]
    return L


def find_peaks(t, imp_dec, par=[35, None, None, None, None]):
    """ Finds peaks in the Level Decay.
        Input: t = time array
               imp_dec = decay array
               par[0] = prominence
               par[1] = horizontal distance
               par[2] = peak width at prominence
               par[3] = plateau_size at peak
               par[4] = height of peak
               - See documentationof scipy.signal.find_peaks() -
        Output: [peak position, peak height], String to explain peak"""

    # Map parameters
    prominence = par[0]
    distance = par[1]
    width = par[2]
    plateau_size = par[3]
    height = par[4]

    # Find peaks
    peaks, par = sg.find_peaks(imp_dec,
                               prominence=prominence,
                               distance=distance,
                               width=width,
                               plateau_size=plateau_size,
                               height=height)

    # Initalize values for output creation
    pos = []
    boxtext = ''
    i = 0
    for p in peaks:
        # Fill output 1 for each el in peaks
        pos.append((t[p], par['prominences'][i]))
        # ax1.axvline(pos,lw=.5,ls='-',c='r')            # Necesarry to draw it in a active figure
        boxtext += ("%.2f s - %.0f dB, Pr - %.0f\n" %    # Create Boxtext for ech in peaks
                    (pos[i][0], imp_dec[p], par['prominences'][i]))
        i += 1

    return pos, boxtext


def linear(t, m, y0):
    """ Linear function for regression fit: y = m*t+y0"""
    return m*t+y0


def fit(t, imp_dec, peak, length):
    """ fit linear function linear() to slice of im_dec.
        Input: t = time array
               imp_dec = complete imulse decay array
               peak    = Main (normaly highest or only) peak
               length  = selected lenght of the slice
        Output: opt = [m, y0] = Optimized parameters for linear()"""

    # Map input variables
    dt = t[1]          # dt
    # (is that dirty and dt should be input or great and dt should not be input in the rest?)
    db0 = peak         # Peakposition in s
    db5 = np.where(np.isclose(imp_dec, max(imp_dec)-5, atol=.01))[0][-1]
    # Last pos where a sample is close (+-.01) at L(peak)-5dB = start of Slice in s
    dbe = peak+length  # End of slice in s

    el5 = db5          # Start of slice in index
    ele = int(dbe/dt)  # End of slice in index

    # Slice the arrays in which the fit should take place
    t_act = t[el5:ele]
    y_act = imp_dec[el5:ele]

    # Get the optimisation
    # Consider adding the R^2 value in return to tell user if chosen length was ok
    opt, _ = op.curve_fit(linear, t_act, y_act)

    return opt


def Txx(lin_par, xx):
    """ Calculate the decay time between Peak-5dB and Peak-(5+xx)dB
    according to the lin_par for the linear() fuction.
        Innput: lin_par = [m, y0] (see linear())
        Output: Decay time for xx"""

    return xx/np.abs(lin_par[0])


# %%

matplotlib.use('TkAgg')


#  functions for figure Handling ####################################################################

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


# lists and dicts ################################################################################
plot_dict = {'Impulsumwandlung': 0,
             'Vergleich Sweep-Impuls': 1, 'Abklingkurve': 2}
fig_lst = [None]*len(plot_dict)
new = [False, False, False]
#                        f0,             t1,             f1
par_sweep = [6.39893794e+01, 5.13363160e+00, 2.19951833e+04]

band_lst = [[16.,       32.],
            [32.,       63.],
            [63.,      125.],
            [125.,     250.],
            [250.,     500.],
            [500.,    1000.],
            [1000.,   2000.],
            [2000.,   4000.],
            [4000.,   8000.],
            [8000.,  16000.],
            [16000., 20000.],
            [65., 22000.]]

gui.theme('LightGreen')

figure_w, figure_h = 1000, 650

# define layout ######################################################################
listbox_values = list(plot_dict)
tab_sweep = [[gui.T('Settings für den Sweep')],
             [gui.T('Sweep f_0'), gui.Input(
                 key='Sweep_f0', default_text="%f" % par_sweep[0])],
             [gui.T('Sweep f_1'), gui.Input(
                 key='Sweep_f1', default_text="%f" % par_sweep[2])],
             [gui.T('T        '), gui.Input(
                 key='Sweep_T', default_text="%f" % par_sweep[1])],
             [gui.B('Anwenden', key='Apl_S')]]

tab_decay = [[gui.T('Settings für die Nachhallzeitanalyse')],
             [gui.T('Bandwahl'),
              gui.Combo([str(band_lst[i][:])for i in range(len(band_lst))],
                        key='Band',
                        default_value=str(band_lst[-1]))],
             [gui.T('-------------------------------------------------------------')],
             [gui.T('Settings für die Automatische Peakerkennung')],
             [gui.Checkbox('Peaks anzeigen', default=False, key='Peaks_show')],
             [gui.T('Prominenz'),
              gui.Slider(range=(5, 100),
                         default_value=35,
                         size=(20, 15),
                         orientation='horizontal',
                         font=('Helvetica', 12),
                         key='Prominence')],
             [gui.T('-------------------------------------------------------------')],
             [gui.T('Settings für die Nachhallauswahl')],
             [gui.Checkbox('Auswertebereich zeigen',
                           default=True, key='Len_show')],
             [gui.T('Nachhall ges'),
              gui.Slider(range=(0., 3.),
                         resolution=.01,
                         default_value=1.2,
                         size=(20, 15),
                         orientation='horizontal',
                         font=('Helvetica', 12),
                         key='Len')],
             [gui.Checkbox('Zeiten berechnen', default=False, key='Txx_calc'),
              gui.Checkbox('Gerade anzeigen', default=False, key='Txx_show')],
             [gui.Output(size=(50, 5))],
             [gui.T('-------------------------------------------------------------')],
             [gui.B('Anwenden', key='Apl_D')],
             [gui.InputText(visible=False, enable_events=True, key='out_path'),
              gui.FileSaveAs(button_text='Auswertebereich in Wave Datei speichern',
                             key='wav_save',
                             file_types=(('Wave', '*.wav'),))]]


col_listbox = [[gui.T('.wav File wählen:')],
               [gui.InputText(key="Path", change_submits=True),
                gui.FileBrowse()],
               [gui.T('-------------------------------------------------------------')],
               [gui.T('Visualisierungstyp wählen:')],
               [gui.Listbox(values=listbox_values,
                            default_values='Abklingkurve',
                            enable_events=True,
                            size=(28, len(listbox_values)),
                            key='Graph_Typ')],
               [gui.T('-------------------------------------------------------------')],
               [gui.TabGroup([[gui.Tab('Nachhall Settings', tab_decay),
                               gui.Tab('Sweep Settings', tab_sweep)]])],
               [gui.T(' ' * 12), gui.Exit(size=(5, 2))]]
col_plot = [[gui.Canvas(key='controls_cv')],
            [gui.Canvas(size=(figure_w, figure_h), key='-CANVAS-')]]

layout = [[gui.Text('Raumakustik aus E-Sweep', font=('current 18'))],
          [gui.Col(col_listbox), gui.Col(col_plot)]]


# create the form and show it without the plot ##############################################
window = gui.Window('Raumakustik aus E-Sweep', layout,
                    grab_anywhere=False, finalize=True)
# figure_agg = None
loaded = False

# The GUI Event Loop
while True:

    # read and eventually print events and values
    event, values = window.read()
    # print(event, values)

    # If user closed window or clicked Exit button
    if event in (gui.WIN_CLOSED, 'Exit'):
        break

    # safes data if user wants to save, has chosen a valid path and data is loaded
    if (event == 'out_path') and (values['out_path'] != '') and loaded:
        wavfile.write(values['out_path'],
                      int(1/dt),
                      imp[int(pos[0][0]/dt): int((pos[0][0]+values['Len'])/dt)])
        print('Gespeichert')

    # Performs Analysis if
    #     a) User loads new File
    #     b) User
    if (event == 'Path'                   # if its required to calc and plot new values
            or (event in ('Apl_S', 'Apl_D') and loaded)
            and not values['Path'] == ''):

        sweep_el = [float(el) for el in [values['Sweep_f0'],
                                         values['Sweep_T'], values['Sweep_f1']]]

        dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f = analysis_all(values['Path'],
                                                                        par_sweep=sweep_el)
        t, imp_dec = analyse_decay(dt, t, imp,
                                   [float(el[:-1])
                                    for el in values['Band'][1:].split()])
        loaded = True
        new = [False, False, False]

    if (event in ('Path', 'Graph_Typ', 'Apl_S', 'Apl_D', 'Apl_P')
            and loaded):

        fig_nr = plot_dict[values['Graph_Typ'][0]]

        if new[fig_nr] == False:
            if fig_nr == 0:
                fig_lst[fig_nr] = create_step_plot(
                    dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f)
            elif fig_nr == 1:
                fig_lst[fig_nr] = create_comp_plot(
                    dt, t, tu, u, i_u, u_f, i_uf, y, y_f, imp, imp_f)
            elif fig_nr == 2:
                pos, boxtext = find_peaks(t, imp_dec, par=[values['Prominence'],
                                                           None, None, None, None])
                fig_lst[fig_nr] = create_deca_plot(t, imp_dec,
                                                   [float(el[:-1]) for el in values['Band'][1:].split()])
                if values['Peaks_show']:
                    for el in pos:
                        fig_lst[fig_nr].axes[0].axvline(
                            el[0], lw=.5, ls='-', c='r')

                        props = dict(boxstyle='round',
                                     facecolor='wheat', alpha=0.5)

                        # place a text box in upper left in axes coords
                        fig_lst[fig_nr].axes[0].text(0, max(imp_dec), boxtext[:-1],  fontsize=8,
                                                     verticalalignment='top', bbox=props)
                if values['Len_show'] and len(pos) == 1:
                    fig_lst[fig_nr].axes[0].axvline(
                        pos[0][0], lw=.75, ls='-', c='r')
                    fig_lst[fig_nr].axes[0].axvline(
                        pos[0][0]+values['Len'], lw=.75, ls='-', c='r')
                elif(values['Len_show'] and len(pos) != 1):
                    print(
                        'Keine Anzeige:\nStellen sie Sicher, dass nur ein Peak erkannt wird!')

                if values['Txx_calc'] and len(pos) == 1:
                    lin_par = fit(t, imp_dec, pos[0][0], values['Len'])

                    print('T20 = %.3f s\nT30 = %.3f s' %
                          (Txx(lin_par, 20), Txx(lin_par, 30)))

                    if values['Txx_show']:
                        fig_lst[fig_nr].axes[0].plot(
                            t, [linear(ti, *lin_par) for ti in t])
                        fig_lst[fig_nr].axes[0].axhline(
                            np.max(imp_dec)-5, lw=.5, ls='--')
                        fig_lst[fig_nr].axes[0].axhline(
                            np.max(imp_dec)-25, lw=.5, ls='--')
                        fig_lst[fig_nr].axes[0].axhline(
                            np.max(imp_dec)-35, lw=.5, ls='--')
                elif(values['Txx_calc'] and len(pos) != 1):
                    print(
                        'Keine Berechnung:\nStellen sie Sicher, dass nur ein Peak erkannt wird!')

            new[fig_nr] = True
            new[2] = False

        fig = fig_lst[fig_nr]      # select wich fig to print
        # fig.axes[0].plot([1.,2.,3.],[15.,16.,17.])
        draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas,
                              fig,
                              window['controls_cv'].TKCanvas)

window.close()
# matplotlib.use('nbAgg')     # Necessarry if you want to run this in notebook and keep working after gui termination
