from datetime import datetime
import matplotlib.pyplot as plt

# from scipy.fft import fft, ifft, fftfreq  # , fftshift
# from scipy.signal.windows import tukey, blackmanharris  # , hann
# import scipy.signal as sg
# import librosa as lr
import numpy as np
# import os

if __name__ == "__main__":
    # from Signal import Signal
    from Transfer_function import TransferFunction
else:
    # from RRIR.Signal import Signal
    from RRIR.Transfer_function import TransferFunction


def dbg_info(txt, verbose=False):
    """Print debug info if global VERBOSE = True

    Parameters
    ----------
    txt: str
        Text to print in debug
    verbose : Bool
        Print takes place when True
    Returns
    -------
    None"""

    if verbose:
        print(str(datetime.now().time()) + ' - ' + txt)


class Measurement:
    """Class for handling a whole measurement with multiple MeasurementPoint s.

    If there are multiple signals (should be),
    averaging before creation of Measurement Point is advised.
    Attributes
    ----------
    m_name: str
       Name of the Measurement (Meta data class is planned)
    d_mic: float
       Distance
    d_probe: float
       Distance
    mp_lst: list(MeasurementPoint)
       List containing all measurementpoints with signals
    n_mp: int
       Over all number of mps
    """
    # Methods
    # -------
    # create_mp(number, _signal, pos) -> None
    #     creates a measurement point
    # del_mp(number) -> None
    #     deletes a measurement point by number
    # plot_overview() -> fig, ax
    #     plots an overview of all measurement points
    #     relative to source and probe
    # average() -> None

    def __init__(self, name, d_mic, d_probe):
        self.m_name = name
        self.d_mic = d_mic
        self.d_probe = d_probe
        self.mp_lst = []
        self.n_mp = 0
        self.average = []

    def create_mp(self, number, _signal, pos):
        """Create measurement point object

        MeasurementPoint holds a specific number and _signal
        object for mp Position pos [x,y]

        Parameters
        ----------
        number: int
            Number of the measurement.
            Norm: 9 MPs with numbering like number block
        d_mic: float
            Distance between source and microphone
        d_probe: float
            Distance between mic and probe"""

        self.mp_lst.append(MeasurementPoint(
            number, (self.d_mic, self.d_probe), _signal, pos))
        self.n_mp += 1

    def del_mp(self, number):
        """Delete MeasurementPoint by number.

        Parameters
        ----------
        number: int
            Number of MeasurementPoint to delete."""

        for i, el in enumerate(self.mp_lst):
            if el.no == number:
                del self.mp_lst[i]
                # return 0

    def plot_overview(self):
        """Plots an overview of Source, Probe and Mics.

        All units in [m

        Returns
        -------
        fig: plt.Figure
            Figure Object handle
        ax: plt.Axis
            Axis Object handle"""

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = []
        y = []
        n = []
        z = []
        # bet = []
        for el in self.mp_lst:
            pos = el.pos
            x.append(pos['x'])
            y.append(pos['y'])
            n.append(pos['no'])
            z.append(self.d_mic)
            # bet.append(str(el.beta_in_deg()))

        ax.scatter(x, y, z)
        ax.scatter(0, 0, self.d_probe)
        ax.set_zlim(0, self.d_probe)
        ax.set_xlabel('Plane Coord. [m]')
        ax.set_ylabel('Plane Coord. [m]')
        ax.set_zlabel('Distance Coord. [m]')

        for i, n in enumerate(n):
            ax.text(x[i], y[i], z[i], str(n))

        ax.text(0, 0, self.d_probe, 'Source')
        ax.text(0, 0, 0, 'Probe')

        return fig, ax

    def average_mp(self):
        """Averages all created MeasurementPoint s

        If required make sure to apply Corrections on a mp level.
        Requires no parameters but needs a full mp_lst.
        Returns nothing, but sets the objects average (TF) attribute."""

        n_mp = len(self.mp_lst)
        sum = np.zeros(len(self.mp_lst[0].tf.hf))

        for el in self.mp_lst:
            sum += el.tf.hf
        average = sum / n_mp
        self.average = TransferFunction(xf=self.mp_lst[0].tf.xf,
                                        hf=average)


class MeasurementPoint:
    """Class for one measuremnet point.

    it should be included in the 'Measurement'-class.
    Norm suggests 3x3 measurement grid with distances .4 m,
    numbered like a phone.

    Attributes
    ----------
    no: int
        Number of measurement point
        Set with init
    x: float
        Position rel to probe normal through source
        Set with init
    y: float
        Position rel to probe normal through source
        Set with init
    tf: TransferFunction
        Transffct obj from mp (if multiple avr)
        Set with init
    d_mic: float
        Distance between source and microphone
        Set with init
    d_probe: float
        Distance between microphone and probe
        Set with init
    beta: float
        Reflection angle
        Set with __geo()
    c_geo: float
        Correction coefficient for sound power distribution
        Set with __c_geo
    """
#   Methods
#     -------
#     calc_c_geo(norm) -> if norm: returns traveled sound distance: float
#         Needs gemometry to be set.
#         Calcs the correction factors for geometrical correction.
#         Sets to attribute if not otherwise specified by norm.
#     calc_c_dir() -> 0
#         Currently not implemented bc its assumed,
#         that omnisource has no directivity.
#         Dummy bc Norm specifies correction.
#     apply_c() -> None
#         Applies all calculated corrections to the MPs signal
#     beta_in_deg() -> float
#         Returns reflection angle in degrees

    def __init__(self, number, distances, transfer_function, pos):
        """Init with good defaults or provided values."""
        self.tf = transfer_function

        self.distances = {'mic': distances[0],
                          'probe': distances[1]}

        self.corrections = {'c_geo': 1.,
                            'c_dir': 1.,
                            'c_gain': 1.,
                            'applied': False}

        self.pos = {'x': pos[0],
                    'y': pos[1],
                    'beta': 2*np.pi,
                    'no': number}

    def __geo_norm(self):
        """Performs geometrical calcs concerning the measurement pos.
           works with:
               self.x, self.y, self.d_mic, self.d_probe
           returns:
               travel_to_r    traveled distance from source to reflection point
               travel_from_r  traveled distance from reflection point to mic
               alpha          angle between probe normal and source-mic line
               beta           angle between probe normal and reflected soundray
           sets:
               self.beta      ang betw probe normal and reflected soundrray"""

        x = self.pos['x']  # x-pos on grid
        y = self.pos['y']  # y-pos on grid

        d_mic = self.distances['mic']      # source - mic
        d_probe = self.distances['probe']  # mic - probe

        r_xy = np.sqrt(x**2 + y**2)     # Distance in mic plane
        r_xyz = np.sqrt(d_mic**2 + r_xy**2)  # Direct distance source -mic

        # Traveled distance to and from reflection point
        r_to_ref_xy = r_xy*(d_mic + d_probe)/(2*d_probe + d_mic)
        travel_to_r = np.sqrt((d_mic + d_probe)**2 + r_to_ref_xy**2)
        travel_from_r = np.sqrt(d_probe**2 + (r_xy - r_to_ref_xy)**2)

        alpha = np.arctan(r_xy/d_mic)
        self.pos['beta'] = np.arctan((d_mic + d_probe)/r_to_ref_xy)

        return travel_to_r, travel_from_r, alpha, self.pos['beta'], r_xyz

    def calc_c_geo(self, norm=True):
        """Calculates c_geo, uses __geo_norm()"""

        # Get geometry
        travel_to_r, travel_from_r, _, _, r_xyz = self.__geo_norm()

        # Set c_geo
        if norm:
            self.corrections['c_geo'] = ((travel_to_r + travel_from_r)
                                         / r_xyz)**2
        else:
            return travel_to_r + travel_from_r

    def apply_c(self):
        """Applys all correction values to the transfer fkt"""
        # Are there already corrections set
        mul = self.corrections['c_geo'] * \
            self.corrections['c_dir'] * \
            self.corrections['c_gain']

        # If no corrections set, do it
        if abs(mul-1) < .02:  # If corrections in .98-1.02
            self.calc_c_geo()
            self.calc_c_dir()

        # If no corrections applied, do it
        if not self.corrections['applied']:
            self.tf.hf = np.copy(self.tf.hf) \
                                 * self.corrections['c_geo'] \
                                 * self.corrections['c_dir'] \
                                 * self.corrections['c_gain']
            self.corrections['applied'] = True
            print('c_geo\t '
                  + str(round(self.corrections['c_geo'],  3))
                  + ';\t c_dir\t '
                  + str(round(self.corrections['c_dir'],  3))
                  + ';\t c_gain\t '
                  + str(round(self.corrections['c_gain'], 3)))

    def calc_c_dir(self):  # , signal_direct, signal_ref):
        """Currently not implemented - placeholder """
        _ = self.pos['x']  # bc of code evaluation reasons
        return 0

    def beta_in_deg(self):
        """Returns reflection angle beta in degree"""
        return 180/np.pi*self.pos['beta']
