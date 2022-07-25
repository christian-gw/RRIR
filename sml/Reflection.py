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
    # from sml.Signal import Signal
    from sml.Transfer_function import TransferFunction


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
    """Class for handling a whole measurement with multiple Measurement points
       containing one signal each.
       If there are multiple signals (should be),
       averaging before creation of Measurement Point is advised.
       Attributes:
         self.m_name   Name of the Measurement (Meta data class is planned)
         self.d_mic    Distance
         self.d_probe  Distance
         self.mp_lst   List containing all measurementpoints with signals
         self.n_mp     Over all number of mps
       Methods:
         self.create_mp()     creates a measurement point
         self.del_mp()        deletes a measurement point by number
         self.plot_overview() plots an overview of all measurement points
                              relative to source and probe
         """

    def __init__(self, name, d_mic, d_probe):
        self.m_name = name
        self.d_mic = d_mic
        self.d_probe = d_probe
        self.mp_lst = []
        self.n_mp = 0
        self.average = []

    def create_mp(self, number, _signal, pos):
        """Create measurement point object with specific number and _signal
        object for mp Position pos [x,y]"""

        self.mp_lst.append(MeasurementPoint(
            number, (self.d_mic, self.d_probe), _signal, pos))
        self.n_mp += 1

    def del_mp(self, number):
        """Delete measurement point object specified by number.
           No reaction if not present."""

        for i, el in enumerate(self.mp_lst):
            if el.no == number:
                del self.mp_lst[i]
                # return 0

    def plot_overview(self):
        """Plots an overview of Source, Probe and Mics. All units in [m]"""
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
        """
        Averages all mps
        If required make sure to apply Corrections on a mp level
        Parameters
        ----------
        Requires full mp_lst

        Returns
        -------
        Averaged tf
        """
        n_mp = len(self.mp_lst)
        sum = np.zeros(len(self.mp_lst[0].tf.hf))

        for el in self.mp_lst:
            sum += el.tf.hf
        average = sum / n_mp
        self.average = TransferFunction(xf=self.mp_lst[0].tf.xf,
                                        hf=average)


class MeasurementPoint:
    """Class for one measuremnet point,
    it should be included in the 'Measurement'-class.
       Norm suggests 3x3 measurement grid with distances .4 m,
       numbered like a phone.

    Attributes:
     self.no         (init)        number of measurement point
     self.x, self.y  (init as pos) position rel to probe normal through source
     self.tf         (init)        transffct obj from mp (if multiple avr)
     self.d_mic      (init)        distance between source and microphone
     self.d_probe    (init)        distance between microphone and probe
     self.beta       (__geo)       reflection angle
     self.c_geo      (__c_geo)     correction coefficient for sound power
                                         distribution

    Public methods:
     self.calc_c_geo()  Calcs self.c_geo, calls self.__c_geo and self.__geo
     self.calc_c_dir()  Currently not implemented bc its assumed, that
                       omnisource has no directivity
    """

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
            self.corrections['c_geo'] = ((travel_to_r + travel_from_r) / r_xyz)**2
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
