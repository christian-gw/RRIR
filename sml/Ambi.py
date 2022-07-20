# %%[markdown]
# # Ambisonics
# Class module to handle Ambisonics Signals
# Classes:
#  - Mic - Microphone, Microphone sensetivity
#  - AmbiMic - Ambisonics Mic, Sensitivity, Corection Matrix
#  - ambiSig - A-Format, B-Format, Directive Signal, rotation
# # Source
#
# 2018_Schulze-Forster_The B-Format – ...
# Recording, Auralization and Absorbtion Measurements.pdf
#
#
# asdf

# %%codecell
if __name__ == "__main__":
    import numpy as np
    import Signal as sg
    from scipy import signal
    import matplotlib.pyplot as plt
else:
    import numpy as np
    import sml.Signal as sg
    from scipy import signal
    import matplotlib.pyplot as plt


# %%codecell
class Mic:
    def __init__(self, alpha):
        self.alpha = alpha

    def __basic_pattern_2d(self, phi):
        """Calculates the gain value at angle phi of Mic of given pattern."""
        return self.alpha + (1 - self.alpha) * np.cos(phi)

    def __basic_pattern_3d(self, phi, theta):
        """Calculates the gain value at angle phi of Mic of given pattern."""
        return self.alpha + (1-self.alpha)*np.sin(theta)*np.cos(phi)

    def plot_directivity_2d(self):
        """Plots directivity using self.__basic_pattern_2d."""
        abszissa = np.linspace(0, 2*np.pi, 360)

        plt.polar(abszissa,
                  abs(self.__basic_pattern_2d(abszissa)))

    def plot_directivity_3d(self):
        """Plots directivity using self.__basic_pattern_2d."""

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        phi = np.linspace(0, 2*np.pi, 360)
        theta = np.linspace(0, 2*np.pi, 360)

        P, T = np.meshgrid(phi, theta)
        R = abs(self.__basic_pattern_3d(P, T))
        X = R*np.sin(T)*np.cos(P)
        Y = R*np.sin(T)*np.sin(P)
        Z = R*np.cos(T)

        ax.plot_surface(X, Y, Z)


class AmbiMic:
    """Class to hold information about the Ambisonics-Mic.
    Takes
      r - Radius of 1.O Ambisonics mic
      a - alpha value which specifies Form"""
    def __init__(self,
                 R,
                 a,
                 c=340.0):
        self.c = c
        self.R = R
        self.a = a

        self._calc_positions()
        self._calc_M_AB()
        self._calc_hw_cor_Gerzon()
        self._calc_hxyz_cor_Gerzon()

    def _calc_M_AB(self):
        """Calculate the Matrix to calc wxyz from lf, rb, lb, rf
        according to 2009 Faller p. 3 assuming coincidence
        [W, X, Y, Z]' = M_AB * (S_LF, S_RB, S_LB, S_RF)"""
        a = self.a

        # Other way around
        # b = (1-a)/np.sqrt(6)
        # M_BA = np.array([[a, b, b, b],
        #                  [a, -b, -b, b],
        #                  [a, -b, b, -b],
        #                  [a, b, -b, -b]])

        b = np.sqrt(6)/(4*(1-a))
        self.M_AB = np.array([[1/4*a, 1/4*a, 1/4*a, 1/4*a],
                             [b, -b, -b, b],
                             [b, -b, b, -b],
                             [b, b, -b, -b]])

    def _calc_positions(self):
        """Calculates the position of the 4 Mics"""
        tilt = 1/np.sqrt(2)
        R = self.R
        s1_pos = (R, np.pi/2 + tilt, 0)
        s2_pos = (R, np.pi/2 - tilt, np.pi/2)
        s3_pos = (R, np.pi/2 + tilt, np.pi)
        s4_pos = (R, np.pi/2 - tilt, 3*np.pi/2)
        self.positions = [s1_pos,
                          s2_pos,
                          s3_pos,
                          s4_pos]

    def _calc_hw_cor_Gerzon(self):
        """Generates the filter for W-Component correction
        according to Gerzon_1975. The formula was used from:
        http://pcfarina.eng.unipr.it/Public/B-format/A2B-conversion/A2B.htm"""

        num = [1/3*np.power(self.R/self.c, 2), self.R/self.c, 1]
        den = [1/3*self.R/self.c]
        self.hw = signal.tf2sos(num,
                                den)

    def _calc_hxyz_cor_Gerzon(self):
        """Generates the filter for xyz-Component correction
        according to Gerzon_1975. The formula was used from:
        http://pcfarina.eng.unipr.it/Public/B-format/A2B-conversion/A2B.htm"""

        num = np.sqrt(6) * \
            np.array([1/3*np.power(self.R/self.c, 2), self.R/self.c, 1])
        den = np.sqrt(6)*np.array([1/3*self.R/self.c])
        self.hxyz = signal.tf2sos(num,
                                  den)


class ambiSig:
    """Signal Class for Ambisonics signal.
    Takes a array of 4 Signals which form a A-Format and create a B-Format.
    Order of signals according to Sennheiser Ambeo VR Manual:
    1   Yellow  FLU (Sennheiser)     RB (Faller)
    2   Red     FRD (Sennheiser)     LB (Faller)
    3   Blue    BLD (Sennheiser)     RF (Faller)
    4   Green   BRU (Sennheiser)     LF (Faller)
    Contains methods to manipulate B-Format:
    ambiSig.rotate_b_format()
    ambiSig.extract_dir_signal()"""

    def __init__(self,
                 Signals: list,
                 mic_settings: AmbiMic):

        self.a_format = Signals

        dt = Signals[0].dt
        self.b_format = []

        for s in self.__create_b_format(mic_settings):
            self.b_format.append(sg.Signal(y=s, dt=dt))

    def __create_b_format(self, mic: AmbiMic):
        """Transfere from A-format to B-format
        Gets a_format and mic_settings."""

        lfu, rfd, lbd, rbu = [s.y for s in self.a_format]

        # Assuming perfect coincidence
        # Truer for small frequencies (d_mic=2 cm ~> 4 kHz max)
        # Result from the Y^m_{l,n} -> Y Matrix
        # (m - direction Spherical Harmonics, l - MicNr., n - Order)
        # [SchulzeForster_2018] 2.1.2
        # Source [1][2] in 2018_Schulze-Forster
        w = lfu + rfd + lbd + rbu
        y = lfu - rfd + lbd - rbu
        z = lfu - rfd - lbd + rbu
        x = lfu + rfd - lbd - rbu

        # Coincidence Correction according to [Faller_2009]
        w = signal.sosfiltfilt(mic.hw, w)
        x = signal.sosfiltfilt(mic.hxyz, x)
        y = signal.sosfiltfilt(mic.hxyz, y)
        z = signal.sosfiltfilt(mic.hxyz, z)

        # Correcting nonperfect coincidence
        # 2018_Schulze-Forster_The B Format recording 2.1.4
        # Source [3][] in 2018_Schulze-Forster
        # Needs AmbiMic Class

        return [w, x, y, z]

    def _create_rot_matrix(self, phi, theta, rad=True):
        """Creates rotation matrix around
        phi - rotation angle angle around Z (up)
        theta - rotation angle around Y (L of X)"""

        if not rad:
            phi, theta = np.radians((phi, theta))

        cp = np.cos(phi)
        ct = np.cos(theta)
        sp = np.sin(phi)
        st = np.sin(theta)

        Rz = np.array([[ct, -st, 0],
                       [st,  ct, 0],
                       [00,   0, 1]])
        Ry = np.array([[cp,  0, sp],
                       [00,  1, 0],
                       [-sp, 0, cp]])
        return np.dot(Rz, Ry)

    def _rotate_b_format(self,
                         angle: np.array = np.eye(3)):
        """Rotate the B-Format to a new coordinate system
        Gets self.B_format and rotates it with rotation matrix angle."""
        self.b_rot = np.dot(angle, self.b_format[1:])

    def _extract_dir_signal(self,
                            angle: np.array = np.eye(3)):
        """Get B-Format and direction angle and find directive signal."""

        # First Sketch: Linearcombination of xyz:
        direction = np.dot(angle, np.array([1., 0., 0.]))
        # print(direction)
        # print(self.b_format[1:])
        weighted = 1/3*direction * self.b_format[1:]
        # print(weighted)
        return sg.Signal(signal_lst_imp=weighted)

    def safe_b_format(self, names: dict):
        """Safes the b_format as 4 files in the local directory.
        Filenames are specified in dictionary name."""
        for key in names:
            self.b_format[key-1].write_wav(names[key])

    def get_one_direction(self, phi, theta, rad=True):
        """Setts up the necesarry rotation matrix
        and extracts a signal like a fig8 mic pointet in that direction"""

        rot_mat = self._create_rot_matrix(phi, theta, rad=rad)
        return self._extract_dir_signal(rot_mat)


def test(i):
    """Used to test the unittest in test_Ambi.py"""
    return i


# %%
if __name__ == "__main__":
    # Microphone settings
    am = AmbiMic(1.47, .5)

    # File settings
    path = """C:/Users/gmeinwieserch/Desktop/20220420_Messung im Gang/"""
    files = {1: """Gang_1.WAV""",
             2: """Gang_2.WAV""",
             3: """Gang_3.WAV""",
             4: """Gang_4.WAV"""}

    # Load Files
    sigs_raw = []
    for key in files:
        f_path = path + files[key]
        sigs_raw.append(sg.Signal(path=path,
                        name=files[key]))

    # Initialisation of Ambisonics Signal
    aSig = ambiSig(sigs_raw, am)
    # print(aSig.b_format)
    od_sig = aSig.get_one_direction(30, 30, rad=False)
    print(od_sig)
    od_sig.write_wav('test.wav')
    od_sig.plot_y_t()
# %%
