# %%
# Has to grow before anything happens

import numpy as np
from RRIR.Ambi import create_rot_matrix
from RRIR.Signal import Signal


def _extract_dir_signal(signal: Signal, rot_mat: np.array = np.eye(3)):
    """Get B-Format and direction rot_mat and find directive signal."""

    # First Sketch: Linearcombination of xyz:
    direction = np.dot(rot_mat, np.array([1., 0., 0.]))

    # print(direction)
    # print(self.b_format[1:])

    weighted = 1/3*direction * signal.b_format[1:]
    # print(weighted)
    return Signal(signal_lst_imp=weighted)
