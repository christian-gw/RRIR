# %%
# Has to grow before anything happens

import numpy as np
from sml.Ambi import create_rot_matrix

def _extract_dir_signal(rot_mat: np.array = np.eye(3)):
    """Get B-Format and direction rot_mat and find directive signal."""

    # First Sketch: Linearcombination of xyz:
    direction = np.dot(rot_mat, np.array([1., 0., 0.]))
    
    # print(direction)
    # print(self.b_format[1:])
    
    weighted = 1/3*direction * self.b_format[1:]
    # print(weighted)
    return sg.Signal(signal_lst_imp=weighted)