# %%
# Learning and validating Rotation Matrizes
import numpy as np


def create_rot_matrix(phi, theta, rad=True):
    """Creates rotation matrix around
    phi - rotation angle angle around Z with X = 0
    theta - rotation angle around x Z=0"""

    if not rad:
        phi, theta = np.radians((phi, theta))

    cp = np.cos(phi)
    ct = np.cos(theta)
    sp = np.sin(phi)
    st = np.sin(theta)

    Rx = np.array([[1, -0, 0],
                   [0,  ct, -st],
                   [00,   st, ct]])
    Rz = np.array([[cp,  -sp, 0],
                   [sp,  cp, 0],
                   [0, 0, 1]])

    # S = np.array([[st*cp, ct*cp, -sp],
    #              [st*sp, ct*sp, cp],
    #              [ct, -st, 0]])
    S = np.dot(Rx, Rz)
    # print(np.round(S, 2))
    return S


def rotate_vector(phi, theta, vec):
    rot_mat = create_rot_matrix(phi, theta, rad=False)
    rot_vec = np.dot(rot_mat, vec)
    return rot_vec


# %%

#               x  y  z
vec = np.array([1, 0, 0])
rot_vec = rotate_vector(phi=-270,
                        theta=-90,
                        vec=vec)

print(np.round(rot_vec, 2))

# %%[markdown]
# Rotate 90° around
# vec = np.array([1, 0, 0])
# assertEqual(rotate_vector(+000, +000, vec), [+1, +0, +0])
# assertEqual(rotate_vector(+090, +000, vec), [+0, +1, +0])
# assertEqual(rotate_vector(+000, +090, vec), [+1, +0, +0])
# assertEqual(rotate_vector(+090, +090, vec), [+0, +0, +1])

# assertEqual(rotate_vector(+270, +000, vec), [+0, -1, +0])
# assertEqual(rotate_vector(-090, +000, vec), [+0, -1, +0])
# assertEqual(rotate_vector(-270, -270, vec), [+0, +0, +1])
# assertEqual(rotate_vector(-270, -090, vec), [+0, +0, -1])

# assertAlmostEqual(rotate_vector(+030, +000, vec), [np.sqrt(3)/2, .5, 0])
# assertAlmostEqual(rotate_vector(+060, +000, vec), [.5, np.sqrt(3)/2, 0])

# vec = np.array([0, 1, 0])
# assertAlmostEqual(rotate_vector(+030, +000, vec), [0, np.sqrt(3)/2, .5, 0])
# assertAlmostEqual(rotate_vector(+060, +000, vec), [0, .5, np.sqrt(3)/2, 0])


# %%
