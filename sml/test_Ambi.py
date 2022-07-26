# %% codecell
import unittest as unit
import numpy as np
from Ambi import test
from Ambi import create_rot_matrix


# Helpers for testing np.array
def test_array_eq(x, y):
    return unit.TestCase.assertIsNone(np.testing.assert_array_equal(x, y))


def test_array_sim(x, y):
    return unit.TestCase.assertIsNone(np.testing.assert_allclose(x,
                                                                 y,
                                                                 atol=.01))


class ambiSig_test(unit.TestCase):
    def test___create_b_format(self):
        self.assertEqual(test(1), 1)
        self.assertAlmostEqual(0, 0)
        self.assertListEqual([0, 0, 0], [0, 0, 0])

    def test_crate_rot_matrix(self):
        def rotate_vector(phi, theta, vec):
            rot_mat = create_rot_matrix(phi, theta, rad=False)
            rot_vec = np.dot(rot_mat, vec)
            # print(rot_vec)
            return rot_vec

        # Rotate 90° around
        vec = np.array([1, 0, 0])
        test_array_sim(rotate_vector(+0, +0, vec),
                       np.array([+1, +0, +0]))
        test_array_sim(rotate_vector(+90, +0, vec),
                       np.array([+0, +1, +0]))
        test_array_sim(rotate_vector(+0, +90, vec),
                       np.array([+1, +0, +0]))
        test_array_sim(rotate_vector(+90, +90, vec),
                       np.array([+0, +0, +1]))

        # Over 90° and negative
        test_array_sim(rotate_vector(+270, +0, vec),
                       np.array([+0, -1, +0]))
        test_array_sim(rotate_vector(-90, +0, vec),
                       np.array([+0, -1, +0]))
        test_array_sim(rotate_vector(-270, -270, vec),
                       np.array([+0, +0, +1]))
        test_array_sim(rotate_vector(-270, -90, vec),
                       np.array([+0, +0, -1]))

        # Non orthogonal
        test_array_sim(rotate_vector(+30, +0, vec),
                       np.array([np.sqrt(3)/2, .5, 0]))
        test_array_sim(rotate_vector(+60, +0, vec),
                       np.array([.5, np.sqrt(3)/2, 0]))

        vec = np.array([0, 1, 0])
        test_array_sim(rotate_vector(+0, +30, vec),
                       np.array([0, np.sqrt(3)/2, .5]))
        test_array_sim(rotate_vector(+0, +60, vec),
                       np.array([0, .5, np.sqrt(3)/2]))


# %%codecell
if __name__ == "__main__":
    unit.main()

# %%
