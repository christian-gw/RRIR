# %% codecell
import unittest as unit
from Ambi import *


class ambiSig_test(unit.TestCase):
    def test___create_b_format(self):
        self.assertEqual(test(1), 1)
        self.assertAlmostEqual(0, 0)
        self.assertListEqual([0, 0, 0], [0, 0, 0])


# %%codecell
if __name__ == "__main__":
    unit.main()
