""" Full system test suite. """

import unittest

import numpy as np

from opendeplete import results
from opendeplete import utilities

class TestUtilities(unittest.TestCase):
    """ Tests the utilities classes.

    This also tests the results read/write code.
    """

    def test_evaluate_single_nuclide(self):
        """ Tests evaluating single nuclide utility code.
        """

        # Load the reference
        res = results.read_results("test/test_reference.h5")

        x, y = utilities.evaluate_single_nuclide(res, "10000", "Xe135")

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        y_ref = [6.6747328233649218e+08, 3.5519259015137175e+14,
                 3.5677583314840250e+14, 3.3822315389988969e+14]

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)

    def test_evaluate_reaction_rate(self):
        """ Tests evaluating reaction rate utility code.
        """

        # Load the reference
        res = results.read_results("test/test_reference.h5")

        x, y = utilities.evaluate_reaction_rate(res, "10000", "Xe135", "(n,gamma)")

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        xe_ref = np.array([6.6747328233649218e+08, 3.5519259015137175e+14,
                           3.5677583314840250e+14, 3.3822315389988969e+14])
        r_ref = np.array([4.0643479973455909e-05, 3.9300637358953226e-05,
                          4.0927995824013084e-05, 4.2909741038149828e-05])

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, xe_ref * r_ref)


    def test_evaluate_eigenvalue(self):
        """ Tests evaluating eigenvalue
        """

        # Load the reference
        res = results.read_results("test/test_reference.h5")

        x, y = utilities.evaluate_eigenvalue(res)

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        y_ref = [1.192198835792166, 1.1637094014049858, 1.1809904752233054, 1.2209960015790184]


        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)

if __name__ == '__main__':
    unittest.main()
