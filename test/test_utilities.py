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

        x, y = utilities.evaluate_single_nuclide(res, "1", "Xe135")

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        y_ref = [6.6747328233649218e+08, 3.5519299354614412e+14,
                 3.6365476945913844e+14, 3.4256390732369456e+14]

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)

    def test_evaluate_reaction_rate(self):
        """ Tests evaluating reaction rate utility code.
        """

        # Load the reference
        res = results.read_results("test/test_reference.h5")

        x, y = utilities.evaluate_reaction_rate(res, "1", "Xe135", "(n,gamma)")

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        xe_ref = np.array([6.6747328233649218e+08, 3.5519299354614412e+14,
                           3.6365476945913844e+14, 3.4256390732369456e+14])
        r_ref = np.array([4.0643598574546534e-05, 3.8854747998041196e-05,
                          3.7250974561260465e-05, 3.5633818370102976e-05])

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, xe_ref * r_ref)


    def test_evaluate_eigenvalue(self):
        """ Tests evaluating eigenvalue
        """

        # Load the reference
        res = results.read_results("test/test_reference.h5")

        x, y = utilities.evaluate_eigenvalue(res)

        x_ref = [0.0, 1296000.0, 2592000.0, 3888000.0]
        y_ref = [1.1921986054449398, 1.1783826979724599, 1.1734272255490044, 1.2198294833989853]

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)


if __name__ == '__main__':
    unittest.main()
