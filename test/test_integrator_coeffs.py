""" Tests for integrator_coeffs.py """

import unittest

import numpy as np

from opendeplete import integrator_coeffs

class TestIntegratorCoefficients(unittest.TestCase):
    """ Tests for Integrator class. """

    def test_stages(self):
        """ Test the stages parameter. """

        coeffs = integrator_coeffs.Integrator()

        coeffs.d = np.array(((1.0, 1.0), (1.0, 1.0)))

        self.assertEqual(coeffs.stages, 2)


if __name__ == '__main__':
    unittest.main()
