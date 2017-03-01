""" Regression tests for integrator.py"""

import os
import unittest

import numpy as np

import opendeplete
from opendeplete import results
from opendeplete import utilities
import test.dummy_geometry as dummy_geometry

class TestIntegratorRegression(unittest.TestCase):
    """ Regression tests for opendeplete.integrator().

    These tests run integrator() on a simple test problem described in
    dummy_geometry.py.
    """

    def test_integrator(self):
        """ Integral regression test of integrator algorithm using CE/CM. """

        settings = opendeplete.Settings()
        settings.dt_vec = [0.75, 0.75]
        settings.output_dir = "test_integrator_regression"

        op = dummy_geometry.DummyGeometry(settings)

        # Perform simulation using the MCNPX/MCNP6 algorithm
        opendeplete.integrate(op, opendeplete.ce_cm_c1, print_out=False)

        # Load the files
        res = results.read_results(settings.output_dir + "/results")

        _, y1 = utilities.evaluate_single_nuclide(res, 0, "1", "1", use_interpolation=False)
        _, y2 = utilities.evaluate_single_nuclide(res, 0, "1", "2", use_interpolation=False)

        # Mathematica solution
        s1 = [1.86872629872102, 1.395525772416039]
        s2 = [2.18097439443550, 2.69429754646747]

        tol = 1.0e-13

        self.assertLess(np.absolute(y1[1] - s1[0]), tol)
        self.assertLess(np.absolute(y2[1] - s1[1]), tol)

        self.assertLess(np.absolute(y1[2] - s2[0]), tol)
        self.assertLess(np.absolute(y2[2] - s2[1]), tol)

        # Delete files
        os.remove(os.path.join(settings.output_dir, "results.h5"))
        os.rmdir(settings.output_dir)


if __name__ == '__main__':
    unittest.main()
