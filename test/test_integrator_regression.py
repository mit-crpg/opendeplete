""" Regression tests for integrator.py"""

import os
import unittest

import numpy as np
from mpi4py import MPI

import opendeplete
from opendeplete import results
from opendeplete import utilities
import test.dummy_geometry as dummy_geometry

class TestIntegratorRegression(unittest.TestCase):
    """ Regression tests for opendeplete.integrator algorithms.

    These tests integrate a simple test problem described in dummy_geometry.py.
    """

    def test_predictor(self):
        """ Integral regression test of integrator algorithm using CE/CM. """

        settings = opendeplete.Settings()
        settings.dt_vec = [0.75, 0.75]
        settings.output_dir = "test_integrator_regression"

        op = dummy_geometry.DummyGeometry(settings)

        # Perform simulation using the predictor algorithm
        opendeplete.predictor(op, print_out=False)

        # Load the files
        res = results.read_results(settings.output_dir + "/results")

        _, y1 = utilities.evaluate_single_nuclide(res, "1", "1")
        _, y2 = utilities.evaluate_single_nuclide(res, "1", "2")

        # Mathematica solution
        s1 = [2.46847546272295, 0.986431226850467]
        s2 = [4.11525874568034, -0.0581692232513460]

        tol = 1.0e-13

        self.assertLess(np.absolute(y1[1] - s1[0]), tol)
        self.assertLess(np.absolute(y2[1] - s1[1]), tol)

        self.assertLess(np.absolute(y1[2] - s2[0]), tol)
        self.assertLess(np.absolute(y2[2] - s2[1]), tol)

        # Delete files

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            os.remove(os.path.join(settings.output_dir, "results.h5"))
            os.rmdir(settings.output_dir)

    def test_cecm(self):
        """ Integral regression test of integrator algorithm using CE/CM. """

        settings = opendeplete.Settings()
        settings.dt_vec = [0.75, 0.75]
        settings.output_dir = "test_integrator_regression"

        op = dummy_geometry.DummyGeometry(settings)

        # Perform simulation using the MCNPX/MCNP6 algorithm
        opendeplete.cecm(op, print_out=False)

        # Load the files
        res = results.read_results(settings.output_dir + "/results")

        _, y1 = utilities.evaluate_single_nuclide(res, "1", "1")
        _, y2 = utilities.evaluate_single_nuclide(res, "1", "2")

        # Mathematica solution
        s1 = [1.86872629872102, 1.395525772416039]
        s2 = [2.18097439443550, 2.69429754646747]

        tol = 1.0e-13

        self.assertLess(np.absolute(y1[1] - s1[0]), tol)
        self.assertLess(np.absolute(y2[1] - s1[1]), tol)

        self.assertLess(np.absolute(y1[2] - s2[0]), tol)
        self.assertLess(np.absolute(y2[2] - s2[1]), tol)

        # Delete files

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            os.remove(os.path.join(settings.output_dir, "results.h5"))
            os.rmdir(settings.output_dir)


if __name__ == '__main__':
    unittest.main()
