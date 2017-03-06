""" Tests for integrator.py """

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import scipy.sparse as sp

import opendeplete
from opendeplete import integrator

class TestIntegrator(unittest.TestCase):
    """ Tests for integrator.py

    It is worth noting that opendeplete.integrate is extremely complex, to
    the point I am unsure if it can be reasonably unit-tested.  For the time
    being, it will be left unimplemented and testing will be done via
    regression (in test_integrator_regression.py)
    """

    def test_compute_x(self):
        """ Test the creation of a substep vector."""
        # TODO : When the EL algorithms get published, switch testing to that.

        coeffs = opendeplete.ce_cm_c1

        # Construct f, x

        mat11 = [[-1.0, 0.0], [-2.0, -3.0]]
        mat12 = [[-4.0, -2.0], [-1.5, -3.0]]

        mat21 = [[-0.5, 0.0], [0.0, -2.0]]
        mat22 = [[-3.1, -2.2], [-0.5, -2.0]]

        x11 = [1.0, 1.0]
        x12 = [0.5, 0.7]

        x21 = [0.65, 0.1]
        x22 = [0.1, 0.2]

        f = []
        x = []

        f.append([sp.csr_matrix(mat11), sp.csr_matrix(mat12)])
        f.append([sp.csr_matrix(mat21), sp.csr_matrix(mat22)])

        x.append([np.array(x11), np.array(x12)])
        x.append([np.array(x21), np.array(x22)])

        dt = 0.1

        class test_operator():
            def form_matrix(self, rates, i):
                return rates[i]

        op = test_operator()

        z = integrator.compute_x(op, coeffs, f, x, dt, 1, print_out=False)

        # Solution from mathematica
        z0 = np.array((0.951229424500714, 0.818730753077982))
        z1 = np.array((0.249202138820186, 0.556735711120623))

        tol = 1.0e-15

        self.assertLess(np.linalg.norm(z[0] - z0), tol)
        self.assertLess(np.linalg.norm(z[1] - z1), tol)

    def test_compute_max_relerr(self):
        """ Test the relative error code for stepsize"""

        v1 = [np.array((1.0 + 3.0e-6, 1.0 + 1.0e-6, 0.1 + 1.0e-6)),
              np.array((2.0, 3.0, 0.01 + 1.0e-6))]
        v2 = [np.array((1.0, 1.0, 0.1)), np.array((2.0, 3.0, 0.01))]

        relerr = integrator.compute_max_relerr(v1, v2)

        self.assertAlmostEqual(relerr, 1.0e-6 / (0.01 + 1.0e-6))

    @patch('opendeplete.integrator.Results')
    def test_compute_results(self, mock_results):
        """ Test the polynomial construction """
        coeffs = opendeplete.ce_cm_c1

        # Construct x

        x11 = [1.0, 1.0]
        x21 = [0.0, 0.0]
        x31 = [0.5, 0.7]
        x41 = [0.0, 0.0]
        x51 = [1.5, 0.2]
        x61 = [0.0, 0.0]
        x71 = [2.5, 0.2]

        x12 = [0.65, 0.1]
        x22 = [0.0, 0.0]
        x32 = [0.1, 0.2]
        x42 = [0.0, 0.0]
        x52 = [0.2, 1.2]
        x62 = [0.0, 0.0]
        x72 = [0.2, 0.2]

        x = []

        x.append([np.array(x11), np.array(x12)])
        x.append([np.array(x21), np.array(x22)])
        x.append([np.array(x31), np.array(x32)])
        x.append([np.array(x41), np.array(x42)])
        x.append([np.array(x51), np.array(x52)])
        x.append([np.array(x61), np.array(x62)])
        x.append([np.array(x71), np.array(x72)])

        op = MagicMock()

        vol_list = [1.0, 1.0]
        nuc_list = ["na", "nb"]
        burn_list = ["a", "b"]

        op.get_results_info.return_value = vol_list, nuc_list, burn_list, burn_list

        results = integrator.compute_results(op, coeffs, x)

        # Assert allocated
        results.allocate.assert_called_once_with(vol_list, nuc_list, burn_list, burn_list, 4)

        # Assert calls
        # Due to how mock handles inputs, assertion of arrays must be through numpy
        calls = [(("a", "na"), np.array((1.0, 1.5, -7.0, 5.0))),
                 (("a", "nb"), np.array((1.0, 0.2, -1.5, 1.0))),
                 (("b", "na"), np.array((0.65, 0.2, -2.25, 1.5))),
                 (("b", "nb"), np.array((0.1, 1.2, -2.3, 1.2)))]

        actual_calls = results.__setitem__.call_args_list

        for i, acall in enumerate(actual_calls):

            a0 = acall[0][0]
            a1 = acall[0][1]

            t0 = calls[i][0]
            t1 = calls[i][1]

            self.assertEqual(a0, t0)
            np.testing.assert_array_almost_equal(a1, t1)

    def test_CRAM16(self):
        """ Test 16-term CRAM. """
        x = np.array([1.0, 1.0])
        mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
        dt = 0.1

        z = integrator.CRAM16(mat, x, dt)

        # Solution from mathematica
        z0 = np.array((0.904837418035960, 0.576799023327476))

        tol = 1.0e-15

        self.assertLess(np.linalg.norm(z - z0), tol)

    def test_CRAM48(self):
        """ Test 48-term CRAM. """
        x = np.array([1.0, 1.0])
        mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
        dt = 0.1

        z = integrator.CRAM48(mat, x, dt)

        # Solution from mathematica
        z0 = np.array((0.904837418035960, 0.576799023327476))

        tol = 1.0e-15

        self.assertLess(np.linalg.norm(z - z0), tol)


if __name__ == '__main__':
    unittest.main()
