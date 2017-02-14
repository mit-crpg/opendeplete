""" The OpenMC wrapper module.

This module implements the OpenDeplete -> OpenMC linkage.
"""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from opendeplete.reaction_rates import ReactionRates

class DummyGeometry:
    """ This is a dummy geometry class with no statistical uncertainty.

    y_1' = sin(y_2) y_1 + cos(y_1) y_2
    y_2' = -cos(y_2) y_1 + sin(y_1) y_2

    y_1(0) = 1
    y_2(0) = 1

    y_1(1.5) ~ 2.3197067076743316
    y_2(1.5) ~ 3.1726475740397628

    """

    def __init__(self):
        """ Dummy function.  All inputs ignored."""
        return

    def initialize(self):
        """ Dummy function.  All inputs ignored."""
        return

    def function_evaluation(self, vec, settings):
        """ Evaluates F(y)

        Parameters
        ----------
        vec : list of numpy.array
            Total atoms to be used in function.
        settings : Settings
            Ignored.

        Returns
        -------
        mat : list of scipy.sparse.csr_matrix
            Matrices for the next step.
        k : float
            Zero.
        rates : ReactionRates
            Reaction rates from this simulation.
        seed : int
            Zero.
        """

        y_1 = vec[0][0]
        y_2 = vec[0][1]

        mat = np.zeros((2, 2))
        a11 = np.sin(y_2)
        a12 = np.cos(y_1)
        a21 = -np.cos(y_2)
        a22 = np.sin(y_1)

        mat = [sp.csr_matrix(np.array([[a11, a12], [a21, a22]]))]

        # Create a fake rates object

        return mat, 0.0, self.reaction_rates, 0

    @property
    def volume(self):
        """
        volume : OrderedDict[float]
            Given a material ID, gives the volume of said material.
        """

        volume = {1 : 0}

        return volume

    @property
    def total_number(self):
        """
        total_number : OrderedDict of int to OrderedDict of str to float
            The number density of a nuclide in a cell multiplied by the volume of
            the cell.  Indexed as total_number[cell ID : int][nuclide : str].
        """

        total_number = OrderedDict()
        total_number["1"] = OrderedDict()
        total_number["1"]["1"] = 1.0
        total_number["1"]["2"] = 1.0

        return total_number

    @property
    def nuc_list(self):
        """
        nuc_list : list of str
            A list of all nuclide names. Used for sorting the simulation.
        """

        return ["1", "2"]

    @property
    def burn_list(self):
        """
        burn_list : list of str
            A list of all cell IDs to be burned.  Used for sorting the simulation.
        """

        return ["1"]

    @property
    def reaction_rates(self):
        """
        reaction_rates : ReactionRates
            Reaction rates from the last operator step.
        """
        cell_to_ind = {"1" : 0}
        nuc_to_ind = {"1" : 0, "2" : 1}
        react_to_ind = {"1" : 0}

        return ReactionRates(cell_to_ind, nuc_to_ind, react_to_ind)

    def start(self):
        """ Returns initial vector.

        Returns
        -------
        list of numpy.array
            Total density for initial conditions.
        """

        return [np.array((1.0, 1.0))]

    def get_non_participating_nuc(self):
        """ Returns a nested dictionary of nuclides not participating.

        Returns
        -------
        not_participating : dict of str to dict of str to float
            Not participating nuclides, indexed by cell id and nuclide id.

        """

        not_participating = {}

        for mat in self.burn_list:

            not_participating[mat] = {}

            # Get all nuclides that don't exist in chain but do in total_number
            for nuc in self.total_number[mat]:
                if nuc not in self.nuc_list:
                    not_participating[mat][nuc] = self.total_number[mat][nuc]

        return not_participating

    def fill_nuclide_list(self):
        """ Dummy function """
        return

    @property
    def n_nuc(self):
        """Number of nuclides considered in the decay chain."""
        return len(self.chain.nuclides)
