""" Tests for nuclide.py. """

import unittest

from opendeplete import nuclide

class TestNuclide(unittest.TestCase):
    """ Tests for the nuclide class. """

    def test_n_decay_paths(self):
        """ Test the decay path count parameter. """

        nuc = nuclide.Nuclide()

        nuc.decay_target = ["a", "b", "c"]

        self.assertEqual(nuc.n_decay_paths, 3)

    def test_n_reaction_paths(self):
        """ Test the reaction path count parameter. """

        nuc = nuclide.Nuclide()

        nuc.reaction_target = ["a", "b", "c"]

        self.assertEqual(nuc.n_reaction_paths, 3)


class TestYield(unittest.TestCase):
    """ Tests for the yield class. """

    def test_n_fis_prod(self):
        """ Test the fission product count parameter. """

        nuc_yield = nuclide.Yield()

        nuc_yield.name = ["a", "b", "c"]

        self.assertEqual(nuc_yield.n_fis_prod, 3)

    def test_n_precursors(self):
        """ Test the fission product count parameter. """

        nuc_yield = nuclide.Yield()

        nuc_yield.precursor_list = ["a", "b", "c"]

        self.assertEqual(nuc_yield.n_precursors, 3)

    def test_n_energies(self):
        """ Test the energy band count parameter. """

        nuc_yield = nuclide.Yield()

        nuc_yield.energy_list = [1.0, 2.0, 3.0]

        self.assertEqual(nuc_yield.n_energies, 3)


if __name__ == '__main__':
    unittest.main()
