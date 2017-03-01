""" Tests for depletion_chain.py"""

from collections import OrderedDict
import unittest

import numpy as np

from opendeplete import depletion_chain
from opendeplete import reaction_rates
from opendeplete import nuclide

class TestDepletionChain(unittest.TestCase):
    """ Tests for DepletionChain class."""

    def test__init__(self):
        """ Test depletion chain initialization."""
        dep = depletion_chain.DepletionChain()

        self.assertIsInstance(dep.nuclides, list)
        self.assertIsInstance(dep.nuclide_dict, OrderedDict)
        self.assertIsInstance(dep.react_to_ind, OrderedDict)

    def test_n_nuclides(self):
        """ Test depletion chain n_nuclides parameter. """
        dep = depletion_chain.DepletionChain()

        dep.nuclides = ["NucA", "NucB", "NucC"]

        self.assertEqual(dep.n_nuclides, 3)

    def test_from_endf(self):
        pass

    def test_xml_read(self):
        """ Read chain_test.xml and ensure all values are correct. """
        # Unfortunately, this routine touches a lot of the code, but most of
        # the components external to depletion_chain.py are simple storage
        # types.

        dep = depletion_chain.DepletionChain.xml_read("chains/chain_test.xml")

        # Basic checks
        self.assertEqual(dep.n_nuclides, 3)

        # A tests
        nuc = dep.nuclides[dep.nuclide_dict["A"]]

        self.assertEqual(nuc.name, "A")
        self.assertEqual(nuc.half_life, 2.36520E+04)
        self.assertEqual(nuc.n_decay_paths, 2)
        self.assertEqual(nuc.decay_target, ["B", "C"])
        self.assertEqual(nuc.decay_type, ["beta1", "beta2"])
        self.assertEqual(nuc.branching_ratio, [0.6, 0.4])
        self.assertEqual(nuc.n_reaction_paths, 1)
        self.assertEqual(nuc.reaction_target, ["C"])
        self.assertEqual(nuc.reaction_type, ["(n,gamma)"])

        # B tests
        nuc = dep.nuclides[dep.nuclide_dict["B"]]

        self.assertEqual(nuc.name, "B")
        self.assertEqual(nuc.half_life, 3.29040E+04)
        self.assertEqual(nuc.n_decay_paths, 1)
        self.assertEqual(nuc.decay_target, ["A"])
        self.assertEqual(nuc.decay_type, ["beta"])
        self.assertEqual(nuc.branching_ratio, [1.0])
        self.assertEqual(nuc.n_reaction_paths, 1)
        self.assertEqual(nuc.reaction_target, ["C"])
        self.assertEqual(nuc.reaction_type, ["(n,gamma)"])

        # C tests
        nuc = dep.nuclides[dep.nuclide_dict["C"]]

        self.assertEqual(nuc.name, "C")
        self.assertEqual(nuc.n_decay_paths, 0)
        self.assertEqual(nuc.n_reaction_paths, 2)
        self.assertEqual(nuc.reaction_target, [None, "A"])
        self.assertEqual(nuc.reaction_type, ["fission", "(n,gamma)"])

        # Yield tests
        self.assertEqual(nuc.yield_energies, [0.0253])
        self.assertEqual(list(nuc.yield_data.keys()), [0.0253])
        self.assertEqual(nuc.yield_data[0.0253],
                         [("A", 0.0292737), ("B", 0.002566345)])

    def test_xml_write(self):
        """Test writing a depletion chain to XML."""

        A = nuclide.Nuclide()
        A.name = "A"
        A.half_life = 2.36520e4
        A.decay_target = ["B", "C"]
        A.decay_type = ["beta1", "beta2"]
        A.branching_ratio = [0.6, 0.4]
        A.reaction_target = ["C"]
        A.reaction_type = ["(n,gamma)"]
        A.reaction_Q = [0.0]

        B = nuclide.Nuclide()
        B.name = "B"
        B.half_life = 3.29040e4
        B.decay_target = ["A"]
        B.decay_type = ["beta", ]
        B.branching_ratio = [1.0]
        B.reaction_target = ["C"]
        B.reaction_type = ["(n,gamma)"]
        B.reaction_Q = [0.0]

        C = nuclide.Nuclide()
        C.name = "C"
        C.reaction_target = [None, "A"]
        C.reaction_type = ["fission", "(n,gamma)"]
        C.reaction_Q = [2.0e8, 0.0]
        C.yield_energies = [0.0253]
        C.yield_data = {0.0253: [("A", 0.0292737), ("B", 0.002566345)]}

        chain = depletion_chain.DepletionChain()
        chain.nuclides = [A, B, C]
        chain.xml_write('test.xml')

        original = open('chains/chain_test.xml', 'r').read()
        chain_xml = open('test.xml', 'r').read()
        self.assertEqual(original, chain_xml)

    def test_form_matrix(self):
        """ Using chain_test, and a dummy reaction rate, compute the matrix. """
        # Relies on test_xml_read passing.

        dep = depletion_chain.DepletionChain.xml_read("chains/chain_test.xml")

        cell_ind = {"10000": 0, "10001": 1}
        nuc_ind = {"A": 0, "B": 1, "C": 2}
        react_ind = dep.react_to_ind

        react = reaction_rates.ReactionRates(cell_ind, nuc_ind, react_ind)

        dep.nuc_to_react_ind = nuc_ind

        react["10000", "C", "fission"] = 1.0
        react["10000", "A", "(n,gamma)"] = 2.0
        react["10000", "B", "(n,gamma)"] = 3.0
        react["10000", "C", "(n,gamma)"] = 4.0

        mat = dep.form_matrix(react[0, :, :])
        # Loss A, decay, (n, gamma)
        mat00 = -np.log(2) / 2.36520E+04 - 2
        # A -> B, decay, 0.6 branching ratio
        mat10 = np.log(2) / 2.36520E+04 * 0.6
        # A -> C, decay, 0.4 branching ratio + (n,gamma)
        mat20 = np.log(2) / 2.36520E+04 * 0.4 + 2

        # B -> A, decay, 1.0 branching ratio
        mat01 = np.log(2)/3.29040E+04
        # Loss B, decay, (n, gamma)
        mat11 = -np.log(2)/3.29040E+04 - 3
        # B -> C, (n, gamma)
        mat21 = 3

        # C -> A fission, (n, gamma)
        mat02 = 0.0292737 * 1.0 + 4
        # C -> B fission
        mat12 = 0.002566345 * 1.0
        # Loss C, fission, (n, gamma)
        mat22 = -1.0 - 4.0

        self.assertEqual(mat[0, 0], mat00)
        self.assertEqual(mat[1, 0], mat10)
        self.assertEqual(mat[2, 0], mat20)
        self.assertEqual(mat[0, 1], mat01)
        self.assertEqual(mat[1, 1], mat11)
        self.assertEqual(mat[2, 1], mat21)
        self.assertEqual(mat[0, 2], mat02)
        self.assertEqual(mat[1, 2], mat12)
        self.assertEqual(mat[2, 2], mat22)

    def test_nuc_by_ind(self):
        """ Test nuc_by_ind converter function. """
        dep = depletion_chain.DepletionChain()

        dep.nuclides = ["NucA", "NucB", "NucC"]
        dep.nuclide_dict = {"NucA" : 0, "NucB" : 1, "NucC" : 2}

        self.assertEqual("NucA", dep.nuc_by_ind("NucA"))
        self.assertEqual("NucB", dep.nuc_by_ind("NucB"))
        self.assertEqual("NucC", dep.nuc_by_ind("NucC"))

if __name__ == '__main__':
    unittest.main()
