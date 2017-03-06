"""depletion_chain module.

This module contains information about a depletion chain.  A depletion chain is
loaded from an .xml file and all the nuclides are linked together.
"""

from collections import OrderedDict, defaultdict
import math
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse as sp

from .nuclide import Nuclide, Yield


class DepletionChain(object):
    """ The DepletionChain class.

    This class contains a full representation of a depletion chain.

    Attributes
    ----------
    n_nuclides : int
        Number of nuclides in chain.
    nuclides : list of Nuclide
        List of nuclides in chain.
    nuclide_dict : OrderedDict of str to int
        Maps a nuclide name to an index in nuclides.
    precursor_dict : OrderedDict of str to int
        Maps a nuclide name to an index in yields.fis_yield_data
    yields : Yield
        Yield object for fission.
    nuc_to_react_ind : OrderedDict of str to int
        Dictionary mapping a nuclide name to an index in ReactionRates.
    react_to_ind : OrderedDict of str to int
        Dictionary mapping a reaction name to an index in ReactionRates.
    """

    def __init__(self):
        self.nuclides = []

        self.nuclide_dict = OrderedDict()
        self.precursor_dict = OrderedDict()

        self.yields = None

        self.nuc_to_react_ind = OrderedDict()
        self.react_to_ind = OrderedDict()

    @property
    def n_nuclides(self):
        """Number of nuclides in chain."""
        return len(self.nuclides)

    def xml_read(self, filename):
        """ Reads a depletion chain xml file.

        Parameters
        ----------
        filename : str
            The path to the depletion chain xml file.

        Todo
        ----
            Allow for branching on capture, etc.
        """

        # Create variables
        self.react_to_ind = OrderedDict()
        self.nuclide_dict = OrderedDict()

        # Load XML tree
        try:
            root = ET.parse(filename)
        except:
            if filename is None:
                print("No chain specified, either manually or in environment variable OPENDEPLETE_CHAIN.")
            else:
                print('Decay chain "', filename, '" is invalid.')
            raise

        # Read nuclide tables
        decay_node = root.find('decay_constants')

        reaction_index = 0

        for nuclide_index, nuclide_node in enumerate(
                decay_node.findall('nuclide_table')):
            nuc = Nuclide()

            # Just set it to zero to ensure it's set
            nuc.yield_ind = 0

            nuc.name = nuclide_node.get('name')

            self.nuclide_dict[nuc.name] = nuclide_index

            # Check for half-life
            if 'half_life' in nuclide_node.attrib:
                nuc.half_life = float(nuclide_node.get('half_life'))

            # Check for decay paths
            for decay_node in nuclide_node.iter('decay_type'):
                nuc.decay_target.append(decay_node.get('target'))
                nuc.decay_type.append(decay_node.get('type'))
                nuc.branching_ratio.append(
                    float(decay_node.get('branching_ratio')))

            # Check for reaction paths
            for reaction_node in nuclide_node.iter('reaction_type'):
                r_type = reaction_node.get('type')

                # Add to total reaction types
                if r_type not in self.react_to_ind:
                    self.react_to_ind[r_type] = reaction_index
                    reaction_index += 1

                nuc.reaction_type.append(r_type)
                # If the type is not fission, get target, otherwise
                # just set the variable to exists.
                if r_type != 'fission':
                    nuc.reaction_target.append(reaction_node.get('target'))
                else:
                    nuc.reaction_target.append(0)
                    nuc.fission_power = float(reaction_node.get('energy'))

            self.nuclides.append(nuc)

        # Read neutron induced fission yields table
        nfy_node = root.find('neutron_fission_yields')

        self.yields = Yield()

        # Create and load all the variables
        n_fis_prod = int(nfy_node.find('nuclides').text)

        temp = nfy_node.find('precursor_name').text
        self.yields.precursor_list = temp.split()

        temp = nfy_node.find('energy').text
        self.yields.energy_list = [float(x) for x in temp.split()]

        self.yields.energy_dict = OrderedDict()

        # Form dictionaries out of inverses of lists
        for energy_index, energy in enumerate(self.yields.energy_list):
            self.yields.energy_dict[energy] = energy_index

        for precursor_index, precursor in enumerate(self.yields.precursor_list):
            self.precursor_dict[precursor] = precursor_index

        # Allocate variables
        self.yields.fis_yield_data = np.zeros((n_fis_prod,
                                               self.yields.n_energies,
                                               self.yields.n_precursors))

        self.yields.fis_prod_dict = OrderedDict()

        # For eac fission product
        for product_index, yield_table_node in enumerate(
                nfy_node.findall('nuclide_table')):
            name = yield_table_node.get('name')
            self.yields.name.append(name)

            nuc_ind = self.nuclide_dict[name]

            self.nuclides[nuc_ind].yield_ind = product_index

            # For each energy (table)
            for fy_table in yield_table_node.findall('fission_yields'):
                energy = float(fy_table.get('energy'))

                energy_index = self.yields.energy_dict[energy]

                self.yields.fis_prod_dict[name] = product_index
                temp = fy_table.find('fy_data').text
                self.yields.fis_yield_data[product_index, energy_index, :] = \
                    [float(x) for x in temp.split()]

    def form_matrix(self, rates):
        """ Forms depletion matrix.

        Parameters
        ----------
        rates : numpy.ndarray
            2D array indexed by nuclide then by cell.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix representing depletion.
        """

        matrix = defaultdict(float)

        for i, nuc in enumerate(self.nuclides):

            if nuc.n_decay_paths != 0:
                # Decay paths
                # Loss
                decay_constant = math.log(2) / nuc.half_life

                if decay_constant != 0.0:
                    matrix[i, i] -= decay_constant

                # Gain
                for j in range(nuc.n_decay_paths):
                    target_nuc = nuc.decay_target[j]

                    # Allow for total annihilation for debug purposes
                    if target_nuc != 'Nothing':
                        k = self.nuclide_dict[target_nuc]

                        branch_val = nuc.branching_ratio[j] * decay_constant

                        if branch_val != 0.0:
                            matrix[k, i] += branch_val

            if nuc.name in self.nuc_to_react_ind:
                # Extract all reactions for this nuclide in this cell
                nuc_ind = self.nuc_to_react_ind[nuc.name]
                nuc_rates = rates[nuc_ind, :]

                for j in range(nuc.n_reaction_paths):
                    path = nuc.reaction_type[j]
                    # Extract reaction index, and then final reaction rate
                    r_id = self.react_to_ind[path]
                    path_rate = nuc_rates[r_id]

                    # Loss term
                    if path_rate != 0.0:
                        matrix[i, i] -= path_rate

                    # Gain term
                    target_nuc = nuc.reaction_target[j]

                    # Allow for total annihilation for debug purposes
                    if target_nuc != 'Nothing':
                        if path != 'fission':
                            k = self.nuclide_dict[target_nuc]
                            if path_rate != 0.0:
                                matrix[k, i] += path_rate
                        else:
                            m = self.precursor_dict[nuc.name]

                            for k in range(self.yields.n_fis_prod):
                                l = self.nuclide_dict[self.yields.name[k]]
                                # Todo energy
                                yield_val = self.yields.fis_yield_data[k, 0, m] * path_rate
                                if yield_val != 0.0:
                                    matrix[l, i] += yield_val

        # Use DOK matrix as intermediate representation, then convert to CSR and return
        matrix_dok = sp.dok_matrix((self.n_nuclides, self.n_nuclides))
        matrix_dok.update(matrix)
        return matrix_dok.tocsr()

    def nuc_by_ind(self, ind):
        """ Extracts nuclides from the list by dictionary key.

        Parameters
        ----------
        ind : str
            Name of nuclide.

        Returns
        -------
        Nuclide
            Nuclide object that corresponds to ind.
        """
        return self.nuclides[self.nuclide_dict[ind]]
