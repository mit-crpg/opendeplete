"""Concentrations module.

Just contains a dictionary of np.arrays to store nuclide concentrations.
"""

import numpy as np


class Concentrations:
    """ The Concentrations class.

    Contains all concentrations from the end of a simulation.

    Attributes
    ----------
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    n_cell : int
        Number of cells.
    n_nuc : int
        Number of nucs.
    number : np.array
        Array storing rates indexed by the above dictionaries.
    """

    def __init__(self):

        self.cell_to_ind = None
        self.nuc_to_ind = None

        self.n_cell = None
        self.n_nuc = None

        self.number = None

    def convert_nested_dict(self, nested_dict):
        """ Converts a nested dictionary to a concentrations.

        This function converts a dictionary of the form
        nested_dict[cell_id][nuc_id] to a concentrations type.  This method
        does not guarantee an order to self.number, and it is fairly
        impractical to do so, so be warned.

        Parameters
        ----------
        nested_dict : Dict[Dict[float]]
            The concentration indexed as nested_dict[cell_id][nuc_id].
        """

        # First, find a complete set of nuclides
        unique_nuc = set()

        for cell_id in nested_dict:
            for nuc in nested_dict[cell_id]:
                unique_nuc.add(nuc)

        # Now, form cell_to_ind, nuc_to_ind
        self.cell_to_ind = {}
        self.nuc_to_ind = {}

        cell_ind = 0
        for cell_id in nested_dict:
            self.cell_to_ind[str(cell_id)] = cell_ind
            cell_ind += 1

        nuc_ind = 0
        for nuc in unique_nuc:
            self.nuc_to_ind[nuc] = nuc_ind
            nuc_ind += 1

        # Set lengths
        self.n_cell = len(self.cell_to_ind)
        self.n_nuc = len(self.nuc_to_ind)

        # Allocate arrays
        self.number = np.zeros((self.n_cell, self.n_nuc))

        # Extract data
        for cell_id in nested_dict:
            for nuc in nested_dict[cell_id]:
                cell_ind = self.cell_to_ind[str(cell_id)]
                nuc_ind = self.nuc_to_ind[nuc]

                self.number[cell_ind, nuc_ind] = nested_dict[cell_id][nuc]

    def __getitem__(self, pos):
        """ Retrieves an item from concentrations.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a cell index and a nuc index.  These
            indexes can be strings (which get converted to integers via the
            dictionaries), integers used directly, or slices.

        Returns
        -------
        np.array
            The value indexed from self.number.
        """

        cell, nuc = pos
        if isinstance(cell, str):
            cell_id = self.cell_to_ind[cell]
        else:
            cell_id = cell
        if isinstance(nuc, str):
            nuc_id = self.nuc_to_ind[nuc]
        else:
            nuc_id = nuc

        return self.number[cell_id, nuc_id]

    def __setitem__(self, pos, val):
        """ Sets an item from concentrations.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a cell index and a nuc index.  These
            indexes can be strings (which get converted to integers via the
            dictionaries), integers used directly, or slices.
        val : float
            The value to set the array to.
        """

        cell, nuc = pos
        if isinstance(cell, str):
            cell_id = self.cell_to_ind[cell]
        else:
            cell_id = cell
        if isinstance(nuc, str):
            nuc_id = self.nuc_to_ind[nuc]
        else:
            nuc_id = nuc

        self.number[cell_id, nuc_id] = val
