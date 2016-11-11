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

        self.number = None

    @property
    def n_cell(self):
        return len(self.cell_to_ind)

    @property
    def n_nuc(self):
        return len(self.nuc_to_ind)

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
        self.cell_to_ind = {str(cell_id): i for i, cell_id in
                            enumerate(nested_dict)}

        self.nuc_to_ind = {nuc: i for i, nuc in enumerate(unique_nuc)}

        # Allocate arrays
        self.number = np.zeros((self.n_cell, self.n_nuc))

        # Extract data
        for cell_id, cell_data in nested_dict.items():
            for nuc in cell_data:
                cell_ind = self.cell_to_ind[str(cell_id)]
                nuc_ind = self.nuc_to_ind[nuc]

                self.number[cell_ind, nuc_ind] = cell_data[nuc]

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
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        return self.number[cell, nuc]

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
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        self.number[cell, nuc] = val
