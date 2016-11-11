"""ReactionRates module.

Just contains a dictionary of np.arrays to store reaction rates.
"""

import numpy as np


class ReactionRates:
    """ The Nuclide class.

    Contains everything in a depletion chain relating to a single nuclide.

    Parameters
    ----------
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    react_to_ind : OrderedDict[int]
        A dictionary mapping reaction name as string to index.

    Attributes
    ----------
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    react_to_ind : OrderedDict[int]
        A dictionary mapping reaction name as string to index.
    n_cell : int
        Number of cells.
    n_nuc : int
        Number of nucs.
    n_react : int
        Number of reactions.
    rates : np.array
        Array storing rates indexed by the above dictionaries.
    """

    def __init__(self, cell_to_ind, nuc_to_ind, react_to_ind):

        self.cell_to_ind = cell_to_ind
        self.nuc_to_ind = nuc_to_ind
        self.react_to_ind = react_to_ind

        self.rates = np.zeros((self.n_cell, self.n_nuc, self.n_react))

    def __getitem__(self, pos):
        """ Retrieves an item from reaction_rates.

        Parameters
        ----------
        pos : tuple
            A three-length tuple containing a cell index, a nuc index, and a
            reaction index.  These indexes can be strings (which get converted
            to integers via the dictionaries), integers used directly, or
            slices.

        Returns
        -------
        np.array
            The value indexed from self.rates.
        """

        cell, nuc, react = pos
        if isinstance(cell, str):
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]
        if isinstance(react, str):
            react = self.react_to_ind[react]

        return self.rates[cell, nuc, react]

    def __setitem__(self, pos, val):
        """ Sets an item from reaction_rates.

        Parameters
        ----------
        pos : tuple
            A three-length tuple containing a cell index, a nuc index, and a
            reaction index.  These indexes can be strings (which get converted
            to integers via the dictionaries), integers used directly, or
            slices.
        val : float
            The value to set the array to.
        """

        cell, nuc, react = pos
        if isinstance(cell, str):
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]
        if isinstance(react, str):
            react = self.react_to_ind[react]

        self.rates[cell, nuc, react] = val

    @property
    def n_cell(self):
        return len(self.cell_to_ind)

    @property
    def n_nuc(self):
        return len(self.nuc_to_ind)

    @property
    def n_react(self):
        return len(self.react_to_ind)
