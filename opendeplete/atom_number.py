"""AtomNumber module.

An ndarray to store atom densities with string, integer, or slice indexing.
"""

import numpy as np


class AtomNumber(object):
    """ AtomNumber module.

    An ndarray to store atom densities with string, integer, or slice indexing.

    Parameters
    ----------
    mat_to_ind : OrderedDict of str to int
        A dictionary mapping material ID as string to index.
    nuc_to_ind : OrderedDict of str to int
        A dictionary mapping nuclide name as string to index.
    volume : OrderedDict of int to float
        Volume of geometry.
    n_mat_burn : int
        Number of materials to be burned.
    n_nuc_burn : int
        Number of nuclides to be burned.

    Attributes
    ----------
    mat_to_ind : OrderedDict of str to int
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict of str to int
        A dictionary mapping nuclide name as string to index.
    volume : numpy.array
        Volume of geometry indexed by mat_to_ind.  If a volume is not found,
        it defaults to 1 so that reading density still works correctly.
    n_mat_burn : int
        Number of materials to be burned.
    n_nuc_burn : int
        Number of nuclides to be burned.
    n_mat : int
        Number of materials.
    n_nuc : int
        Number of nucs.
    number : numpy.array
        Array storing total atoms indexed by the above dictionaries.
    """

    def __init__(self, mat_to_ind, nuc_to_ind, volume, n_mat_burn, n_nuc_burn):

        self.mat_to_ind = mat_to_ind
        self.nuc_to_ind = nuc_to_ind

        self.volume = np.ones(self.n_mat)

        for mat in volume:
            ind = self.mat_to_ind[str(mat)]
            self.volume[ind] = volume[mat]

        self.n_mat_burn = n_mat_burn
        self.n_nuc_burn = n_nuc_burn

        self.number = np.zeros((self.n_mat, self.n_nuc))

    def __getitem__(self, pos):
        """ Retrieves total atom number from AtomNumber.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a material index and a nuc index.
            These indexes can be strings (which get converted to integers via
            the dictionaries), integers used directly, or slices.

        Returns
        -------
        numpy.array
            The value indexed from self.number.
        """

        mat, nuc = pos
        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        return self.number[mat, nuc]

    def __setitem__(self, pos, val):
        """ Sets total atom number into AtomNumber.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a material index and a nuc index.
            These indexes can be strings (which get converted to integers via
            the dictionaries), integers used directly, or slices.
        val : float
            The value to set the array to.
        """

        mat, nuc = pos
        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        self.number[mat, nuc] = val

    def get_atom_density(self, mat, nuc):
        """ Accesses atom density instead of total number.

        Parameters
        ----------
        mat : str, int or slice
            Material index.
        nuc : str, int or slice
            Nuclide index.

        Returns
        -------
        numpy.array
            The density indexed.
        """

        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        return self[mat, nuc] / self.volume[mat]

    def set_atom_density(self, mat, nuc, val):
        """ Sets atom density instead of total number.

        Parameters
        ----------
        mat : str, int or slice
            Material index.
        nuc : str, int or slice
            Nuclide index.
        val : numpy.array
            Array of values to set.
        """

        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        self[mat, nuc] = val * self.volume[mat]

    def get_mat_slice(self, mat):
        """ Gets atom quantity indexed by mats for all burned nuclides

        Parameters
        ----------
        mat : str, int or slice
            Material index.

        Returns
        -------
        numpy.array
            The slice requested.
        """

        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]

        return self[mat, 0:self.n_nuc_burn]

    def set_mat_slice(self, mat, val):
        """ Sets atom quantity indexed by mats for all burned nuclides

        Parameters
        ----------
        mat : str, int or slice
            Material index.
        val : numpy.array
            The slice to set.
        """

        if isinstance(mat, str):
            mat = self.mat_to_ind[mat]

        self[mat, 0:self.n_nuc_burn] = val

    @property
    def n_mat(self):
        """Number of cells."""
        return len(self.mat_to_ind)

    @property
    def n_nuc(self):
        """Number of nucs."""
        return len(self.nuc_to_ind)
