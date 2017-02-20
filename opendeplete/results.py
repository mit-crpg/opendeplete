""" The results module.

Contains results generation and saving capabilities.
"""

from collections import OrderedDict

import numpy as np
import h5py

from .reaction_rates import ReactionRates

RESULTS_VERSION = 1

class Results(object):
    """ Contains output of opendeplete.

    Attributes
    ----------
    k : list of float
        Eigenvalue at beginning, end of step.
    seeds : list of int
        Seeds for each substep.
    time : list of float
        Time at beginning, end of step, in seconds.
    n_cell : int
        Number of cells.
    n_nuc : int
        Number of nuclides.
    p_terms : int
        Polynomial order.
    rates : list of ReactionRates
        The reaction rates for each substep.
    volume : OrderedDict of int to float
        Dictionary mapping cell id to volume.
    final_stage : int
        Index of final stage
    cell_to_ind : OrderedDict of str to int
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict of str to int
        A dictionary mapping nuclide name as string to index.
    data : numpy.array
        Number density polynomial coefficients, stored by cell, then by
        nuclide.
    """

    def __init__(self):
        self.k = None
        self.seeds = None
        self.time = None
        self.p_terms = None
        self.rates = None
        self.volume = None
        self.final_stage = None

        self.cell_to_ind = None
        self.nuc_to_ind = None

        self.data = None

    def allocate(self, operator, p_terms):
        """ Allocates memory of Results.

        Parameters
        ----------
        operator : Function
            The operator used to generate these results.
        p_terms : int
            Terms of polynomial.
        """
        self.p_terms = p_terms
        self.volume = operator.geometry.volume

        # Get mapping dictionaries from operator
        self.cell_to_ind, self.nuc_to_ind = get_dict(operator.total_number, operator.burn_list)

        # Create polynomial storage array
        self.data = np.zeros((self.n_cell, self.n_nuc, self.p_terms))

    @property
    def n_cell(self):
        """Number of cells."""
        return len(self.cell_to_ind)

    @property
    def n_nuc(self):
        """Number of nuclides."""
        return len(self.nuc_to_ind)

    def __getitem__(self, pos):
        """ Retrieves an item from results.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a cell index and a nuc index.  These
            indexes can be strings (which get converted to integers via the
            dictionaries), integers used directly, or slices.

        Returns
        -------
        numpy.array
            The polynomial coefficients at the index of interest.
        """

        cell, nuc = pos
        if isinstance(cell, str):
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        return self.data[cell, nuc, :]

    def __setitem__(self, pos, val):
        """ Sets an item from results.

        Parameters
        ----------
        pos : tuple
            A two-length tuple containing a cell index and a nuc index.  These
            indexes can be strings (which get converted to integers via the
            dictionaries), integers used directly, or slices.
        val : numpy.array
            The value to set the polynomial to.
        """

        cell, nuc = pos
        if isinstance(cell, str):
            cell = self.cell_to_ind[cell]
        if isinstance(nuc, str):
            nuc = self.nuc_to_ind[nuc]

        self.data[cell, nuc, :] = val

    def evaluate(self, cell, nuc, time):
        """ Evaluate a polynomial for a given cell-nuclide combination.

        Parameters
        ----------
        cell : int or str
            Cell index to evaluate at.
        nuc : int or str
            Nuclide to evaluate at.
        time : numpy.array
            Time at which to evaluate the polynomial.

        Returns
        -------
        numpy.array
            The polynomial value corresponding to time.
        """

        # Convert time into unitless time
        time_unitless = (time - self.time[0]) / (self.time[1] - self.time[0])

        return np.polynomial.polynomial.polyval(time_unitless, self[cell, nuc])

    def create_hdf5(self, handle):
        """ Creates file structure for a blank HDF5 file.

        Parameters
        ----------
        handle : h5py.File or h5py.Group
            An hdf5 file or group type to store this in.
        """

        # Create and save the 5 dictionaries:
        # quantities
        #   self.cell_to_ind -> self.volume (TODO: support for changing volumes)
        #   self.nuc_to_ind
        # reactions
        #   self.rates[0].nuc_to_ind (can be different from above, above is superset)
        #   self.rates[0].react_to_ind
        # these are shared by every step of the simulation, and should be deduplicated.

        # Store concentration cell and nuclide dictionaries (along with volumes)

        handle.create_dataset("version", data=RESULTS_VERSION)
        handle.create_dataset("final index", data=self.final_stage)

        cell_list = sorted(self.cell_to_ind.keys())
        nuc_list = sorted(self.nuc_to_ind.keys())
        rxn_list = sorted(self.rates[0].react_to_ind.keys())

        n_cells = len(cell_list)
        n_nuc_number = len(nuc_list)
        n_nuc_rxn = len(self.rates[0].nuc_to_ind)
        n_rxn = len(rxn_list)
        p_terms = self.p_terms
        n_stages = len(self.rates)

        cell_group = handle.create_group("cells")

        for cell in cell_list:
            cell_single_group = cell_group.create_group(cell)
            cell_single_group.attrs["index"] = self.cell_to_ind[cell]
            cell_single_group.attrs["volume"] = self.volume[int(cell)]

        nuc_group = handle.create_group("nuclides")

        for nuc in nuc_list:
            nuc_single_group = nuc_group.create_group(nuc)
            nuc_single_group.attrs["atom number index"] = self.nuc_to_ind[nuc]
            if nuc in self.rates[0].nuc_to_ind:
                nuc_single_group.attrs["reaction rate index"] = self.rates[0].nuc_to_ind[nuc]

        rxn_group = handle.create_group("reactions")

        for rxn in rxn_list:
            rxn_single_group = rxn_group.create_group(rxn)
            rxn_single_group.attrs["index"] = self.rates[0].react_to_ind[rxn]

        # Construct array storage

        handle.create_dataset("number", (0, n_cells, n_nuc_number, p_terms),
                              maxshape=(None, n_cells, n_nuc_number, p_terms),
                              compression="gzip",
                              shuffle=True,
                              dtype='float64')

        handle.create_dataset("reaction rates", (0, n_stages, n_cells, n_nuc_rxn, n_rxn),
                              maxshape=(None, n_stages, n_cells, n_nuc_rxn, n_rxn),
                              compression="gzip",
                              shuffle=True,
                              dtype='float64')

        handle.create_dataset("eigenvalues", (0, n_stages),
                              maxshape=(None, n_stages), dtype='float64')

        handle.create_dataset("seeds", (0, n_stages), maxshape=(None, n_stages), dtype='int64')

        handle.create_dataset("time", (0, 2), maxshape=(None, 2), dtype='float64')

    def to_hdf5(self, handle, index):
        """ Converts results object into an hdf5 object.

        Parameters
        ----------
        handle : h5py.File or h5py.Group
            An hdf5 file or group type to store this in.
        index : int
            What step is this?
        """

        if "/number" not in handle:
            self.create_hdf5(handle)

        # Grab handles
        number_dset = handle["/number"]
        rxn_dset = handle["/reaction rates"]
        eigenvalues_dset = handle["/eigenvalues"]
        seeds_dset = handle["/seeds"]
        time_dset = handle["/time"]

        # Get number of results stored
        number_shape = list(number_dset.shape)
        number_results = number_shape[0]

        if number_results < index:
            # Extend first dimension by 1
            number_shape[0] = index
            number_dset.resize(number_shape)

            rxn_shape = list(rxn_dset.shape)
            rxn_shape[0] = index
            rxn_dset.resize(rxn_shape)

            eigenvalues_shape = list(eigenvalues_dset.shape)
            eigenvalues_shape[0] = index
            eigenvalues_dset.resize(eigenvalues_shape)

            seeds_shape = list(seeds_dset.shape)
            seeds_shape[0] = index
            seeds_dset.resize(seeds_shape)

            time_shape = list(time_dset.shape)
            time_shape[0] = index
            time_dset.resize(time_shape)

        # Add data
        n_stages = len(self.rates)
        number_dset[index-1, :, :, :] = self.data
        for i in range(n_stages):
            rxn_dset[index-1, i, :, :, :] = self.rates[i].rates
        eigenvalues_dset[index-1, :] = self.k
        seeds_dset[index-1, :] = self.seeds
        time_dset[index-1, :] = self.time

    def from_hdf5(self, handle, index):
        """ Loads results object from HDF5.

        Parameters
        ----------
        handle : h5py.File or h5py.Group
            An hdf5 file or group type to load from.
        index : int
            What step is this?
        """

        # Get final stage
        self.final_stage = handle["/final index"].value

        # Grab handles
        number_dset = handle["/number"]
        rxn_dset = handle["/reaction rates"]
        eigenvalues_dset = handle["/eigenvalues"]
        seeds_dset = handle["/seeds"]
        time_dset = handle["/time"]

        self.data = number_dset[index, :, :, :]
        self.k = eigenvalues_dset[index, :]
        self.seeds = seeds_dset[index, :]
        self.time = time_dset[index, :]
        self.p_terms = number_dset.shape[3]
        n_stages = rxn_dset.shape[1]

        # Reconstruct dictionaries
        self.volume = OrderedDict()
        self.cell_to_ind = OrderedDict()
        self.nuc_to_ind = OrderedDict()
        rxn_nuc_to_ind = OrderedDict()
        rxn_to_ind = OrderedDict()

        for cell in handle["/cells"]:
            cell_handle = handle["/cells/" + cell]
            vol = cell_handle.attrs["volume"]
            ind = cell_handle.attrs["index"]

            self.volume[cell] = vol
            self.cell_to_ind[cell] = ind

        for nuc in handle["/nuclides"]:
            nuc_handle = handle["/nuclides/" + nuc]
            ind_atom = nuc_handle.attrs["atom number index"]
            self.nuc_to_ind[nuc] = ind_atom

            if "reaction rate index" in nuc_handle.attrs:
                rxn_nuc_to_ind[nuc] = nuc_handle.attrs["reaction rate index"]

        for rxn in handle["/reactions"]:
            rxn_handle = handle["/reactions/" + rxn]
            rxn_to_ind[rxn] = rxn_handle.attrs["index"]

        self.rates = []
        # Reconstruct reactions
        for i in range(n_stages):
            rate = ReactionRates(self.cell_to_ind, rxn_nuc_to_ind, rxn_to_ind)

            rate.rates = handle["/reaction rates"][index, i, :, :, :]
            self.rates.append(rate)



def get_dict(nested_dict, burn_list):
    """ Given an operator nested dictionary, output indexing dictionaries.

    These indexing dictionaries map cell IDs and nuclide names to indices
    inside of Results.data.

    Parameters
    ----------
    nested_dict : OrderedDict of str to OrderedDict of str to Float
        Dictionary with first index corresponding to cell, second corresponding
        to nuclide, maps to total atom quantity.
    burn_list : list of int
        A list of all cell IDs to be burned.

    Returns
    -------
    cell_to_ind : OrderedDict of str to int
        Maps cell strings to index in array.
    nuc_to_ind : OrderedDict of str to int
        Maps nuclide strings to index in array.
    """

    # First, find a complete set of nuclides
    unique_nuc = set()

    for cell_id in nested_dict:
        if cell_id in burn_list:
            for nuc in nested_dict[cell_id]:
                unique_nuc.add(nuc)

    # Now, form cell_to_ind, nuc_to_ind
    cell_to_ind = OrderedDict()

    i = 0
    for cell_id in nested_dict:
        if cell_id in burn_list:
            cell_to_ind[str(cell_id)] = i
            i += 1

    nuc_to_ind = OrderedDict()
    for i, nuc in enumerate(unique_nuc):
        nuc_to_ind[nuc] = i

    return cell_to_ind, nuc_to_ind

def write_results(result, filename, index):
    """ Outputs result to an .hdf5 file.

    Parameters
    ----------
    result : Results
        Object to be stored in a file.
    filename : String
        Target filename, without extension.
    index : int
        What step is this?
    """

    if index == 1:
        file = h5py.File(filename + ".h5", "w")
    else:
        file = h5py.File(filename + ".h5", "a")

    result.to_hdf5(file, index)

    file.close()


def read_results(filename):
    """ Reads out a list of results objects from an hdf5 file.

    Parameters
    ----------
    filename : str
        The filename to read from, without extension.

    Returns
    -------
    results : list of Results
        The result objects.
    """

    file = h5py.File(filename + ".h5", "r")

    assert file["/version"].value == RESULTS_VERSION

    # Grab handles
    number_dset = file["/number"]

    # Get number of results stored
    number_shape = list(number_dset.shape)
    number_results = number_shape[0]

    results = []

    for i in range(number_results):
        result = Results()
        result.from_hdf5(file, i)
        results.append(result)

    file.close()

    return results
