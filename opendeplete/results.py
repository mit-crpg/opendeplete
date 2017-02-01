""" The results module.

Contains results generation and saving capabilities.
"""

import lzma
import numpy as np
import dill

class Results(object):
    """

    Attributes
    ----------
    k : List[float]
        Eigenvalue at beginning, end of step.
    seeds : List[int]
        Seeds for each substep.
    time : List[float]
        Time at beginning, end of step, in seconds.
    n_cell : int
        Number of cells.
    n_nuc : int
        Number of nuclides.
    p_order : int
        Polynomial order.
    rates : List[reaction_rates.ReactionRates]
        The reaction rates for each substep.
    volume : OrderedDict[float]
        Dictionary mapping cell id to volume.
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    data : np.array
        Number density polynomial coefficients, stored by cell, then by
        nuclide.
    """

    def __init__(self, op, p_order):
        self.k = []
        self.seeds = []
        self.time = []
        self.p_order = p_order
        self.rates = []
        self.volume = op.geometry.volume

        # Get mapping dictionaries from op
        self.cell_to_ind, self.nuc_to_ind = get_dict(op.total_number)

        # Create polynomial storage array
        self.data = np.zeros((self.n_cell, self.n_nuc, self.p_order))

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
        np.array
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
        val : float
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
        cell : Int or String
            Cell index to evaluate at.
        nuc : Int or String
            Nuclide to evaluate at.
        time : np.array
            Time at which to evaluate the polynomial.

        Returns
        -------
        np.array
            The polynomial value corresponding to time.
        """

        # Convert time into unitless time
        time_unitless = (time - self.time[0]) / (self.time[1] - self.time[0])

        return np.polynomial.polynomial.polyval(time_unitless, self[cell, nuc])

def get_dict(nested_dict):
    """ Given an operator nested dictionary, output indexing dictionaries.

    These indexing dictionaries map cell IDs and nuclide names to indices
    inside of Results.data.

    Parameters
    ----------
    nested_dict : OrderedDict[OrderedDict[Float]]
        Dictionary with first index corresponding to cell, second corresponding
        to nuclide, maps to total atom quantity.

    Returns
    -------
    cell_to_ind : Dict
        Maps cell strings to index in array.
    nuc_to_ind : Dict
        Maps nuclide strings to index in array.
    """

    # First, find a complete set of nuclides
    unique_nuc = set()

    for cell_id in nested_dict:
        for nuc in nested_dict[cell_id]:
            unique_nuc.add(nuc)

    # Now, form cell_to_ind, nuc_to_ind
    cell_to_ind = {str(cell_id): i for i, cell_id in enumerate(nested_dict)}

    nuc_to_ind = {nuc: i for i, nuc in enumerate(unique_nuc)}

    return cell_to_ind, nuc_to_ind

def write_results(results, ind):
    """ Outputs results to a .pkl file.

    Parameters
    ----------
    results : Results
        Object to be stored in a file.
    ind : Int
        Integer corresponding to file name ("step" + ind + ".pklz")
    """

    # dill results
    output = lzma.open('step' + str(ind) + '.pklz', 'wb')

    dill.dump(results, output)

    output.close()


def read_results(filename):
    """ Reads out a results object from a compressed file.

    Parameters
    ----------
    filename : str
        The filename to read from.

    Returns
    -------
    results : Results
        The result object encapsulated.
    """

    # Undill results
    handle = lzma.open(filename, 'rb')

    results = dill.load(handle)

    handle.close()

    return results

def evaluate_result_list(results, n_points, use_interpolation=True):
    """ Evaluates all nuclides in all cells using a given results list.

    Parameters
    ----------
    results : List[Results]
        The results to extract data from.  Must be sorted and continuous.
    n_points : Int
        Number of points, equally spaced, to evaluate on.
    use_interpolation : Bool
        Whether or not to use the algorithm-defined interpolation.
        n_points will be ignored.

    Returns
    -------
    time : np.array
        Time vector.
    concentration : Dict[Dict[np.array]]
        Nested dictionary (indexed by cell, then nuclide) containing values.
    """

    cell_to_ind = results[0].cell_to_ind
    nuc_to_ind = results[0].nuc_to_ind

    if use_interpolation:
        # Get time vector
        time_final = results[-1].time[1]

        time = np.linspace(0, time_final, n_points)

        concentration = {}

        for cell in cell_to_ind:
            concentration[cell] = {}
            for nuc in nuc_to_ind:
                concentration[cell][nuc] = np.zeros(n_points)


                # Evaluate value in each region
                for result in results:
                    ind1 = np.argmax(time >= result.time[0])
                    ind2 = np.argmax(time >= result.time[1])

                    if ind1 == ind2:
                        # ind2 is probably the end
                        ind2 = len(time)

                    if ind2 == 0 and ind1 > 1:
                        ind2 = len(time)-1

                    concentration[cell][nuc][ind1:ind2] = \
                        result.evaluate(cell, nuc, time[ind1:ind2])
    else:
        n_points = len(results) + 1
        time = np.zeros(n_points)
        concentration = {}

        for cell in cell_to_ind:
            concentration[cell] = {}
            for nuc in nuc_to_ind:
                concentration[cell][nuc] = np.zeros(n_points)

                i = 0

                # Evaluate value in each region
                for result in results:

                    time[i] = result.time[0]
                    time[i + 1] = result.time[1]

                    poly = result[cell, nuc]

                    concentration[cell][nuc][i] = poly[0]
                    concentration[cell][nuc][i + 1] = np.sum(poly)

                    i += 1

    return time, concentration
