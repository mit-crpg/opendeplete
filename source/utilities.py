""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import numpy as np
import os
import fnmatch
import pickle
import results


def get_eigval(directory):
    """ Get eigenvalues as a function of time.

    Parameters
    ----------
    directory : str
        Directory to read results from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : np.array
        Eigenvalue for each step.
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = np.zeros(count)
    time = np.zeros(count)

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            result = results.read_results(directory + '/' + file)

            # Extract results
            val[ind] = result.k
            time[ind] = result.time
    return time, val


def get_atoms(directory, cell_list, nuc_list):
    """ Get total atom count as a function of time.

    Parameters
    ----------
    directory : str
        Directory to read results from.
    cell_list : List[int]
        List of cell IDs to extract data from.
    nuc_list : List[str]
        List of nuclides to extract data from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : Dict[Dict[np.array]]
        Total number of atoms, indexed [cell id : int][nuclide : str]
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = {}
    time = np.zeros(count)

    for cell in cell_list:
        val[cell] = {}
        for nuc in nuc_list:
            val[cell][nuc] = np.zeros(count)

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            result = results.read_results(directory + '/' + file)

            for cell in cell_list:
                if str(cell) in result.num[0].cell_to_ind:
                    for nuc in nuc_list:
                        if nuc in result.num[0].nuc_to_ind:
                            val[cell][nuc][ind] = result.num[0][str(cell), nuc]
            time[ind] = result.time
    return time, val


def get_atoms_volaveraged(directory, cell_list, nuc_list):
    """ Get volume averaged atom count as a function of time.

    This function sums the atom concentration from each cell and then divides
    by the volume sum.

    Parameters
    ----------
    directory : str
        Directory to read results from.
    cell_list : List[int]
        List of cell IDs to average.
    nuc_list : List[str]
        List of nuclides to extract data from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : Dict[np.array]
        Volume averaged atoms, indexed [nuclide : str]
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = {}
    time = np.zeros(count)

    for nuc in nuc_list:
        val[nuc] = np.zeros(count)

    # Calculate volume of cell_list
    # Load first result
    result_0 = results.read_results(directory + '/step0.pklz')
    vol = 0.0
    for cell in cell_list:
        if cell in result_0.volume:
            vol += result_0.volume[cell]

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            result = results.read_results(directory + '/' + file)

            for cell in cell_list:
                if str(cell) in result.num[0].cell_to_ind:
                    for nuc in nuc_list:
                        if nuc in result.num[0].nuc_to_ind:
                            val[nuc][ind] += result.num[0][str(cell), nuc]/vol
            time[ind] = result.time
    return time, val

def get_reaction_rate(directory, cell_list, nuc_list, reaction):
    """ Gets the reaction rate.

    The reaction rate is specifically result.rate_bar * result.concentration,
    as reaction rates are divided by atom density prior to utilization.

    Parameters
    ----------
    directory : str
        Directory to read results from.
    cell_list : List[int]
        List of cell IDs to extract data from.
    nuc_list : List[str]
        List of nuclides to extract data from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : Dict[Dict[np.array]]
        Reaction rate, indexed [cell id : int][nuclide : str]
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = {}
    time = np.zeros(count)

    for cell in cell_list:
        val[cell] = {}
        for nuc in nuc_list:
            val[cell][nuc] = np.zeros(count)

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            result = results.read_results(directory + '/' + file)

            for cell in cell_list:
                if str(cell) in result.num[0].cell_to_ind:
                    for nuc in nuc_list:
                        if nuc in result.num[0].nuc_to_ind:
                            val[cell][nuc][ind] = \
                                result.num[0][str(cell), nuc] * \
                                result.rate_bar[str(cell), nuc, reaction]
            time[ind] = result.time
    return time, val
