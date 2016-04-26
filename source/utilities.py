""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import numpy as np
import os
import fnmatch
import pickle
import results
import scipy
import scipy.stats


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


def get_eigval_average(dir_list):
    """ Get eigenvalues as a function of time for a set of simulations.

    This function extracts the eigenvalue from several different simulation
    directories and merges them together.  It is assumed that each directory
    was run precisely identically.

    Parameters
    ----------
    directory : List[str]
        List of directories to read from.

    Returns
    -------
    time : np.array
        Time for each step.
    mu : np.array
        Eigenvalue average for each step.
    std_val : np.array
        Eigenvalue standard deviation for each step.
    p_value : np.array
        Shapiro-Wilk p-value
    """

    # First, calculate how many step files are in each folder

    count_list = [0 for directory in dir_list]
    for i in range(len(dir_list)):
        directory = dir_list[i]
        for file in os.listdir(directory):
            if fnmatch.fnmatch(file, 'step*'):
                count_list[i] += 1

    # Allocate result
    count = min(count_list)
    val = np.zeros((count, len(dir_list)))
    time = np.zeros(count)

    # Read in file, get eigenvalue, close file

    for i in range(len(dir_list)):
        directory = dir_list[i]
        for file in os.listdir(directory):
            if fnmatch.fnmatch(file, 'step*'):
                # Get ind (files will be found out of order)
                name = file.split(".")
                ind = int(name[0][4::])

                # Do not extract data past the end of the minimum number of run
                # steps.
                if ind >= count:
                    continue

                # Read file
                result = results.read_results(directory + '/' + file)

                # Extract results
                val[ind, i] = result.k
                time[ind] = result.time

    # Perform statistics on result
    r_stats = scipy.stats.describe(val, axis=1)

    mu = r_stats.mean
    std_val = np.sqrt(r_stats.variance) / np.sqrt(len(dir_list))
    p_val = [scipy.stats.shapiro(b)[1] for b in val]

    return time, mu, std_val, p_val


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
