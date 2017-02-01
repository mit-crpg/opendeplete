""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import os
import fnmatch

from .results import read_results

def load_directory(directory):
    """ Reads all resutls files from a directory and stores them in a list.

    Parameters
    ----------
    directory : str
        Directory to read results from.

    Returns
    -------
    results : List[opendeplete.Results]
        The list of results.
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    results = []

    # Read in file, get results
    for i in range(count):
        file = 'step' + str(i + 1) + '.pklz'
        # Read file
        results.append(read_results(directory + '/' + file))

    return results
