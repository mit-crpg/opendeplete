""" The results module.

Contains results generation and saving capabilities.
"""

import lzma
import pickle

import reaction_rates
import concentrations


class Results:
    """ The Results class.

    This class contains the full output of a single timestep of OpenDeplete.

    Parameters
    ----------
    op : function.Operator
        The operator used to generate these results.
    eigvl : float
        The eigenvalue of the problem.
    d_vec : List[List[numpy.array]]
        List of each substep number density arrays.
    rates : List[reaction_rates.ReactionRates]
        The reaction rates for each substep.
    weights : List[float]
        Weights for each substep to get average rates.
    seeds : List[int]
        Seeds for each substep.
    time : float
        Time at beginning of step.

    Attributes
    ----------
    k : float
        Eigenvalue at beginning of step.
    num : List[concentrations.Concentrations]
        List of total_number.
    rates : List[reaction_rates.ReactionRates]
        The reaction rates for each substep.
    weights : List[float]
        Weights for each substep to get average rates.
    rate_bar : reaction_rates.ReactionRates
        The average reaction rate throughout a timestep.
    seeds : List[int]
        Seeds for each substep.
    time : float
        Time at beginning of step.
    volume : OrderedDict[float]
        Dictionary mapping cell id to volume.
    """

    def __init__(self, op, eigvl, d_vec, rates, weights, seeds, time):
        # Fills a new Results array with data from geometry
        self.k = eigvl

        # It is assumed that num[0] is the beginning-of-time value
        self.num = []

        for vec in d_vec:
            # Set number densities of the operator to d_vec to get ids as well.
            op.set_density(vec)

            # op.total_number is already in the right format:
            concentration = concentrations.Concentrations()
            concentration.convert_nested_dict(op.total_number)
            self.num.append(concentration)

        # Extract rates
        self.rates = rates

        # Save weights
        self.weights = weights

        # Calculate rate_bar
        self.rate_bar = merge_results(rates, weights)

        # Save seeds
        self.seeds = seeds

        # Save time
        self.time = time

        # Save volume
        self.volume = op.geometry.volume


def merge_results(rates_array, weights_array):
    """ Merges rates by weights.

    Parameters
    ----------
    rates_array : List[reaction_rates.ReactionRates]
        The reaction rates for each substep.
    weights_array : List[float]
        The weights of each substep.

    Returns
    -------
    r_bar : reaction_rates.ReactionRates
        Merged reaction rates.
    """

    # First, create an empty rate object
    r_bar = reaction_rates.ReactionRates(rates_array[0].cell_to_ind,
                                         rates_array[0].nuc_to_ind,
                                         rates_array[0].react_to_ind)

    # Then, merge results
    for i in range(len(weights_array)):
        r_bar.rates += rates_array[i].rates

    return r_bar


def write_results(operator, eigvl, d_vec, rates, weights, seeds, time, ind):
    """ Outputs results to a .pkl file.

    Parameters
    ----------
    operator : function.Operator
        The operator used to generate these results.
    eigvl : float
        The eigenvalue of the problem.
    d_vec : List[List[numpy.array]]
        List of each substep number density arrays.
    rates : List[Dict[Dict[Dict[float]]]]
        The reaction rates for each substep.  Indexed
        rates[substep][cell id : int][nuclide : str][reaction path : str].
    weights : List[float]
        Weights for each substep to get average rates.
    seeds : List[int]
        Seeds for each substep.
    time : float
        Time at beginning of step.
    ind : int
        Timestep index.
    """

    # Create results
    res = Results(operator, eigvl, d_vec, rates, weights, seeds, time)

    # Pickle results
    output = lzma.open('step' + str(ind) + '.pklz', 'wb')

    pickle.dump(res, output)

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

    # Unpickle results
    handle = lzma.open(filename, 'rb')

    results = pickle.load(handle)

    handle.close()

    return results
