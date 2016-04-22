""" The results module.

Contains results generation and saving capabilities.
"""


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
    rates : List[Dict[Dict[Dict[float]]]]
        The reaction rates for each substep.  Indexed
        rates[substep][cell id : int][nuclide : str][reaction path : str].
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
    num : List[OrderedDict[OrderedDict[float]]]
        List of total_number, indexed as
        [substep : int][cell id : int][nuclide name : str].
    rates : List[Dict[Dict[Dict[float]]]]
        The reaction rates for each substep.  Indexed:
        rates[substep][cell id : int][nuclide : str][reaction path : str].
    weights : List[float]
        Weights for each substep to get average rates.
    rate_bar : Dict[Dict[Dict[float]]]
        The average reaction rate throughout a timestep. Indexed
        rate_bar[cell id : int][nuclide : str][reaction path : str].
    seeds : List[int]
        Seeds for each substep.
    time : float
        Time at beginning of step.
    """

    def __init__(self, op, eigvl, d_vec, rates, weights, seeds, time):
        import copy
        # Fills a new Results array with data from geometry
        self.k = eigvl

        # It is assumed that num[0] is the beginning-of-time value
        self.num = []

        for vec in d_vec:
            # Set number densities of the operator to d_vec to get ids as well.
            op.set_density(vec)

            # op.total_number is already in the right format:
            self.num.append(copy.deepcopy(op.total_number))

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


def extract_rates(op):
    """ Extracts rates from an operator.

    Parameters
    ----------
    op : function.Operator
        The operator to extract rates from.

    Returns
    -------
    rates : Dict[Dict[Dict[float]]]
        Reaction rates, indexed as
        rates[cell id : int][nuclide name : str][reaction path : str].
    """

    rates = {}
    # Extract results
    for cell in op.burn_list:
        rates[cell] = {}
        for nuclide in op.chain.nuclides:
            name = nuclide.name
            rates[cell][name] = {}
            for i in range(nuclide.n_reaction_paths):
                rates[cell][name][nuclide.reaction_type[i]] = \
                    op.reaction_rates[cell].rate[name][i]
    return rates


def merge_results(rates_array, weights_array):
    """ Merges rates by weights.

    Parameters
    ----------
    rates_array : List[Dict[Dict[Dict[float]]]]
        The reaction rates for each substep.  Indexed:
        rates[substep][cell id : int][nuclide : str][reaction path : str].
    weights_array : List[float]
        The weights of each substep.

    Returns
    -------
    rates : Dict[Dict[Dict[float]]]
        Reaction rates, indexed as
        rates[cell id : int][nuclide name : str][reaction path : str].
    """

    import copy

    # Calculates the merged rates
    rates = {}

    # For each simulation
    for i in range(len(weights_array)):
        # For each cell
        for c in rates_array[i]:
            if c not in rates:
                rates[c] = {}

            # For each nuclide
            for n in rates_array[i][c]:
                if n not in rates[c]:
                    rates[c][n] = {}

                # For each reaction
                for r in rates_array[i][c][n]:
                    if r not in rates[c][n]:
                        rates[c][n][r] = \
                            rates_array[i][c][n][r] * weights_array[i]
                    else:
                        rates[c][n][r] += \
                            rates_array[i][c][n][r] * weights_array[i]
    return rates


def write_results(op, eigvl, d_vec, rates, weights, seeds, time, ind):
    """ Outputs results to a .pkl file.

    Parameters
    ----------
    op : function.Operator
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

    import pickle

    # Create results
    res = Results(op, eigvl, d_vec, rates, weights, seeds, time)

    # Pickle resutls
    output = open('step' + str(ind) + '.pkl', 'wb')

    pickle.dump(res, output)

    output.close()
