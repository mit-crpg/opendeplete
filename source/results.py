class Results:

    def __init__(self, geo, eigvl, d_vec, rates, weights, seeds, time):
        import copy
        # Fills a new Results array with data from geometry
        self.seed = None
        self.k = eigvl

        # It is assumed that num[0] is the beginning-of-time value
        self.num = []

        for vec in d_vec:
            # Set number densities of geo to d_vec to get ids as well.
            geo.set_density(vec)

            # geo.total_number is already in the right format:
            self.num.append(copy.deepcopy(geo.total_number))

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

def extract_rates(geo):
    rates = {}
    # Extract results
    for cell in geo.burn_list:
        rates[cell] = {}
        for nuclide in geo.chain.nuclides:
            name = nuclide.name
            rates[cell][name] = {}
            for i in range(nuclide.n_reaction_paths):
                rates[cell][name][nuclide.reaction_type[i]] = geo.reaction_rates[cell].rate[name][i]
    return rates

def merge_results(rates_array, weights_array):
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
                        rates[c][n][r] = rates_array[i][c][n][r] * weights_array[i]
                    else:
                        rates[c][n][r] += rates_array[i][c][n][r] * weights_array[i]
    return rates

def write_results(geo, eigvl, d_vec, rates, weights, seeds, time, ind):
    import pickle
    # Create results
    res = Results(geo, eigvl, d_vec, rates, weights, seeds, time)

    # Pickle resutls
    output = open('step' + str(ind) + '.pkl', 'wb')

    pickle.dump(res, output)

    output.close()
