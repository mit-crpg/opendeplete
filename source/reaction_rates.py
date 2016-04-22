"""ReactionRates module.

Just contains a dictionary of np.arrays to store reaction rates.
"""

import numpy as np


class ReactionRates:
    """ The Nuclide class.

    Contains everything in a depletion chain relating to a single nuclide.

    Parameters
    ----------
    chain : depletion_chain.DepletionChain
        The depletion chain to construct a reaction rate table from.

    Attributes
    ----------
    rate : Dict[numpy.array]
        Dictionary of reaction rate vectors, indexed by nuclide name.
    """

    def __init__(self, chain):

        self.rate = {}
        """dict: Dictionary of fission products to indexes."""

        for nuclide in chain.nuclides:
            self.rate[nuclide.name] = np.zeros(nuclide.n_reaction_paths)
