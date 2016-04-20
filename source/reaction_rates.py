"""ReactionRates module.

Just contains a dictionary of np.arrays to store reaction rates.
"""

import numpy as np


class ReactionRates:
    def __init__(self, chain):
        self.rate = {}
        """dict: Dictionary of fission products to indexes."""

        for nuclide in chain.nuclides:
            self.rate[nuclide.name] = np.zeros(nuclide.n_reaction_paths)
