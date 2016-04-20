"""Nuclide module.

Contains the per-nuclide components of a depletion chain.
"""


class Nuclide:

    def __init__(self):
        # Information about the nuclide
        self.name = None
        """str: Name of nuclide."""
        self.half_life = None
        """float: Half life of nuclide."""

        # Decay paths
        self.n_decay_paths = None
        """int: Number of possible decay pathways."""
        self.decay_target = None
        """list: List of names of nuclides it decays to."""
        self.decay_type = None
        """list: The name of the decay method."""
        self.branching_ratio = None
        """list: Branching ratio for each decay path."""

        # Reaction paths and rates
        self.n_reaction_paths = None
        """int: Number of possible reaction pathways."""
        self.reaction_target = None
        """list: List of names of targets of reactions."""
        self.reaction_type = None
        """list: The name of the reaction pathway."""

        self.yield_ind = None
        """int: Index in yield tables."""

        self.fission_power = None
        """float: Fission energy release, MeV."""


class Yield:

    def __init__(self):
        self.n_fis_prod = None
        """int: Number of fission products."""
        self.n_precursors = None
        """int: Number of precursor nuclides."""
        self.n_energies = None
        """int: Number of energies."""

        self.name = None
        """list: Names of products."""
        self.precursor_list = None
        """list: Names of precursors."""
        self.energy_list = None
        """list: Energy list."""

        self.fis_prod_dict = None
        """dict: Dictionary of fission products to indexes."""
        self.energy_dict = None
        """dict: Dictionary of energies to indexes."""

        self.fis_yield_data = None
        """np.array: Fission yield data."""
