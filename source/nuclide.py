"""Nuclide module.

Contains the per-nuclide components of a depletion chain.
"""


class Nuclide:
    """ The Nuclide class.

    Contains everything in a depletion chain relating to a single nuclide.

    Attributes
    ----------
    name : str
        Name of nuclide.
    half_life : float
        Half life of nuclide.
    n_decay_paths : int
        Number of decay pathways.
    decay_target : List[str]
        Names of targets nuclide can decay to.
    branching_ratio : List[float]
        Branching ratio for each target.
    n_reaction_paths : int
        Number of possible reaction pathways.
    reaction_target : List[str]
        List of names of targets of reactions.
    reaction_type : List[str]
        List of names of reactions.
    yield_ind : int
        Index in nuclide.Yield table.
    fission_power : float
        Energy released in a fission, MeV.
    """

    def __init__(self):
        # Information about the nuclide
        self.name = None
        self.half_life = None

        # Decay paths
        self.decay_target = []
        self.decay_type = []
        self.branching_ratio = []

        # Reaction paths and rates
        self.reaction_target = []
        self.reaction_type = []

        self.yield_ind = None

        self.fission_power = 0.0

    @property
    def n_decay_paths(self):
        return len(self.decay_target)

    @property
    def n_reaction_paths(self):
        return len(self.reaction_target)


class Yield:
    """ The Yield class.

    Contains a complete description of fission for a decay chain.

    Attributes
    ----------
    n_fis_prod : int
        Number of fission products.
    n_precursors : int
        Number of precursor nuclides.
    n_energies
        Number of energies.
    name : List[str]
        Name of fission products.
    precursor_list : List[str]
        Name of precursors.
    energy_list : List[float]
        Energy list.
    fis_prod_dict : Dict[int]
        Maps fission product name to index in fis_yield_data.
    energy_dict : Dict[int]
        Maps an energy to an index in fis_yield_data.
    fis_yield_data : np.array
        Fission yield data, indexed as [product ind, energy ind, nuclide ind].
    """

    def __init__(self):
        self.name = []
        self.precursor_list = []
        self.energy_list = []

        self.fis_prod_dict = None
        self.energy_dict = None

        self.fis_yield_data = None

    @property
    def n_fis_prod(self):
        return len(self.name)

    @property
    def n_precursors(self):
        return len(self.precursor_list)

    @property
    def n_energies(self):
        return len(self.energy_list)
