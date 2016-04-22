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
        self.n_decay_paths = None
        self.decay_target = None
        self.decay_type = None
        self.branching_ratio = None

        # Reaction paths and rates
        self.n_reaction_paths = None
        self.reaction_target = None
        self.reaction_type = None

        self.yield_ind = None

        self.fission_power = None


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
        self.n_fis_prod = None
        self.n_precursors = None
        self.n_energies = None

        self.name = None
        self.precursor_list = None
        self.energy_list = None

        self.fis_prod_dict = None
        self.energy_dict = None

        self.fis_yield_data = None
