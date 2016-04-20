import openmc
import os
import time
from subprocess import call
from results import *


class Settings:
    def __init__(self):
        # OpenMC specific
        self.cross_sections = None
        """str: Path to cross sections."""
        self.chain_file = None
        """str: Path to depletion chain XML."""
        self.openmc_call = None
        """list of str: Contains the way to call openmc (see subprocess.call)"""
        self.particles = None
        """int: Number of neutrons per batch."""
        self.batches = None
        """int: Number of batches total."""
        self.inactive = None
        """int: Number of inactive batches."""

        # Depletion problem specific
        self.power = None
        """float: Power in MeV/cm."""
        self.dt_vec = None
        """np.array: Array of time steps, in seconds."""
        self.output_dir = None
        """str: Directory to output results."""


def initialize(geo, settings):
    # Clear out OpenMC
    openmc.reset_auto_material_id()
    openmc.reset_auto_surface_id()
    openmc.reset_auto_cell_id()
    openmc.reset_auto_universe_id()

    # Set up data
    geo.load_participating(settings.cross_sections)
        
    # Get initial conditions
    geo.generate_initial_number_density()
    
    # Output initial model
    geo.generate_geometry_xml()
    geo.generate_materials_xml()
    geo.generate_settings_xml(settings)
    
    geo.load_depletion_data(settings.chain_file)
    geo.generate_tally_xml()
    
    return geo.total_density_list()

def function_evaluation(geo, v2, settings):
    # Update status
    geo.set_density(v2)
    
    # Recreate model
    geo.generate_materials_xml()
    geo.generate_tally_xml()
    geo.generate_settings_xml(settings)
    
    # Run model
    devnull = open(os.devnull, 'w')

    t1 = time.time()
    call(settings.openmc_call)

    statepoint_name = "statepoint." + str(settings.batches) + ".h5"
    
    # Extract results
    t2 = time.time()
    k = geo.unpack_tallies_and_normalize(statepoint_name, settings.power)
    t3 = time.time()
    os.remove(statepoint_name) 
    mat = geo.depletion_matrix_list()
    t4 = time.time()

    rates = extract_rates(geo)

    print("Time to openmc: ", t2-t1)
    print("Time to unpack: ", t3-t2)
    print("Time to matrix: ", t4-t3)
    
    return mat, k, rates, geo.seed
