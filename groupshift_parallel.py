import numpy as np
import pickle

from tqdm import tqdm
import pandas as pd
import warnings
import os
from groupshift_model import *


#===================================================================
# Parallelizing
#===================================================================

import concurrent.futures
import hashlib

# Function to run a simulation and pickle the result
def run_and_pickle(sim):
    sim.initializeSim()
    sim.run_simulation()
    sim.pickle()
    sim.clearSim()

def make_seed(param_id, simnum):
    key = f"{param_id}-{simnum}"
    hash_bytes = hashlib.sha256(key.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:4], 'little')
    return seed

def ensure_folder_exists(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)

def setupSims(num_sims, num_nodes, num_groups, timesteps, dims, lowvalence, highvalence, temp, SCOPE, GLEAN, SHIFT, foldername):
    G = np.array([])
    N = np.array([])
    N_adj = np.array([])
    C = np.array([])
    offset = np.random.randint(1, 1_000_000)
    simlist = []
    for simnum in range(num_sims):
        offset_simnum = offset + simnum
        simulation = GroupshiftSim(G, N, N_adj, C, num_nodes, dims, num_groups, timesteps, (lowvalence, highvalence), offset_simnum, SCOPE, GLEAN, SHIFT, folder = foldername, filename = f'simulation{simnum}')
        simlist.append(simulation)

    return simlist


def simSetup():
    sims_to_run = []
    aWidth_list = [5]
    rWidth_list = [10]
    aAmp_list = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #rAmp_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 3.0, 10.0]
    rAmp_list = [3.0]

    for aWidth in aWidth_list:
        for rWidth in rWidth_list:
            for aAmp in aAmp_list:
                for rAmp in rAmp_list: 
                    SCOPE = IngroupOnly()
                    GLEAN = Attract(aWidth, aAmp)
                    SHIFT = Repulse(rWidth, rAmp)
                    foldername = f'/l/nx/data/groupshift/{trial_folder_name}/aWidth{aWidth}_aAmp{aAmp}_rWidth{rWidth}_rAmp{rAmp}'
                    ensure_folder_exists(foldername)
                    sims_to_run += setupSims(num_sims, num_nodes, num_groups, timesteps, dims, lowvalence, highvalence, temp, SCOPE, GLEAN, SHIFT, foldername)

    return sims_to_run
#===================================================================
# MAIN
#===================================================================

if __name__ == "__main__":
    print("Setting up basic variables...")
    # >>> Set up sim params common across all sims
    num_sims = 5
    num_workers = 16
    num_nodes = 1000
    num_groups = 2
    timesteps = 10000
    dims = 1
    lowvalence = 0
    highvalence = 100
    temp = 20

    trial_folder_name = "trial2_july7_2025"

    print(f"Setting up simulations to run, for {trial_folder_name}...")

    sims_to_run = simSetup()

    print(f'Running {len(sims_to_run)} simulations...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit each simulation to be run and pickled in parallel
        futures = [executor.submit(run_and_pickle, sim) for sim in sims_to_run]
        
        # Optionally, wait for all simulations to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise an exception if the simulation failed
            except Exception as exc:
                print(f'Simulation generated an exception: {exc}')
else:
    from matplotlib import pyplot as plt
    import seaborn as sns