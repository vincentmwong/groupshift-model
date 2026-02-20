import numpy as np
import pickle

from tqdm import tqdm
import pandas as pd
import warnings
import os
from datetime import datetime
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

def run_and_collect(sim):
    sim.initializeSim()
    sim.run_simulation()

    rows = sim.outputMetrics()

    sim.clearSim()

    return rows

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

def setupSims(num_sims, sim_config, SCOPE, GLEAN, SHIFT, foldername=None):
    num_nodes = sim_config["num_nodes"]
    dims = sim_config["dims"]
    lowvalence = sim_config["lowvalence"]
    highvalence = sim_config["highvalence"]
    temp = sim_config["temp"]
    timesteps = sim_config["timesteps"]
    num_groups = sim_config["num_groups"]
    init_method = "initgauss"
    G = np.array([])
    N = np.array([])
    N_adj = np.array([])
    C = np.array([])
    offset = np.random.randint(1, 1_000_000)

    simlist = []
    for simnum in range(num_sims):
        offset_simnum = offset + simnum
        simulation = GroupshiftSim(G, N, N_adj, C, num_nodes, dims, num_groups, timesteps, (lowvalence, highvalence), offset_simnum, temp, init_method, SCOPE, GLEAN, SHIFT, folder = foldername, filename = f'simulation{simnum}')
        simlist.append(simulation)

    return simlist


def sims_to_run(num_sims_per_type, sim_config):
    sims_to_run = []

    aWidth_list = [4.0]
    rWidth_list = [4.0]
    aAmp_list = [0.2]
    rAmp_list = [1.0, 3.0, 10.0]
    sample_type_list = ["random", "extremity", "proximity"]

    for aWidth in aWidth_list:
        for rWidth in rWidth_list:
            for aAmp in aAmp_list:
                for rAmp in rAmp_list: 
                    for sample_type in sample_type_list:
                        SCOPE = AsymmetricSample(sim_config["group_sample_size"], sim_config["group_sample_size"], sample_type)
                        GLEAN = Attract(aWidth, aAmp)
                        SHIFT = Repulse(rWidth, rAmp)
                        sims_to_run += setupSims(num_sims_per_type, sim_config, SCOPE, GLEAN, SHIFT)

    return sims_to_run
#===================================================================
# MAIN
#===================================================================

if __name__ == "__main__":
    print("Setting up basic variables...")
    # >>> Set up sim params common across all sims
    num_sims_per_type = 50
    num_workers = 16
    sim_config = {
        "num_nodes": 1000,
        "num_groups": 2,
        "timesteps": 10000,
        "dims": 1,
        "lowvalence": 0,
        "highvalence": 100,
        "temp": 100, 
        "group_sample_size": 100
    }

    sims_to_run = sims_to_run(num_sims_per_type, sim_config)

    print(f'Running {len(sims_to_run)} simulations...')

    output_folder = "/l/nx/data/groupshift/simdata/"
    os.makedirs(output_folder, exist_ok=True)

    # Construct filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename = f"results_{timestamp}.csv"

    output_path = os.path.join(output_folder, filename)
    first_write = True

    #all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_and_collect, sim) for sim in sims_to_run]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result_rows = future.result()
                #all_rows.extend(result_rows)

                df_partial = pd.DataFrame(result_rows)
                df_partial.to_csv(
                    output_path,
                    mode='a',
                    header=first_write,
                    index=False
                )

                first_write = False
            except Exception as exc:
                print(f'Simulation generated an exception: {exc}')
    
    # df = pd.DataFrame(all_rows)
    # df.to_csv(output_path, index=False)
    # print("Results saved:", output_path)

    print("All simulations completed.")

else:
    from matplotlib import pyplot as plt
    import seaborn as sns