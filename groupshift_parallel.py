import numpy as np
import pickle

from tqdm import tqdm
import pandas as pd
import warnings
import os


#===============================================================================
# Mechanics
#===============================================================================

class MechanicBase:
    def __init__(self):
        pass

    def apply(self, N, N_adj, G, *args, **kwargs):
        raise NotImplementedError("This should be implemented by subclasses")


class DoNothing(MechanicBase):
    def __init__(self):
        pass

    def apply(self, *args, **kwargs):
        pass

#----------------------------------------------------------
# SCOPE mechanics
#----------------------------------------------------------

class IngroupOnly(MechanicBase):
    def __init__(self):
        pass

    def apply(self, G, N, N_adj, affected_nodes, t):
        members = []
        _, num_groups, _, _ = G.shape
        for node_idx in affected_nodes:
            members_by_group = []
            for gi in range(num_groups):
                # Get the group the node is in at time t
                g = int(N_adj[node_idx, t])

                if g==gi:
                    # Get all members of the group that the node belongs to
                    members_idx = np.where(N_adj[:, t] == g)[0]
                else:
                    members_idx = np.array([])

                if not isinstance(members_idx, np.ndarray):
                    warnings.warn("Array of members selected for scoping is not a NumPy array.", UserWarning)
                members_by_group.append(members_idx)
            members.append(members_by_group)
        return members

class AsymmetricSample(MechanicBase):
    def __init__(self, ingroup_sample_size, outgroup_sample_size, sample_type):
        self.ingroupSS = ingroup_sample_size
        self.outgroupSS = outgroup_sample_size
        self.sample_type = sample_type # "random", "extremity", "proximity"

    def apply(self, G, N, N_adj, affected_nodes, t):
        members = []
        _, num_groups, _, _ = G.shape
        for node_idx in affected_nodes:
            members_by_group = []
            # Get the group the node is in at time t
            g = int(N_adj[node_idx, t])
            for gi in range(num_groups):
                if g==gi:
                    # Get ingroup members
                    members_idx = np.where(N_adj[:, t] == g)[0]
                    match self.sample_type:
                        case "random":
                            members_idx = np.random.choice(members_idx, self.ingroupSS, replace=False) #ISSUE: not protected for sample size > num group members
                        case "extremity":
                            #Grab opinion vector
                            ingroup_opinions = N[members_idx].mean(axis=1)
                            extremity_scores = np.abs(ingroup_opinions - 50)
                            total = extremity_scores.sum()
                            if total == 0:
                                extremity_probs = None  # Defensively random sample if no deviation from center
                            else:
                                #Normalize to probabilities
                                extremity_probs = extremity_scores / total
                            members_idx = np.random.choice(members_idx, self.ingroupSS, replace = False, p = extremity_probs)
                        case "proximity":
                            distances = np.abs(N[members_idx] - N[node_idx]).mean(axis=1)
                            proximity_scores = np.exp(-distances)
                            weights = np.exp(-1 * proximity_scores)
                            proximity_probs = weights / weights.sum()
                            members_idx = np.random.choice(members_idx, self.ingroupSS, replace = False, p = proximity_probs)
                        case _:
                            raise ValueError(f"Sample type {self.sample_type} not recognized.")
                else:
                    # Get outgroup members
                    members_idx = np.where(N_adj[:, t] == gi)[0]
                    match self.sample_type:
                        case "random":
                            members_idx = np.random.choice(members_idx, self.outgroupSS, replace=False) #ISSUE: not protected for sample size > num group members
                        case "extremity":
                            outgroup_opinions = N[members_idx].mean(axis=1)
                            extremity_scores = np.abs(outgroup_opinions - 50)
                            total = extremity_scores.sum()
                            if total == 0:
                                extremity_probs = None  # Defensively random sample if no deviation from center
                            else:
                                #Normalize to probabilities
                                extremity_probs = extremity_scores / total
                            members_idx = np.random.choice(members_idx, self.outgroupSS, replace = False, p = extremity_probs)
                        case "proximity":
                            distances = np.abs(N[members_idx] - N[node_idx]).mean(axis=1)
                            proximity_scores = np.exp(-distances)
                            weights = np.exp(-1 * proximity_scores)
                            proximity_probs = weights / weights.sum()
                            members_idx = np.random.choice(members_idx, self.outgroupSS, replace = False, p = proximity_probs)
                        case _:
                            raise ValueError(f"Sample type {self.sample_type} not recognized.")
                members_by_group.append(members_idx)
            members.append(members_by_group)
        return members

#----------------------------------------------------------
# GLEAN mechanics
#----------------------------------------------------------

class TakeMean(MechanicBase):
    def __init__(self):
        pass

    def apply(self, G, N, N_adj, affected_nodes, t, scope):
        for i, node_idx in enumerate(affected_nodes):
            relevant_scope = scope[i] #Get the scope array associated with the node being affected
            effects = np.zeros_like(G[node_idx, :, :, t]) # (G , dims) matrix
            for g, members_idx in enumerate(relevant_scope):
                if len(members_idx) <= 0:
                    effects[g] = G[node_idx, g, :, t-1] # Maintain previous perception
                else:
                    membervalences = N[members_idx]
                    effects[g] = np.mean(membervalences, axis=0) #Sum over dimensions of the sample
            G[node_idx, :, :, t] = effects

class LagMean(MechanicBase):
    def __init__(self, reluctance):
        self.reluctance = reluctance

    def apply(self, G, N, N_adj, affected_nodes, t, scope):
        for i, node_idx in enumerate(affected_nodes):
            relevant_scope = scope[i] #Get the scope array associated with the node being affected
            effects = G[node_idx, :, :, t-1] # (G , dims) matrix
            for g, members_idx in enumerate(relevant_scope):
                if len(members_idx) <= 0: #Skipping over cases where members_idx is blank, i.e. IngroupOnly()
                    continue
                membervalences = N[members_idx]
                newmean = np.mean(membervalences, axis=0) #Get next mean
                effects[g,:] = effects[g,:] + self.reluctance * (newmean - effects[g,:])
            G[node_idx, :, :, t] = effects

class Attract(MechanicBase):
    def __init__(self, aWidth, aAmp, normalize = False):
        self.aWidth = aWidth
        self.aAmp = aAmp
        self.normalize = normalize

    def apply(self, G, N, N_adj, affected_nodes, t, scope):
        for i, node_idx in enumerate(affected_nodes):
            relevant_scope = scope[i] #Get the scope array associated with the node being affected
            effects = np.zeros_like(G[node_idx, :, :, t]) #(G, dims)
            for g, members_idx in enumerate(relevant_scope):
                if len(members_idx) == 0:
                    effects[g] = G[node_idx, g, :, t-1]
                else:
                    membervalences = N[members_idx]
                    op = G[node_idx, g, :, t-1]
                    pointing = membervalences - op
                    distance = np.abs(membervalences - op)
                    delta_op = pointing * self.aAmp * np.exp(-1 * (1/self.aWidth) * distance)
                    if self.normalize:
                        delta_op = delta_op * (1/len(members_idx))
                    effects[g] = G[node_idx, g, :, t-1] + delta_op.sum(axis=0) #Sums over dims
            G[node_idx, :, :, t] = effects
#----------------------------------------------------------
# SHIFT mechanics
#----------------------------------------------------------

class Repulse(MechanicBase):
    def __init__(self, rWidth, rAmp):
        self.rWidth = rWidth
        self.rAmp = rAmp

    def apply(self, G, N, N_adj, affected_nodes, t):
        _, num_groups, _, _ = G.shape
        for node_idx in affected_nodes:
            effects = np.zeros_like(G[node_idx, :, :, t])
            for g1 in range(num_groups):
                # Calculate repulsion from all other groups based on the previous group's opinion
                op = G[node_idx, g1, :, t-1]
                aggregated_delta = np.zeros(op.shape)

                for g2 in range(num_groups):
                    if g1 != g2:  # We don't want self-repulsion
                        pointing = -(G[node_idx, g2, :, t-1] - op)
                        distance = np.abs(G[node_idx, g2, :, t-1] - op)
                        delta_op = pointing * self.rAmp * np.exp(-1 * (1/self.rWidth) * distance)
                        aggregated_delta += delta_op
                effects[g1] = aggregated_delta
            G[node_idx, :, :, t] = G[node_idx, :, :, t] + effects

class RepulseSpecific(MechanicBase):
    def __init__(self, rWidth):
        self.rWidth = rWidth

    def apply(self, G, N, N_adj):
        raise ValueError("This mechanic is not yet implemented.")
        pass

#===============================================================================
# Simulation class
#===============================================================================
class GroupshiftSim():
    def __init__(self, G, N, N_adj, C, num_nodes, dims, num_groups, timesteps, opinion_range, simnum, SCOPE, GLEAN, SHIFT, folder = None, filename = None):
        # This function assumes the following shapes:
        # N = np.random.uniform(0, 100, size=(num_nodes, dims))    #   Node opinion matrix
        # N_adj = np.zeros((num_nodes, timesteps))    #   N_adj matrix
        # G = np.zeros((num_nodes, num_groups, dims, timesteps)) # Group matrix, with each node having a group perception

        self.G = G                                                              # Group perception matrix
        self.N = N                                                              # Node opinion matrix
        self.N_adj = N_adj                                                      # Node affiliation matrix
        self.C = C                                                              # Affected nodes per timestep
        self.lowvalence = opinion_range[0]
        self.highvalence = opinion_range[1]
        self.SCOPE = SCOPE
        self.GLEAN = GLEAN
        self.SHIFT = SHIFT

        self.folder = folder
        self.filename = filename
        self.simnum = simnum

        self.num_nodes = num_nodes
        self.dims = dims
        self.num_groups = num_groups
        self.timesteps = timesteps
        
        #Pull out num_nodes, num_groups, and dims from the arrays
        #self.num_nodes, self.dims = N.shape
        #_, self.num_groups, _, self.timesteps = G.shape

    def initializeSim(self, seed = None):
        if seed is None:
            #seed = int(f"{os.getpid()}{self.simnum}")
            seed = self.simnum
        np.random.seed(seed)

        # Create N, the node opinion vector. Shape: [node opinion, dimension]
        self.N = np.random.uniform(self.lowvalence, self.highvalence, size=(self.num_nodes, self.dims))

        # Create N_adj, the node group affiliation matrix. Shape: [node affiliation, timestep]
        self.N_adj = np.zeros((self.num_nodes, self.timesteps))
        self.N_adj[:,0] = np.random.randint(0,self.num_groups,size=(self.num_nodes))

        # Create G, the group perception matrix. Shape: [node id, group id, dimension, perception at timestep]
        # Initialize first timestep as the mean of node opinions in each group
        # self.G = np.zeros((self.num_nodes, self.num_groups, self.dims, self.timesteps))
        # sums = np.zeros((self.num_groups, self.dims))
        # np.add.at(sums, self.N_adj[:, 0].astype(int), self.N)
        # counts = np.bincount(self.N_adj[:, 0].astype(int), minlength=self.num_groups)  # Calculate the number of nodes in each group
        # counts[counts == 0] = 1  # Avoid division by zero by replacing zero counts with one (the corresponding sums are zero, so the division result will still be zero)
        # mean_values = sums / counts[:, np.newaxis] # Calculate the mean of node values for each group
        # self.G[:, :, :, 0] = np.tile(mean_values, (self.num_nodes, 1, 1)).reshape(self.num_nodes, self.num_groups, self.dims)

        # Create G, the group perception matrix. Shape: [node id, group id, dimension, perception at timestep]
        # Initialize first timestep randomly
        self.G = np.zeros((self.num_nodes, self.num_groups, self.dims, self.timesteps))
        self.G[:, :, :, 0] = np.random.uniform(self.lowvalence, self.highvalence, size=(self.num_nodes, self.num_groups, self.dims))

        # >>> Create node change order
        temp = 20 #Number of nodes to change per timestep
        self.C = np.random.randint(0,self.num_nodes,size=(self.timesteps, temp))

    def carryOver(self, t):
        # Take assignments from the previous timestep for all nodes
        self.N_adj[:, t] = self.N_adj[:, t-1]

        #Carry over everyone's perceptions to the next time step
        self.G[:,:,:,t] = self.G[:,:,:,t-1]

    def switch_group(self, t, affected_nodes):
        for node_idx in affected_nodes:
            # Node to be updated
            node_values = self.N[node_idx]

            # Compute distances of this node to all group centroids from the previous timestep
            distances = np.linalg.norm(node_values - self.G[node_idx, :, :, t-1], axis=1)

            # Find the closest group
            closest_group = np.argmin(distances)

            # Update group assignment for this node
            self.N_adj[node_idx, t] = closest_group

    def run_simulation(self, *mechanic_params):
        for t in range(1,self.timesteps):
            #Define the nodes affected this timestep
            affected_nodes = self.C[t, :]

            #Perform the carryover mechanic
            self.carryOver(t)

            #Perform the switch groups mechanic
            self.switch_group(t, affected_nodes)

            # Set the scope of the glean effects
            scope = self.SCOPE.apply(self.G, self.N, self.N_adj, affected_nodes, t)

            #Perform glean mechanics
            self.GLEAN.apply(self.G, self.N, self.N_adj, affected_nodes, t, scope)

            #Perform shift mechanics
            self.SHIFT.apply(self.G, self.N, self.N_adj, affected_nodes, t)

            #Lastly, clamp the values down
            for i in affected_nodes:
                self.G[i, :, :, t] = np.clip(self.G[i, :, :, t], self.lowvalence, self.highvalence)

    #----------------------------------------------------------
    # Non-sim functions
    #----------------------------------------------------------

    def plot_group_values(self):
        # Create a figure and axis
        fig, ax = plt.subplots(self.dims, figsize=(10, 5 * self.dims))

        # If there's only one dimension, make ax an array for consistency
        if self.dims == 1:
            ax = [ax]

        # For each dimension and each group
        for d in range(self.dims):
            #perceptionlines = {}
            for group in range(self.num_groups):
                #Calculate mean opinions of constituents
                mean_opinions = []
                for t in range(self.timesteps):
                    group_members = self.N[self.N_adj[:, t] == group]
                    mean_opinion = np.mean(group_members[:, d])
                    mean_opinions.append(mean_opinion)
                ax[d].plot(mean_opinions, '-.', label=f'Mean Opinion Group {group}')

                # Calculate and plot the mean perception of the constituents for the current group over time
                for perception_group in range(self.num_groups):

                    mean_perceptions = []
                    for t in range(self.timesteps):
                        # Get the members of the group
                        group_members = np.where(self.N_adj[:, t] == group)[0]
                        if len(group_members) > 0:
                            mean_perception = np.mean(self.G[group_members, perception_group, d, t])
                        else:
                            mean_perception = np.nan
                        mean_perceptions.append(mean_perception)

                    # Set line style based on whether the group is perceiving itself or another group
                    if group == perception_group:
                        linestyle = '-'
                    else:
                        linestyle = '--'

                    ax[d].plot(mean_perceptions, label=f'Group {group} perception of Group {perception_group}', linestyle=linestyle, alpha=0.6)
                    #perceptionlines[f'Group {group} perception of Group {perception_group}'] = mean_perceptions

                    # if perception_group == group:
                    #   diffs = np.array(mean_perceptions) - np.array(mean_opinions)
                    #   print( np.mean( diffs[self.timesteps - 500:] ) )

            ax[d].set_title(f'Dimension {d + 1}')
            ax[d].set_xlabel('Time')
            ax[d].set_ylabel('Value')
            ax[d].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax[d].set_ylim((self.lowvalence, self.highvalence))

        plt.tight_layout()
        plt.show()

        #return perceptionlines

    def plot_indiv_values(self, node_num, dim_num = 0):
        # Create a figure for the plot
        plt.figure(figsize=(10, 6))

        # Loop through each group and plot the timeseries
        for g in range(self.num_groups):
            # Extract the time series for the current group (assuming the last dimension is time)
            timeseries = self.G[node_num, g, dim_num, :]

            # Plot the time series
            plt.plot(timeseries, label=f'Group {g}')  # You can customize the label if necessary

        # Add plot labels and legend
        plt.xlabel('Time')
        plt.ylabel(f'Perception for Node {node_num}, Dimension {dim_num}')
        plt.title(f'Time Series of Perceptions for Node {node_num + 1} (Dimension {dim_num + 1})')
        plt.ylim((0,100))
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def pickle(self):
        if self.folder is not None and self.filename is not None:
            with open(f'{self.folder}/{self.filename}.pkl', 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Folder and filename must be specified for saving.")
        
    def clearSim(self):
        self.G = None
        self.N = None
        self.N_adj = None
        self.C = None


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