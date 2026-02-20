import numpy as np
import pickle

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
        self.name = "ingrouponly"
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

# class AsymmetricSampleOriginal(MechanicBase):
#     def __init__(self, ingroup_sample_size, outgroup_sample_size, sample_type):
#         self.ingroupSS = ingroup_sample_size
#         self.outgroupSS = outgroup_sample_size
#         self.sample_type = sample_type # "random", "extremity", "proximity"

#     def apply(self, G, N, N_adj, affected_nodes, t):
#         members = []
#         _, num_groups, _, _ = G.shape
#         for node_idx in affected_nodes:
#             members_by_group = []
#             # Get the group the node is in at time t
#             g = int(N_adj[node_idx, t])
#             for gi in range(num_groups):
#                 if g==gi:
#                     # Get ingroup members
#                     members_idx = np.where(N_adj[:, t] == g)[0]
#                     actual_sample_size = len(members_idx) if len(members_idx) < self.ingroupSS else self.ingroupSS
#                     match self.sample_type:
#                         case "random":
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace=False) #ISSUE: not protected for sample size > num group members
#                         case "extremity":
#                             #Grab opinion vector
#                             ingroup_opinions = N[members_idx].mean(axis=1)
#                             extremity_scores = np.abs(ingroup_opinions - 50)
#                             total = extremity_scores.sum()
#                             if total == 0:
#                                 extremity_probs = None  # Defensively random sample if no deviation from center
#                             else:
#                                 #Normalize to probabilities
#                                 extremity_probs = extremity_scores / total
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace = False, p = extremity_probs)
#                         case "proximity":
#                             distances = np.abs(N[members_idx] - N[node_idx]).mean(axis=1)
#                             proximity_scores = np.exp(-distances)
#                             weights = np.exp(-1 * proximity_scores)
#                             proximity_probs = weights / weights.sum()
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace = False, p = proximity_probs)
#                         case _:
#                             raise ValueError(f"Sample type {self.sample_type} not recognized.")
#                 else:
#                     # Get outgroup members
#                     members_idx = np.where(N_adj[:, t] == gi)[0]
#                     actual_sample_size = len(members_idx) if len(members_idx) < self.outgroupSS else self.outgroupSS
#                     match self.sample_type:
#                         case "random":
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace=False) #ISSUE: not protected for sample size > num group members
#                         case "extremity":
#                             outgroup_opinions = N[members_idx].mean(axis=1)
#                             extremity_scores = np.abs(outgroup_opinions - 50)
#                             total = extremity_scores.sum()
#                             if total == 0:
#                                 extremity_probs = None  # Defensively random sample if no deviation from center
#                             else:
#                                 #Normalize to probabilities
#                                 extremity_probs = extremity_scores / total
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace = False, p = extremity_probs)
#                         case "proximity":
#                             distances = np.abs(N[members_idx] - N[node_idx]).mean(axis=1)
#                             proximity_scores = np.exp(-distances)
#                             weights = np.exp(-1 * proximity_scores)
#                             proximity_probs = weights / weights.sum()
#                             members_idx = np.random.choice(members_idx, actual_sample_size, replace = False, p = proximity_probs)
#                         case _:
#                             raise ValueError(f"Sample type {self.sample_type} not recognized.")
#                 members_by_group.append(members_idx)
#             members.append(members_by_group)
#         return members

def vectorized_node_sampling(P, set1_indices, set2_indices, is_in_set1, X, Y):
    # Gumbel-max trick function
    # Purpose: This function uses the Gumbel-max trick to randomly sample nodes in bulk. 
    # This function is basically for choosing a number of ingroup and outgroup members per node 
    # where each node's sampling is independent with a different weighted probability, P_x, 
    # without replacement. Instead of running np.random.choice() for each node x group, which is very slow, 
    # we can use the Gumbel-max trick to randomly generate values in bulk. 
    """
    Returns:
        members: list of length N
            each entry = [ingroup_samples, outgroup_samples]
    """
    N, M = P.shape

    # Gumbel noise
    noise = -np.log(-np.log(np.random.uniform(0, 1, size=(N, M))))
    scores = np.log(P + 1e-20) + noise

    # Split scores
    scores_s1 = scores[:, set1_indices]
    scores_s2 = scores[:, set2_indices]

    members = []

    for i in range(N):
        if is_in_set1[i]:
            # ingroup = set1
            own_scores = scores_s1[i]
            other_scores = scores_s2[i]

            own_idx = np.argpartition(-own_scores, X)[:X]
            other_idx = np.argpartition(-other_scores, Y)[:Y]

            own = set1_indices[own_idx]
            other = set2_indices[other_idx]
        else:
            # ingroup = set2
            own_scores = scores_s2[i]
            other_scores = scores_s1[i]

            own_idx = np.argpartition(-own_scores, X)[:X]
            other_idx = np.argpartition(-other_scores, Y)[:Y]

            own = set2_indices[own_idx]
            other = set1_indices[other_idx]

        members.append([own, other])

    return members


class AsymmetricSample(MechanicBase):
    def __init__(self, ingroup_sample_size, outgroup_sample_size, sample_type):
        self.ingroupSS = ingroup_sample_size
        self.outgroupSS = outgroup_sample_size
        self.sample_type = sample_type # "random", "extremity", "proximity"
        self.name = "asymmetricsample"

    def apply(self, G, N, N_adj, affected_nodes, t):
        members = []
        _, num_groups, _, _ = G.shape

        group_ids = N_adj[:, t]
        set1_indices = np.where(group_ids == 0)[0]
        set2_indices = np.where(group_ids == 1)[0]

        opinions = N.mean(axis=1)

        # Number of affected nodes
        num_aff = len(affected_nodes)

        if self.sample_type == "random":
            # Uniform probability over all nodes
            P = np.ones((num_aff, N.shape[0]))
            P /= P.sum(axis=1, keepdims=True)

        elif self.sample_type == "extremity":
            # Extremity-based probabilities (same for all nodes)
            opinions = N.mean(axis=1)
            extremity = np.abs(opinions - 50)

            # Avoid divide-by-zero
            extremity = extremity + 1e-12
            base_probs = extremity / extremity.sum()

            # Tile across affected nodes
            P = np.tile(base_probs, (num_aff, 1))

        elif self.sample_type == "proximity":
            # Node-specific proximity probabilities
            all_opinions = N.mean(axis=1)

            # Extract opinions of affected nodes
            node_opinions = all_opinions[affected_nodes]

            # Compute distance matrix:
            # rows = affected nodes
            # columns = all agents
            distances = np.abs(node_opinions[:, None] - all_opinions[None, :])

            # Convert to proximity weights
            proximity_scores = np.exp(-distances)

            # Normalize row-wise
            P = proximity_scores / proximity_scores.sum(axis=1, keepdims=True)

        else:
            raise ValueError(f"Sample type {self.sample_type} not recognized.")

        node_groups = group_ids[affected_nodes]
        is_in_set1 = (node_groups == 0)

        members = vectorized_node_sampling( P, set1_indices, set2_indices, is_in_set1, self.ingroupSS, self.outgroupSS )
        return members

#----------------------------------------------------------
# GLEAN mechanics
#----------------------------------------------------------

class TakeMean(MechanicBase):
    def __init__(self):
        self.name = "takemean"
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
        self.name = "lagmean"

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
        self.name = "attract"

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

#=============================================================================================================================================
# Simulation class
#=============================================================================================================================================
class GroupshiftSim():
    def __init__(self, G, N, N_adj, C, num_nodes, dims, num_groups, timesteps, opinion_range, simnum, temp, init_method, SCOPE, GLEAN, SHIFT, folder = None, filename = None):
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
        self.temp = temp
        self.init_method = init_method

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
        # G allows for different initialization methods for testing purposes
        if self.init_method == "initmeans":
            # ------ Initialize first perception values as the mean of node opinions in each group
            self.G = np.zeros((self.num_nodes, self.num_groups, self.dims, self.timesteps))
            sums = np.zeros((self.num_groups, self.dims))
            np.add.at(sums, self.N_adj[:, 0].astype(int), self.N)
            counts = np.bincount(self.N_adj[:, 0].astype(int), minlength=self.num_groups)  # Calculate the number of nodes in each group
            counts[counts == 0] = 1  # Avoid division by zero by replacing zero counts with one (the corresponding sums are zero, so the division result will still be zero)
            mean_values = sums / counts[:, np.newaxis] # Calculate the mean of node values for each group
            self.G[:, :, :, 0] = np.tile(mean_values, (self.num_nodes, 1, 1)).reshape(self.num_nodes, self.num_groups, self.dims)
        elif self.init_method == "initrand":
            # ------ Initialize first perception values randomly
            # Create G, the group perception matrix. Shape: [node id, group id, dimension, perception at timestep]
            self.G = np.zeros((self.num_nodes, self.num_groups, self.dims, self.timesteps))
            self.G[:, :, :, 0] = np.random.uniform(self.lowvalence, self.highvalence, size=(self.num_nodes, self.num_groups, self.dims))
        elif self.init_method == "initgauss":
            # ------ Initialize the first perception values
            # Create G
            mu = np.array([46, 54])      # mean perception for group 0 and 1
            sigma = 10                  # or whatever spread you want

            self.G = np.zeros((self.num_nodes, self.num_groups, self.dims, self.timesteps))

            for g in range(self.num_groups):
                self.G[:, g, :, 0] = np.random.normal(
                    loc=mu[g],
                    scale=sigma,
                    size=(self.num_nodes, self.dims)
                )
        else:
            raise ValueError(f"Simulation initialization type {self.init_method} not recognized.")

        # >>> Create node change order
        self.C = np.random.randint(0,self.num_nodes,size=(self.timesteps, self.temp))

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

    def outputMetrics(self, dim=0, end_window=500):
        """
        Compute per-group summary metrics for this simulation.
        Returns a list of dicts (one per group).
        """

        T = self.timesteps
        end_slice = slice(T - end_window, T)

        rows = []

        for group in range(self.num_groups):

            # --- membership ---
            members_init = np.where(self.N_adj[:, 0] == group)[0]
            members_fin  = np.where(self.N_adj[:, T - 1] == group)[0]

            if len(members_fin) == 0:
                continue

            # --- true opinions ---
            true_vals_fin = self.N[members_fin, dim]
            true_mean = np.mean(true_vals_fin)
            true_var  = np.var(true_vals_fin)

            # --- perceptions ---
            percept_init = np.mean(
                self.G[members_init, group, dim, 0]
            )

            percept_fin = np.mean(
                self.G[members_fin, group, dim, end_slice]
            )

            outgroup = 1 - group

            percept_outgroup_fin = np.mean(
                self.G[members_fin, outgroup, dim, end_slice]
            )

            # --- store ---
            rows.append({
                "simnum": self.simnum,
                "init_type": self.init_method,
                "temp": self.temp,
                "sample_method": self.SCOPE.sample_type,
                "aAmp": self.GLEAN.aAmp, 
                "aWidth": self.GLEAN.aWidth,
                "rAmp": self.SHIFT.rAmp,
                "rWidth": self.SHIFT.rWidth,
                "GroupID": group,
                "true_mean": float(true_mean),
                "true_var": float(true_var),
                "group_size_init": int(len(members_init)),
                "group_size_fin": int(len(members_fin)),
                "percept_init": float(percept_init),
                "percept_fin": float(percept_fin),
                "percept_outgroup_fin": float(percept_outgroup_fin),
            })

        return rows



if __name__ == '__main__':
    print("This is the GroupshiftSim class. Please run groupshift_parallel.py to execute simulations.")
else: 
    from matplotlib import pyplot as plt
    import seaborn as sns