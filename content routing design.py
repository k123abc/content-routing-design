#%%
# 
#   Written by "Kais Suleiman" (ksuleiman.weebly.com)
#
#   Notes:
#
#   - The contents of this script have been explained in details 
#   in Chapter 6 of the thesis:
#    
#       Kais Suleiman, "Popular Content Distribution in Public 
#       Transportation Using Artificial Intelligence Techniques.", 
#       Ph.D. thesis, University of Waterloo, Ontario, Canada, 2019.
#
#   - Notice that the very beginning and the very end of this script
#   address the cluster extraction function and operations explained 
#   in details at the end of Chapter 4 of the aforementioned thesis.
#   - Similar to the thesis, the functions to be used throughout 
#   content distribution tasks are introduced first.
#   - Simpler but still similar variable names have been used throughout 
#   this script instead of the mathematical notations used in the thesis.
#   - The assumptions used in the script are the same as those used in 
#   the thesis including those related to the case study considered 
#   representing the Grand River Transit bus service offered throughout 
#   the Region of Waterloo, Ontario, Canada.
#   - Figures are created throughout this script to aid 
#   in thesis visualizations and other forms of results sharing.

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt
from ismember import ismember
import geopy.distance

#%%

#   The extractCluster function:

def extractCluster(period_start_time, period_end_time, \
                   connectivities, time_delta, minimum_contact_duration, \
                       maximum_number_of_hops):
    
    period_connectivities = \
        np.zeros((np.shape(connectivities[0, :, :])[0], \
                  np.shape(connectivities[0, :, :])[0]))
    
    for i in range(np.shape(connectivities[0, :, :])[0]):
    
        for t in range(period_start_time * 6, period_end_time * 6):
        
            if t == period_start_time * 6:
          
                period_connectivities[i, :] = \
                    connectivities[t, i, :]
            
            else:
            
                period_connectivities[i, :] = \
                    period_connectivities[i, :] + \
                        connectivities[t, i, :]
    
        previous_cluster_members[i, :] = \
            np.where(period_connectivities[i, :] * time_delta >= \
                     minimum_contact_duration)

    next_cluster_members = np.copy(previous_cluster_members)

    for n in range(maximum_number_of_hops // 2 - 1):
    
        for i in range(np.shape(connectivities[0, :, :])[0]):
        
            for j in range(np.shape(previous_cluster_members[i, :])[1]):
            
                next_cluster_members[i, :] = \
                    (next_cluster_members[i, :]).union( \
                                                       previous_cluster_members[ \
                                                                                previous_cluster_members[i, 0, j]])
    
        previous_cluster_members = np.copy(next_cluster_members)

    _, cluster_indices = \
        np.unique(next_cluster_members, axis = 0, return_index = True)
    cluster_members = \
        next_cluster_members[np.sort(cluster_indices)]
 
    cluster_sizes = np.zeros((np.shape(cluster_indices)[0],1))

    for i in range(np.shape(cluster_indices)[0]):
    
        cluster_sizes[i] = \
            np.shape(cluster_members[i, :])[1]

    biggest_cluster = \
        np.where(cluster_sizes == np.amax(cluster_sizes))

    biggest_cluster = biggest_cluster[0]
    
    return biggest_cluster, cluster_members

#%%

connectivities = np.load('connectivities.npy', allow_pickle = True)
modified_data = np.load('modified_data.npy', allow_pickle = True)
synthetic_lats = np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)

#   Extracting the first biggest cluster connectivities:
#
#   The first biggest cluster is chosen within the specified
#   period start and end times given the first minimum discontinuous
#   contact duration. The next biggest clusters are chosen afterwards
#   according to the minimum contact durations schedule shown at the
#   end of this script.

period_start_time = 16 * 60
period_end_time = 18 * 60
time_delta = 10
minimum_contact_duration = 20 * 60
maximum_number_of_hops = 20

biggest_cluster, cluster_members = \
    extractCluster(period_start_time, period_end_time, \
                   connectivities, time_delta, \
                       minimum_contact_duration, \
                           maximum_number_of_hops)

biggest_cluster_connectivities = \
    np.zeros((period_end_time * 6 - period_start_time * 6 + 1, \
              np.shape(cluster_members)[1], \
              np.shape(cluster_members)[1]))

for t in range(period_end_time * 6 - period_start_time * 6 + 1):
    
    biggest_cluster_connectivities[t, :, :] = \
        connectivities[period_start_time * 6 + t - 1, \
                       np.transpose(cluster_members[biggest_cluster, :]), \
                           np.transpose(cluster_members[biggest_cluster, :])]

biggest_cluster_members = cluster_members[biggest_cluster, :]

np.savez('biggest_cluster_data.npy', biggest_cluster_connectivities, \
         biggest_cluster_members, period_start_time, period_end_time)

#   Visualizing the first biggest cluster at the busiest time:

busiest_time = (period_start_time + period_end_time) / 2 * 6

figure, ax = plt.subplots(nrows = 1, ncols = 1)

ax.set_title('First biggest cluster @ 5:00 PM')

map = plt.imread('python_map.jpg')
ax.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])

ax.scatter(synthetic_lons[:, busiest_time], synthetic_lats[:, busiest_time], \
           color = 'yellow', marker = 'o', edgecolors = 'black')
ax.scatter(synthetic_lons[biggest_cluster_members, busiest_time], \
           synthetic_lats[biggest_cluster_members, busiest_time], \
               color = 'red', marker = 'o', edgecolors = 'black')

ax.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])

ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')  

figure.tight_layout()
figure.savefig('first biggest cluster at 5 pm.png', dpi = 500)

#%%

#   The divideData function:

def divideData(number_of_nodes, \
               period_start_time, period_end_time, \
                   biggest_cluster_connectivities, \
                       numbers_of_data_segments, n):

    total_time = period_end_time * 6 - period_start_time * 6 + 1
    
    initial_node_data_sizes = np.zeros((1, number_of_nodes))
    
    for i in range(number_of_nodes):
        
        for t in range(total_time):
            
            if t == 1:
                
                initial_node_data_sizes[i] = \
                    np.sum(biggest_cluster_connectivities[t, i, :])
                
            else:
                
                initial_node_data_sizes[i] = \
                    initial_node_data_sizes[i] + \
                        np.sum(biggest_cluster_connectivities[t, i, :])
    
    initial_node_data_sizes = \
        np.floor(numbers_of_data_segments[n] * \
                 initial_node_data_sizes / np.sum(initial_node_data_sizes))

    return initial_node_data_sizes

#%%

#   The allocateData function:

def allocateData(number_of_nodes, initial_node_data_sizes):
    
    segment_allocations = \
        np.zeros((number_of_nodes, np.sum(initial_node_data_sizes)))
                 
    for i in range(number_of_nodes):
        
        if i == 1:
            
            segment_allocations[i, \
                                list(range(initial_node_data_sizes[i]))] = \
                np.ones((1, len(range(initial_node_data_sizes[i]))))
            last_segment_allocation = initial_node_data_sizes[i]
            
        else:
            
            segment_allocations[i, \
                list(range(last_segment_allocation + 1, \
                           last_segment_allocation + initial_node_data_sizes[i]))] = \
                np.ones((1, len(range(last_segment_allocation + 1, \
                                      last_segment_allocation + initial_node_data_sizes[i]))))
            
            last_segment_allocation = \
                last_segment_allocation + initial_node_data_sizes[i]
    
    return segment_allocations

#%%

#   The computeFeatures function:

def computeFeatures(number_of_nodes, \
                    biggest_cluster_connectivities, time_index, \
                        segment_allocations): 

    node_features = np.zeros((number_of_nodes, 4))
    
    for i in range(number_of_nodes):
            
        node_features[i, 0] = \
            np.shape(np.where(biggest_cluster_connectivities[time_index, i, :] == 1))[0] \
                / number_of_nodes
            
        if node_features[i, 0] != 0:
            
            node_segments = \
                np.where(segment_allocations[i, :] == 1)
                
            node_neighbors = \
                np.where(biggest_cluster_connectivities[time_index, i, :] == 1)
                
            neighbors_neighbors = np.empty((0,), int)
                
            for j in range(np.shape(node_neighbors)[0]):
                    
                neighbor_missing_segments = np.where(segment_allocations[node_neighbors[j], :] == 0)
                    
                if neighbor_missing_segments.size == 0:
                
                    node_features[i, 1] = \
                        (node_features[i, 1] * j + \
                         len(node_segments[ismember(node_segments, \
                                                    np.sort(neighbor_missing_segments))]) \
                             / len(neighbor_missing_segments)) / (j + 1)
                    
                if j == 1:
                        
                    	shared_missing_segments = \
                            node_segments[ismember(node_segments, \
                                                   np.sort(neighbor_missing_segments))]
                        
                else:
                        
                    shared_missing_segments = \
                        shared_missing_segments[ismember(shared_missing_segments, \
                                                         np.sort(node_segments[ismember(node_segments, \
                                                                                        np.sort(neighbor_missing_segments))]))]
                
                np.append(neighbors_neighbors, \
                          np.where(biggest_cluster_connectivities \
                                   [time_index, node_neighbors[j], :] \
                                       == 1), axis = 0)
                
            node_features[i, 2] = \
                len(shared_missing_segments) / \
                    np.shape(segment_allocations)[1]
                
            node_features[i, 3] = 1 - \
                len(neighbors_neighbors.difference([node_neighbors, i])) \
                    / number_of_nodes

    return node_features

#%%

#   The controlRange function:

def controlRange(transmitter_neighbors, synthetic_lats, \
                 synthetic_lons, biggest_cluster_members, \
                     transmitter, time_index, period_start_time, \
                         node_statuses):

    distances = np.zeros((np.shape(transmitter_neighbors)[0], 1))
                    
    for j in range(len(transmitter_neighbors)):
                        
        	distances[j] = \
                geopy.distance.geodesic( \
                                        (synthetic_lats[ \
                                                        biggest_cluster_members[transmitter], time_index + \
                                                            period_start_time * 6 - 1], \
                                         synthetic_lons[ \
                                                        biggest_cluster_members[transmitter], time_index + \
                                                            period_start_time * 6 - 1]), \
                                            (synthetic_lats[ \
                                                            biggest_cluster_members[transmitter_neighbors[j]], \
                                                                time_index + period_start_time * 6 - 1], \
                                             synthetic_lons[ \
                                                            biggest_cluster_members[transmitter_neighbors[j]], \
                                                                time_index + period_start_time * 6 - 1])).km
                    
    eliminated_neighbors = np.empty((0,), int)
                    
    for j in range(len(transmitter_neighbors)):
        
        if (node_statuses[transmitter_neighbors[j], 0] == 1) \
            or (node_statuses[transmitter_neighbors[j], 2] == 1):
                
                np.append(eliminated_neighbors, \
                          np.where(distances >= distances[j]), \
                              axis = 0)
                       
        transmitter_neighbors[eliminated_neighbors] = np.empty((0,), int)
        
        distances[eliminated_neighbors] = np.empty((0,), int)

    return transmitter_neighbors, distances

#%%

#   The targetSegments function:

def targetSegments(segment_allocations, \
                   transmitter, transmitter_neighbors, node_statuses):
    
    targeted_segments = np.empty((0,), int)
    
    transmitter_segments = \
        np.where(segment_allocations[transmitter, :] == 1)
                                
    for j in range(len(transmitter_neighbors)):
        
        if node_statuses[transmitter_neighbors[j], 1] == 0:
            
            neighbor_missing_segments = \
                np.where(segment_allocations[transmitter_neighbors[j], :] == 0)
                
            np.append(targeted_segments, \
                      transmitter_segments[ismember(transmitter_segments, \
                                                    np.sort(neighbor_missing_segments))])
    
    _, idx = \
        np.unique(targeted_segments, axis = 0, return_index = True)
    targeted_segments = \
        targeted_segments[np.sort(idx)]

    return targeted_segments

#%%

#   The transmitData function:

def transmitData(node_statuses, transmitter, transmitter_neighbors, \
                 segment_allocations, targeted_segments, \
                     actions_history, time_index, \
                         period_start_time, distances, \
                             number_of_nodes):

    node_statuses[transmitter, 0] = 1
                            
    segment_popularities = \
        np.sum(segment_allocations[transmitter_neighbors[ \
                                                         np.where(node_statuses[transmitter_neighbors, 1] == 0)], :], \
               axis = 0)
            
    chosen_segment = \
        targeted_segments[np.where(segment_popularities[targeted_segments] == \
                                   np.amin(segment_popularities[targeted_segments]))]
                            
    chosen_segment = chosen_segment[0]
                            
    for k in range(len(transmitter_neighbors)):
        
        if (node_statuses[transmitter_neighbors[k], 1] == 0) \
            and (segment_allocations \
                 [transmitter_neighbors[k], chosen_segment] == 0):
                    
            node_statuses[transmitter_neighbors[k], [1, 2]] = \
                np.ones((1, 2))
                                    
            segment_allocations \
                [transmitter_neighbors[k], chosen_segment] = 1
                                    
            actions_history = np.vstack((actions_history, \
                                         [time_index + period_start_time * 6 - 1, transmitter, \
                                          np.amax(distances) * 1000, chosen_segment, \
                                              np.zeros((1, number_of_nodes))]))
                                    
            actions_history[-1, 3 + transmitter_neighbors[k]] = 1
                            
    node_statuses[transmitter_neighbors, 1] = \
        np.ones((len(transmitter_neighbors), 1))

    return segment_allocations, actions_history, node_statuses

#%%

#   The followPolicy function:

def followPolicy(period_start_time, period_end_time, number_of_nodes, \
                 biggest_cluster_connectivities, segment_allocations, \
                     omegas, synthetic_lats, synthetic_lons, \
                         biggest_cluster_members):

    actions_history = np.empty((0,), int)
    
    total_time = period_end_time * 6 - period_start_time * 6 + 1
    
    for t in range(total_time):
        
        node_statuses = np.zeros((number_of_nodes, 3))
        
        #   node_statuses(:,1) ==> indicate whether nodes are transmitting or not
        #   node_statuses(:,2) ==> indicate whether nodes are covered or not
        #   node_statuses(:,3) ==> indicate whether nodes are receiving or not
        
        #   Computing node features:
        
        node_features = computeFeatures(number_of_nodes, \
            biggest_cluster_connectivities, t, segment_allocations)
        
        utility_function = \
            node_features * omegas
        
        transmissions_order = np.argsort(utility_function)
        transmissions_order = transmissions_order[::-1]
        
        for i in range(number_of_nodes):
 
            transmitter = transmissions_order[i]
            
            if node_statuses[transmitter, 1] == 1:
                
                continue
                
            else:
                
                transmitter_neighbors = \
                    np.where(biggest_cluster_connectivities \
                             [t, transmitter, :] == 1)
                
                if (np.all(node_statuses[transmitter_neighbors, 1]) == 1) \
                    or transmitter_neighbors.size == 0:
                    
                    continue
                    
                else:
                    
                    #   Controlling range:
                    
                    transmitter_neighbors, distances = \
                        controlRange(transmitter_neighbors, \
                                     synthetic_lats, synthetic_lons, \
                                         biggest_cluster_members, \
                                             transmitter, t, \
                                                 period_start_time, \
                                                     node_statuses)
                    
                    if transmitter_neighbors.size == 0:
                        
                        continue
                        
                    else:
                        
                        #   Targeting segments:
                        
                        targeted_segments = \
                            targetSegments(segment_allocations, \
                                           transmitter, \
                                               transmitter_neighbors, \
                                                   node_statuses)
                        
                        if targeted_segments.size == 0:
                            
                            continue
                            
                        else:
                            
                            #   Transmitting data:
                            
                            segment_allocations, \
                             actions_history, node_statuses = \
                                transmitData(node_statuses, \
                                             transmitter, \
                                                 transmitter_neighbors, \
                                                     segment_allocations, \
                                                         targeted_segments, \
                                                             actions_history, t, \
                                                                 period_start_time, \
                                                                     distances, number_of_nodes)

    return segment_allocations, actions_history, node_statuses
        
#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import matplotlib.pyplot as plt

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
numbers_of_data_segments = list(range(50, 2050, 50))
segments_number_variation = np.empty((0,), int)
minimum_segments_exchanged_over_distributed_ratio = 18

#   Estimating the maximum number of data segments to be exchanged
#   using the naive-policy under a minimum 
#   "segments_exchanged_over_distributed_ratio":

for n in range(len(numbers_of_data_segments)):
    
    #   Dividing data:
    
    initial_node_data_sizes = \
        divideData(number_of_nodes, \
                   period_start_time, period_end_time, \
                       biggest_cluster_connectivities, \
                           numbers_of_data_segments, n)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)
    
    #   Specifying the naive policy weights:
    
    omegas = [10, 0, 0, 0]
    
    #   Notice that the naive policy is choosing the node with the \
    #   highest number of neighbors normalized by the total number \
    #   of nodes to start transmission first while ignoring the other \
    #   features.
    
    #   Following the specified policy:

    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, biggest_cluster_connectivities, \
                         segment_allocations, omegas, synthetic_lats, \
                             synthetic_lons, biggest_cluster_members)
    
    segments_number_variation = np.vstack((segments_number_variation, \
                                           [numbers_of_data_segments[n], np.sum(initial_node_data_sizes), \
                                            np.sum(np.sum(segment_allocations)) - np.sum(initial_node_data_sizes)]))
             
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format(n / np.shape(numbers_of_data_segments)[0] \
                                * 100)))
    sys.stdout.flush()

figure, ax = plt.subplots(nrows = 1, ncols = 1)

ax.set_title('Number of data segments\ninitially-distributed effect')
ax.plot(segments_number_variation[:, 0], \
         segments_number_variation[:, 2] / \
             segments_number_variation[:, 1], \
             '-*k')
ax.plot(np.vstack((0, segments_number_variation[:, 0])), \
        np.ones((1, 1 + len(numbers_of_data_segments))) * \
                minimum_segments_exchanged_over_distributed_ratio, \
                    '--r')

ax.set_xlabel('Number of data segments\ninitially-distributed')
ax.set_ylabel('Segments exchanged\nover initially-distributed ratio')
ax.grid(color = 'k', linestyle = '--', linewidth = 1)

figure.tight_layout()
figure.savefig('segments number variation.png', dpi = 500)

np.save('segments_number_variation.npy', segments_number_variation)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
synthetic_lats = np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)

number_of_visualizations = 6
number_of_iterations = 100
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150
visualizations_data = \
    np.zeros((number_of_iterations * number_of_visualizations, 4 + 1))

#   Building search space 3D-visualizations data:

for v in range(number_of_visualizations):
    
    for iteration in range(number_of_iterations):
        
        #   Dividing data:
    
        initial_node_data_sizes = \
            divideData(number_of_nodes, \
                       period_start_time, \
                           period_end_time, \
                               biggest_cluster_connectivities, \
                                   [number_of_data_segments], 0)
    
        #   Allocating data:
    
        segment_allocations = \
            allocateData(number_of_nodes, initial_node_data_sizes)
        
        #   Specifying visualization slice policy weights:
        
        if v == 0:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[0, 1], 0] = [0, 0]
        
        if v == 1:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[0, 2], 0] = [0, 0]
                                 
        if v == 2:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[0, 3], 0] = [0, 0]
        
        if v == 3:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[1, 2], 0] = [0, 0]
        
        if v == 4:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[1, 3], 0] = [0, 0]
        
        if v == 5:
            
            omegas = \
                np.random.randint(-10, 10, size = (4, 1))
            omegas[[2, 3], 0] = [0, 0]
        
        #   Following the specified policy:
        
        segment_allocations, actions_history, \
            node_statuses = \
                followPolicy(period_start_time, \
                             period_end_time, \
                                 number_of_nodes, \
                                     biggest_cluster_connectivities, \
                                         segment_allocations, \
                                             omegas, \
                                                 synthetic_lats, \
                                                     synthetic_lons, \
                                                         biggest_cluster_members)
        
        visualizations_data[v * number_of_iterations + iteration, :] = \
            [omegas.transpose(), len(actions_history)]
                     
        sys.stdout.write('\r' + \
                         str("Please wait ... {:.2f}%". \
                             format((v * number_of_iterations + iteration) / \
                                    (number_of_visualizations * number_of_iterations) * 100)))
        
        sys.stdout.flush()

np.save('visualizations_data.npy', visualizations_data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

visualizations_data = \
    np.load('visualizations_data.npy', allow_pickle = True)

#   Visualizing search space 3D-data:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Search space with ' + \
              r'$\omega_1$' + ' & ' + r'$\omega_2$' + ' set to 0')
ax1.set_xlabel(r'$\omega_3$')
ax1.set_ylabel(r'$\omega_4$')

v = 0
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 2]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 3]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure1.colorbar(ax1.tripcolor(x, y, z), orientation = 'vertical')

figure1.tight_layout()
figure1.savefig('search space with zero omega 1 and 2.png', dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Search space with ' + \
              r'$\omega_1$' + ' & ' + r'$\omega_3$' + ' set to 0')
ax2.set_xlabel(r'$\omega_2$')
ax2.set_ylabel(r'$\omega_4$')

v = 1
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 1]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 3]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure2.colorbar(ax2.tripcolor(x, y, z), orientation = 'vertical')

figure2.tight_layout()
figure2.savefig('search space with zero omega 1 and 3.png', dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Search space with ' + \
              r'$\omega_1$' + ' & ' + r'$\omega_4$' + ' set to 0')
ax3.set_xlabel(r'$\omega_2$')
ax3.set_ylabel(r'$\omega_3$')

v = 2
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 1]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 2]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure3.colorbar(ax3.tripcolor(x, y, z), orientation = 'vertical')

figure3.tight_layout()
figure3.savefig('search space with zero omega 1 and 4.png', dpi = 500)

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Search space with ' + \
              r'$\omega_2$' + ' & ' + r'$\omega_3$' + ' set to 0')
ax4.set_xlabel(r'$\omega_1$')
ax4.set_ylabel(r'$\omega_4$')

v = 3
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 0]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 3]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure4.colorbar(ax4.tripcolor(x, y, z), orientation = 'vertical')

figure4.tight_layout()
figure4.savefig('search space with zero omega 2 and 3.png', dpi = 500)

figure5, ax5 = plt.subplots(nrows = 1, ncols = 1)

ax5.set_title('Search space with ' + \
              r'$\omega_2$' + ' & ' + r'$\omega_4$' + ' set to 0')
ax5.set_xlabel(r'$\omega_1$')
ax5.set_ylabel(r'$\omega_3$')

v = 3
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 0]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 2]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure5.colorbar(ax5.tripcolor(x, y, z), orientation = 'vertical')

figure5.tight_layout()
figure5.savefig('search space with zero omega 2 and 4.png', dpi = 500)

figure6, ax6 = plt.subplots(nrows = 1, ncols = 1)

ax6.set_title('Search space with ' + \
              r'$\omega_3$' + ' & ' + r'$\omega_4$' + ' set to 0')
ax6.set_xlabel(r'$\omega_1$')
ax6.set_ylabel(r'$\omega_2$')

v = 5
x = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 0]
y = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 1]
z = visualizations_data[list(range(v * 100 + 1, (v + 1) * 100)), 4]

figure6.colorbar(ax6.tripcolor(x, y, z), orientation = 'vertical')

figure6.tight_layout()
figure6.savefig('search space with zero omega 3 and 4.png', dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_iterations = 100
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150
regression_data = np.zeros((number_of_iterations, 4 + 1))

#   Generating initial regression data:

for iteration in range(number_of_iterations):
    
    #   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, \
                                         period_start_time, \
                                             period_end_time, \
                                                 biggest_cluster_connectivities, \
                                                     [number_of_data_segments], 0)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)

    #   Specifying the random policy weights:
    
    omegas = np.random.randint(-10, 10, size = (4, 1))
    
    #   Following the specified policy:

    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, biggest_cluster_connectivities, \
                         segment_allocations, omegas, \
                             synthetic_lats, synthetic_lons, \
                                 biggest_cluster_members)               
    
    regression_data[iteration, :] = \
        [omegas.transpose(), len(actions_history)]
    
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format((iteration + 1) / number_of_iterations * 100)))
        
    sys.stdout.flush()

np.save('regression_data.npy', regression_data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
regression_data = \
    np.load('regression_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_iterations = 900
number_of_random_points = 1000
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150

#   Bayesian optimization using GP regression:

optimized_gp_regression_data = np.copy(regression_data)
optimized_gp_execution_times = np.zeros((number_of_iterations, 1))

for iteration in range(number_of_iterations):
    
    for i in range(10):
        
        tic = time.time()
        
        kernel = DotProduct() + WhiteKernel()
        gp_model = GaussianProcessRegressor(kernel = kernel, \
                                       random_state = 0). \
            fit(optimized_gp_regression_data[:, [0, 1, 2, 3]], \
                optimized_gp_regression_data[:, 4])

        toc = time.time() - tic
        
        optimized_gp_execution_times[iteration] = \
            (i * optimized_gp_execution_times[iteration] + toc) / (i + 1)
 
    #   Using UCB:
    
    random_data = \
        np.zeros((number_of_random_points, 4 + 1))

    for i in range(number_of_random_points):
            
        omegas = np.random.randint(-10, 10, size = (4, 1))

        random_data[i, [0, 1, 2, 3]] = omegas
        
        prediction, standard_deviation = \
            gp_model.predict(random_data[i, [0, 1, 2, 3]], \
                             return_std = True)            
        
        random_data[i, 4] = \
            np.round(prediction + standard_deviation)
    
    best_random_point = random_data[np.where(random_data[:, 4] == \
                                             np.amax(random_data[:, 4])), :]
              
    #   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, \
                                         period_start_time, \
                                             period_end_time, \
                                                 biggest_cluster_connectivities, \
                                                     [number_of_data_segments], 0)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)
    
    #   Specifying the best policy weights found so far:
    
    omegas = best_random_point[0, [0, 1, 2, 3]].transpose()
    
    #   Following the specified policy weights:

    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, biggest_cluster_connectivities, \
                         segment_allocations, omegas, synthetic_lats, \
                             synthetic_lons, biggest_cluster_members)               
    
    optimized_gp_regression_data = \
        np.vstack((optimized_gp_regression_data, \
                   [omegas.transpose(), len(actions_history)]))

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format((iteration + 1) / number_of_iterations * 100)))
        
    sys.stdout.flush()
    
np.savez('optimized_gp_data.npy', optimized_gp_regression_data, \
         optimized_gp_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
import math
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
regression_data = \
    np.load('regression_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_iterations = 900
number_of_random_points = 1000
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150

#   Bayesian optimization using RF regression:

optimized_rf_regression_data = np.copy(regression_data)
optimized_rf_execution_times = np.zeros((number_of_iterations, 1))

for iteration in range(number_of_iterations):
    
    for i in range(10):
        
        tic = time.time()
        
        for j in range(10):
            
            globals()['tree_model_' + str(j)] = \
                DecisionTreeRegressor(random_state = 0, \
                                      min_samples_leaf = j). \
                    fit(optimized_rf_regression_data[:, [0, 1, 2, 3]], \
                        optimized_rf_regression_data[:, 4])
        
        toc = time.time() - tic
        
        optimized_rf_execution_times[iteration] = \
            (i * optimized_rf_execution_times[iteration] + toc) / (i + 1)
 
    #   Using UCB:
    
    random_data = np.zeros((number_of_random_points, 4))

    for i in range(number_of_random_points):
        
        omegas = np.random.randint(-10, 10, size = (4, 1))

        random_data[i, [0, 1, 2, 3]] = omegas
        
        tree_predictions = np.zeros((1, 10))
        
        for j in range(10):
            
            tree_predictions[j] = \
                globals()['tree_model_' + str(j)]. \
                    predict(globals()['tree_model_' + str(j)], \
                            random_data[i, [0, 1, 2, 3]])
        
        prediction = np.mean(tree_predictions)
        
        standard_deviation = \
            math.sqrt(1 / 10 * np.sum(pow(tree_predictions, 2)) - \
                                      pow(prediction, 2))
        
        random_data[i, 4] = \
            np.round(prediction + standard_deviation)
    
    best_random_point = random_data[np.where(random_data[:, 4] == \
                                             np.amax(random_data[:, 4])), :]

    #   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, \
                                         period_start_time, \
                                             period_end_time, \
                                                 biggest_cluster_connectivities, \
                                                     [number_of_data_segments], 0)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)
    
    #   Specifying the best policy weights found so far:
    
    omegas = best_random_point[0, [0, 1, 2, 3]].transpose()
    
    #   Following the specified policy weight:

    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, biggest_cluster_connectivities, \
                         segment_allocations, omegas, synthetic_lats, \
                             synthetic_lons, biggest_cluster_members)               
    
    optimized_rf_regression_data = \
        np.vstack((optimized_rf_regression_data, \
                   [omegas.transpose(), len(actions_history)])) 
                
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format((iteration + 1) / number_of_iterations * 100)))
        
    sys.stdout.flush()

np.savez('optimized_rf_data.npy', optimized_rf_regression_data, \
         optimized_rf_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
regression_data = \
    np.load('regression_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_iterations = 900
number_of_random_points = 1000
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150

#   Bayesian optimization using NN regression:

optimized_nn_regression_data = np.copy(regression_data)
optimized_nn_execution_times = np.zeros((number_of_iterations, 1))

for iteration in range(number_of_iterations):
    
    for i in range(10):
        
        tic = time.time()
        
        nn_model = Sequential()
        nn_model.add(Dense(10, kernel_initializer = 'normal', \
                           input_dim = 4, activation = 'tanh'))
        nn_model.add(Dense(10, kernel_initializer = 'normal', \
                           activation = 'tanh'))
        nn_model.add(Dense(10, kernel_initializer = 'normal', \
                           activation = 'tanh'))
        nn_model.add(Dense(1, kernel_initializer = 'normal', \
                           activation = 'linear'))
        
        nn_model.compile(loss = 'mean_squared_error', \
                         optimizer = 'adam', metrics = ['mse','mae'])
            
        nn_model.fit(optimized_nn_regression_data[:, [0, 1, 2, 3]].\
                     transpose(), \
                     optimized_nn_regression_data[:, 4].transpose(), \
                         epochs = 150, batch_size = 50)

        #   Replacing the last hidden layer:
            
        new_nn_model = Sequential()
        
        for layer in nn_model.layers[:-1]:
            new_nn_model.add(layer)
        
        new_nn_model_predictions = \
            new_nn_model.predict(optimized_nn_regression_data[:, \
                                                          [0, 1, 2, 3]]. \
                                 transpose())
        
        l_model = LinearRegression(). \
            fit(new_nn_model_predictions.transpose(), \
                optimized_nn_regression_data[:, 4])
        
        toc = time.time() - tic
        
        optimized_nn_execution_times[iteration] = \
            (i * optimized_nn_execution_times[iteration] + toc) / (i + 1)
    
    #   Using UCB:
    
    random_data = \
        np.zeros((number_of_random_points, 4))

    for i in range(number_of_random_points):
        
        omegas = np.random.randint(-10, 10, size = (4, 1))

        random_data[i, [0, 1, 2, 3]] = omegas
        
        prediction = \
            l_model.predict( \
                            new_nn_model.predict \
                                (random_data[i, [0, 1, 2, 3]].transpose()).tranpose())
        
        random_data[i, 4] = np.round(prediction)

    best_random_point = random_data[np.where(random_data[:, 4] == \
                                             np.amax(random_data[:, 4])), :]

    #   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, \
                                         period_start_time, \
                                             period_end_time, \
                                                 biggest_cluster_connectivities, \
                                                     [number_of_data_segments], 0)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)
    
    #   Specifying the best policy weights found so far:
    
    omegas = best_random_point[0, [0, 1, 2, 3]].transpose()
    
    #   Following the specified policy weights:
   
    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, biggest_cluster_connectivities, \
                         segment_allocations, omegas, synthetic_lats, \
                             synthetic_lons, biggest_cluster_members)               
        
    optimized_nn_regression_data = \
        np.vstack((optimized_nn_regression_data, \
        [omegas.tranpose(), len(actions_history)]))
             
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format((iteration + 1) / number_of_iterations * 100)))
        
    sys.stdout.flush()

np.savez('optimized_nn_data.npy', optimized_nn_regression_data, \
         optimized_nn_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np

optimized_gp_data = \
    np.load('optimized_gp_data.npy', allow_pickle = True)
optimized_rf_data = \
    np.load('optimized_rf_data.npy', allow_pickle = True)
optimized_nn_data = \
    np.load('optimized_nn_data.npy', allow_pickle = True)

#   Cleaning the execution-time results of the GP regression,
#   the RF regression and the NN regression used for Bayesian optimization:

#   Removing outliers:

outlier_removing_window = 10

for i in range(outlier_removing_window, \
               len(optimized_gp_execution_times) - \
                   outlier_removing_window - 1):
    
    if (optimized_gp_execution_times[i, 0] > \
        np.percentile(optimized_gp_execution_times \
                      [list(range(i - outlier_removing_window, \
                                  i + outlier_removing_window)), 0], 75) + \
            1.5 * (np.percentile(optimized_gp_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 75) - \
                   np.percentile(optimized_gp_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 25))) or \
        (optimized_gp_execution_times[i, 0] < \
         np.percentile(optimized_gp_execution_times \
                       [list(range(i - outlier_removing_window, \
                                   i + outlier_removing_window)), 0], 25) - \
             1.5 * (np.percentile(optimized_gp_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 75) - \
                    np.percentile(optimized_gp_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 25))):
            
            optimized_gp_execution_times[i, 0] = \
                np.mean(optimized_gp_execution_times \
                        [list(range(i - outlier_removing_window, \
                                    i + outlier_removing_window)), 0])

for i in range(outlier_removing_window, \
               len(optimized_rf_execution_times) - \
                   outlier_removing_window - 1):
    
    if (optimized_rf_execution_times[i, 0] > \
        np.percentile(optimized_rf_execution_times \
                      [list(range(i - outlier_removing_window, \
                                  i + outlier_removing_window)), 0], 75) + \
            1.5 * (np.percentile(optimized_rf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 75) - \
                   np.percentile(optimized_rf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 25))) or \
        (optimized_rf_execution_times[i, 0] < \
         np.percentile(optimized_rf_execution_times \
                       [list(range(i - outlier_removing_window, \
                                   i + outlier_removing_window)), 0], 25) - \
             1.5 * (np.percentile(optimized_rf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 75) - \
                    np.percentile(optimized_rf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 25))):
            
            optimized_rf_execution_times[i, 0] = \
                np.mean(optimized_rf_execution_times \
                        [list(range(i - outlier_removing_window, \
                                    i + outlier_removing_window)), 0])

for i in range(outlier_removing_window, \
               len(optimized_nn_execution_times) - \
                   outlier_removing_window - 1):

    if (optimized_nn_execution_times[i, 0] > \
        np.percentile(optimized_nn_execution_times \
                      [list(range(i - outlier_removing_window, \
                                  i + outlier_removing_window)), 0], 75) + \
            1.5 * (np.percentile(optimized_nn_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 75) - \
                   np.percentile(optimized_nn_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 25))) or \
        (optimized_nn_execution_times[i, 0] < \
         np.percentile(optimized_nn_execution_times \
                       [list(range(i - outlier_removing_window, \
                                   i + outlier_removing_window)), 0], 25) - \
             1.5 * (np.percentile(optimized_nn_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 75) - \
                    np.percentile(optimized_nn_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 25))):
            
            optimized_nn_execution_times[i, 0] = \
                np.mean(optimized_nn_execution_times \
                        [list(range(i - outlier_removing_window, \
                                    i + outlier_removing_window)), 0])

#   Smoothing data:

moving_average_window = 10;

for i in range(moving_average_window, \
               len(optimized_gp_execution_times) - \
                   moving_average_window - 1):
    
    optimized_gp_execution_times[i, 0] = \
        np.mean(optimized_gp_execution_times \
                [list(range(i - moving_average_window, \
                            i + moving_average_window)), 0])

for i in range(moving_average_window, \
               len(optimized_rf_execution_times) - \
                   moving_average_window - 1):
    
    optimized_rf_execution_times[i, 0] = \
        np.mean(optimized_rf_execution_times \
                [list(range(i - moving_average_window, \
                            i + moving_average_window)), 0])

for i in range(moving_average_window, \
               len(optimized_nn_execution_times) - \
                   moving_average_window - 1):
    
    optimized_nn_execution_times[i, 0] = \
        np.mean(optimized_nn_execution_times \
                [list(range(i - moving_average_window, \
                            i + moving_average_window)), 0])

optimized_gp_smoothed_execution_times = \
    np.copy(optimized_gp_execution_times)

optimized_rf_smoothed_execution_times = \
    np.copy(optimized_rf_execution_times)

optimized_nn_smoothed_execution_times = \
    np.copy(optimized_nn_execution_times)

np.save('optimized_gp_smoothed_execution_times.npy', \
        optimized_gp_smoothed_execution_times)

np.save('optimized_rf_smoothed_execution_times.npy', \
        optimized_rf_smoothed_execution_times)

np.save('optimized_nn_smoothed_execution_times.npy', \
        optimized_nn_smoothed_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

optimized_gp_data = \
    np.load('optimized_gp_data.npy', allow_pickle = True)
optimized_rf_data = \
    np.load('optimized_rf_data.npy', allow_pickle = True)
optimized_nn_data = \
    np.load('optimized_nn_data.npy', allow_pickle = True)
optimized_gp_smoothed_execution_times = \
    np.load('optimized_gp_smoothed_execution_times.npy', \
            allow_pickle = True)
optimized_rf_smoothed_execution_times = \
    np.load('optimized_rf_smoothed_execution_times.npy', \
            allow_pickle = True)
optimized_nn_smoothed_execution_times = \
    np.load('optimized_nn_smoothed_execution_times.npy', \
            allow_pickle = True)

#   Comparing the GP regression, the RF regression
#   and the NN regression used for Bayesian optimization:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Bayesian optimization performances')
ax1.set_xlabel('Iteration index')
ax1.set_ylabel('Number of data segments exchanged')

ax1.plot(list(range(np.shape(optimized_gp_regression_data)[0])), \
        optimized_nn_regression_data[:, -1], '-ob', \
            label = 'w/ Neural-network-regression')
ax1.plot(list(range(np.shape(optimized_gp_regression_data)[0])), \
        optimized_rf_regression_data[:, -1], '-*k', \
            label = 'w/ Random-forest-regression')
ax1.plot(list(range(np.shape(optimized_gp_regression_data)[0])), \
        optimized_gp_regression_data[:, -1], '--r', \
            label = 'w/ Gaussian-processes-regression')
ax1.set_xlim([0, np.shape(optimized_gp_regression_data)[0]])
ax1.ylim([np.amin( \
                  [np.amin(optimized_gp_regression_data[:, -1]), \
                   np.amin(optimized_rf_regression_data[:, -1]), \
                       np.amin(optimized_nn_regression_data[:, -1])]) * 0.9, \
          np.amax( \
                  [np.amax(optimized_gp_regression_data[:, -1]), \
                   np.amax(optimized_rf_regression_data[:, -1]), \
                       np.amax(optimized_nn_regression_data[:, -1])]) * 1.1])
ax1.legend(loc = 'best')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('gp vs. rf vs. nn - data segments exchanged.png', \
                dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Bayesian optimization performances')
ax2.set_xlabel('Number of observations')
ax2.set_ylabel('Execution time (msec)')
ax2.plot(list(range(np.shape(optimized_gp_execution_times)[0])), \
         np.mean(optimized_nn_smoothed_execution_times, axis = 1) * \
             1000, '-ob', label = 'w/ Neural-network-regression')
ax2.plot(list(range(np.shape(optimized_gp_execution_times)[0])), \
         np.mean(optimized_rf_smoothed_execution_times, axis = 1) * \
             1000, '-*k', label = 'w/ Random-forest-regression')
ax2.plot(list(range(np.shape(optimized_gp_execution_times)[0])), \
         np.mean(optimized_gp_smoothed_execution_times, axis = 1) * \
             1000, '--r', label = 'w/ Gaussian-processes-regression')
ax2.set_xlim([0, np.shape(optimized_gp_execution_times)[0]])
ax2.set_ylim([np.amin( \
                      [np.amin(np.mean(optimized_gp_smoothed_execution_times, axis = 1)), \
                       np.amin(np.mean(optimized_rf_smoothed_execution_times, axis = 1)), \
                           np.amin(np.mean(optimized_nn_smoothed_execution_times, axis = 1))]) * 1000 * 0.9, \
              np.amax( \
                      [np.amax(np.mean(optimized_gp_smoothed_execution_times, axis = 1)), \
                       np.amax(np.amean(optimized_rf_smoothed_execution_times, axis = 1)), \
                           np.amax(np.mean(optimized_nn_smoothed_execution_times, axis = 1))]) * 1000 * 1.1])
ax2.legend(loc = 'best')
ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure2.tight_layout()
figure2.savefig('gp vs. rf vs. nn - execution times.png', \
                dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Policy weights under GP regression')
ax3.set_xlabel('Iteration index')
ax3.set_ylabel('Weight value')
ax3.set_xlim([1, np.shape(optimized_gp_regression_data)[0]])
ax3.set_ylim([-10, 10])

for i in range(np.shape(optimized_gp_regression_data)[1] - 1):

    ax3.plot(list(range(np.shape(optimized_gp_regression_data)[0])), \
             optimized_gp_regression_data[:, i], \
                 color = np.random.uniform(0, 1, size = (1, 3)))

ax3.grid(color = 'k', linestyle = '--', linewidth = 1)

figure3.tight_layout()
figure3.savefig('policy weights under GP regression.png', \
                dpi = 500)

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Policy weights under RF regression')
ax4.set_xlabel('Iteration index')
ax4.set_ylabel('Weight value')
ax4.set_xlim([1, np.shape(optimized_rf_regression_data)[0]])
ax4.set_ylim([-10, 10])

for i in range(np.shape(optimized_rf_regression_data)[1] - 1):

    ax4.plot(list(range(np.shape(optimized_rf_regression_data)[0])), \
             optimized_rf_regression_data[:, i], \
                 color = np.random.uniform(0, 1, size = (1, 3)))

ax4.grid(color = 'k', linestyle = '--', linewidth = 1)

figure4.tight_layout()
figure4.savefig('policy weights under RF regression.png', \
                dpi = 500)

figure5, ax5 = plt.subplots(nrows = 1, ncols = 1)

ax5.set_title('Policy weights under NN regression')
ax5.set_xlabel('Iteration index')
ax5.set_ylabel('Weight value')
ax5.set_xlim([1, np.shape(optimized_nn_regression_data)[0]])
ax5.set_ylim([-10, 10])

for i in range(np.shape(optimized_nn_regression_data)[1] - 1):

    ax5.plot(list(range(np.shape(optimized_nn_regression_data)[0])), \
             optimized_nn_regression_data[:, i], \
                 color = np.random.uniform(0, 1, size = (1, 3)))

ax5.grid(color = 'k', linestyle = '--', linewidth = 1)

figure5.tight_layout()
figure5.savefig('policy weights under NN regression.png', \
                dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
import math
import sys

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
regression_data = \
    np.load('regression_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_iterations = 900
number_of_random_points = 1000
number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150

#   Bayesian optimization using batch-based RF regression:

#   Notice that the data batch used includes 100 samples. If a bigger
#   data batch is needed, then we need the "regression_data" found 
#   previously to have the larger number of samples needed beforehand.

optimized_brf_regression_data = np.copy(regression_data)
optimized_brf_execution_times = \
    np.zeros((number_of_iterations, 1))

for iteration in range(number_of_iterations):
    
    for i in range(10):
        
        tic = time.time()
        
        for j in range(10):
            
            globals()['tree_model_' + str(j)] = \
                DecisionTreeRegressor(random_state = 0, \
                                      min_samples_leaf = j). \
                    fit(optimized_brf_regression_data[list(range(iteration, 99 + iteration)), [0, 1, 2, 3]], \
                        optimized_brf_regression_data[list(range(iteration, 99 + iteration)), 4])
          
        toc = time.time() - tic
        
        optimized_brf_execution_times[iteration] = \
            (i * optimized_brf_execution_times[iteration] + toc) / (i + 1)

    #   Using UCB:
    
    random_data = \
        np.zeros((number_of_random_points, 4 + 1))

    for i in range(number_of_random_points):
        
        for j in range(4):
            
            average = \
                np.mean(optimized_brf_regression_data \
                [list(range(iteration, 99 + iteration)), j])
            
            standard_deviation = \
                np.std(optimized_brf_regression_data[ \
                                                     list(range(iteration, 99 + iteration)), j])

            min_batch_omega = \
                np.round(max(average - standard_deviation, -10))
            
            max_batch_omega = \
                np.round(min(average + standard_deviation, 10))
            
            omegas[j, 0] = \
                np.random.randint(min_batch_omega, max_batch_omega)
     
        random_data[i, [0, 1, 2, 3]] = omegas
        
        for j in range(10):
            
            tree_predictions[j] = \
                globals()['tree_model_' + str(j)]. \
                    predict(globals()['tree_model_' + str(j)], \
                            random_data[i, [0, 1, 2, 3]])
        
        prediction = np.mean(tree_predictions)
        
        standard_deviation = \
            math.sqrt(1 / 10 * np.sum(pow(tree_predictions, 2)) - \
                                      pow(prediction, 2))
        
        random_data[i, 4] = \
            np.round(prediction + standard_deviation)
        
    best_random_point = random_data[np.where(random_data[:, 4] == \
                                             np.amax(random_data[:, 4])), :]

    #   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, \
                                         period_start_time, \
                                             period_end_time, \
                                                 biggest_cluster_connectivities, \
                                                     [number_of_data_segments], 0)
    
    #   Allocating data:
    
    segment_allocations = \
        allocateData(number_of_nodes, initial_node_data_sizes)
    
    #   Specifying the best policy weights found so far:
    
    omegas = best_random_point[0, [0, 1, 2, 3]].transpose()
    
    #   Following the specified policy weights:
   
    segment_allocations, actions_history, node_statuses = \
        followPolicy(period_start_time, period_end_time, \
                     number_of_nodes, 
                     biggest_cluster_connectivities, \
                         segment_allocations, omegas, synthetic_lats, \
                             synthetic_lons, biggest_cluster_members)      

    optimized_brf_regression_data = \
        np.vstack((optimized_brf_regression_data, \
                   [omegas.transpose(), len(actions_history)]))
    
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%". \
                         format((iteration + 1) / number_of_iterations * 100)))
        
    sys.stdout.flush()
    
np.savez('optimized_brf_data.npy', optimized_brf_regression_data, \
         optimized_brf_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np

optimized_rf_data = \
    np.load('optimized_rf_data.npy', allow_pickle = True)
optimized_brf_data = \
    np.load('optimized_brf_data.npy', allow_pickle = True)

#   Cleaning the execution-time results of the RF regression,
#   and the batch-based RF regression used for Bayesian optimization:

#   Removing outliers:

outlier_removing_window = 10

for i in  range(outlier_removing_window, \
                len(optimized_rf_execution_times) - \
                    outlier_removing_window - 1):
    
    if (optimized_rf_execution_times[i, 0] > \
        np.percentile(optimized_rf_execution_times \
                      [list(range(i - outlier_removing_window, \
                                  i + outlier_removing_window)), 0], 75) + \
            1.5 * (np.percentile(optimized_rf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 75) - \
                   np.percentile(optimized_rf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 25))) or \
        (optimized_rf_execution_times[i, 0] < \
         np.percentile(optimized_rf_execution_times \
                       [list(range(i - outlier_removing_window, \
                                   i + outlier_removing_window)), 0], 25) - \
             1.5 * (np.percentile(optimized_rf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 75) - \
                    np.percentile(optimized_rf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 25))):
            
            optimized_rf_execution_times[i, 0] = \
                np.mean(optimized_rf_execution_times \
                        [list(range(i - outlier_removing_window, \
                                    i + outlier_removing_window)), 0])

for i in range(outlier_removing_window, \
               len(optimized_brf_execution_times) - \
                   outlier_removing_window - 1):

    if (optimized_brf_execution_times[i, 0] > \
        np.percentile(optimized_brf_execution_times \
                      [list(range(i - outlier_removing_window, \
                                  i + outlier_removing_window)), 0], 75) + \
            1.5 * (np.percentile(optimized_brf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 75) - \
                   np.percentile(optimized_brf_execution_times \
                                 [list(range(i - outlier_removing_window, \
                                             i + outlier_removing_window)), 0], 25))) or \
        (optimized_brf_execution_times[i, 0] < \
         np.percentile(optimized_brf_execution_times \
                       [list(range(i - outlier_removing_window, \
                                   i + outlier_removing_window)), 0], 25) - \
             1.5 * (np.percentile(optimized_brf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 75) - \
                    np.percentile(optimized_brf_execution_times \
                                  [list(range(i - outlier_removing_window, \
                                              i + outlier_removing_window)), 0], 25))):
            
            optimized_brf_execution_times[i, 0] = \
                np.mean(optimized_brf_execution_times \
                        [list(range(i - outlier_removing_window, \
                                    i + outlier_removing_window)), 0])

#   Smoothing data:

moving_average_window = 10

for i in range(moving_average_window, \
               len(optimized_rf_execution_times) - \
                   moving_average_window - 1):
    
    optimized_rf_execution_times[i, 0] = \
        np.mean(optimized_rf_execution_times \
                [list(range(i - moving_average_window, \
                            i + moving_average_window)), 0])

for i in range(moving_average_window, \
               len(optimized_brf_execution_times) - \
                   moving_average_window - 1):
    
    optimized_brf_execution_times[i, 0] = \
        np.mean(optimized_brf_execution_times \
                [list(range(i - moving_average_window, \
                            i + moving_average_window)), 0])

optimized_rf_smoothed_execution_times = \
    np.copy(optimized_rf_execution_times)

optimized_brf_smoothed_execution_times = \
    np.copy(optimized_brf_execution_times)

np.save('optimized_rf_smoothed_execution_times.npy', \
        optimized_rf_smoothed_execution_times)

np.save('optimized_brf_smoothed_execution_times.npy', \
        optimized_brf_smoothed_execution_times)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

optimized_rf_data = \
    np.load('optimized_rf_data.npy', allow_pickle = True)
optimized_brf_data = \
    np.load('optimized_brf_data.npy', allow_pickle = True)
optimized_rf_smoothed_execution_times = \
    np.load('optimized_rf_smoothed_execution_times.npy', \
            allow_pickle = True)
optimized_brf_smoothed_execution_times = \
    np.load('optimized_brf_smoothed_execution_times.npy', \
            allow_pickle = True)

#   Comparing the all-data RF regression and
#   the batch-based RF regression used for Bayesian optimization:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Bayesian optimization performances')
ax1.set_xlabel('Iteration index')
ax1.set_ylabel('Number of data segments exchanged')
ax1.plot(list(range(np.shape(optimized_rf_regression_data)[0])), \
         optimized_rf_regression_data[:, -1], \
             '-ob', label = 'w/ Random-forest-regression')
ax1.plot(list(range(np.shape(optimized_rf_regression_data)[0])), \
         optimized_brf_regression_data \
             [list(range(np.shape(optimized_rf_regression_data)[0])), -1], \
                 '-*k', label = 'w/ Batch-based random-forest-regression')
ax1.set_xlim([0, np.shape(optimized_rf_regression_data)[0]])
ax1.set_ylim([np.amin( \
                 [np.amin(optimized_rf_regression_data[:, -1]), \
                  np.amin(optimized_brf_regression_data[:, -1])]) * 0.9, \
             np.amax( \
                     [np.amax(optimized_rf_regression_data[:, -1]), \
                      np.amax(optimized_brf_regression_data[:, -1])]) * 1.1])
ax1.legend(loc = 'best')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('rf vs. brf - data segments exchanged.png', \
                dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Bayesian optimization performances')
ax2.set_xlabel('Number of observations')
ax2.set_ylabel('Execution time (msec)')
ax2.plot(list(range(np.shape(optimized_rf_execution_times)[0])), \
         np.mean(optimized_rf_smoothed_execution_times, axis = 1) * 1000, \
             '-*k', label = 'w/ Random-forest-regression')
ax2.plot(list(range(np.shape(optimized_brf_execution_times)[0])), \
         np.mean(optimized_brf_smoothed_execution_times, axis = 1) * 1000, \
             '--r', label = 'w/ Batch-based random-forest-regression')
ax2.set_xlim([0, np.shape(optimized_rf_execution_times)[0]])
ax2.set_ylim([np.amin( \
                      [np.amin(np.mean(optimized_rf_smoothed_execution_times, axis = 1)), \
                       np.amin(np.mean(optimized_brf_smoothed_execution_times, axis = 1))]) * 1000 * 0.9, \
              np.amax( \
                      [np.amax(np.mean(optimized_rf_smoothed_execution_times, axis = 1)), \
                       np.amax(np.mean(optimized_brf_smoothed_execution_times, axis = 1))]) * 1000 * 1.1])
ax2.legend(loc = 'best')
ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure2.tight_layout()
figure2.savefig('rf vs. brf - execution times.png', \
                dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Policy weights under RF regression')
ax3.set_xlabel('Iteration index')
ax3.set_ylabel('Weight value')
ax3.set_xlim([1, np.shape(optimized_rf_regression_data)[0]])
ax3.set_ylim([-10, 10])

for i in range(np.shape(optimized_rf_regression_data)[1] - 1):

    ax3.plot(list(range(np.shape(optimized_rf_regression_data)[0])), \
             optimized_rf_regression_data[:, i], \
                 color = np.random.uniform(0, 1, size = (1, 3)))

ax3.grid(color = 'k', linestyle = '--', linewidth = 1)

figure3.tight_layout()
figure3.savefig('policy weights under RF regression.png', \
                dpi = 500)

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Policy weights under batch-based RF regression')
ax4.set_xlabel('Iteration index')
ax4.set_ylabel('Weight value')
ax4.set_xlim([1, np.shape(optimized_rf_regression_data)[0]])
ax4.set_ylim([-10, 10])

for i in range(np.shape(optimized_brf_regression_data)[1] - 1):

    ax4.plot(list(range(np.shape(optimized_brf_regression_data)[0])), \
             optimized_brf_regression_data[:, i], \
                 color = np.random.uniform(0, 1, size = (1, 3)))

ax4.grid(color = 'k', linestyle = '--', linewidth = 1)

figure4.tight_layout()
figure4.savefig('policy weights under batch-based RF regression.png', \
                dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt
import math

biggest_cluster_data = \
    np.load('biggest_cluster_data.npy', allow_pickle = True)

optimized_gp_data = \
    np.load('optimized_gp_data.npy', allow_pickle = True)
optimized_rf_data = \
    np.load('optimized_rf_data.npy', allow_pickle = True)
optimized_nn_data = \
    np.load('optimized_nn_data.npy', allow_pickle = True)
optimized_brf_data = \
    np.load('optimized_brf_data.npy', allow_pickle = True)

regression_data = \
    np.load('modified_data.npy', allow_pickle = True)
synthetic_lats = \
    np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = \
    np.load('synthetic_lons.npy', allow_pickle = True)

number_of_nodes = np.shape(biggest_cluster_connectivities[0, :, :])[0]
number_of_data_segments = 150

#   Comparing worst-policy, naive-policy and best-policy performances: 

worst_gp_performance = \
    optimized_gp_regression_data[ \
                                 np.where(optimized_gp_regression_data[:, -1] == \
                                          np.amin(optimized_gp_regression_data[:, -1])), :]
worst_gp_performance = worst_gp_performance[0, :]

worst_rf_performance = \
    optimized_rf_regression_data[ \
                                 np.where(optimized_rf_regression_data[:, -1] == \
                                          np.amin(optimized_rf_regression_data[:, -1])), :]
worst_rf_performance = worst_rf_performance[0, :]

worst_nn_performance = \
    optimized_nn_regression_data[ \
                                 np.where(optimized_nn_regression_data[:, -1] == \
                                          np.amin(optimized_nn_regression_data[:, -1])), :]
worst_nn_performance = worst_nn_performance[0, :]

worst_brf_performance = \
    optimized_brf_regression_data[ \
                                  np.where(optimized_brf_regression_data[:, -1] == \
                                           np.amin(optimized_brf_regression_data[:, -1])), :]
worst_brf_performance = worst_brf_performance[0, :]

worst_performances = \
    np.vstack((worst_gp_performance, \
               worst_rf_performance, \
                   worst_nn_performance, \
                       worst_brf_performance))

worst_policy_omegas = \
    worst_performances[ \
                       np.where(worst_performances[:, -1] == \
                                np.amin(worst_performances[:, -1])), \
                           list(range(len(worst_performances) - 1))]

naive_policy_omegas = [10, 0, 0, 0]

best_gp_performance = \
    optimized_gp_regression_data[ \
                                 np.where(optimized_gp_regression_data[:, -1] == \
                                          np.amax(optimized_gp_regression_data[:, -1])), :]
best_gp_performance = best_gp_performance[0, :]

best_rf_performance = \
    optimized_rf_regression_data[ \
                                 np.where(optimized_rf_regression_data[:, -1] == \
                                          np.amax(optimized_rf_regression_data[:, -1])), :]
best_rf_performance = best_rf_performance[0, :]

best_nn_performance = \
    optimized_nn_regression_data[ \
                                 np.where(optimized_nn_regression_data[:, -1] == \
                                          np.amax(optimized_nn_regression_data[:, -1])), :]
best_nn_performance = best_nn_performance[0, :]

best_brf_performance = \
    optimized_brf_regression_data[ \
                                  np.where(optimized_brf_regression_data[:, -1] == \
                                           np.amax(optimized_brf_regression_data[:, -1])), :]
best_brf_performance = best_brf_performance[0, :]

best_performances = \
    np.vstack((best_gp_performance, \
               best_rf_performance, \
                   best_nn_performance, \
                       best_brf_performance))

best_policy_omegas = \
    best_performances[ \
                      np.where(best_performances[:, -1] == \
                               np.amax(best_performances[:, -1])), \
                          list(range(len(worst_performances) - 1))]

#   Dividing data:
    
initial_node_data_sizes = divideData(number_of_nodes, \
                                     period_start_time, \
                                         period_end_time, \
                                             biggest_cluster_connectivities, \
                                                 [number_of_data_segments], 0)

#   Allocating data:
    
initial_segment_allocations = \
	allocateData(number_of_nodes, initial_node_data_sizes)

#   Worst-policy simulation:

#   Specifying the worst policy weights found so far:
    
omegas = worst_policy_omegas.transpose()
    
#   Following the specified policy weights:

final_segment_allocations = np.copy(initial_segment_allocations)

final_segment_allocations, worst_policy_actions_history, node_statuses = \
    followPolicy(period_start_time, period_end_time, number_of_nodes, \
                 biggest_cluster_connectivities, \
                     final_segment_allocations, omegas, synthetic_lats, \
                         synthetic_lons, biggest_cluster_members) 

worst_policy_segment_allocations = \
    np.copy(final_segment_allocations)

#   Naive-policy simulation:

#   Specifying the naive policy weights:
    
omegas = naive_policy_omegas.transpose()
    
#   Following the specified policy weights:

final_segment_allocations = np.copy(initial_segment_allocations)

final_segment_allocations, naive_policy_actions_history, node_statuses = \
    followPolicy(period_start_time, period_end_time, number_of_nodes, \
                 biggest_cluster_connectivities, \
                     final_segment_allocations, omegas, synthetic_lats, \
                         synthetic_lons, biggest_cluster_members) 

naive_policy_segment_allocations = np.copy(final_segment_allocations)

#   Best-policy simulation:

#   Specifying the best policy weights found so far:
    
omegas = best_policy_omegas.transpose()
    
#   Following the specified policy weights:

final_segment_allocations = np.copy(initial_segment_allocations)

final_segment_allocations, best_policy_actions_history, node_statuses = \
    followPolicy(period_start_time, period_end_time, number_of_nodes, \
                 biggest_cluster_connectivities, \
                     final_segment_allocations, omegas, synthetic_lats, \
                         synthetic_lons, biggest_cluster_members)

best_policy_segment_allocations = np.copy(final_segment_allocations)

#   Visualizing the different policy performances:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Data exchages under best policy @ 5:00 PM')

map = plt.imread('python_policies_map.jpg')
ax1.imshow(map, \
           extent = [np.amin(synthetic_lons[ \
                                            biggest_cluster_members, 17 * 60 * 6 + 3]), \
                     np.amax(synthetic_lons[ \
                                            biggest_cluster_members, 17 * 60 * 6 + 3]), \
                         np.amin(synthetic_lats[ \
                                                biggest_cluster_members, 17 * 60 * 6 + 3]), \
                             np.amax(synthetic_lats[ \
                                                    biggest_cluster_members, 17 * 60 * 6 + 3])])

ax1.scatter(synthetic_lons[biggest_cluster_members, 17 * 60 * 6 + 3], \
            synthetic_lats[biggest_cluster_members, 17 * 60 * 6 + 3], \
                color = 'black', marker = '.')

for i in range(np.shape(biggest_cluster_members)[1]):
    
    relevant_actions_history = \
        best_policy_actions_history \
            [np.where(best_policy_actions_history[:, 0] == \
                      17 * 60 * 6 + 3), :]
    
    relevant_actions_history = \
        relevant_actions_history \
        [np.where(relevant_actions_history[:, 1] == i), :]
    
    if relevant_actions_history.size() != 0:
        
        transmitter = i
        
        ax1.scatter( \
                    synthetic_lons[ \
                                   biggest_cluster_members[transmitter], \
                                       17 * 60 * 6 + 3], \
                        synthetic_lats[ \
                                       biggest_cluster_members[transmitter], \
                                           17 * 60 * 6 + 3], \
                            color = 'red', marker = '.')
        
        #   Displaying coverage areas:
        
        broadcasting_range = \
            relevant_actions_history[0, 2] / 1000
        earth_radius = 6371
        
        transmitter_lon = \
            synthetic_lons[ \
                           biggest_cluster_members[transmitter], \
                               17 * 60 * 6 + 3]
        transmitter_lat = \
            synthetic_lats[ \
                           biggest_cluster_members[transmitter], \
                               17 * 60 * 6 + 3]
        
        transmitter_x = earth_radius * \
            np.cos(transmitter_lat * np.pi / 180) * \
                np.cos(transmitter_lon * np.pi / 180)
        transmitter_y = earth_radius * \
            np.cos(transmitter_lat * np.pi / 180) * \
                np.sin(transmitter_lon * np.pi / 180)
        
        transmitter_xs = transmitter_x + \
            broadcasting_range * \
                np.cos(list(range(np.pi / 50, 2 * np.pi + 1)))
        transmitter_ys = transmitter_y + \
            broadcasting_range * \
                np.sin(list(range(np.pi / 50, 2 * np.pi + 1)))
        
        transmitter_lons = \
            np.degrees(np.arctan(transmitter_ys / transmitter_xs))
        transmitter_lats = \
            np.degrees(np.arccos(transmitter_ys / \
                              (earth_radius * \
                               np.degrees(np.arcsin(transmitter_lons)))))
        
        #   Correcting coverage areas:
        
        for j in range(len(transmitter_lons)):
            
            distance = \
                geopy.distance.geodesic( \
                                        [transmitter_lat, transmitter_lon], \
                                            [transmitter_lats[j], transmitter_lons[j]]).km
            
            if distance > broadcasting_range:
                
                theta = \
                    math.atan2((transmitter_ys[j] - transmitter_y), \
                               (transmitter_xs[j] - transmitter_x))
                
                transmitter_xs[j] = \
                    transmitter_x + \
                        broadcasting_range / distance * \
                            broadcasting_range * np.cos(theta)
                transmitter_ys[j] = \
                    transmitter_y + \
                        broadcasting_range / distance * \
                            broadcasting_range * np.sin(theta)
                
                transmitter_lons[j] = \
                    np.degrees(np.arctan(transmitter_ys[j] / \
                                    transmitter_xs[j]))
                transmitter_lats[j] = \
                    np.degrees(np.arccos(transmitter_ys[j] / \
                                      (earth_radius * \
                                       np.degrees(np.arcsin(transmitter_lons[j])))))
        
        ax1.plot(transmitter_lons, transmitter_lats, '-r')
        
        for j in range(np.shape(relevant_actions_history)[0]):
            
            receiver = \
                np.where(relevant_actions_history[j, 4:] == 1)
            
            ax1.plot( \
                [synthetic_lons[ \
                                biggest_cluster_members[transmitter], 17 * 60 * 6 + 3], \
                 synthetic_lons[ \
                                biggest_cluster_members[receiver], 17 * 60 * 6 + 3]], \
                    [synthetic_lats[ \
                                    biggest_cluster_members[transmitter], 17 * 60 * 6 + 3], \
                     synthetic_lats[ \
                                    biggest_cluster_members[receiver], 17 * 60 * 6 + 3]], \
                        'r')
            
            ax1.scatter( \
                        synthetic_lons[ \
                                       biggest_cluster_members[receiver], \
                                           17 * 60 * 6 + 3], \
                            synthetic_lats[ \
                                           biggest_cluster_members[receiver], \
                                               17 * 60 * 6 + 3], \
                                color = 'green', marker = '.')

ax1.set_xlim([np.amin(synthetic_lons[ \
                                     biggest_cluster_members, 17 * 60 * 6 + 3]), \
              np.amax(synthetic_lons[ \
                                     biggest_cluster_members, 17 * 60 * 6 + 3])])
ax1.set_ylim([np.amin(synthetic_lats[ \
                                     biggest_cluster_members, 17 * 60 * 6 + 3]), \
              np.amax(synthetic_lats[ \
                                     biggest_cluster_members, 17 * 60 * 6 + 3])])
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')

ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('data exchanges at 5.png', dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Zoomed-in data exchages\nunder best policy @ 5:00 PM')

map = plt.imread('python_zoomed_in_policies_map.jpg')
ax2.imshow(map, extent = [-80.5, -80.48, 43.44, 43.46])

ax2.scatter(synthetic_lons[biggest_cluster_members, 17 * 60 * 6 + 3], \
            synthetic_lats[biggest_cluster_members, 17 * 60 * 6 + 3], \
                color = 'black', marker = '.')

for i in range(np.shape(biggest_cluster_members)[1]):
    
    relevant_actions_history = \
        best_policy_actions_history \
            [np.where(best_policy_actions_history[:, 0] == \
                      17 * 60 * 6 + 3), :]
    
    relevant_actions_history = \
        relevant_actions_history \
            [np.where(relevant_actions_history[:, 1] == i), :]
    
    if relevant_actions_history.size() != 0:
        
        transmitter = i
        
        ax2.scatter( \
                    synthetic_lons[ \
                                   biggest_cluster_members[transmitter], \
                                       17 * 60 * 6 + 3], \
                        synthetic_lats[ \
                                       biggest_cluster_members[transmitter], \
                                           17 * 60 * 6 + 3], \
                            color = 'red', marker = '.')
        
        #   Displaying coverage areas:
        
        broadcasting_range = \
            relevant_actions_history[0, 2] / 1000
        earth_radius = 6371
        
        transmitter_lon = \
            synthetic_lons[ \
                           biggest_cluster_members[transmitter], \
                               17 * 60 * 6 + 3]
        transmitter_lat = \
            synthetic_lats[ \
                           biggest_cluster_members[transmitter], \
                               17 * 60 * 6 + 3]
        
        transmitter_x = earth_radius * \
            np.cos(transmitter_lat * np.pi / 180) * \
                np.cos(transmitter_lon * np.pi / 180)
        transmitter_y = earth_radius * \
            np.cos(transmitter_lat * np.pi / 180) * \
                np.sin(transmitter_lon * np.pi / 180)
        
        transmitter_xs = transmitter_x + \
            broadcasting_range * \
                np.cos(list(range(np.pi / 50, 2 * np.pi + 1)))
        transmitter_ys = transmitter_y + \
            broadcasting_range * \
                np.sin(list(range(np.pi / 50, 2 * np.pi + 1)))
        
        transmitter_lons = \
            np.degrees(np.arctan(transmitter_ys / transmitter_xs))
        transmitter_lats = \
            np.degrees(np.arccos(transmitter_ys / \
                              (earth_radius * np.degrees(np.arcsin(transmitter_lons)))))
        
        #   Correcting coverage areas:
        
        for j in range(len(transmitter_lons)):
            
            distance = \
                geopy.distance.geodesic( \
                                        [transmitter_lat, transmitter_lon], \
                                            [transmitter_lats[j], transmitter_lons[j]]).km
            
            if distance > broadcasting_range:
                
                theta = \
                    math.atan2((transmitter_ys[j] - transmitter_y), \
                               (transmitter_xs[j] - transmitter_x))
                
                transmitter_xs[j] = \
                    transmitter_x + \
                        broadcasting_range / distance * \
                            broadcasting_range * np.cos(theta)
                transmitter_ys[j] = \
                    transmitter_y + \
                        broadcasting_range / distance * \
                            broadcasting_range * np.sin(theta)
                
                transmitter_lons[j] = \
                    np.degrees(np.arctan(transmitter_ys[j] / transmitter_xs[j]))
                transmitter_lats[j] = np.degrees(np.arccos(transmitter_ys[j] / \
                                                        (earth_radius * np.degrees(np.arcsin(transmitter_lons[j])))))
        
        ax2.plot(transmitter_lons, transmitter_lats, '-r')
        
        for j in range(np.shape(relevant_actions_history)[0]):
            
            receiver = \
                np.where(relevant_actions_history[j, 4:] == 1)
            
            ax2.plot( \
                     [synthetic_lons[ \
                                     biggest_cluster_members[transmitter], 17 * 60 * 6 + 3], \
                      synthetic_lons[ \
                                     biggest_cluster_members[receiver], 17 * 60 * 6 + 3]], \
                         [synthetic_lats[ \
                                         biggest_cluster_members[transmitter], 17 * 60 * 6 + 3], \
                          synthetic_lats[ \
                                         biggest_cluster_members[receiver], 17 * 60 * 6 + 3]], \
                             'r')
            
            ax2.scatter( \
                        synthetic_lons[ \
                                       biggest_cluster_members[receiver], \
                                           17 * 60 * 6 + 3], \
                            synthetic_lats[ \
                                           biggest_cluster_members[receiver], \
                                               17 * 60 * 6 + 3], \
                                color = 'green', marker = '.')

ax2.set_xlim([-80.5, -80.48])
ax2.set_ylim([43.44, 43.46])
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure2.tight_layout()
figure2.savefig('zoomed-in data exchanges at 5.png', dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Initial data segment allocations')
ax3.set_xlabel('Segment index')
ax3.set_ylabel('Node index')

temp = \
    ax3.imshow(initial_segment_allocations, \
               interpolation = 'nearest', aspect = 'auto')
figure3.colorbar(temp, ax = ax3)

figure3.savefig('initial segment allocations.png', dpi = 500)

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Worst-policy data segment allocations')
ax4.set_xlabel('Segment index')
ax4.set_ylabel('Node index')

temp = \
    ax4.imshow(worst_policy_segment_allocations, \
               interpolation = 'nearest', aspect = 'auto')
figure4.colorbar(temp, ax = ax4)

figure4.savefig('worst policy segment allocations.png', dpi = 500)

figure5, ax5 = plt.subplots(nrows = 1, ncols = 1)

ax5.set_title('Naive-policy data segment allocations')
ax5.set_xlabel('Segment index')
ax5.set_ylabel('Node index')

temp = \
    ax5.imshow(naive_policy_segment_allocations, \
               interpolation = 'nearest', aspect = 'auto')
figure5.colorbar(temp, ax = ax5)

figure5.savefig('naive policy segment allocations.png', dpi = 500)

figure6, ax6 = plt.subplots(nrows = 1, ncols = 1)

ax6.set_title('Best-policy data segment allocations')
ax6.set_xlabel('Segment index')
ax6.set_ylabel('Node index')

temp = \
    ax6.imshow(best_policy_segment_allocations, \
               interpolation = 'nearest', aspect = 'auto')
figure6.colorbar(temp, ax = ax6)

figure6.savefig('best policy segment allocations.png', dpi = 500)

_, idx = \
    np.unique(worst_policy_actions_history[:, 0], axis = 0, \
              return_index = True)
worst_policy_transmission_times = \
    worst_policy_actions_history[np.sort(idx), 0]
            
worst_policy_data_collected = \
    np.zeros((np.shape(worst_policy_transmission_times)[0], 1))
worst_policy_data_accumulated = \
    np.zeros((np.shape(worst_policy_transmission_times)[0], 1))

for i in range(np.shape(worst_policy_transmission_times)[0]):
    
    worst_policy_data_collected[i] = \
        np.shape(worst_policy_actions_history[ \
                                              np.where(worst_policy_actions_history[:, 0] == \
                                                       worst_policy_transmission_times[i]), :])[0]

    worst_policy_data_accumulated[i] = \
        np.sum(worst_policy_data_collected)

_, idx = \
    np.unique(naive_policy_actions_history[:, 0], axis = 0, \
              return_index = True)
naive_policy_transmission_times = \
    naive_policy_actions_history[np.sort(idx), 0]    
naive_policy_data_collected = \
    np.zeros((np.shape(naive_policy_transmission_times)[0], 1))
naive_policy_data_accumulated = \
    np.zeros((np.shape(naive_policy_transmission_times)[0], 1))

for i in range(np.shape(naive_policy_transmission_times)[0]):
    
    naive_policy_data_collected[i] = \
        np.shape(naive_policy_actions_history[ \
                                              np.where(naive_policy_actions_history[:, 0] == \
                                                       naive_policy_transmission_times[i]), :])[0]

    naive_policy_data_accumulated[i] = \
        np.sum(naive_policy_data_collected)

_, idx = \
    np.unique(best_policy_actions_history[:, 0], axis = 0, \
              return_index = True)
best_policy_transmission_times = \
    best_policy_actions_history[np.sort(idx), 0]    
best_policy_data_collected = \
    np.zeros((np.shape(best_policy_transmission_times)[0], 1))
best_policy_data_accumulated = \
    np.zeros((np.shape(best_policy_transmission_times)[0], 1))

for i in range(np.shape(best_policy_transmission_times)[0]):
    
    best_policy_data_collected[i] = \
        np.shape(best_policy_actions_history[ \
                                             np.where(best_policy_actions_history[:, 0] == \
                                                      best_policy_transmission_times[i]), :])[0]

    best_policy_data_accumulated[i] = \
        np.sum(best_policy_data_collected)

figure7, ax7 = plt.subplots(nrows = 1, ncols = 1)

ax7.set_title('Instantaneous data segments\nexchanged vs. time')
ax7.set_xlabel('Time (Hour)')
ax7.set_ylabel('Instantaneous data\nsegments exchanged')

ax7.plot(worst_policy_transmission_times / (60 * 6), \
         worst_policy_data_collected, \
             '--r', label = 'Worst-policy')
ax7.plot(naive_policy_transmission_times / (60 * 6), \
         naive_policy_data_collected, \
             '-*b', label = 'Naive-policy')
ax7.plot(best_policy_transmission_times / (60 * 6), \
         best_policy_data_collected, \
             '-og', label = 'Best-policy')

ax7.set_xlim([0, \
              np.amax(np.amax(np.amax(worst_policy_data_collected)))])
ax7.set_ylim([np.amax(naive_policy_data_collected), \
              np.amax(best_policy_data_collected) * 1.1])
ax7.legend(loc = 'best')
ax7.grid(color = 'k', linestyle = '--', linewidth = 1)

figure7.tight_layout()
figure7.savefig('instantaneous data segments exchanged vs. time', \
                dpi = 500)

figure8, ax8 = plt.subplots(nrows = 1, ncols = 1)

ax8.set_title('Cumulative data segments\nexchanged vs. time')
ax8.set_xlabel('Time (Hour)')
ax8.set_ylabel('Cumulative data\nsegments exchanged')

ax8.plot(worst_policy_transmission_times / (60 * 6), \
         worst_policy_data_accumulated, \
             '--r', label = 'Worst-policy')
ax8.plot(naive_policy_transmission_times / (60 * 6), \
         naive_policy_data_accumulated, \
             '-*b', label = 'Naive-policy')
ax8.plot(best_policy_transmission_times / (60 * 6), \
         best_policy_data_accumulated, \
             '-og', label = 'Best-policy')
ax8.legend(loc = 'best')
ax8.grid(color = 'k', linestyle = '--', linewidth = 1)

figure8.tight_layout()
figure8.savefig('cumulative data segments exchanged vs. time', \
                dpi = 500)

data_initially_distributed_percentages = \
    initial_node_data_sizes.transpose() / \
        np.sum(initial_node_data_sizes) * 100

worst_policy_data_exchanged_percentages = \
    np.sum(worst_policy_segment_allocations, axis = 1) / \
        np.sum(initial_node_data_sizes) * 100 - \
            data_initially_distributed_percentages

worst_policy_data_finally_distributed_percentages = \
    np.ones((number_of_nodes, 1)) * 100 - \
        (data_initially_distributed_percentages + \
         worst_policy_data_exchanged_percentages)

naive_policy_data_exchanged_percentages = \
    np.sum(naive_policy_segment_allocations, axis = 1) / \
        np.sum(initial_node_data_sizes) * 100 - \
            data_initially_distributed_percentages

naive_policy_data_finally_distributed_percentages = \
    np.ones((number_of_nodes, 1)) * 100 - \
        (data_initially_distributed_percentages + \
         naive_policy_data_exchanged_percentages)

best_policy_data_exchanged_percentages = \
    np.sum(best_policy_segment_allocations, axis = 1) / \
        np.sum(initial_node_data_sizes) * 100 - \
            data_initially_distributed_percentages

best_policy_data_finally_distributed_percentages = \
    np.ones((number_of_nodes, 1)) * 100 - \
        (data_initially_distributed_percentages + \
         best_policy_data_exchanged_percentages)

figure9, ax9 = plt.subplots(nrows = 1, ncols = 1)

ax9.set_title( \
              'Node data segment category\npercentages under worst-policy')

cumulative_bottom = np.arange(number_of_nodes) * 0
cumulative_bottom = cumulative_bottom.astype(float)

ax9.bar(np.arange(number_of_nodes) + 1, \
        data_initially_distributed_percentages, \
            bottom = cumulative_bottom, color = 'blue', \
                label = 'Initially-distributed')
    
cumulative_bottom += \
    data_initially_distributed_percentages

ax9.bar(np.arange(number_of_nodes) + 1, \
        worst_policy_data_exchanged_percentages, \
            bottom = cumulative_bottom, color = 'green', \
                label = 'Exchanged')

cumulative_bottom += \
    worst_policy_data_exchanged_percentages

ax9.bar(np.arange(number_of_nodes) + 1, \
        worst_policy_data_finally_distributed_percentages, \
            bottom = cumulative_bottom, color = 'yellow', \
                label = 'Finally-distributed')
        
ax9.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')

ax9.set_xlim([0, number_of_nodes + 1])
ax9.set_xlabel('Node index')
ax9.set_ylabel('Data segment\ncategory percentage')
ax9.grid(color = 'k', linestyle = '--', linewidth = 1)

figure9.tight_layout()
figure9.savefig('segment category percentages under worst policy', \
                dpi = 500)

figure10, ax10 = plt.subplots(nrows = 1, ncols = 1)

ax10.set_title('Node data segment category\npercentages under naive-policy')

cumulative_bottom = np.arange(number_of_nodes) * 0
cumulative_bottom = cumulative_bottom.astype(float)

ax10.bar(np.arange(number_of_nodes) + 1, \
        data_initially_distributed_percentages, \
            bottom = cumulative_bottom, color = 'blue', \
                label = 'Initially-distributed')
    
cumulative_bottom += \
    data_initially_distributed_percentages

ax10.bar(np.arange(number_of_nodes) + 1, \
        naive_policy_data_exchanged_percentages, \
            bottom = cumulative_bottom, color = 'green', \
                label = 'Exchanged')

cumulative_bottom += \
    naive_policy_data_exchanged_percentages

ax10.bar(np.arange(number_of_nodes) + 1, \
        naive_policy_data_finally_distributed_percentages, \
            bottom = cumulative_bottom, color = 'yellow', \
                label = 'Finally-distributed')
        
ax10.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')

ax10.set_xlim([0, number_of_nodes + 1])
ax10.set_xlabel('Node index')
ax10.set_ylabel('Data segment\ncategory percentage')
ax10.grid(color = 'k', linestyle = '--', linewidth = 1)

figure10.tight_layout()
figure10.savefig('segment category percentages under naive policy', \
                dpi = 500)

figure11, ax11 = plt.subplots(nrows = 1, ncols = 1)

ax11.set_title('Node data segment category\npercentages under best-policy')

cumulative_bottom = np.arange(number_of_nodes) * 0
cumulative_bottom = cumulative_bottom.astype(float)

ax11.bar(np.arange(number_of_nodes) + 1, \
        data_initially_distributed_percentages, \
            bottom = cumulative_bottom, color = 'blue', \
                label = 'Initially-distributed')
    
cumulative_bottom += \
    data_initially_distributed_percentages

ax11.bar(np.arange(number_of_nodes) + 1, \
        best_policy_data_exchanged_percentages, \
            bottom = cumulative_bottom, color = 'green', \
                label = 'Exchanged')

cumulative_bottom += \
    best_policy_data_exchanged_percentages

ax11.bar(np.arange(number_of_nodes) + 1, \
        best_policy_data_finally_distributed_percentages, \
            bottom = cumulative_bottom, color = 'yellow', \
                label = 'Finally-distributed')
        
ax11.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')

ax11.set_xlim([0, number_of_nodes + 1])
ax11.set_xlabel('Node index')
ax11.set_ylabel('Data segment\ncategory percentage')
ax11.grid(color = 'k', linestyle = '--', linewidth = 1)

figure11.tight_layout()
figure11.savefig('segment category percentages under best policy', \
                dpi = 500)

data_rates = list(range(3, 30, 3))

data_initially_distributed = \
    np.sum(np.sum(initial_segment_allocations)) \
        * 10 * data_rates * 1024 / 8 / 1024 / 1024

data_after_exchanges = \
    np.sum(np.sum(best_policy_segment_allocations)) \
        * 10 * data_rates * 1024 / 8 / 1024 / 1024

data_finally_distributed = \
    number_of_nodes * np.sum(np.sum(initial_segment_allocations)) \
        * 10 * data_rates * 1024 / 8 / 1024 / 1024

figure12, ax12 = plt.subplots(nrows = 1, ncols = 1)

ax12.set_title('Data category sizes\nunder best-policy')

ax12.plot(data_rates, data_initially_distributed,'--k', \
          label = 'Initially-distributed')
ax12.plot(data_rates, data_after_exchanges,'-+b', \
          label = 'After-exchanges')
ax12.plot(data_rates, data_finally_distributed,'-xr', \
          label = 'After-final-distributions')

ax12.set_xlim([3, 27])
ax12.set_xlabel('Data rate (Mbps)')
ax12.set_ylabel('Data size (GB)')
ax12.legend(loc = 'best')
ax12.grid(color = 'k', linestyle = '--', linewidth = 1)

figure12.tight_layout()
figure12.savefig('data category sizes under best policy', \
                dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

connectivities = np.load('connectivities.npy', allow_pickle = True)
biggest_cluster_data= \
    np.load('biggest_cluster_data.npy', allow_pickle = True)
modified_data = np.load('modified_data.npy', allow_pickle = True)
synthetic_lats= np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)

#   Extracting next biggest clusters according
#   to the minimum contact durations schedule:

#   Notice that the first biggest cluster is chosen within 
#   the specified period start and end times given the first 
#   minimum discontinuous contact duration as shown at 
#   the beginning of this script (i.e. 20 min). Also notice 
#   that the last minimum discontinuous contact duration of 
#   the schedule should guarantee that all nodes are chosen 
#   eventually to be part of a cluster. Consider that 
#   the schedule given here does not necessarily lead to this 
#   eventual inclusion of all nodes; it is just presented as 
#   a sample for explanation purposes.

period_start_time = 16 * 60
period_end_time = 18 * 60
time_delta = 10
minimum_contact_durations_schedule = [15, 15, 15] * 60
maximum_number_of_hops = 20

for k in range(len(minimum_contact_durations_schedule)):
    
    #   Removing connectivities of previous biggest clusters:
    
    for t in range(27 * 60 * 6):
        
        connectivities[t, biggest_cluster_members, :] = \
            np.zeros((1, \
                      np.shape(biggest_cluster_members)[0], \
                      np.shape(connectivities)[2]))
        connectivities[t, :, biggest_cluster_members] = \
            np.zeros((1, \
                      np.shape(connectivities)[1], \
                      np.shape(biggest_cluster_members)[0]))
    
    #   Extracting the next biggest cluster:
    
    biggest_cluster, cluster_members = \
        extractCluster(period_start_time, period_end_time, \
                       connectivities, time_delta, \
                           minimum_contact_durations_schedule[k], \
                               maximum_number_of_hops)
    
    biggest_cluster_members = \
        cluster_members[biggest_cluster]
    
    #   Visualizing the biggest cluster at the busiest time:
    
    busiest_time = \
        (period_start_time + period_end_time) / 2 * 6
    
    figure, ax = plt.subplots(nrows = 1, ncols = 1)
    
    if k == 1:
        
        ax.set_title('Second biggest cluster @ 5:00 PM')
    
    if k == 2:
        
        ax.set_title('Third biggest cluster @ 5:00 PM')
    
    if k == 3:
        
        ax.set_title('Fourth biggest cluster @ 5:00 PM')

    map = plt.imread('python_map.jpg')
    ax.imshow(map, \
              extent = [np.amin(modified_data[:, 5]), \
                        np.amax(modified_data[:, 5]), \
                            np.amin(modified_data[:, 4]), \
                                np.amax(modified_data[:, 4])])

    ax.scatter(synthetic_lons[:, busiest_time], \
               synthetic_lats[:, busiest_time], \
                   color = 'yellow', marker = 'o', \
                       edgecolors = 'black')
    
    ax.scatter(synthetic_lons[ \
                              biggest_cluster_members, busiest_time], \
               synthetic_lats[ \
                              biggest_cluster_members, busiest_time], \
                   color = 'red', marker = 'o', \
                       edgecolors = 'black')
    
    ax.set_xlim([np.amin(modified_data[:, 5]), \
                 np.amax(modified_data[:, 5])])
    ax.set_ylim([np.amin(modified_data[:, 4]), \
                 np.amax(modified_data[:, 4])])
    
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    
    figure.tight_layout()
        
    if k == 1:

        figure.savefig('second biggest cluster at 5 pm.png', dpi = 500)
    
    if k == 2:
        
        figure.savefig('third biggest cluster at 5 pm.png', dpi = 500)
     
    if k == 3:
        
        figure.savefig('fourth biggest cluster at 5 pm.png', dpi = 500)

#%%

#   Written by "Kais Suleiman" (ksuleiman.weebly.com)