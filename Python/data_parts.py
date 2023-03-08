from pathlib import Path
import json
import time
from logger import logging_setup
logger = logging_setup(__name__)

## TODO: 
# - Sanitiy check for loaded neighborhood data
# - Check if nodes are in a row
# - Check if nodes start at 0 or 1

def weights_to_line(weights, index):
    # Add Weight at the beginning
    if weights is None:
        line = f"{0} "
    else:
        line = f"{weights[index]} "

    return line

def embeddings_to_line(line, edge, edge_emb, FeatureCounter, first_run, groupinfo=None):
    # edge = [node1, node2]
    # edge to node
    node1, node2 = edge[0], edge[1]
    
    # Log features
    if first_run:
        # -1 necessary because 0 is included because this is the first feature
        logger.info(f"Features {FeatureCounter}-{FeatureCounter+len(edge_emb[node1])-1}: Edge Embeddings Node 1")
        logger.info(f"Features {FeatureCounter+len(edge_emb[node1])-1}-"
            f"{FeatureCounter+len(edge_emb[node1])+len(edge_emb[node2])-1}: Edge Embeddings Node 2")

    # List node embeddings
    # Embeddings node 1
    for val in edge_emb[node1]:
        if val != 0.0:
            line += f"{FeatureCounter}:{val} "
        FeatureCounter += 1

    # Embeddings node 2
    for val in edge_emb[node2]:
        if val != 0.0:
            line += f"{FeatureCounter}:{val} "
        FeatureCounter += 1

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, len(edge_emb[node1])+len(edge_emb[node2]))
    
    return line, FeatureCounter, groupinfo

def ids_to_line(line, edge, FeatureCounter, number_of_nodes, first_run, groupinfo=None):
    if first_run:
        logger.info(f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Node IDs")

    # Add Node IDs
    node1, node2 = edge[0], edge[1]
    if node1 < node2:
        line += f"{FeatureCounter+node1}:1 "
        line += f"{FeatureCounter+node2}:1 "
    else:
        line += f"{FeatureCounter+node2}:1 "
        line += f"{FeatureCounter+node1}:1 "

    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo

def neighborhood_to_line(line, node, FeatureCounter, number_of_nodes, first_run, groupinfo=None, loaded_data=None):
    node = str(int(node))
    # Log features
    if first_run:
        logger.info(f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Neighborhood of edge nodes")

    #Find out if nodes are neighbors of edge nodes and add to line
    for neighbor_node in loaded_data[node]:
        line += f"{FeatureCounter+int(neighbor_node)}:1 "
    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo

def recent_neighborhood_to_line(line, node, FeatureCounter, number_of_nodes, first_run, groupinfo=None, loaded_data=None):
    node = str(int(node))
    # Log features
    if first_run:
        logger.info(f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Neighborhood of edge nodes")

    #Find out if nodes are neighbors of edge nodes and add to line
    for neighbor_node in loaded_data[node]:
        line += f"{FeatureCounter+int(neighbor_node)}:1 "
    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo


def neighborhood_data_loader(edge_set, number_of_nodes, config):
    start_time = time.time()
    # Set the path to the folder containing the JSON file
    file_path = Path(f"{config['FOLDERNAMES']['neighborhood_folder']}/{config['FILENAMES']['neighborhood_file']}")

    # Check if the file exists in the folder
    if file_path.is_file():
        # Open the file and load the JSON data into a dictionary
        with file_path.open(mode='r') as f:
            data = json.load(f)
        logger.info('Neighborhood data loaded from file.')
    else:
        data = {}
        logger.info('Neighborhood data not found. Creating new data.')
        set_tensor = set(map(tuple, edge_set.numpy()))
        # Create data
        for node in range(0, number_of_nodes):
            data[node] = {}
            for i in range(0, number_of_nodes): # i = potential neighbor node
                if (((node, i) in set_tensor or (i, node) in set_tensor) and node != i):
                    data[node][i] = 1
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
        logger.info(f'Neighborhood data creation execution time: {time.time() - start_time:.2f} seconds for {number_of_nodes} nodes')
    return data

def add_groupinfo(groupinfo, number_elements):
    groupinfo[1].extend([str(groupinfo[0]) for i in range(number_elements)])
    groupinfo[0] += 1

    return groupinfo