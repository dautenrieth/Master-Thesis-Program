import random
from ogb.linkproppred import PygLinkPropPredDataset
from pathlib import Path
import torch
import time
import os
from configparser import ConfigParser, ExtendedInterpolation
import utils as ut
import commandline as cl
import data_parts as dp


from logger import logging_setup
logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')

## TODO:
# - Randomise data --> DONE
# - Neighbourhood data is loaded multiple times

def create_data_files(config: ConfigParser):
    d_name = config['STANDARD']['graph_name']
    
    # Save data config to file
    ut.data_config_log()

    logger.info("create_data_files started")
    # load dataset (first time running will download zip file otherwise existing file in dataset folder will be used)
    dataset = PygLinkPropPredDataset(name = d_name)
    # split loaded data
    split_edges = dataset.get_edge_split()
    train_edges, valid_edges, test_edges = split_edges["train"], split_edges["valid"], split_edges["test"]
    
    # Get list of 128-bit edge embeddings (see documentation)
    edge_emb = dataset[0]['x']

    # Call rest of logic for different filenames & data
    number_of_nodes = edge_emb.shape[0]

    # Check if file with negative samples exists
    if os.path.exists('neg_samples.txt'):
        neg_samples = ut.read_neg_edges()
        if len(neg_samples) != len(train_edges['edge']):
            # Create negative samples
            neg_samples = create_neg_samples(train_edges, valid_edges, test_edges, number_of_nodes=number_of_nodes)
            # Write negative samples to file
            ut.write_neg_edges(neg_samples)
    else:
        # Create negative samples
        neg_samples = create_neg_samples(train_edges, valid_edges, test_edges, number_of_nodes=number_of_nodes)
        # Write negative samples to file
        ut.write_neg_edges(neg_samples)
    
    # Create train data
    data_lines = data_structure(pos_edge=train_edges['edge'], neg_edge=neg_samples, edge_emb=edge_emb, number_of_nodes=number_of_nodes, weights=train_edges['weight'])
    # Randomise data
    random.shuffle(data_lines)
    path = Path(f"{config['PATHS']['libfm_path']}/{config['FILENAMES']['libfm_train']}")
    ut.write_file(data_lines, path)
    # Print number of available postive and negative samples
    print(f"Number of positive samples: {len(train_edges['edge'])}, Number of negative samples: {len(neg_samples)}")

    # Create valid data
    data_lines = data_structure(pos_edge=valid_edges['edge'], neg_edge=valid_edges['edge_neg'], edge_emb=edge_emb, number_of_nodes=number_of_nodes, weights=valid_edges['weight'])
    path = Path(f"{config['PATHS']['libfm_path']}/{config['FILENAMES']['libfm_valid']}")
    ut.write_file(data_lines, path)
    # Print number of available postive and negative samples
    print(f"Number of positive samples: {len(valid_edges['edge'])}, Number of negative samples: {len(valid_edges['edge_neg'])}")

    # Create test data
    data_lines = data_structure(pos_edge=test_edges['edge'], neg_edge=test_edges['edge_neg'], edge_emb=edge_emb, number_of_nodes=number_of_nodes, weights=test_edges['weight'])
    path = Path(f"{config['PATHS']['libfm_path']}/{config['FILENAMES']['libfm_test']}")
    ut.write_file(data_lines, path)

    # Print number of available postive and negative samples
    print(f"Number of positive samples: {len(test_edges['edge'])}, Number of negative samples: {len(test_edges['edge_neg'])}")

    # Convert to binary files
    path_to_libfm = Path(config['PATHS']['libfm_path'])
    cl.convert_to_binary(config['FILENAMES']['libfm_train'], path_to_libfm)
    cl.convert_to_binary(config['FILENAMES']['libfm_valid'], path_to_libfm)
    cl.convert_to_binary(config['FILENAMES']['libfm_test'], path_to_libfm)

    #Transpose binary files
    cl.transpose_binary(config['FILENAMES']['libfm_train'], path_to_libfm)
    cl.transpose_binary(config['FILENAMES']['libfm_valid'], path_to_libfm)
    cl.transpose_binary(config['FILENAMES']['libfm_test'], path_to_libfm)

    # Copy data files from libfm folder to data folder
    ut.copy_data_files()

    return


def data_structure(pos_edge, neg_edge, edge_emb, number_of_nodes, weights = None):
    start_time = time.time()
    # Create positive samples
    sample_data = create_sample_data(edge_set=pos_edge, edge_emb=edge_emb, number_of_nodes=number_of_nodes, weights=weights, x="pos")
    # Create negative samples
    sample_data.extend(create_sample_data(edge_set=neg_edge, edge_emb=edge_emb, number_of_nodes = number_of_nodes, x = "neg"))
    end_time = time.time()
    # Log the elapsed time as a message
    logger.info(f'Creation of data execution time: {end_time - start_time:.2f} seconds for {len(sample_data)} lines')
    return sample_data
    
   
def create_sample_data(edge_set, edge_emb, number_of_nodes, weights = None, x = None):
    sample_data = []
    
    first_run = True
    

    if config['DATASTRUCTURE'].getboolean('Grouping'):
        groupinfo = [0, []] # (group index, list of group numbers)
    else:
        groupinfo = None

    if config['DATASTRUCTURE'].getboolean('Neighborhood'):
        loaded_nh = dp.neighborhood_data_loader(edge_set, number_of_nodes, config)

    # Create samples
    for index, edge in enumerate(edge_set):

        FeatureCounter = 0
        line = dp.weights_to_line(weights, index)
        if first_run:
            logger.info(f"Target: Weight")

        if config['DATASTRUCTURE'].getboolean('Embeddings'):
            line, FeatureCounter, groupinfo = dp.embeddings_to_line(line, edge, edge_emb, FeatureCounter, first_run, groupinfo)

        if config['DATASTRUCTURE'].getboolean('NodeIDs'):
            line, FeatureCounter, groupinfo = dp.ids_to_line(line, edge, FeatureCounter, number_of_nodes, first_run, groupinfo)

        if config['DATASTRUCTURE'].getboolean('Neighborhood'):
            line, FeatureCounter, groupinfo = dp.neighborhood_to_line(line, edge[0], FeatureCounter, number_of_nodes, first_run, groupinfo, loaded_nh)
            line, FeatureCounter, groupinfo = dp.neighborhood_to_line(line, edge[1], FeatureCounter, number_of_nodes, first_run, groupinfo, loaded_nh)
        
        # Add line to list
        sample_data.append(line)
        if first_run:
            first_run = False
            # Write group info list to file
            if config['DATASTRUCTURE'].getboolean('Grouping'):
                path = Path(f"{config['PATHS']['libfm_path']}/{config['FILENAMES']['groups']}")
                ut.write_file(groupinfo[1], path)
                
        # Print progress
        if (index+1) % 100 == 0:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")},000 - {__name__} - DEBUG - "{index+1}" lines generated', end="\r")

        if config['NUMBERINSTANCES'].getboolean('ALL') is False:
            if index+1 >= config['NUMBERINSTANCES'].getint('NUMBER'):
                break

    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")},000 - {__name__} - DEBUG - "{index+1}" lines generated')
    return sample_data


def create_neg_samples(train_edges, valid_edges, test_edges, number_of_nodes=None, number_of_neg_samples=None):
    start_time = time.time()

    # Merge all edges
    merged_tensor = torch.cat((train_edges['edge'], valid_edges['edge'], test_edges['edge'], valid_edges['edge_neg'], test_edges['edge_neg']), dim=0)
    # Convert tensor to set
    merged_tensor = set(map(tuple, merged_tensor.numpy()))

    # Get negative edges when not in train, valid or test
    if number_of_neg_samples is None:
        number_of_neg_samples = len(train_edges['edge']) # Equal to number of positive edges
    neg_samples = set()
    for index in range(number_of_neg_samples):
        # edge = [node1, node2]
        # Create random edge
        # Generate two random integers between 1 and number_of_nodes
        node1 = random.randint(0, number_of_nodes-1)
        node2 = random.randint(0, number_of_nodes-1)

        # Check if edge is not in merged_tensor nor in neg_samples
        while (node1, node2) in merged_tensor or (node1, node2) in neg_samples:
            node1 = random.randint(0, number_of_nodes-1)
            node2 = random.randint(0, number_of_nodes-1)
        # Add edge to neg_samples
        neg_samples.add((node1, node2))

        # Print progress
        if index+1 % 100 == 0:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")},000 - {__name__} - DEBUG - "{index+1}/{number_of_neg_samples}"', end="\r")
        
        # Convert set to tensor
    neg_samples = torch.tensor(list(neg_samples))
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")},000 - {__name__} - DEBUG - "{index+1}/{number_of_neg_samples}"')
    # Log the elapsed time as a message
    logger.info(f'create_neg_samples-function execution time: {time.time() - start_time:.2f} seconds for {number_of_neg_samples} edges')
    return neg_samples