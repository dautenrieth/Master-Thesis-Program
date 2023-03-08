import os
from pathlib import Path
from logger import logging_setup
from configparser import ConfigParser, ExtendedInterpolation
import subprocess

from utils import load_file
import clear_outputs as co

# Setup for module
logger = logging_setup(__name__)



# def external_program_call(command: str, path: Path):

#     logger.info(f'Running command: {command}')
#     output_stream = os.popen(f'cd{path} && {command}')
#     out = output_stream.read()
#     logger.info(f'Finished running libfm')

#     process_output(out)

#     return

def external_program_call(command: str, path: Path):

    # Delete prediction file to prevent wrong evaluation
    co.delete_prediction_file()

    logger.info(f'Running command: {command}')
    process = subprocess.Popen(f"cd {path} && {command}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    logger.info(f'Finished running libfm')
    print(out)
    process_output(out)

    if err:
        logger.error(f"Error output: {err}")

    return

def process_output(out):
    # Function can be used to process output from libfm
    return

def convert_to_binary(filename: str, path: Path):
    # Function can be used to convert the files to binary
    logger.info(f'Convert file {filename} to binary')
    # Cut the file extension
    filename_wo_ext = filename.split('.')[0]
    # Create the command
    command = f'convert --ifile {filename} --ofilex {filename_wo_ext}.x --ofiley {filename_wo_ext}.y'
    output_stream = os.popen(f'cd{path} && {command}')
    out = output_stream.read()
    print(out)
    if 'num_rows' not in out:
        logger.error(f'Error in converting {filename} to binary:\nOutput till \
            determination:\n {out}\nPlease check filename and path.')
        raise Exception(f'Error in converting {filename} to binary. See log for more information')
    logger.info(f'Finished running convert function')
    return

def transpose_binary(filename: str, path: Path):
    # Function can be used to transpose the files to binary
    logger.info(f'Transpose binary file {filename}')
    # Cut the file extension
    filename_wo_ext = filename.split('.')[0]

    ## Convert x
    # Create the command
    command = f'transpose --ifile {filename_wo_ext}.x --ofile {filename_wo_ext}.xt'
    output_stream = os.popen(f'cd{path} && {command}')
    out = output_stream.read()

    if 'num_rows' not in out:
        logger.error(f'Error in transposing {filename} to binary:\nOutput till \
            determination:\n {out}\nPlease check filename and path.')
        raise Exception(f'Error in transposing {filename} to binary. See log for more information')

    ## Convert y
    # command = f'transpose --ifile {filename_wo_ext}.y --ofile {filename_wo_ext}.yt'
    # output_stream = os.popen(f'cd{path} && {command}')
    # out = output_stream.read()

    # if 'num_rows' not in out:
    #     logger.error(f'Error in transposing {filename} to binary:\nOutput till \
    #         determination:\n {out}\nPlease check filename and path.')
    #     raise Exception(f'Error in transposing {filename} to binary. See log for more information')

    logger.info(f'Finished running transpose function')
    return