# Author: Ellsworth McCullough
# File Name: hpfc_standard
# description Standard functions and classes used throughout hpfc

# Library imports
import os
from pathlib import Path


# Object used to hold values needed for regression
class HPFCData:
    def __init__(self, index, heights, values):
        self.index = index
        self.heights = heights
        self.values = values


# Displays an error message based on given values
def error_message(func, error):

    # Activates if func and error are strings
    if isinstance(func, str) and isinstance(error, str):
        print('FUNC : ' + func)
        print('ERROR: ' + error)
    # Activates if func or error are not strings
    else:
        # Activates if func is not a string
        if not isinstance(func, str):
            error_message(error_message.__name__,
                          'Inputted variable, func, is not the correct type')
        # Activates if error is not a string
        if not isinstance(error, str):
            error_message(error_message.__name__,
                          'Inputted variable, error, is not the correct type')


# Generates a filepath, if val is 0 will generate a filepath to last model in Data/Models
# if val is greater than 0, will generate a filepath to model_val in Data/Models
# if val is less than 0, will generate a filepath to model_(n - val), with n representing
# last model in models
# if load is false, will not check if filepath exists
# Assumes files have not been otherwise altered
# Returns generated file path
def generate_file_path(val=0, load=False):

    # Generates filename based on amount of files in directory
    file_name = 'model_'

    # Retrieves existing models
    existing_models = os.listdir('Data/Models')

    # Activates if we are loading a file and there are none to load
    if load and len(existing_models) == 0:
        return 0

    # Activates if val is default or negative
    # If negative, use would be to retrieve file
    if val <= 0:

        # Generates filename
        file_name += str(len(existing_models) + val) + '.pickle'

    # Activates if val is greater than 0
    else:

        # Generates filename
        file_name += str(val) + '.pickle'

    # Generates path to new file
    path = Path('.')
    path = path / 'Data' / 'Models' / file_name

    # Activates if load is true and file to load
    # does not exist
    if load and not os.path.isfile(path):
        return 0

    return path
