# Author: Ellsworth McCullough
# File Name: hpfc
# Description:
# hpfc stands for Height Prediction based on Facial Characteristics
# This file contains the start of the hpfc program
# This program takes in photograph and related heights and produces models
# based off of those.

# hpfc imports
from hpfc_standard import error_message
from preprocessing import preprocessing
from supervised_regression import supervised_regression
from postprocessing import postprocessing
from data_output import data_output

# library imports
import os
import pandas as pd


# Start of the program, acts as an overall manager function
# manages model creation iterations
def hpfc():

    # Path to arguments file to be retrieved from
    arg_path = 'arguments.txt'

    # Retrieves arguments to be inputted into functions
    args = argument_retrieval(arg_path)

    # Activates if arguments retrieved
    if not args.empty:

        # Loops the amount of times given in iterations
        for i in range(int(args.at[0, 'iterations'])):
            hfpc_iteration(i, args)

    # Activates if arguments not retrieved
    else:
        error_message(hpfc.__name__,
                      'No arguments retrieved')


# A hpfc iteration, produces a model and compares it to previous
# best model if possible, if last iteration produces graphs related
# to current best model
def hfpc_iteration(iteration, args):

    # Retrieves the data to be modeled upon
    # s_data is sample data, t_data is testing data
    data_s, data_t = preprocessing(args)

    # Retrieves the model from the supervised regression
    models = supervised_regression(data_s, args)

    # Retrieves the best model
    best_model = postprocessing(data_s, data_t, models, args)

    # Activates if best model is not empty
    if not best_model.empty:

        # Activates if current iteration is the last iteration
        if int(args.at[0, 'iterations']) == (iteration + 1):
            # Outputs all relevant graphs about the created model
            # and saves it if new
            data_output(best_model, data_t)
    # Activates if no best model
    else:
        error_message(hfpc_iteration.__name__,
                      'No best model retrieved')


# Retrieves the arguments
# Assumes that a file named arguments.txt is in the directory containing hpfc.py
# arguments.txt should contain lines such as: argument_name:argument_value\n
# No  spaces should be in the file
# Assumes all values given in arguments is correct, does no error checking
# Returns a pandas DataFrame, will be empty if error occurred in retrieval
def argument_retrieval(arg_path):

    # Used to hold the name of each argument
    arg_names = []
    # Used tp hold the value of each argument
    arg_values = []

    # Initial dataframe creation
    args = pd.DataFrame()

    # Activates if the arg_path exists and is a file
    if os.path.exists(arg_path) and os.path.isfile(arg_path):

        # Opens arg_path file in read access
        # Closes file at end
        with open(arg_path, "r") as file:

            # Reads one line of the file to start loop
            contents = file.readline()

            # Reads file one line at a time
            # Reads until end of file
            while contents != '':

                # Splits the contents by the colon character
                # content[0] is the argument name
                # content[1] is the argument value
                contents = contents.split(":")

                # Appends the argument name
                arg_names.append(contents[0])
                # Appends the argument value
                # Strips the end of line character
                arg_values.append(contents[1].strip('\n'))

                # Reads one line of the file and loops
                contents = file.readline()

            # Activates if argValues and argNames have equal lengths, implying that
            # argument retrieval did not fail
            if len(arg_values) == len(arg_names):
                # Iterates through all elements in argsNames and argsValues
                for i in range(len(arg_values)):
                    # Inserts argNames and argValues into args
                    args.insert(i, arg_names[i], [arg_values[i]])

            # Activates if argValues and argNames don't have the same length
            else:
                error_message(argument_retrieval.__name__,
                              'Unequal lengths for argNames and argValues, ' +
                              'meaning error in retrieving arguments')
    # Activates if argument file does not exist
    else:
        # Activates if the item at the end of arg_path does not exist
        if not os.path.exists(arg_path):
            error_message(argument_retrieval.__name__,
                          arg_path + ' does not exist in the directory')
        # Activates if arg_path is not a file
        else:
            error_message(argument_retrieval.__name__,
                          arg_path + ' is not a file')

    # Returns created DataFrame
    # args be empty if error occurred
    return args


# Activates if file is ran as script
if __name__ == "__main__":
    hpfc()
