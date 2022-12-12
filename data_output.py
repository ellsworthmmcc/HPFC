# Author: Ellsworth McCullough
# File Name: data_output
# Description:
# Saves current model if it is new, prints out
# graphs related to model

# hpfc imports
from hpfc_standard import generate_file_path

# Library Imports
import matplotlib.pyplot as plt
import pickle


# Saves current model if new, prints out graphs
# related to current model and given data
def data_output(model, data):

    # Saves the current model
    save_model(model)

    # Graphs prediction and actual
    prediction_graph(model, data)
    residual_graph(model, data)

    plt.show()


# Saves the current model, generates model name based
# on amount of current models existing
def save_model(model):

    # Activates if current model is new
    if model.at[0, 'New']:

        # Generates a file path to the new model
        path = generate_file_path()

        # Creates and dumps to new file at
        # designated filepath
        with path.open('wb+') as fp:
            pickle.dump(model, fp)


# Generates a graph of model prediction and data values compared
def prediction_graph(model, data):

    # Retrieves model
    model_actual = model.at[0, 'Model']

    # Activates if model is graphable given the current data
    if model_actual.n_features_in_ == len(data.values[0]):

        # Retrieves prediction and actual
        prediction = model_actual.predict(data.values)
        actual = data.heights

        # Array to hold x values
        x_arr = []

        # Generate x values based on length of prediction
        for i in range(len(prediction)):
            x_arr.append(i + 1)

        # Sets settings of current plot
        plt.figure(1)
        plt.style.use('seaborn')
        plt.title(model.at[0, 'Title'])
        plt.ylabel('Height (in cm)')
        plt.xlabel('Number of Samples')

        # Plots prediction
        plt.plot(x_arr, prediction, label='prediction', color='blue', alpha=0.7)
        plt.scatter(x_arr, prediction, color='blue', alpha=0.9, s=60)

        # Plots actual
        plt.plot(x_arr, actual, label='actual', color='orange', alpha=0.7)
        plt.scatter(x_arr, actual, color='orange', alpha=0.7, s=60)

        # Generates generic plot settings
        graph_end()
    # Activates if model is not graphable
    else:
        print('\nNumber of model features, ', model_actual.n_features_in_,
              ' does not match amount of features of current dataset, ', len(data.values[0]), '\n')


def residual_graph(model, data):

    # Retrieves model
    model_actual = model.at[0, 'Model']

    # Activates if model is graphable given the current data
    if model_actual.n_features_in_ == len(data.values[0]):

        # Retrieves prediction and actual
        prediction = model_actual.predict(data.values)
        actual = data.heights

        # Generates residuals
        residuals = prediction - actual

        # Array to hold x values
        x_arr = []

        # Generate x values based on length of prediction
        for i in range(len(prediction)):
            x_arr.append(i + 1)

        # Sets settings of current plot
        plt.figure(2)
        plt.style.use('seaborn')
        plt.title(model.at[0, 'Title'] + ' Residuals compared to Predicted')
        plt.ylabel('Residuals')
        plt.xlabel('Predicted Values')

        # Plots residuals
        plt.scatter(prediction, residuals, label='residuals', color='blue', alpha=0.9, s=40)

        # Generates generic plot settings
        graph_end()
    # Activates if model is not graphable
    else:
        print('\nNumber of model features, ', model_actual.n_features_in_,
              ' does not match amount of features of current dataset, ', len(data.values[0]), '\n')
        return


# Function used to avoid code redundancies, sets settings that are generic to every plot/graph
def graph_end():

    plt.legend()
    plt.grid(color='black', linestyle='-', linewidth=0.2, alpha=0.2)
