# Author: Ellsworth McCullough
# File Name: supervised_regression
# Description:
# Creates three models based on given data and returns them


# Library imports
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import BayesianRidge as BayRid
from sklearn.linear_model import ARDRegression as ArdReg
import pandas as pd


# Generates three regression models
# Returns a pandas dataframe containing those models
def supervised_regression(s_data, args):

    # Arrays to hold models
    models = pd.DataFrame()

    # Retrieves linear model
    if eval(args.at[0, 'lin_model']):
        models['Linear Model'] = [lin_model(s_data)]

    # Retrieves bayesian model
    if eval(args.at[0, 'bay_ridge_model']):
        models['Bayesian Ridge Model'] = [baye_ridge_model(s_data, args)]

    # Retrieves ard model
    if eval(args.at[0, 'ard_model']):
        models['ARD Model'] = [ard_model(s_data, args)]

    return models


# Creates a linear regression model based on the given data
# and returns it
def lin_model(s_data):

    # Generating linear regression model
    return LinReg().fit(s_data.values, s_data.heights)


# Creates a bayesian ridge regression model based on the given data
# and returns it, retrieves number of iterations from args
def baye_ridge_model(s_data, args):

    # Generating bayesian ridge linear regression model
    # Max used to prevent errors from having too many iterations
    return BayRid(n_iter=max(int(args.at[0, 'n_iter']), len(s_data.values)))\
        .fit(s_data.values, s_data.heights)


# Creates an automatic relevance determination (ARD) model on the given data
# and returns it, retrieves number of iterations from args
def ard_model(s_data, args):

    # Generating bayesian ard linear regression model
    # Max used to prevent errors from having too many iterations
    return ArdReg(n_iter=max(int(args.at[0, 'n_iter']), len(s_data.values)))\
        .fit(s_data.values, s_data.heights)
