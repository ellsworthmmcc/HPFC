# Author: Ellsworth McCullough
# File Name: postprocessing
# Description:
# Finds the best model based on metrics given through arguments,
# compares best model to previous best model if possible.
# Returns the best current model and associated metrics

# hpfc imports
from hpfc_standard import error_message
from hpfc_standard import generate_file_path

# Library imports
from sklearn import metrics
import pandas as pd
import pickle


# Compares given models and previous best model
# Returns best model and related metrics
def postprocessing(data_s, data_t, models, args):

    # Activates if there are models
    if not models.empty:

        # Retrieves best model based on ones given
        best_model = model_metrics(data_s, data_t, models, args)

        # Retrieves the best model between current model and previous
        # best model if possible
        return compare_to_prev(best_model, args)
    # Activates if there are no models to process
    else:
        error_message(postprocessing.__name__,
                      'No models to process')
        # Returns False when no models to evaluate
        return pd.DataFrame()


# Compares given models on a set number of metrics,
# Returns best model in pandas dataframe object
def model_metrics(data_s, data_t, models, args):

    # Retrieves actual height results for sample and testing data
    actual_s = data_s.heights
    actual_t = data_t.heights

    # Sets all best metrics to values they cannot be, representing unfilled
    best_model = 0
    best_mae = -1
    best_r2 = 2
    best_mse = -1
    best_rmse = -1
    best_ofs = -1

    print('\nModel Metrics\n')

    # Goes through each model and calculates relevant metrics
    for i, model in enumerate(models):
        print(model)

        # Retrieves prediction values for the model
        prediction_s = models.at[0, model].predict(data_s.values)
        prediction_t = models.at[0, model].predict(data_t.values)

        # Calculates the mean_absolute_error
        # Closer to 0 the better
        mae = metrics.mean_absolute_error(actual_t, prediction_t)

        # Calculates r2 score
        # Closer to 1 the better
        r2 = metrics.r2_score(actual_t, prediction_t)

        # Calculates mean squared error
        # Closer to 0 the better
        mse = metrics.mean_squared_error(actual_t, prediction_t, squared=True)

        # Calculates root mean squared error
        # closer to 0 the better
        rmse = metrics.mean_squared_error(actual_t, prediction_t, squared=False)
        rmse_sam = metrics.mean_squared_error(actual_s, prediction_s, squared=False)

        # Calculates the overfitting score
        # closer to 0 the better
        ofs = abs(rmse - rmse_sam)

        # Prints relevant metrics for the model
        print(model, ' mae score: ', mae)
        print(model, ' r2 score:  ', r2)
        print(model, ' mse score: ', mse)
        print(model, ' rmse score:', rmse)
        print(model, ' ofs score: ', ofs)
        print()

        # Activates if there is not best model yet
        if i == 0:
            best_model = model
            best_mae = mae
            best_r2 = r2
            best_mse = mse
            best_rmse = rmse
            best_ofs = ofs
        # Activates if there is an existing best model
        else:
            # Determines best model between current model and previous
            # best model
            best_model, best_mae, best_r2,\
            best_mse, best_rmse, best_ofs = \
                best_model_evaluator(best_model, best_mae, best_r2,
                                     best_mse, best_rmse, best_ofs,
                                     model, mae, r2,
                                     mse, rmse, ofs, args)

    # Error in performing operation in dataframe assignment operation
    # so variable created to be used instead
    best_model_actual = models.at[0, best_model]

    # Creates dataframe of best model and corresponding metrics
    best_model_df = pd.DataFrame()
    best_model_df['Title'] = [best_model]
    best_model_df['Model'] = [best_model_actual]
    best_model_df['New'] = [True]
    best_model_df['MAE'] = [best_mae]
    best_model_df['R2'] = [best_r2]
    best_model_df['MSE'] = [best_mse]
    best_model_df['RMSE'] = [best_rmse]
    best_model_df['OFS'] = [best_ofs]

    # Generates and prints out data related to naive_model for comparison
    naive_model(actual_s, actual_t)

    # Prints name of best model
    print('Best Model: ', best_model)

    return best_model_df


# Compares current best model to previous best model if possible
# Returns the better of the two, if there is no previous returns current
# Returns a pandas dataframe object
def compare_to_prev(best_model, args):

    # Retrieves previous model
    prev_best_model, prev_exists = load_prev_best()

    # Activates if previous model exists
    if prev_exists:
        # Retrieves metrics for best model
        best_mae = best_model.at[0, 'MAE']
        best_r2 = best_model.at[0, 'R2']
        best_mse = best_model.at[0, 'MSE']
        best_rmse = best_model.at[0, 'RMSE']
        best_ofs = best_model.at[0, 'OFS']

        # Retrieves metrics for previous best model
        mae = prev_best_model.at[0, 'MAE']
        r2 = prev_best_model.at[0, 'R2']
        mse = prev_best_model.at[0, 'MSE']
        rmse = prev_best_model.at[0, 'RMSE']
        ofs = prev_best_model.at[0, 'OFS']

        # Retrieves best model between current and previous
        best_model, best_mae, best_r2, best_mse, best_rmse, best_ofs = \
            best_model_evaluator(best_model, best_mae, best_r2,
                                 best_mse, best_rmse, best_ofs,
                                 prev_best_model, mae, r2, mse, rmse, ofs, args)

        # Prints previous model if previous model retrieved
        if best_model.at[0, 'Model'] == prev_best_model.at[0, 'Model']:
            print('\n Using previous best model\n')
            print('Model: ', best_model.at[0, 'Title'])
            print('MAE:  ', best_model.at[0, 'MAE'])
            print('R2:   ', best_model.at[0, 'R2'])
            print('MSE:  ', best_model.at[0, 'MSE'])
            print('RMSE: ', best_model.at[0, 'RMSE'])
            print('OFS:  ', best_model.at[0, 'OFS'])

    return best_model


# Retrieves the previous best model
# Assumes the files in data/models has not been altered
# model represented as dataframe object
# Returns model and boolean equal to true if model retrieved,
# Returns 0 and boolean equal to false if model not retrieved
def load_prev_best():

    # Retrieves file path to load
    path = generate_file_path(-1, True)

    # Activates if path exists
    if path != 0:
        # Opens file at path
        with path.open('rb') as fp:

            # Retrieves previous model
            prev_model = pickle.load(fp)
            prev_model.at[0, 'New'] = False
            return prev_model, True
    # Activates if path does not exist
    else:
        return 0, False


# Determine which model is better,
# returns best model and related metrics
def best_model_evaluator(best_model, best_mae, best_r2,
                         best_mse, best_rmse, best_ofs,
                         model, mae, r2, mse, rmse, ofs, args):

    # Holds the points of the evaluated model
    metrics_better_weighted = 0

    # Retrieves the weights for each metric
    mae_weight = float(args.at[0, 'mae'])
    r2_weight = float(args.at[0, 'r2'])
    mse_weight = float(args.at[0, 'mse'])
    rmse_weight = float(args.at[0, 'rmse'])
    ofs_weight = float(args.at[0, 'ofs'])

    # Total possible points
    total_possible_better = mae_weight + r2_weight\
                            + mse_weight + rmse_weight\
                            + ofs_weight

    # Checks which mae is lesser
    if mae < best_mae:
        metrics_better_weighted += mae_weight

    # Checks which r2 is closer to 1
    if abs(r2 - 1) < abs(best_r2 - 1):
        metrics_better_weighted += r2_weight

    # Checks which mse is lesser
    if mse < best_mse:
        metrics_better_weighted += mse_weight

    # Checks which rmse is lesser
    if rmse < best_rmse:
        metrics_better_weighted += rmse_weight

    # Checks which ofs is lesser
    if ofs < best_ofs:
        metrics_better_weighted += ofs_weight

    # Activates if evaluated metrics are better than previous model best metrics
    # Attributed points greater than points not attributed
    if metrics_better_weighted > (
            total_possible_better - metrics_better_weighted):
        return model, mae, r2, mse, rmse, ofs
    # If best metrics better than evaluated metrics
    else:
        return best_model, best_mae, best_r2, best_mse, best_rmse, best_ofs


# Generates a naive prediction based on worldwide average human height
# Prints out information related to Naive model and returns it in dataframe object
def naive_model(actual_s, actual_t):

    # Generating naive prediction method to compare to
    average_height = 165
    human_avg_prediction_t = []
    human_avg_prediction_s = []

    # Creates naive model prediction for testing data
    for i in range(len(actual_t)):
        human_avg_prediction_t.append(average_height)

    # Creates naive model prediction for sample data
    for i in range(len(actual_s)):
        human_avg_prediction_s.append(average_height)

    # Retrieves naive model metrics
    hum_mae = metrics.mean_absolute_error(actual_t, human_avg_prediction_t)
    hum_r2 = metrics.r2_score(actual_t, human_avg_prediction_t)
    hum_mse = metrics.mean_squared_error(actual_t, human_avg_prediction_t, squared=True)
    hum_rmse = metrics.mean_squared_error(actual_t, human_avg_prediction_t, squared=False)
    hum_rmse_sam = metrics.mean_squared_error(actual_s, human_avg_prediction_s, squared=False)
    hum_ofs = abs(hum_rmse - hum_rmse_sam)

    # Prints naive model metrics
    print('Naive model')
    print('Naive model mae score:  ', hum_mae)
    print('Naive model r2 score:   ', hum_r2)
    print('Naive model mse score:  ', hum_mse)
    print('Naive model rmse score: ', hum_rmse)
    print('Naive model ofs score:  ', hum_ofs)
    print()

    # Generates naive dataframe
    naive = pd.DataFrame()
    naive['Title'] = ['Naive']
    naive['MAE'] = [hum_mae]
    naive['R2'] = [hum_r2]
    naive['MSE'] = [hum_mse]
    naive['RMSE'] = [hum_rmse]
    naive['OFS'] = [hum_ofs]

    return naive
