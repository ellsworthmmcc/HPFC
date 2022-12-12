# Height-Prediction-based-on-Facial-Characteristics
Author: Ellsworth McCullough
## Github Link: https://github.com/ellsworthmmcc/HPFC

## How to use:

## Argumets Explanation:

example; valid values; description

iterations; 0+; amount of hpfc iterations, essentially amount of models to produce

ordered; True/False; whether the dataset should be ordered before modeling and graphing, ordering from lowest to highest heights

directory; file path; the directory containing the pictures

pc; 1+; the amount of principle components to use during feature reduction

fla; 468; the amount of features produced by MediaPipe, recommended not to change this value

ran; 1+; the value that determines how much of the dataset becomes sample data and how much becomes testing data
a good estimate is about 1/ran becomes testing data, and the rest becomes sample data

lin_model; True/False; whether to generate a Linear Regression Model

bay_ridge_model; True/False; whether to generate a Bayesian Ridge Model

ard_model; True/False; whether to generate an Automatic Relevance Determination Model

n_iter; 1+; amount of iterations to be performed at a mimimum in the generation of the bayesian models

mae; 0.0+; mean absolute error, weight used in model comparison for this metric

r2; 0.0+; r2 score, weight used in model comparison for this metric

mse; 0.0+; mean squared error, weight used in model comparison for this metric

rmse; 0.0+; root mean squared error, weight used in model comparison for this metric

ofs; 0.0+; overfitting score, weight used in model comparison for this metric
