# Height-Prediction-based-on-Facial-Characteristics
Author: Ellsworth McCullough
## Github Link: https://github.com/ellsworthmmcc/HPFC

## How to use:

Assumes you are on Windows 10, using PyCharmm, and that you have winRAR.

Assumes you have about 1 Gigbyte of space available and are using PyCharm.

### Step 1:
Download from GitHub
![download zip](https://user-images.githubusercontent.com/81348353/207201784-3dcf7ab3-9b0f-4454-bfac-4252dea9e47b.PNG)

### Step 2:
Decompress the downloaded folder using winRar wherever you please.

### Step 3:
Open PyCharm and select the open option.
.![PyCharm Open](https://user-images.githubusercontent.com/81348353/207202408-e78ee0fc-3de7-497e-a2ae-11e0cd7926f1.PNG)

### Step 4:
From there find the decompressed folder, select it with your mouse, and then click ok to open it using PyCharm.
![Opening Folder](https://user-images.githubusercontent.com/81348353/207202664-134e2390-574a-4feb-bf13-1cdd2ce706a0.PNG)

### Step 5:
From here, to get to running the program, double click hpfc.py on the left sidebar to open it.
![hpfc sidebar](https://user-images.githubusercontent.com/81348353/207202806-6109e338-8f39-404e-a122-780ab956a2e4.PNG)

### Step 6:
The next step is to either press run button in the topright of the pycharm window or use the keybinding "Shift+F10" to run the program.
![run pycharm](https://user-images.githubusercontent.com/81348353/207203066-36608691-0415-4958-bda6-9a712847f9f8.PNG)

From here the program should run and go through 1 iteration. This is unlikely to produce a better model, so it should instead most likely display the previous best's graphs with matplotlib windows and the metrics inside the console.

### Step 7:
If you would like to attempt to produce new better models you can change the settings contained with arguments.txt to make that possible. An explanation for how to do that and what each argument is will be below.

## Arguments Explanation:
### Changing Arguments

When changing arguments it is best to not anything unnessecary spaces or deviate from the valid values for each argument described below. Arguments should be kept on their seperate lines and when changing the user should not unnessecary lines or remove arguments entirely.

### Arguments Meaning

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
