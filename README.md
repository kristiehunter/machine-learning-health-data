# Machine Learning Health Data

This project can be used on any data set, but I used it on data collected from a Garmin smart watch, Oura ring, some manually collected data, and Lose It nutrition tracking.

The goal of this project is to optimize health habits by finding the features that have the most impact on a specific target and building/training a model on those feature points.  I have a passion in exploring personalized healthcare and am a strong believer in the idea that there is no one-size-fits-all approach to health living.

This program is an exploratory algorithm, so it is not designed to be quick or efficient.  It loops through different types of null data replacement methods, scaling methods, number of features and different classification models to determine what best suits the data provided.

This is a classification algorithm that uses Logistic Regression, Naive Bayes, Decision Tree, Support Vector Machine, Neural Network, and K-Nearest Neighbours to predict a specified target value.

## How to Run

### 1. Upload input data
In the `/data` folder, upload a csv called "masterdata.csv" with headers and with numerical values for all features.

### 2. Update the configuration files

The configuration file contains 3 parameters to determine what the data target is and what the classification values should be.  The health data used as an input is continuouse data, and the algorithm is trying to determine an increase or decrease in the value based on other features, therefore the following variables need to be included.

`TARGET` : String value for the column of the target value in the masterdata.csv file.\
`TARGET_MIN` : Numerical value for the minimum value in the TARGET column.  This enables the algorithm to create buckets for classification.\
`TARGET_MAX` : Numerical value for the maximum value in the TARGET column.  This enables the algorithm to create buckets for classification.

### 3. Running the program

The program is executable by running `main.py` in a code editor, or by using `python main.py` on the command line.

## Outputs to Expect

The algorithm produces prediction files, model parameter selection files, and some Exploratory Data Analysis images (feature in progress).

The prediction files will be in the `/data` folder with the names `TARGET_training_predictions.csv` and `TARGET_testing_predictions.csv` where TARGET is the target variable provided in the configuration file.  The actual target values will be included in the prediction files in order to run further analysis on the accuracy of the model and potential outliers in the data set.

The model parameter selection file will be saves in the `/data` folder with the name `TARGET_best_estimators.json` and it will contain the model parameters determined to the best way to predict the target data based on the procedure outlined below in Section 4.  Once this file exists, there is a supplemental file called `best_model.py` that can be run at any time using the `masterdata.csv` and the best estimator JSON file.