import pandas as pd

# Load the avocado dataset
data = pd.read_csv('avocadoNew.csv', sep=";", header='infer')
# Normalize 'Total Volume' by dividing by 1,000,000 for easier interpretation
data['Total Volume'] /= 1000000

# Define polynomial regression functions
def linear(w0, w1):
    return lambda x: w0 + w1 * x

def quadratic(w0, w1, w2):
    return lambda x: w0 + w1 * x + w2 * (x ** 2)

def cubic(w0, w1, w2, w3):
    return lambda x: w0 + w1 * x + w2 * (x ** 2) + w3 * (x ** 3)

def quartic(w0, w1, w2, w3, w4):
    return lambda x: w0 + w1 * x + w2 * (x ** 2) + w3 * (x ** 3) + w4 * (x ** 4)

import numpy as np

# Define the cost function to calculate the sum of squared differences
def costFunction(predictedValues, actualValues):
    predictedValues = np.array(predictedValues)
    actualValues = np.array(actualValues)
    differencesSquared = (predictedValues - actualValues) ** 2
    return sum(differencesSquared)

# Function to compute weights using the normal equation for linear regression
def normalEquation(X, y):
    XTranspose = np.transpose(X)  # Transpose of X
    XTransposeX = np.dot(XTranspose, X)  # Dot product of X^T and X
    XTransposey = np.dot(XTranspose, y)  # Dot product of X^T and y
    return np.linalg.solve(XTransposeX, XTransposey)  # Solve for weights

# Five-fold cross-validation function
def fiveFoldCrossValidation(X, y):
    costsTraining = []  # Store training costs for each fold
    costsTest = []      # Store testing costs for each fold
    foldSize = len(y) // 5  # Determine the size of each fold

    for i in range(5):
        # Split the data into training and testing sets for the current fold
        testData = data['AveragePrice'].values[foldSize*i:foldSize*(i+1)]
        actualValues = y[foldSize*i:foldSize*(i+1)]
        
        # Combine training data from all other folds
        trainingData = np.concatenate((X[:foldSize*i], X[foldSize*(i+1):]))
        trainingValues = np.concatenate((y[:foldSize*i], y[foldSize*(i+1):]))

        # Calculate weights using the normal equation
        weights = normalEquation(trainingData, trainingValues)

        # Generate predictions for both training and testing data
        predictions = []
        for dataPoint in np.concatenate((data['AveragePrice'].values[:foldSize*i], 
                                           data['AveragePrice'].values[foldSize*(i+1):], 
                                           testData)):
            # Determine the type of polynomial based on the number of weights
            if len(weights) == 2:
                predictions.append(linear(weights[0], weights[1])(dataPoint))
            elif len(weights) == 3:
                predictions.append(quadratic(weights[0], weights[1], weights[2])(dataPoint))
            elif len(weights) == 4:
                predictions.append(cubic(weights[0], weights[1], weights[2], weights[3])(dataPoint))
            elif len(weights) == 5:
                predictions.append(quartic(weights[0], weights[1], weights[2], weights[3], weights[4])(dataPoint))

        # Calculate costs for training and testing data
        costsTraining.append(costFunction(predictions[:-foldSize], trainingValues))
        costsTest.append(costFunction(predictions[-foldSize:], actualValues))

    return costsTraining, costsTest  # Return costs for training and testing

# Prepare the input data
X = data['AveragePrice']
y = data['Total Volume']

# Perform five-fold cross-validation for different polynomial degrees
linearData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X)), y)
quadraticData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2)), y)
cubicData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2, X**3)), y)
quarticData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2, X**3, X**4)), y)

# Prepare data for tabular display
from tabulate import tabulate
table_data = [
    ['Linear'] + linearData[0] + linearData[1] + [np.mean(linearData[0])] + [np.mean(linearData[1])],
    ['Quadratic'] + quadraticData[0] + quadraticData[1] + [np.mean(quadraticData[0])] + [np.mean(quadraticData[1])],
    ['Cubic'] + cubicData[0] + cubicData[1] + [np.mean(cubicData[0])] + [np.mean(cubicData[1])],
    ['Quartic'] + quarticData[0] + quarticData[1] + [np.mean(quarticData[0])] + [np.mean(quarticData[1])],
]
# Define table headers for better clarity
headers = ['', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean Training', 'Mean Testing']

# Print the results in a tabulated format
print(tabulate(table_data, headers=headers))

# Calculate the weights for the quadratic model for prediction
quadraticWeights = normalEquation(np.column_stack((np.ones(X.shape[0]), X, X**2)), y)
quadraticModel = quadratic(quadraticWeights[0], quadraticWeights[1], quadraticWeights[2])

# Function to predict total volume sold based on price
def predict(price):
    return round(quadraticModel(price), 2)

# Example prediction for avocados sold at $2.00
print(f'Avocados sold at $2.00 in million lbs: {predict(2)}')
