"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Train a linear regression model from scratch

Assignment Information:
    Assignment:     9.1.2 TP3 Team 2
    Team ID:        LC5 - 25
    Author:         Baron Chiang, chian163@purdue.edu
    Date:           10/29/2026

Contributors:
    Tanmay Deshmukh, deshmu62@purdue.edu
    Leo Cohen, cohen180@purdue.edu

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import numpy as np
import pandas as pd


def calculate_sigmoid(z):
    # Clip the input so that we don't get crazy sigmoid function outputs.
    z = np.clip(z, -10, 10)

    # Calculate sigmoid
    result = 1 / (1 + np.exp(-z))

    return result


def predict_proba(X, w, b):
    # Equation 9.7
    z = np.dot(X, w) + b

    probabilities = calculate_sigmoid(z)

    return probabilities


def predict_labels(X, w, b, threshold=0.5):
    # Get probability from X, w, and b values. 
    probability = predict_proba(X, w, b)

    # Find where all of the probabilities are above the threshold and place as 1, otherwise place as 0.
    predicted_labels = np.where(probability >= threshold, 1, 0)

    return predicted_labels


def compute_loss_and_grads(X, y, w, b):
    # Calculate the probability and place in y_hat.
    y_hat = predict_proba(X, w, b)
    loss = 0

    # Making sure that we do not get log(0) errors.
    y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)

    m = X.shape[0]

    # Equation 9.5
    loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))

    # Equations 9.8 and 9.9
    dw = 1 / m * np.dot(X.T, (y_hat - y))
    db = 1 / m * np.sum(y_hat - y)

    return loss, dw, db


def train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate, num_iterations):
    # Getting randomized initial weights so we can break symmetry and initializing other variables needed.
    n_features = X_train.shape[1]
    w = np.random.randn(n_features) * 0.01
    b = 0.0

    # Initializing the best of the weights, biases, and losses so we can find what is better. 
    w_best = w.copy()
    b_best = b
    best_val_loss = float('inf')

    loss_history = []
    val_loss_history = []

    for i in range(num_iterations):
        loss, dw, db = compute_loss_and_grads(X_train, y_train, w, b)
        loss_history.append(loss)

        # Changing weights and biases by the change times learning_rate.
        w -= learning_rate * dw
        b -= learning_rate * db

        val_loss, _, _ = compute_loss_and_grads(X_val, y_val, w, b)
        val_loss_history.append(val_loss)

        # Checking if the loss value is less than best, then make best equal to loss and get weight and bias from that iteration. 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            w_best = w.copy()
            b_best = b

        # Printing out where model is in training.
        if i % 100 == 0:
            print(f"Iteration {i}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    return w_best, b_best, loss_history, val_loss_history

# From checkpoint 3 task 1.
def calculate_metrics(predicted_labels, true_labels):
    accuracy = float(np.mean(predicted_labels == true_labels))

    error = 1.0 - accuracy

    return accuracy, error


def evaluate_logistic_regression(X, y, w, b):
    # Essentially calculating the accuracy and error of model when inputed with a table and weights and biases.
    predicted_labels = predict_labels(X, w, b, 0.5)
    accuracy, error = calculate_metrics(predicted_labels, y)

    return accuracy, error


# Function from Checkpoint 2 Task 3.
def scale_features(df, features_cols):
    for col in features_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val


# Function to load dataset from csv file.
def load_dataset(file_path, feature_cols, label_col, shuffle, seed):
    df = pd.read_csv(file_path)

    scale_features(df, feature_cols)

    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()

    if shuffle:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(y))
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    return X, y


# Splitting the dataset into 3 sections for training, validation, and testing.
def train_val_test_split(X, y, train_ratio, val_ratio, test_ratio):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios do not equal 1")

    # Get number of rows from the X variable.
    num_rows = X.shape[0]

    # Get indexes for where training and validation sets are going to end.
    end_train = int(num_rows * train_ratio)
    end_val = end_train + int(num_rows * val_ratio)

    # Initialize 6 different output variables and place all of the values from test set into it.
    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    y_train = y[:end_train].astype(int)
    y_val = y[end_train:end_val].astype(int)
    y_test = y[end_val:].astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    file_path = input("Enter the path to the feature dataset: ")
    shuffle = input("Shuffle the dataset? (yes/no): ")
    seed = int(input("Enter a seed for loading the dataset: "))

    if shuffle == "yes":
        shuffle = True
    else:
        shuffle = False
    np.random.seed(seed)

    X, y = load_dataset(file_path, ["hue_mean", "hue_std", 
                                        "saturation_mean", "saturation_std", 
                                        "value_mean", "value_std", 
                                        "number_lines", "has_circle", "has_triangle"], "ClassId", shuffle, seed)

    (X_train, X_val, X_test,
        y_train, y_val, y_test) = train_val_test_split(X, y, 0.8, 0.1, 0.1)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    print("Data loaded and split into\n")
    print("Training set:")
    print(f"  size: {X_train.shape[0]}")
    print("Validation set:")
    print(f"  size: {X_val.shape[0]}")
    print("Test set:")
    print(f"  size: {X_test.shape[0]}")

    learning_rate = 0.01
    num_iterations = 5000

    print(f"\nTraining logistic regression model with learning rate = {learning_rate} and num_iterations = {num_iterations}...\n")

    w_best, b_best, loss_history, val_loss_history = train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate, num_iterations)
    print("\nFinal Model Evaluation with the best parameters:")
    print("--------------------------------------------------")
    accuracy, error = evaluate_logistic_regression(X_train, y_train, w_best, b_best)
    print(f"Training Set - Accuracy: {accuracy * 100:.2f}%, Error Rate: {error * 100:.2f}%")
    accuracy, error = evaluate_logistic_regression(X_val, y_val, w_best, b_best)
    print(f"Validation Set - Accuracy: {accuracy * 100:.2f}%, Error Rate: {error * 100:.2f}%")
    accuracy, error = evaluate_logistic_regression(X_test, y_test, w_best, b_best)
    print(f"Test Set - Accuracy: {accuracy * 100:.2f}%, Error Rate: {error * 100:.2f}%")

    print("\nFinal Parameters:")
    print(f"Weights shape: {w_best.shape}")
    print(f"Bias: {b_best:.6f}")
    print(f"Final training loss: {loss_history[-1]:.6f}")
    print(f"\nFeature Importance (absolute weight values):")
    for i, w in enumerate(w_best):
        print(f"Feature {i}: {abs(w):.6f}")

if __name__ == "__main__":
    main()
