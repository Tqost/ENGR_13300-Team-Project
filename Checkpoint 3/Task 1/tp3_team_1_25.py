"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    This program creates a KNN machine learning algorithm.

Assignment Information:
    Assignment:     TP3 Team 1
    Team ID:        LC5 - 25 
    Author:         Baron Chiang, chian163@purdue.edu
    Date:           10/28/2026

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

import pandas as pd
import numpy as np
from collections import Counter

def scale_features(df, features_cols):
    for col in features_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val


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

def train_val_test_split(X, y, train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio != 0:
        raise ValueError("Ratios do not equal 1")
    
    num_rows = X.shape[0]

    end_train = int(num_rows * train_ratio)
    end_val = end_train + int(num_rows * val_ratio)

    X_train = X[:end_train].astype(int)
    X_val = X[end_train:end_val].astype(int)
    X_test = X[end_val:].astype(int)

    y_train = y[:end_train].astype(int)
    y_val = y[end_train:end_val].astype(int)
    y_test = y[end_val].astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test


def knn_single_prediction(new_example, X_train, y_train, k):
    distances = np.linalg.norm(X_train - new_example, axis=1)

    dist_label_tuples = list(zip(distances, y_train))

    dist_label_tuples.sort(key=lambda x: x[0])

    k_nearest_labels = [label for _, label in dist_label_tuples[:k]]

    most_common = Counter(k_nearest_labels).most_common(1)
    predicted_label = int(most_common[0][0])

    return predicted_label


def predict_labels_knn(X_new, X_train, y_train, k):
    n_new = X_new.shape[0]
    predicted_labels = np.zeros(n_new, dtype=int)

    for i in range(n_new):
        predicted_labels[i] = knn_single_prediction(X_new[i], X_train, y_train, k)

    return predicted_labels


def calculate_metrics(predicted_labels, true_labels):
    accuracy = float(np.mean(predicted_labels == true_labels))

    error = 1.0 - accuracy

    return accuracy, error


def main():
    file_path = input("Enter the path to the feature dataset: ")
    shuffle = input("Shuffle the dataset? (yes/no): ")
    if shuffle == "yes":
        seed = input("Enter a seed for loading the dataset: ")
        shuffle = True
    else:
        shuffle = False
    k = int(input("Enter the value of k to use for KNN: "))

    X, y = load_dataset(file_path, ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"], "ClassId", shuffle, seed)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.8, 0.1, 0.1)
    print("Data loaded and split into\n")
    print("Training set:")
    print(f"  size: {X_train.shape[0]}")
    print("Validation set:")
    print(f"  size: {X_val.shape[0]}")
    print("Test set:")
    print(f"  size: {X_test.shape[0]}")
    
    print(f"Evaluating KNN model on Training set with k = {k}...\n")
    true_labels = y_train
    predicted_labels = predict_labels_knn(X_train, X_train, y_train, k)
    accuracy, error = calculate_metrics(predicted_labels, true_labels)
    print("--- Model Performance on Training Set ---")
    print(f"Training Set Accuracy: {accuracy:.4f}")
    print(f"Training Set Error Rate: {error:.4f}\n")

    print(f"Evaluating KNN model on validation set with k = {k}...\n")
    true_labels = y_val
    predicted_labels = predict_labels_knn(X_val, X_train, y_train, k)
    accuracy, error = calculate_metrics(predicted_labels, true_labels)
    print("--- Model Performance on Validation Set ---")
    print(f"Training Set Accuracy: {accuracy:.4f}")
    print(f"Training Set Error Rate: {error:.4f}\n")

    print(f"Evaluating KNN model on test set with k = {k}...\n")
    true_labels = y_test
    predicted_labels = predict_labels_knn(X_test, X_train, y_train, k)
    accuracy, error = calculate_metrics(predicted_labels, true_labels)
    print("--- Model Performance on Test Set ---")
    print(f"Training Set Accuracy: {accuracy:.4f}")
    print(f"Training Set Error Rate: {error:.4f}\n")


if __name__ == "__main__":
    main()
