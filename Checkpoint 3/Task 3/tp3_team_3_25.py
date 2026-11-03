"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Loads the extracted image feature dataset, scales and optionally
    shuffles it, and splits it into training, validation, and test sets.
    Evaluates a K-nearest neighbors classifier across several k values,
    plots the training and validation accuracy and error rates, and
    selects the best k by validation accuracy. Trains a logistic
    regression model with gradient descent, plots the training and
    validation loss, and compares the accuracy of both models on the
    validation and test sets.

Assignment Information:
    Assignment:     9.1.2 TP3 Team 3
    Team ID:        LC5 - 25
    Author:         Leo Cohen, cohen180@purdue.edu
    Date:           10/29/2026

Contributors:
    Tanmay Deshmukh, deshmu62@purdue.edu
    Baron Chiang, chian163@purdue.edu
    
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
from collections import Counter
import matplotlib.pyplot as plt


def evaluate_knn_model(k_values, X_train, y_train, X_val, y_val):
    #Evaluates the KNN classifier across multiple k values using training and validation sets.
    metrics = {
        "acc": {"train_acc": [], "val_acc": []},
        "error": {"train_error": [], "val_error": []}
    }
    
    for k in k_values:
        # Predict labels on the training and validation sets
        train_preds = predict_labels_knn(X_train, X_train, y_train, k)
        val_preds = predict_labels_knn(X_val, X_train, y_train, k)
        
        # Calculate metrics for training set
        train_acc, train_error = calculate_metrics(y_train, train_preds)
        
        # Calculate metrics for validation set
        val_acc, val_error = calculate_metrics(y_val, val_preds)
        
        # Append results to the appropriate lists in metrics
        metrics["acc"]["train_acc"].append(train_acc)
        metrics["acc"]["val_acc"].append(val_acc)
        metrics["error"]["train_error"].append(train_error)
        metrics["error"]["val_error"].append(val_error)
        
    return metrics



def knn_single_prediction(new_example, X_train, y_train, k):
    # Calculate the distances from the training set to the new image.
    distances = np.linalg.norm(X_train - new_example, axis=1)

    # Place all of the distances and what the image is in a list of tuples.
    dist_label_tuples = list(zip(distances, y_train))

    # Sort by distance in ascending order.
    dist_label_tuples.sort(key=lambda x: x[0])

    # Get all of the nearest labels within k.
    k_nearest_labels = [label for _, label in dist_label_tuples[:k]]

    # Determine the most common label from a majority 
    # vote in the list of k nearest labels.
    most_common = Counter(k_nearest_labels).most_common(1)
    predicted_label = np.int64(most_common[0][0])

    return predicted_label

def predict_labels_knn(X_new, X_train, y_train, k):
    # Get dimension of the X_new array
    n_new = X_new.shape[0]
    predicted_labels = np.zeros(n_new, dtype=int)

    # Go through each value within X_new and use the knn_single_prediction function. 
    for i in range(n_new):
        predicted_labels[i] = knn_single_prediction(X_new[i], X_train, y_train, k)

    return predicted_labels

def calculate_metrics(predicted_labels, true_labels):
    accuracy = float(np.mean(predicted_labels == true_labels))

    error = 1.0 - accuracy

    return accuracy, error

def plot_knn_performance(metrics, k_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy vs. k
    ax1.plot(k_values, metrics["acc"]["train_acc"], marker='o', label='Training Accuracy')
    ax1.plot(k_values, metrics["acc"]["val_acc"], marker='o', label='Validation Accuracy')
    ax1.set_title('Accuracy vs. k')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Error Rate vs. k
    ax2.plot(k_values, metrics["error"]["train_error"], marker='o', label='Training Error')
    ax2.plot(k_values, metrics["error"]["val_error"], marker='o', label='Validation Error')
    ax2.set_title('Error Rate vs. k')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Error Rate')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('train_val.png')
    plt.close()

def select_best_k(k_values, metrics):
    # Select the best k based on validation accuracy
    val_acc = metrics["acc"]["val_acc"]
    best_k_index = np.argmax(val_acc)
    best_k = k_values[best_k_index]
    return best_k

def evaluate_knn_test_set(X_train, y_train, X_test, y_test, best_k):
    # Evaluate the KNN model on the test set using the best k
    test_preds = predict_labels_knn(X_test, X_train, y_train, best_k)
    test_accuracy, test_error = calculate_metrics(test_preds, y_test)
    return test_accuracy, test_error 

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

def evaluate_logistic_regression(X, y, w, b):
    # Essentially calculating the accuracy and error of model when inputed with a table and weights and biases.
    predicted_labels = predict_labels(X, w, b, 0.5)
    accuracy, error = calculate_metrics(predicted_labels, y)

    return accuracy, error

def evaluate_logistic_model(X_train, y_train, X_val, y_val, X_test, y_test, w, b):
    metrics = {
        "train": {"accuracy": 0.0, "error": 0.0},
        "val": {"accuracy": 0.0, "error": 0.0},
        "test": {"accuracy": 0.0, "error": 0.0},
    }
    for name, X, y in [("train", X_train, y_train),
                       ("val", X_val, y_val),
                       ("test", X_test, y_test)]:
        acc, err = evaluate_logistic_regression(X, y, w, b)
        metrics[name]["accuracy"] = acc
        metrics[name]["error"] = err
    return metrics

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

        # Printing out where model is in training.
        if i % 100 == 0:
            print(f"Iteration {i}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    return w, b, loss_history, val_loss_history

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

def plot_logistic_training(loss_history, val_loss_history):
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Loss vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.close()

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
    shuffle = input("Shuffle the dataset? (yes/no): ").strip().lower() == "yes"
    seed = int(input("Enter a seed for loading the dataset: "))

    # Use the same column names you used in your 9.1.1 main (see the CSV header)
    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std",
                    "value_mean", "value_std", "number_lines", "has_circle",
                    "has_triangle"]
    label_col = "ClassId"

    X, y = load_dataset(file_path, feature_cols, label_col, shuffle, seed)
    np.random.seed(seed)  # reproducible weight initialization
    (X_train, X_val, X_test,
        y_train, y_val, y_test) = train_val_test_split(X, y, 0.8, 0.1, 0.1)

    print("\nData loaded and split into\n")
    print(f"Training set:\n  size: {len(y_train)}")
    print(f"Validation set:\n  size: {len(y_val)}")
    print(f"Test set:\n  size: {len(y_test)}")

    # ----- KNN -----
    k_values = [1, 3, 5, 7, 9, 15, 31, 53]
    knn_metrics = evaluate_knn_model(k_values, X_train, y_train, X_val, y_val)
    plot_knn_performance(knn_metrics, k_values)

    print("\nKNN Validation Curve Summary")
    for i, k in enumerate(k_values):
        print(f"k = {k}: train accuracy = {knn_metrics['acc']['train_acc'][i]:.4f}, "
              f"validation accuracy = {knn_metrics['acc']['val_acc'][i]:.4f}")

    best_k = select_best_k(k_values, knn_metrics)
    print(f"\nBased on the plots, the best k appears to be: {best_k}")

    knn_test_acc, knn_test_err = evaluate_knn_test_set(X_train, y_train, X_test, y_test, best_k)
    print("\n--- Final KNN Model Performance ---")
    print(f"Test Set Accuracy: {knn_test_acc:.4f}")
    print(f"Test Set Error Rate: {knn_test_err:.4f}")

    # ----- Logistic regression -----
    learning_rate = 0.01
    num_iterations = 5000
    print(f"\nTraining logistic regression model with learning rate = {learning_rate} "
          f"and num_iterations = {num_iterations}...\n")

    w, b, loss_history, val_loss_history = train_logistic_regression(
        X_train, y_train, X_val, y_val, learning_rate, num_iterations)

    lr = evaluate_logistic_model(X_train, y_train, X_val, y_val, X_test, y_test, w, b)

    print("\nFinal Logistic Regression Evaluation:")
    print("-" * 50)
    for key, label in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        acc, err = lr[key]["accuracy"], lr[key]["error"]
        print(f"{label} Set - Accuracy: {acc:.4f} ({acc * 100:.2f}%), "
              f"Error Rate: {err:.4f} ({err * 100:.2f}%)")

    plot_logistic_training(loss_history, val_loss_history)

    best_val_acc = knn_metrics["acc"]["val_acc"][k_values.index(best_k)]
    print("\nModel Summary")
    print(f"Best KNN validation accuracy: {best_val_acc:.4f}")
    print(f"Logistic regression validation accuracy: {lr['val']['accuracy']:.4f}")
    print(f"KNN test accuracy: {knn_test_acc:.4f}")
    print(f"Logistic regression test accuracy: {lr['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()