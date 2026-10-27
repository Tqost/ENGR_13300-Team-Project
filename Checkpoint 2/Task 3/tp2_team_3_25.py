"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    This program reads an image feature dataset from a CSV file, optionally applies feature scaling,
    and generates 2D or 3D scatter plots based on user-selected features.

Assignment Information:
    Assignment:     tp2 team 3
    Team ID:        LC5 - 25 
    Author:         Leo Cohen, cohen180@purdue.edu
    Date:           10/22/2026

Contributors:
    Baron Chiang, chian163@purdue.edu
    Tanmay Deshmukh, deshmu62@purdue.edu

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

# This program reads an image feature dataset from a CSV file, optionally applies feature scaling,
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Function to scale the features of the dataset using standardization (z-score normalization)
def scale_features(df, features_cols):
    for col in features_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val

# Main function to handle user input, feature scaling, and plotting
def main():
    # Prompt the user for the CSV file path and whether to apply feature scaling
    csv_path = input("Enter the name of CSV image features file:\n")
    apply_scaling_input = input("Do you want to apply feature scaling? (yes/no):\n")

    # Define the feature columns to be used for scaling and plotting
    features_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std",
                 "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"]
    df = pd.read_csv(csv_path)

    # Read the CSV file into a DataFrame
    if apply_scaling_input.strip().lower() == "yes":
        print("Applying feature scaling to the dataset...")
        scale_features(df, features_cols)
        print("Feature scaling applied successfully.")
    else:
        print("Proceeding without feature scaling.")
    # Display the shape and head of the DataFrame to the user
    print()
    print("Feature Dataset Shape:")
    print(df.shape)
    print()
    print("Feature Dataset Head:")
    print(df.head())
    print()
    # Prompt the user for the features to use for the x, y, and optionally z axes in the scatter plot
    x_feat = input("Enter the column to use as the x-axis feature:\n").strip()
    print()
    y_feat = input("Enter the column to use as the y-axis feature:\n").strip()
    print()
    z_input = input("Enter the column to use as the z-axis feature (or press Enter to skip):\n").strip()

    plt.figure(figsize=(8, 6))
    # Separate the DataFrame into two groups based on the ClassId for plotting
    group0 = df[df["ClassId"] == 0]
    group1 = df[df["ClassId"] == 1]

    if z_input.strip() == "":
        plt.scatter(group0[x_feat], group0[y_feat], c='red', alpha=0.6, label='Deer Warning (ClassId 0)')
        plt.scatter(group1[x_feat], group1[y_feat], c='blue', alpha=0.6, label='Left Turn Ahead (ClassId 1)')
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.title(f"2D Feature Space: {x_feat} vs. {y_feat}")
        plt.grid(True)
    else:
        ax = plt.axes(projection='3d')
        ax.scatter(group0[x_feat], group0[y_feat], group0[z_input], c='red', alpha=0.6, label='Deer Warning (ClassId 0)')
        ax.scatter(group1[x_feat], group1[y_feat], group1[z_input], c='blue', alpha=0.6, label='Left Turn Ahead (ClassId 1)')
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_zlabel(z_input)
        ax.set_title(f"3D Feature Space: {x_feat} vs. {y_feat} vs. {z_input}")
        # Set the title and labels for the 3D scatter plot  
    plt.legend()
    # Save the scatter plot to a file and display it to the user
    output_filename = "KNN_feature_space.png"
    plt.savefig(output_filename)
    print()
    print(f"Scatter plot saved to {output_filename}")

    plt.show()



    
if __name__ == "__main__":
    main()
