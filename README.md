# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:38:11 2024

@author: ASUS
"""

import numpy as np
import pandas as pd

# Get number of alternatives and criteria from user
E = int(input("Enter the number of alternatives: "))  # Number of alternatives
V = int(input("Enter the number of criteria: "))  # Number of criteria

# Subjective weight matrix (user-provided)
subv = np.array(
    list(map(float, input(f"Enter the {V} subjective weights separated by space: ").split()))
)
assert subv.shape[0] == V, "The number of subjective weights must match the number of criteria."

# Interaction matrix between criteria (user-provided)
print(f"Enter the {V}x{V} interaction matrix row by row (separated by spaces):")
VVT = np.array([list(map(float, input().split())) for _ in range(V)])
assert VVT.shape == (V, V), "The interaction matrix must be square with dimensions (VxV)."

# Decision matrix (user-provided)
print(f"Enter the {E}x{V} decision matrix row by row (separated by spaces):")
My_CellEV = np.array([list(map(float, input().split())) for _ in range(E)])
assert My_CellEV.shape == (E, V), "The decision matrix must have dimensions (E x V)."

weighted_My_CellEV = My_CellEV.copy()

print("Weighted_My_CellEV Array:")
print(weighted_My_CellEV)

# Sensitivity input value
EVT = weighted_My_CellEV
print("EVT Array:")
print(EVT)

# Modify interaction matrix for calculations
AA = -1 * VVT
np.fill_diagonal(AA, 1)
BB = EVT
C = AA.T
DD = np.sum(BB, axis=0)

myweightE = []
for i in range(E):
    D = EVT[i, :]  # Get the i-th row from EVT
    assert C.shape[0] == D.shape[0], "C and D must have compatible dimensions"
    myweightE.append(np.linalg.solve(C, D))  # Solve for weights

myweightE = np.array(myweightE)  # Convert to numpy array

# Convert to a pandas DataFrame and save to Excel
df = pd.DataFrame(myweightE)

file_path = r'C:\Users\ASUS\OneDrive\Desktop\weights.xlsx'  # Define a valid file path
df.to_excel(file_path, index=False)
print(f'The weight values have been saved to {file_path}')

# Logistic Normalization Function
def logistic_normalization(matrix):
    return 1 / (1 + np.exp(-matrix))

# Normalize the decision matrix
normalized_matrix = logistic_normalization(myweightE)
print('Normalized matrix:')
print(normalized_matrix)

df = pd.DataFrame(normalized_matrix)
file_path = r'C:\Users\ASUS\OneDrive\Desktop\normalized.xlsx'
df.to_excel(file_path, index=False)

# Weighted myweightE calculation
weighted_myweightE = subv * normalized_matrix
print('Weighted myweightE are:')
print(weighted_myweightE)

df = pd.DataFrame(weighted_myweightE)
file_path = r'C:\Users\ASUS\OneDrive\Desktop\weighted_normalized.xlsx'
df.to_excel(file_path, index=False)

print(f'The weight values have been saved to {file_path}')

# Compute event scores (sum of weighted myweightE per row)
event_scores = np.sum(weighted_myweightE, axis=1)
print('Event scores:')
print(event_scores)

# Rank the event scores in descending order
sorted_indices = np.argsort(event_scores)[::-1]  # Indices of sorted scores
sorted_scores = event_scores[sorted_indices]  # Sorted scores

# Create an array to hold ranks
ranks = np.zeros_like(event_scores, dtype=int)

# Assign ranks
for i, idx in enumerate(sorted_indices):
    ranks[idx] = i + 1  # Assign ranks starting from 1

print("Event rankings:", ranks)

# Display ranked scores with their original indices
ranked_scores_with_indices = list(zip(sorted_scores, ranks[sorted_indices]))
print("Ranked scores with their original ranks:", ranked_scores_with_indices)
