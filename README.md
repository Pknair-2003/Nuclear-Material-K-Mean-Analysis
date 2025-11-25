# Nuclear-Material-K-Mean-Analysis
K-means clustering of nuclear material samples using U-235 enrichment and Pu-239 impurity features. Includes EDA, visualization, elbow-method K selection, and centroid interpretation.

This project performs a simple K-means clustering analysis on nuclear material samples characterized by two measurements:

U-235 Enrichment (%): the percentage of Uranium-235 isotope in the material

Pu-239 Impurity (%): the percentage of Plutonium-239 isotope present as an impurity

These two values act as features, meaning numerical inputs that describe each sample. By applying K-means clustering, the code groups samples into categories based on similarity in these features. This approach is commonly used in data science to discover patterns in datasets without needing labeled examples.

The code does the following at a high level:

Loads the dataset into a Pandas DataFrame

Prints basic information about the dataset

Visualizes distributions of U-235 enrichment and Pu-239 impurity

Creates a scatter plot to observe natural grouping

Uses the elbow method to determine a good number of clusters

Applies K-means clustering with that optimal K

Plots the resulting clusters and their centroids

Prints summary statistics for each cluster

The dataset used in this project can be accessed here:
CSV File:
"C:\Users\pknai\Downloads\U_Pu_comp.csv"

