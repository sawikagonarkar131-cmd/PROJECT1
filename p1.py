# customer segmentation project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

print("--- first 5 rows of data ---")
print(df.head())

print("\nMissing values in dataset:")
print(df.isnull().sum())

X = df.iloc[:, [3, 4]].values

wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plotting the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS values')
plt.show()

print("Looking at the graph, the 'elbow' bends at 5. So we choose K=5.")


# 4. Training the K-Means model on the data with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_means = kmeans.fit_predict(X)


# 5. Visualizing the final clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='blue', label='Cluster 1 (Careful)')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='green', label='Cluster 2 (Standard)')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='orange', label='Cluster 3 (Target Group)')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=50, c='magenta', label='Cluster 5 (Sensible)')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='cyan', label='Cluster 4 (Spendthrift)')

# plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='*', label='Centroids')
plt.title('Customer Groups Based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 6. NEW: Testing with User Input
print("\n--- Model Testing with Custom Input ---")

# dictionary to map cluster numbers to human readable names
cluster_names = {
    0: "Cluster 1 (Careful - High Income, Low Spend)",
    1: "Cluster 2 (Standard - Mid Income, Mid Spend)",
    2: "Cluster 3 (Target Group - High Income, High Spend)",
    3: "Cluster 4 (Spendthrift - Low Income, High Spend)",
    4: "Cluster 5 (Sensible - Low Income, Low Spend)"
}

while True:
    print("\nEnter details to predict customer cluster (or type 'exit' to quit):")
    
    income_input = input("Enter Annual Income in k$ (e.g., 80): ")
    if income_input.lower() == 'exit':
        break
        
    spending_input = input("Enter Spending Score 1-100 (e.g., 75): ")
    if spending_input.lower() == 'exit':
        break
    
    try:
        # converting inputs to float/int
        user_income = float(income_input)
        user_spending = float(spending_input)
        
        # reshaping data because model expects a 2D array [[income, spending]]
        new_data = np.array([[user_income, user_spending]])
        
        # predicting the cluster
        predicted_cluster = kmeans.predict(new_data)[0]
        
        print(f"--> Result: This customer belongs to {cluster_names[predicted_cluster]}")
        
    except ValueError:
        print("Invalid input! Please enter numbers only.")

print("\nProgram closed. Thank you!")