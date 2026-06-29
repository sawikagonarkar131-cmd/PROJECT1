# customer segmentation project

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Mall_Customers.csv"
ELBOW_CHART = BASE_DIR / "elbow_method.png"
CLUSTER_CHART = BASE_DIR / "customer_clusters.png"


def load_customer_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print("--- first 5 rows of data ---")
    print(df.head())

    print("\nMissing values in dataset:")
    print(df.isnull().sum())
    return df


def build_elbow_chart(customer_features):
    wcss = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(customer_features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o", color="red")
    plt.title("The Elbow Method for Optimal K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS values")
    plt.tight_layout()
    plt.savefig(ELBOW_CHART)
    plt.close()

    print(f"\nElbow chart saved as: {ELBOW_CHART.name}")
    print("Looking at the graph, the elbow bends at 5. So we choose K=5.")


def train_model(customer_features):
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
    cluster_numbers = kmeans.fit_predict(customer_features)
    return kmeans, cluster_numbers


def build_cluster_chart(customer_features, cluster_numbers, kmeans):
    cluster_styles = {
        0: ("blue", "Cluster 1 (Careful)"),
        1: ("green", "Cluster 2 (Standard)"),
        2: ("orange", "Cluster 3 (Target Group)"),
        3: ("cyan", "Cluster 4 (Spendthrift)"),
        4: ("magenta", "Cluster 5 (Sensible)"),
    }

    plt.figure(figsize=(10, 7))

    for cluster_number, (color, label) in cluster_styles.items():
        plt.scatter(
            customer_features[cluster_numbers == cluster_number, 0],
            customer_features[cluster_numbers == cluster_number, 1],
            s=50,
            c=color,
            label=label,
        )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c="black",
        marker="*",
        label="Centroids",
    )
    plt.title("Customer Groups Based on Income and Spending")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CLUSTER_CHART)
    plt.close()

    print(f"Cluster chart saved as: {CLUSTER_CHART.name}")


def run_prediction_loop(kmeans):
    cluster_names = {
        0: "Cluster 1 (Careful - High Income, Low Spend)",
        1: "Cluster 2 (Standard - Mid Income, Mid Spend)",
        2: "Cluster 3 (Target Group - High Income, High Spend)",
        3: "Cluster 4 (Spendthrift - Low Income, High Spend)",
        4: "Cluster 5 (Sensible - Low Income, Low Spend)",
    }

    print("\n--- Model Testing with Custom Input ---")

    while True:
        print("\nEnter details to predict customer cluster (or type 'exit' to quit):")

        income_input = input("Enter Annual Income in k$ (e.g., 80): ").strip()
        if income_input.lower() == "exit":
            break

        spending_input = input("Enter Spending Score 1-100 (e.g., 75): ").strip()
        if spending_input.lower() == "exit":
            break

        try:
            user_income = float(income_input)
            user_spending = float(spending_input)
        except ValueError:
            print("Invalid input! Please enter numbers only.")
            continue

        if user_spending < 1 or user_spending > 100:
            print("Invalid spending score! Please enter a value from 1 to 100.")
            continue

        new_data = np.array([[user_income, user_spending]])
        predicted_cluster = kmeans.predict(new_data)[0]

        print(f"--> Result: This customer belongs to {cluster_names[predicted_cluster]}")

    print("\nProgram closed. Thank you!")


def main():
    df = load_customer_data()
    customer_features = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    build_elbow_chart(customer_features)
    kmeans, cluster_numbers = train_model(customer_features)
    build_cluster_chart(customer_features, cluster_numbers, kmeans)
    run_prediction_loop(kmeans)


if __name__ == "__main__":
    main()
