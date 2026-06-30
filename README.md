# Customer Segmentation using K-Means

This is a simple machine learning project that groups mall customers based on their annual income and spending score.

The project uses the K-Means clustering algorithm to find different types of customers. It also lets you enter your own income and spending score to check which customer group a person may belong to.

## What This Project Does

- Reads customer data from `Mall_Customers.csv`
- Checks the first few rows of the dataset
- Checks if there are any missing values
- Uses annual income and spending score for clustering
- Uses the elbow method to choose the best number of clusters
- Trains a K-Means model with 5 clusters
- Shows a graph of the customer groups
- Allows custom input to predict a customer's cluster

## Dataset

The script expects a file named:

```text
Mall_Customers.csv
```

The main columns used in this project are:

- Annual Income
- Spending Score

These two values are used to group customers into different segments.

## Customer Groups

The model divides customers into 5 groups:

- Careful: High income, low spending
- Standard: Medium income, medium spending
- Target Group: High income, high spending
- Spendthrift: Low income, high spending
- Sensible: Low income, low spending

## Requirements

Install the required Python libraries before running the project:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## How To Run

Make sure `p1.py` and `Mall_Customers.csv` are in the same folder.

Then run:

```bash
python p1.py
```

The program will show:

1. The first 5 rows of the dataset
2. Missing value information
3. The elbow method graph
4. The final customer cluster graph
5. A custom input section where you can test new customer details

## Example Input

```text
Enter Annual Income in k$ (e.g., 80): 80
Enter Spending Score 1-100 (e.g., 75): 75
```

Example output:

```text
Result: This customer belongs to Cluster 3 (Target Group - High Income, High Spend)
```

## How To Exit

When the program asks for input, type:

```text
exit
```

This will close the program.

## About

This project is useful for understanding basic customer segmentation. It shows how businesses can group customers by their income and spending behavior, which can help with marketing and business decisions.
[Untitled design (1).pptx](https://github.com/user-attachments/files/29496425/Untitled.design.1.pptx)
