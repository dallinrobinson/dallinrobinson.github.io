# Customer Segmentation Made Simple: K-Means in Python

Ever wonder how companies like Spotify recommend playlists or how stores know which coupons to send you? **Customer segmentation** is the key. In this tutorial, we’ll walk through how to use **K-Means clustering** in Python to group customers into meaningful segments.

By the end, you’ll know how to:

- Load and prepare a dataset of customers
- Run K-Means clustering with `scikit-learn`
- Visualize your results in a scatterplot
- Summarize the cluster centers in a tidy Markdown table

---

## Why Customer Segmentation Matters

Businesses rarely treat all customers the same. Instead, they group customers into _segments_ based on characteristics like age, spending, or habits. This allows for:

- **Personalized marketing** (ads that feel relevant, not random)
- **Smarter product design** (different products for different groups)
- **Better resource allocation** (focus on high-value customers)

And the cool part? You don’t need dozens of variables to see interesting patterns. Sometimes just **age** and **annual spending** are enough to show natural groupings.

---

## Step 1: Load the Data

Let’s start with a simple dataset of fictional customers:

```python
import pandas as pd

# Sample dataset
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6],
    "Age": [19, 35, 26, 27, 50, 45],
    "AnnualSpend": [15_000, 35_000, 18_000, 22_000, 55_000, 48_000]
}

df = pd.DataFrame(data)
print(df)
```

This gives us:

| CustomerID | Age | AnnualSpend |
| ---------- | --- | ----------- |
| 1          | 19  | 15000       |
| 2          | 35  | 35000       |
| 3          | 26  | 18000       |
| 4          | 27  | 22000       |
| 5          | 50  | 55000       |
| 6          | 45  | 48000       |

---

## Step 2: Run K-Means

We’ll use `scikit-learn`’s implementation of K-Means.

```python
from sklearn.cluster import KMeans

X = df[["Age", "AnnualSpend"]]

# Create a KMeans model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print(df)
```

Now each customer has a **cluster label** indicating their segment.

---

## Step 3: Visualize the Clusters

A scatterplot helps us see how the groups form.

```python
import matplotlib.pyplot as plt

plt.scatter(df["Age"], df["AnnualSpend"], c=df["Cluster"], cmap="viridis", s=100)
plt.xlabel("Age")
plt.ylabel("Annual Spending ($)")
plt.title("Customer Segments (K-Means)")
plt.show()
```

**Tip**: Each color shows a different segment. Try changing `n_clusters` to 3 or 4 and see how the picture changes.

_Example visualization:_
![K-Means clusters scatterplot](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png "Scatterplot showing two clusters")

_Alt text: Scatterplot with customers grouped into two clusters by spending and age._

---

## Step 4: Interpret the Cluster Centers

K-Means calculates **cluster centers** (the “average” customer in each group).

```python
centers = pd.DataFrame(kmeans.cluster_centers_, columns=["Age", "AnnualSpend"])
print(centers)
```

Here’s what the cluster centroids might look like:

| Cluster | Avg Age | Avg Annual Spend |
| ------- | ------- | ---------------- |
| 0       | 25.5    | 19,000           |
| 1       | 47.5    | 51,500           |

This tells a clear story:

- **Cluster 0** → Younger customers with lower spending
- **Cluster 1** → Older customers with higher spending

---

## Wrapping Up

K-Means gives us a **quick, data-driven way** to segment customers. Even with just two variables, we can uncover meaningful groups that businesses can target differently.

---

## Call to Action

Now that you’ve seen how simple customer segmentation can be, try it yourself!

- Use a different dataset—like survey responses, your Spotify listening habits, or even your classmates’ study hours.
- Experiment with different numbers of clusters (`n_clusters`).
- Share your visualization with a peer and explain what the groups might mean.

For further reading, check out the [scikit-learn K-Means documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means).

---
