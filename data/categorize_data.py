import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/cleaned_data.csv")

# use attendance rates as clustering features
# student counts are kept separately as descriptors - mixing them with rates skews clusters
ATTENDANCE_COLS = [
    "2021_2022_attendance_rate",
    "2020_2021_attendance_rate",
    "2019_2020_attendance_rate",
]

DESCRIPTOR_COLS = [
    "2021_2022_student_count",
    "2020_2021_student_count",
    "2019_2020_student_count",
]

X = df[ATTENDANCE_COLS].values

# PCA to decorrelate year-over-year attendance columns before clustering
pca = PCA(random_state=42)
X_pca = pca.fit_transform(X)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = np.searchsorted(cumvar, 0.95) + 1
print(f"PCA: keeping {n_components} component(s) ({cumvar[n_components-1]*100:.1f}% variance explained)")

X_reduced = X_pca[:, :n_components]

# elbow + silhouette to pick k
K_RANGE = range(2, 9)
inertias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_reduced)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_reduced, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="orange")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/elbow_silhouette.png", dpi=150)
plt.close()

best_k = list(K_RANGE)[np.argmax(silhouettes)]
print(f"Optimal k: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_reduced)

cluster_profiles = df.groupby("cluster")[ATTENDANCE_COLS + DESCRIPTOR_COLS].mean()
avg_attendance = cluster_profiles[ATTENDANCE_COLS].mean(axis=1)
rank = avg_attendance.rank(ascending=False).astype(int)


def make_label(r, k):
    if r == 1:
        return f"cluster_{r}_high_attendance"
    elif r == k:
        return f"cluster_{r}_low_attendance"
    else:
        return f"cluster_{r}_mid_attendance"


attendance_labels = {idx: make_label(rank[idx], best_k) for idx in avg_attendance.index}
df["cluster_label"] = df["cluster"].map(attendance_labels)

print("\nCluster profiles - attendance rates:")
print(cluster_profiles[ATTENDANCE_COLS].round(4))

print("\nCluster profiles - student counts:")
print(cluster_profiles[DESCRIPTOR_COLS].round(4))

print("\nCluster sizes:")
print(df["cluster_label"].value_counts())

category_dist = (
    df.groupby(["cluster_label", "category"])
    .size()
    .reset_index(name="count")
    .sort_values(["cluster_label", "count"], ascending=[True, False])
)
print("\nCategory distribution per cluster:")
print(category_dist.to_string(index=False))

# scatter using first 2 PCA components
coords = X_pca[:, :2]
var_explained = pca.explained_variance_ratio_[:2]

plt.figure(figsize=(9, 6))
for cluster_id, label in attendance_labels.items():
    mask = df["cluster"] == cluster_id
    plt.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.5, s=18)

plt.title("K-Means Clusters (PCA 2D projection)")
plt.xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/clusters_pca.png", dpi=150)
plt.close()

out_path = f"{OUTPUT_DIR}/categorized_data.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
